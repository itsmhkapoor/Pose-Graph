"""Code adapted from SuperGlue
    https://github.com/magicleap/SuperGluePretrainedNetwork
    and training script from https://github.com/HeatherJiaZG/SuperGlue-pytorch

    Modified by: Mohit Kapoor"""

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

def transform_graph(keypoints, num_keyframes, R, t):
    """Transform a posegraph with Rotation (R) and translation (t)"""
    keypoints = keypoints[0:2,:]
    dist_array = t.repeat(1,num_keyframes)
    new_keypoints = torch.mm(R.cuda(), keypoints)+dist_array.cuda()
    return new_keypoints

def rigidH(x, y, w):
    """Compute rigid homographic transformation from posegraph x to y"""
    xt = torch.transpose(x,0,1)
    c = w * xt
    num_pts = x.size()[1]
    c1 = torch.cat((c, torch.zeros(num_pts,3, device=torch.device('cuda'), dtype=torch.float64)), 1)
    c2 = torch.cat((torch.zeros(num_pts,3, device=torch.device('cuda'), dtype=torch.float64), c), 1)
    C = torch.cat((c1,c2), dim=0)

    d_1 = torch.reshape(y[0, :],(-1,1))
    d_2 = torch.reshape(y[1, :],(-1,1))
    d = torch.cat((d_1, d_2), dim=0)

    z, _ = torch.lstsq(d, C)

    A = torch.tensor([[z[0][0],z[1][0]],[z[3][0],z[4][0]]])
    t = torch.tensor([[z[2][0]],[z[5][0]]])

    u, sd, v = torch.svd(A, some=False)
    R = torch.mm(u , torch.mm(torch.tensor([[1, 0], [0, torch.det(torch.mm(u, v.t()))]]) , v.t()))
    return R, t

def ransacRigidH(x1, x2, w):
    """For RANSAC to compute weighted mean error"""
    x1 = torch.transpose(x1,0,1)
    x2 = torch.transpose(x2,0,1)
    num_pts = x1.size()[1]  # Total number of points
    # homogenize
    x1[2, :] = torch.ones(1, num_pts)
    x2[2, :] = torch.ones(1, num_pts)
    R, t = rigidH(x1, x2, w)
    x2_hat = transform_graph(x1, num_pts, R, t)
    dist_error = x2_hat-x2[0:2,:]
    w_ssd = torch.norm(dist_error, dim=0) * torch.reshape(w, (-1,))
    mean_ssd = torch.mean(w_ssd)

    return mean_ssd, R, t

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeyframeEncoder(nn.Module):
    """ Encoding of Keyframes using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        # inputs = [kpts, scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)

    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class Matcher(nn.Module):
    """

    Given two sets of image (keyframe) locations and descriptors, we determine the
    correspondences by:
      1. Keyframe Encoding 
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keyframe_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeyframeEncoder(
            self.config['descriptor_dim'], self.config['keyframe_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)


    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
        kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()
        dscores0, dscores1 = data['scores0'].double(), data['scores1'].double()

        kpts0 = torch.reshape(kpts0, (1, -1, 3))
        kpts1 = torch.reshape(kpts1, (1, -1, 3))
        desc0 = torch.reshape(desc0, (1, 256, -1))
        desc1 = torch.reshape(desc1, (1, 256, -1))
        dscores0 = torch.reshape(dscores0, (1, -1))
        dscores1 = torch.reshape(dscores1, (1, -1))
        print("SuperGlue begins")

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        all_matches = data['all_matches']
        all_matches = torch.reshape(all_matches, (1, -1, 2))

        # Keyframe encoder
        desc0 = desc0 + self.kenc(kpts0, dscores0)
        desc1 = desc1 + self.kenc(kpts1, dscores1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches 
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid0 = mutual0
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # RANSAC while training
        valid0 = indices0 > -1
        mkpts0 = kpts0[valid0]
        mkpts1 = kpts1[0, indices0[valid0], :] # N'x3 size
        weights = mscores0[valid0].unsqueeze(1)

        w_mean_ssd, R, t = ransacRigidH(mkpts0, mkpts1, weights)

        loss = []
        for i in range(len(all_matches[0])):
            x = all_matches[0][i][0]
            y = all_matches[0][i][1]
            loss.append(-torch.log( scores[0][x][y].exp() ))
        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        loss_mean = torch.mean(torch.stack(loss)) + 0.05*w_mean_ssd # Modified Loss function where lambda is 0.05
        loss_mean = torch.reshape(loss_mean, (1, -1))

        return {
            'matches0': indices0[0], # use -1 for invalid match
            'matches1': indices1[0], # use -1 for invalid match
            'matching_scores0': mscores0[0],
            'matching_scores1': mscores1[0],
            'loss': loss_mean[0],
            'rotation': R,
            'translation': t,
            'skip_train': False
        }


