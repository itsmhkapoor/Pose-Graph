from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import msgpack
import json
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import fnmatch
from scipy.spatial.distance import cdist
import numpy.ma as ma
import statistics
import os
import cv2
import imageio
from natsort import natsorted


from models.matcher import Matcher

torch.set_grad_enabled(False)

if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

def ransacRigidH(x1, x2, thresh):
    """For RANSAC post-processing. Threshold is chosen to be 0.5m for indoor setting"""
    iter = 1000
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    # homogenize (to set angle coordinate to 1)
    x1[2, :] = np.ones([1, x1.shape[1]])
    x2[2, :] = np.ones([1, x2.shape[1]])

    num_pts = x1.shape[1]  # Total number of points

    best_num_inliers = 0
    best_inliers = []
    best_R = 0
    best_t = 0
    for i in range(iter):
        indices = np.random.choice(num_pts, size=3, replace=False)
        R, t = rigidH(x1[:, indices], x2[:, indices])

        x2_hat = transform_graph(x1, num_pts, R, t)
        dist_error = x2_hat-x2[0:2,:]
        ssd = np.linalg.norm(dist_error, axis=0)

        inliers = (ssd<thresh)

        num_inliers = np.sum(inliers)
        if (num_inliers > best_num_inliers):
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_R = R
            best_t = t
            best_error = ssd


    return best_inliers, best_R, best_t

def rigidH(x, y):
    """Compute rigid homographic transformation from posegraph x to y"""
    c = np.transpose(x)
    c1 = np.append(c, np.zeros([3,3]), axis=0)
    c2 = np.append(np.zeros([3,3]), c, axis=0)
    C = np.append(c1,c2, axis=1)
    
    d_1 = np.transpose(y[0, :])
    d_2 = np.transpose(y[1, :])

    d = np.append(np.reshape(d_1,[3,-1]), np.reshape(d_2,[3,-1]), axis=0)
    
    z = np.linalg.solve(C, d)
    
    A = np.array([[z[0][0],z[1][0]],[z[3][0],z[4][0]]])
    
    t = np.array([[z[2][0]],[z[5][0]]])
    
    u, sd, vh = np.linalg.svd(A, full_matrices=True)
    R = np.matmul(u , np.matmul(np.array([[1, 0], [0, np.linalg.det(np.matmul(u,vh))]]) , vh))
    return R, t

def transform_graph(keypoints, num_keyframes, R, t):
    """Transform a posegraph with the applied Rotation (R) and translation (t)"""
    keypoints = keypoints[0:2,:]
    dist_array = np.transpose([t] * num_keyframes).reshape(2,-1)
    new_keypoints = np.matmul(R, keypoints)+dist_array
    return new_keypoints

def compute_matches(kp1, kp2):
    """Compute ground truth one-to-one matches between two localized posegraphs"""
    threshold = 0.2
    distances = cdist(kp1, kp2)
    mask_distances = ma.masked_greater(distances, threshold)
    
    for i in range(distances.shape[0]):
        mask_distances[i, :] = ma.masked_greater(mask_distances[i, :], np.amin(mask_distances[i, :]))
    for i in range(distances.shape[1]):
        mask_distances[:, i] = ma.masked_greater(mask_distances[:, i], np.amin(mask_distances[:, i]))
    row, col = np.where(mask_distances.mask == False)
    matches = np.append(row.reshape(-1, 1), col.reshape(-1, 1), axis=1)
    return matches

def mod_pose_graph(keypoints, num_keyframes, angle, dist_x, dist_y, dist_z):
    """Modify the given pose graph with ground truth transformation for training"""
    dist = np.array([dist_x,dist_y,dist_z])
    dist_array = np.transpose([dist] * num_keyframes)
    rotation_matrix = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    new_keypoints = np.matmul(rotation_matrix,np.transpose(keypoints))+dist_array
    return np.transpose(new_keypoints)

def extract_descriptor(base_folder):
    """Used to generate mean pooled image descriptors"""
    desc_path = f'{base_folder}/descriptors/'
    desc_final = np.empty([256,1])
    for f in fnmatch.filter(natsorted(os.listdir(desc_path)),'descriptor*'):
        desc_temp = np.load(f'{desc_path}{f}')
        desc_temp = np.average(desc_temp, axis=1)
        desc_final = np.append(desc_final, desc_temp.reshape(-1,1), axis=1)
    desc_final = np.delete(desc_final, 0, axis=1)
    return desc_final

def extract_score(base_folder):
    """Used to generate mean pooled image scores"""
    score_path = f'{base_folder}/descriptors/'
    score_final = np.empty([1,1])
    for f in fnmatch.filter(natsorted(os.listdir(score_path)),'score*'):
        score_temp = np.load(f'{score_path}{f}')
        score_temp = np.average(score_temp, axis=1)
        score_final = np.append(score_final, score_temp.reshape(-1,1), axis=1)
    score_final = np.delete(score_final, 0, axis=1)
    return score_final


if __name__ == '__main__':
    ####################### Parse inputs #############################
    parser = argparse.ArgumentParser(
        description='Pose Graph Matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.0,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    
    """Specify model path here"""
    model_path = "path/to/model.pth"
    
    config = {
        'matcher': {
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matcher(config.get('matcher', {}))
    checkpoint = torch.load(model_path) # load model checkpoint
    matching.load_state_dict(checkpoint['model_state_dict'])
    matching.eval()
    print("Model loaded from: ",model_path)

    data_path = 'path/to/test/dataset'
    files = []
    files += [data_path + f for f in natsorted(os.listdir(data_path))]
    files = files[-7:-1]
    angle_error = []
    distance_error = []
    
    min_angle = (0*math.pi)/180
    max_angle = (10*math.pi)/180
    fig = plt.figure()
    cmap = plt.get_cmap('gnuplot')
    for row in range(1):
        idx1 = np.random.randint(6)
        test = [0,1,2,3,4,5]
        test.remove(idx1)
        idx2 = np.random.choice(test)
        # load keypoints and descriptors from npy files
        base_folder1 = files[idx1]
        base_folder2 = files[idx2]
        kf1 = np.load(f'{base_folder1}/std_cam/keypoints.npy')
        kf2 = np.load(f'{base_folder2}/std_cam/keypoints.npy')
        kf1 = np.delete(kf1, 0, axis=1)
        kf2 = np.delete(kf2, 0, axis=1)
        desc_final1 = extract_descriptor(base_folder1)
        desc_final2 = extract_descriptor(base_folder2)
        scores1 = extract_score(base_folder1)
        scores2 = extract_score(base_folder2)

        for col in range(1):
            # Sub graph section
            indices1 = np.random.choice(kf1.shape[0], size=512, replace=False)
            indices2 = np.random.choice(kf2.shape[0], size=512, replace=False)
            kf1_mod = kf1[indices1, :]
            kf2_mod = kf2[indices2, :]
            desc_final1_mod = desc_final1[:, indices1]
            desc_final2_mod = desc_final2[:, indices2]
            scores1_mod = scores1[:, indices1]
            scores2_mod = scores2[:, indices2]

            # matches for sub graph
            all_matches = compute_matches(kf1_mod, kf2_mod)
            angle = (30*math.pi)/180
            dist_x = np.random.uniform(0, 1)
            dist_y = np.random.uniform(0, 1)
            dist_z = np.random.normal(0, 0.0001)

            # kf1_mod = mod_pose_graph(kf1_mod, kf1_mod.shape[0], angle, dist_x, dist_y, dist_z)
            kf2_mod = mod_pose_graph(kf2_mod, kf2.shape[0], angle, dist_x, dist_y, dist_z)

            data = {
                    'keypoints0': torch.from_numpy(kf1_mod).cuda(),
                    'keypoints1': torch.from_numpy(kf2_mod).cuda(),
                    'descriptors0': torch.from_numpy(desc_final1_mod).cuda(),
                    'descriptors1': torch.from_numpy(desc_final2_mod).cuda(),
                    'scores0': torch.from_numpy(scores1_mod).cuda(),
                    'scores1': torch.from_numpy(scores2_mod).cuda(),
                    'all_matches': torch.from_numpy(all_matches).cuda()

                }

            # pred using matcher
            pred = matching(data)

            matches = pred['matches0'].cpu().numpy()
            confidence = pred['matching_scores0'].cpu().numpy()

            valid = matches > -1

            mkpts0 = kf1_mod[valid]

            mkpts1 = kf2_mod[matches[valid]]
            rms_applied = compute_rms(mkpts0, mkpts1, angle, dist_x, dist_y)

            n_matches = mkpts0.shape[0]

            match_ratio = all_matches.shape[0] / 512

            # Run RANSAC
            inliers, _, _ = ransacRigidH(mkpts0, mkpts1, 0.5)

            val_mkp1 = mkpts0[inliers]
            val_mkp2 = mkpts1[inliers]
            R, t = fit_model(val_mkp1, val_mkp2)

            pred_angle = math.acos(R[0][0])
            ang_error = abs(pred_angle - angle)

            dist_error = np.linalg.norm(np.array([t[0][0] - dist_x, t[1][0] - dist_y]))

            rms_pred = compute_rms(mkpts0, mkpts1, pred_angle, t[0][0], t[1][0])

            """Accuracy Section"""
            row_id = np.where(valid == True)[0]
            col_id = np.asarray(matches[valid])
            pred_matches = np.append(row_id.reshape(-1, 1), col_id.reshape(-1, 1), axis=1)
            count = 0
            for index1 in range(n_matches):
                for index2 in range(all_matches.shape[0]):
                    if (np.all(pred_matches[index1] == all_matches[index2])):
                        count = count + 1
            accuracy = count / all_matches.shape[0]

            values = [rms_applied, n_matches, match_ratio, ang_error, dist_error, rms_pred, accuracy]
            data_file.append(values)
            
            # plot
            plt.ylabel('y [m]')
            plt.xlabel('x [m]')
            plt.plot(mkpts0[:, 0], mkpts0[:, 1], 'ro')
            plt.plot(mkpts1[:, 0], mkpts1[:, 1], 'bo')
            for i in range(mkpts0.shape[0]):
                plt.plot([mkpts0[i, 0], mkpts1[i, 0]], [mkpts0[i, 1], mkpts1[i, 1]], color='black')
            plt.savefig('../final_results/seq_1_weighted_ransac/0.2/30_before.png') # matching threshold is 0.2
            fig.clf()
            plt.ylabel('y [m]')
            plt.xlabel('x [m]')
            plt.plot(val_mkp1[:, 0], val_mkp1[:, 1], 'ro')
            plt.plot(val_mkp2[:, 0], val_mkp2[:, 1], 'bo')
            for i in range(val_mkp1.shape[0]):
                plt.plot([val_mkp1[i, 0], val_mkp2[i, 0]], [val_mkp1[i, 1], val_mkp2[i, 1]], color='black')
            plt.savefig('../final_results/seq_1_weighted_ransac/0.2/30_after.png')








    

