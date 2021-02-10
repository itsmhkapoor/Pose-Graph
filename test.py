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

# from models.superglueForTest import SuperGlue
# from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

def ransac8pF(x1, x2, thresh):
    iter = 1000
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    # homogenize
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
    c = np.transpose(x)
    c1 = np.append(c, np.zeros([3,3]), axis=0)
    c2 = np.append(np.zeros([3,3]), c, axis=0)
    C = np.append(c1,c2, axis=1)
    # print(C)
    d_1 = np.transpose(y[0, :])
    d_2 = np.transpose(y[1, :])

    d = np.append(np.reshape(d_1,[3,-1]), np.reshape(d_2,[3,-1]), axis=0)
    # print(d)
    z = np.linalg.solve(C, d)
    # print(z)
    A = np.array([[z[0][0],z[1][0]],[z[3][0],z[4][0]]])
    # print(A.shape)
    t = np.array([[z[2][0]],[z[5][0]]])
    # print(t.shape)
    u, sd, vh = np.linalg.svd(A, full_matrices=True)
    R = np.matmul(u , np.matmul(np.array([[1, 0], [0, np.linalg.det(np.matmul(u,vh))]]) , vh))
    return R, t

def transform_graph(keypoints, num_keyframes, R, t):
    keypoints = keypoints[0:2,:]
    dist_array = np.transpose([t] * num_keyframes).reshape(2,-1)
    new_keypoints = np.matmul(R, keypoints)+dist_array
    return new_keypoints

def compute_matches(kp1, kp2):
    threshold = 0.2
    distances = cdist(kp1, kp2)
    mask_distances = ma.masked_greater(distances, threshold)
    # print(mask_distances)
    for i in range(distances.shape[0]):
        mask_distances[i, :] = ma.masked_greater(mask_distances[i, :], np.amin(mask_distances[i, :]))
    for i in range(distances.shape[1]):
        mask_distances[:, i] = ma.masked_greater(mask_distances[:, i], np.amin(mask_distances[:, i]))
    row, col = np.where(mask_distances.mask == False)
    matches = np.append(row.reshape(-1, 1), col.reshape(-1, 1), axis=1)
    return matches

def gen_matrix(connection, num_keyframes):
    matrix_self = np.zeros([num_keyframes, num_keyframes])

    for index in range(num_keyframes):
        child = np.asarray(connection[index][0],dtype=int).reshape(1,-1)
        parent = np.asarray(connection[index][1], dtype=int).reshape(1,-1)
        conn_indices = np.append(parent, child)
        conn_indices = [x for x in conn_indices if x < num_keyframes] # check if parent not out of bounds
        # print(parent)
        matrix_self[index,conn_indices] = 1./(2*len(conn_indices))
    return matrix_self

def mod_pose_graph(keypoints, num_keyframes, angle, dist_x, dist_y, dist_z):

    dist = np.array([dist_x,dist_y,dist_z])
    dist_array = np.transpose([dist] * num_keyframes)
    rotation_matrix = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    new_keypoints = np.matmul(rotation_matrix,np.transpose(keypoints))+dist_array
    return np.transpose(new_keypoints)

def extract_descriptor(base_folder):
    desc_path = f'{base_folder}/descriptors/'
    desc_final = np.empty([256,1])
    for f in fnmatch.filter(natsorted(os.listdir(desc_path)),'descriptor*'):
        desc_temp = np.load(f'{desc_path}{f}')
        desc_temp = np.average(desc_temp, axis=1)
        desc_final = np.append(desc_final, desc_temp.reshape(-1,1), axis=1)
    desc_final = np.delete(desc_final, 0, axis=1)
    return desc_final

def extract_score(base_folder):
    score_path = f'{base_folder}/descriptors/'
    score_final = np.empty([1,1])
    for f in fnmatch.filter(natsorted(os.listdir(score_path)),'score*'):
        score_temp = np.load(f'{score_path}{f}')
        score_temp = np.average(score_temp, axis=1)
        score_final = np.append(score_final, score_temp.reshape(-1,1), axis=1)
    score_final = np.delete(score_final, 0, axis=1)
    return score_final

class MapLocationExtractor():

    def __init__(self, msg_path):
        self.bin_fn = msg_path
        # self.video_path = '/home/mkapoor/slam_files/openvslam/build/aist_living_lab_1/video.mp4'
        self.base_dir = '/home/mkapoor/slam_files/openvslam/build/'

    def forward(self, plot=False):

        # Read file as binary and unpack data using MessagePack library
        with open(self.bin_fn, "rb") as f:
            data = msgpack.unpackb(f.read(), use_list=False, raw=False)

        # The point data is tagged "landmarks"

        # Create files with keypoints for keyframes
        # point_cloud = data["landmarks"]
        # for id, para in point_cloud.items():  # accessing keys
        #     print(para, end=',')
        # points_keyframe(point_cloud)
        # print(max_kf,min_kf)

        key_frames = data["keyframes"]

        print("Point cloud has {} keyframes.".format(len(key_frames)))

        key_frame = {int(k): v for k, v in key_frames.items()}

        if plot:
            x = []
            y = []
            z = []
            t = []
            for key in sorted(key_frame.keys()):
                point = key_frame[key]
                trans_cw = np.asarray(point["trans_cw"])
                rot_cw = np.asarray(point["rot_cw"])

                rigid_cw = RigidTransform(rot_cw, trans_cw)

                pos = np.matmul(rigid_cw.rotation, trans_cw)

                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                t.append(float(point["ts"]))

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.scatter(x, z)
            plt.show()

            plt.ylabel('Height')
            plt.xlabel('Time')
            plt.scatter(x=t, y=y)
            print(t)

            # # new a figure and set it into 3d
            fig = plt.figure()

            plt.show()


        else:

                count = 0
                connections = []
                keyfrm_center = np.empty([3, 1])
                for key in sorted(key_frame.keys()):
                    point = key_frame[key]
                    # for id, para in point.items():  # accessing keys
                    #     print(id, end=',')

                    # position capture
                    # print("Child", point["span_children"])
                    # print("Parent", point["span_parent"])
                    child_parent = [point["span_children"], point["span_parent"]]
                    connections.append(child_parent)
                    trans_cw = np.asarray(point["trans_cw"])
                    rot_cw = np.asarray(point["rot_cw"])

                    rigid_cw = RigidTransform(rot_cw, trans_cw)

                    pos = np.matmul(rigid_cw.rotation, trans_cw)
                    keyfrm_center = np.append(keyfrm_center, pos.reshape(-1, 1), axis=1)

        #             # image capture
        #             success, image = vidcap.read()
        #
        #             if not success:
        #                 print("capture failed")
        #             else:
        #                 cv2.imwrite(os.path.join(video_name, str(count) +".jpg"), image)
        #
        #             count+=1
        #
        # print(connections)

        keydata = {}
        keyfrm_center = np.delete(keyfrm_center, 0 , axis=1)
        keydata['keyframes'] = keyfrm_center
        keydata['connections'] = connections
        print("Finished")
        # print(np.shape(keyfrm_center))
        return keydata
        # print((keyfrm_center))
        # np.save('keyframes.npy', keyfrm_center)

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


if __name__ == '__main__':
    ####################### Parse inputs #############################
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_glob', type=str, default='*.png',
                        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--nms_dist', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: False)')

    ########## Superglue parse ################
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.0,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)
    ########################### Parse ends ##################################
    # dataset_path = '/home/mkapoor/slam_files/openvslam'
    # msg_path = '/home/mkapoor/slam_files/openvslam/build/aist_living_lab_1_map.msg'
    # keyframe_obj = MapLocationExtractor(msg_path)
    # keyframe_data = keyframe_obj.forward()
    # keyframes = keyframe_data['keyframes']
    # # keyframe is 3xN matrix
    # # connections is union of parent and child
    # self_prob_matrix = gen_matrix(keyframe_data['connections'], np.shape(keyframes)[1])
    # print(self_prob_matrix)
    #################### SuperPoint ########################################
    # vs = VideoStreamer(dataset_path, camid = 0, height = 480, width = 640, skip = 1, opt.img_glob)
    # print('==> Loading pre-trained SuperPoint network.')
    # # This class runs the SuperPoint network and processes its outputs.
    # fe = SuperPointFrontend(weights_path=opt.weights_path,
    #                         nms_dist=opt.nms_dist,
    #                         conf_thresh=opt.conf_thresh,
    #                         nn_thresh=opt.nn_thresh,
    #                         cuda=opt.cuda)
    # print('==> Successfully loaded pre-trained SuperPoint network.')
    #
    # print('==> Running Demo.')
    # desc_final = np.empty([256, 1])
    # scores = np.empty([1, 1])
    #
    # while True:
    #
    #     start = time.time()
    #
    #     # Get a new image.
    #     img, status = vs.next_frame()
    #     if status is False:
    #         break
    #
    #     # Get points and descriptors.
    #     start1 = time.time()
    #     pts, desc, heatmap = fe.run(img)
    #     end1 = time.time()
    #
    #     # Add code for average pool of descriptor and scores.
    #     confidence = np.delete(pts, [0, 1], axis=0)
    #
    #     indices = np.random.choice(confidence.shape[1], size=10, replace=False)
    #
    #     confidence = confidence[:, indices]
    #     pool_score = np.average(confidence, axis=1)
    #     scores = np.append(scores, pool_score.reshape(-1, 1), axis=1)
    #
    #     desc = desc[:, indices]
    #     pool_desc = np.average(desc, axis=1)
    #     desc_final = np.append(desc_final, pool_desc.reshape(-1, 1), axis=1)
    #
    #     end = time.time()
    #     net_t = (1. / float(end1 - start))
    #     total_t = (1. / float(end - start))
    #
    # desc_final = np.delete(desc_final, 0, axis=1)
    # scores = np.delete(scores, 0, axis=1)
    # print('==> Finshed SuperPoint.')


    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    model_path = "/scratch_net/munzekonza/mkapoor/model_ckpt/seq_1_weighted_ransac/model_rand_512_L_0.05_epoch_100.pth"
    print("Model loaded from: ",model_path)
    # matching = SuperGlue(config.get('superglue', {}))
    matching = torch.load(model_path)
    matching.eval()

    data_path = '/srv/beegfs-benderdata/scratch/posegraph/data/cold/freiburg/'
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

            # pred using superglue
            pred = matching(data)

            # print matches

            matches = pred['matches0'].cpu().numpy()
            confidence = pred['matching_scores0'].cpu().numpy()
            # scores = pred['scores'].cpu().numpy()
            valid = matches > -1
            # mkpts0 = np.where(valid)
            mkpts0 = kf1_mod[valid]

            mkpts1 = kf2_mod[matches[valid]]
            # Run RANSAC

            R = pred['rotation'].cpu().numpy()
            t = pred['translation'].cpu().numpy()
            pred_angle = math.acos(R[0][0])
            ang_error = pred_angle - angle
            angle_error.append(abs(ang_error))
            dist_error = np.linalg.norm(np.array([t[0][0] - dist_x, t[1][0] - dist_y]))
            distance_error.append(dist_error)
            plt.ylabel('y [m]')
            plt.xlabel('x [m]')
            plt.plot(mkpts0[:, 0], mkpts0[:, 1], 'ro')
            plt.plot(mkpts1[:, 0], mkpts1[:, 1], 'bo')
            for i in range(mkpts0.shape[0]):
                plt.plot([mkpts0[i, 0], mkpts1[i, 0]], [mkpts0[i, 1], mkpts1[i, 1]], color=cmap(0))
            plt.savefig('/scratch_net/munzekonza/mkapoor/results/seq_1_weighted_ransac/0.05/30.png')


    # np.save(f'/scratch_net/munzekonza/mkapoor/results/seq_1_weighted_ransac/0.2/dist_error_{min_angle}_{max_angle}.npy',distance_error)
    # np.save(f'/scratch_net/munzekonza/mkapoor/results/seq_1_weighted_ransac/0.2/angle_error_{min_angle}_{max_angle}.npy', angle_error)
    #
    # print(statistics.mean(distance_error))
    # print(statistics.mean(angle_error))







    

