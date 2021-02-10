import numpy as np
import numpy.ma as ma
import torch
import os
import cv2
import math
import datetime
import msgpack
import glob
import time
import fnmatch
from natsort import natsorted

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
# from autolab_core import RigidTransform
from mpl_toolkits.mplot3d import axes3d

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

def mod_pose_graph(keypoints, num_keyframes):
    angle = np.random.uniform(0,2*math.pi)
    dist_x = np.random.uniform(0,1)
    dist_y = np.random.uniform(0, 1)
    dist_z = np.random.normal(0,0.0001)
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

def compute_matches(kp1, kp2):
    threshold = 0.2
    distances = cdist(kp1, kp2)
    distances[distances > threshold] == np.nan
    nn12 = np.nanargmin(distances, axis=1)
    nn21 = np.nanargmin(distances, axis=0)
    idx = np.arange(kp1.shape[0])
    valid_matches = (idx == nn21[nn12])
    matches = np.append(idx[valid_matches].reshape(-1, 1), nn12[valid_matches].reshape(-1, 1), axis=1)
    return matches

# class MapLocationExtractor():
#
#     def __init__(self, msg_path):
#         self.bin_fn = msg_path
#         # self.video_path = '/home/mkapoor/slam_files/openvslam/build/aist_living_lab_1/video.mp4'
#         self.base_dir = '/home/mkapoor/slam_files/openvslam/build/'
#
#     def forward(self, plot=False):
#
#         # Read file as binary and unpack data using MessagePack library
#         with open(self.bin_fn, "rb") as f:
#             data = msgpack.unpackb(f.read(), use_list=False, raw=False)
#
#         # The point data is tagged "landmarks"
#
#         # Create files with keypoints for keyframes
#         # point_cloud = data["landmarks"]
#         # for id, para in point_cloud.items():  # accessing keys
#         #     print(para, end=',')
#         # points_keyframe(point_cloud)
#         # print(max_kf,min_kf)
#
#         key_frames = data["keyframes"]
#
#         print("Point cloud has {} keyframes.".format(len(key_frames)))
#
#         key_frame = {int(k): v for k, v in key_frames.items()}
#
#         if plot:
#             x = []
#             y = []
#             z = []
#             t = []
#             for key in sorted(key_frame.keys()):
#                 point = key_frame[key]
#                 trans_cw = np.asarray(point["trans_cw"])
#                 rot_cw = np.asarray(point["rot_cw"])
#
#                 rigid_cw = RigidTransform(rot_cw, trans_cw)
#
#                 pos = np.matmul(rigid_cw.rotation, trans_cw)
#
#                 x.append(pos[0])
#                 y.append(pos[1])
#                 z.append(pos[2])
#                 t.append(float(point["ts"]))
#
#             plt.xlabel('X')
#             plt.ylabel('Y')
#             plt.scatter(x, z)
#             plt.show()
#
#             plt.ylabel('Height')
#             plt.xlabel('Time')
#             plt.scatter(x=t, y=y)
#             print(t)
#
#             # # new a figure and set it into 3d
#             fig = plt.figure()
#
#             plt.show()
#
#
#         else:
#
#                 count = 0
#                 connections = []
#                 keyfrm_center = np.empty([3, 1])
#                 for key in sorted(key_frame.keys()):
#                     point = key_frame[key]
#                     # for id, para in point.items():  # accessing keys
#                     #     print(id, end=',')
#
#                     # position capture
#                     # print("Child", point["span_children"])
#                     # print("Parent", point["span_parent"])
#                     child_parent = [point["span_children"], point["span_parent"]]
#                     connections.append(child_parent)
#                     trans_cw = np.asarray(point["trans_cw"])
#                     rot_cw = np.asarray(point["rot_cw"])
#
#                     rigid_cw = RigidTransform(rot_cw, trans_cw)
#
#                     pos = np.matmul(rigid_cw.rotation, trans_cw)
#                     keyfrm_center = np.append(keyfrm_center, pos.reshape(-1, 1), axis=1)
#
#         #             # image capture
#         #             success, image = vidcap.read()
#         #
#         #             if not success:
#         #                 print("capture failed")
#         #             else:
#         #                 cv2.imwrite(os.path.join(video_name, str(count) +".jpg"), image)
#         #
#         #             count+=1
#         #
#         # print(connections)
#
#         keydata = {}
#         keyfrm_center = np.delete(keyfrm_center, 0 , axis=1)
#         keydata['keyframes'] = keyfrm_center
#         keydata['connections'] = connections
#         print("Finished")
#         # print(np.shape(keyfrm_center))
#         return keydata
#         # print((keyfrm_center))
#         # np.save('keyframes.npy', keyfrm_center)

class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, data_path, nfeatures):

        self.files = []
        self.files += [data_path + f for f in natsorted(os.listdir(data_path))]
        self.nfeatures = nfeatures
        # self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        # print(self.files)
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return 8

    def __getitem__(self, idx):

		# load precalculated correspondences
		# data = np.load(self.files[idx], allow_pickle=True)

        # if idx < 8:
        #     idx2 = np.random.randint(8)
        # else:
        #     idx2 = np.random.randint(8,16)

        idx2 = np.random.randint(8)
        start = time.time()
        base_folder1 = self.files[idx]
        base_folder2 = self.files[idx2]
        kf1 = np.load(f'{base_folder1}/std_cam/keypoints.npy')
        kf2 = np.load(f'{base_folder2}/std_cam/keypoints.npy')
        kf1 = np.delete(kf1, 0, axis=1)
        kf2 = np.delete(kf2, 0, axis=1)
        desc_final1 = extract_descriptor(base_folder1)
        desc_final2 = extract_descriptor(base_folder2)
        scores1 = extract_score(base_folder1)
        scores2 = extract_score(base_folder2)

        # Sub graph section
        indices1 = np.random.choice(kf1.shape[0], size=512, replace=False)
        indices2 = np.random.choice(kf2.shape[0], size=512, replace=False)
        kf1 = kf1[indices1, :]
        kf2 = kf2[indices2, :]
        desc_final1 = desc_final1[:, indices1]
        desc_final2 = desc_final2[:, indices2]
        scores1 = scores1[:, indices1]
        scores2 = scores2[:, indices2]

        # Fake zero desciptors and scores
        # desc_final1 = np.zeros([256, 512])
        # desc_final2 = np.zeros([256, 512])
        # scores1 = np.ones([1,512])
        # scores2 = np.ones([1, 512])

        # matches for sub graph
        all_matches = compute_matches(kf1,kf2)
        kf2 = mod_pose_graph(kf2, kf2.shape[0])
        
        # print shapes
        end = time.time()
        print("Matches: ", all_matches.shape)
        print("Kf1: ", kf1.shape)
        print("Kf2: ", kf2.shape)
        print("Desc1: ", desc_final1.shape)
        print("Desc2: ", desc_final2.shape)
        print("Scores1: ", scores1.shape)
        print("Scores2: ", scores2.shape)
        print("Load time: ", (end-start))
        # kf1 = np.transpose(keyframe_data['keyframes']) #shape after transpose Nx3
        # kf2 = mod_pose_graph(kf1, np.shape(kf1)[0]) #shape Nx3
        #
        # self_prob_matrix = gen_matrix(keyframe_data['connections'], np.shape(kf1)[0]) #shape NxN
        #
        # ########1
        # vs = VideoStreamer(self.img_path, camid = 0, height = 480, width = 640, skip = 1, img_glob = '*.jpg')
        # print('==> Loading pre-trained SuperPoint network.')
        # # This class runs the SuperPoint network and processes its outputs.
        # fe = SuperPointFrontend(weights_path='superpoint_v1.pth',
        #                         nms_dist=4,
        #                         conf_thresh=0.015,
        #                         nn_thresh=0.7,
        #                         cuda=False) #Change cuda to True later
        # print('==> Successfully loaded pre-trained SuperPoint network.')
        #
        # print('==> Running Demo.')
        # desc_final1 = np.empty([256, 1])
        # scores1 = np.empty([1, 1])
        # desc_final2 = np.empty([256, 1])
        # scores2 = np.empty([1, 1])
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
        #     indices1 = np.random.choice(confidence.shape[1], size=10, replace=False)
        #     indices2 = np.random.choice(confidence.shape[1], size=10, replace=False)
        #
        #     confidence1 = confidence[:, indices1]
        #     pool_score1 = np.average(confidence1, axis=1)
        #     scores1 = np.append(scores1, pool_score1.reshape(-1, 1), axis=1)
        #
        #     confidence2 = confidence[:, indices2]
        #     pool_score2 = np.average(confidence2, axis=1)
        #     scores2 = np.append(scores2, pool_score2.reshape(-1, 1), axis=1)
        #
        #     desc1 = desc[:, indices1]
        #     pool_desc1 = np.average(desc1, axis=1)
        #     desc_final1 = np.append(desc_final1, pool_desc1.reshape(-1, 1), axis=1)
        #
        #     desc2 = desc[:, indices2]
        #     pool_desc2 = np.average(desc2, axis=1)
        #     desc_final2 = np.append(desc_final2, pool_desc2.reshape(-1, 1), axis=1)
        #
        #     end = time.time()
        #     net_t = (1. / float(end1 - start))
        #     total_t = (1. / float(end - start))
        #
        # desc_final1 = np.delete(desc_final1, 0, axis=1)
        # scores1 = np.delete(scores1, 0, axis=1)
        # desc_final2 = np.delete(desc_final2, 0, axis=1)
        # scores2 = np.delete(scores2, 0, axis=1)
        #
        # print('==> Finshed SuperPoint.')
        #
        #
        # kf1 = kf1.reshape((-1, kf1.shape[0], kf1.shape[1])) #BxNx3
        # kf2 = kf2.reshape((-1, kf2.shape[0], kf2.shape[1]))
        # desc_final1 = desc_final1.reshape((-1, desc_final1.shape[0], desc_final1.shape[1]))
        # desc_final2 = desc_final2.reshape((-1, desc_final2.shape[0], desc_final2.shape[1]))

        # dummy = np.array([i for i in range(np.shape(kf1)[1])])
        # all_matches = np.transpose([dummy] * 2)
        # all_matches = all_matches.reshape((-1, all_matches.shape[0], all_matches.shape[1])) #shape BxNx2
        # print(np.shape(all_matches))
        # return {'kp1': kp1_np / max_size, 'kp2': kp2_np / max_size, 'descs1': descs1 / 256., 'descs2': descs2 / 256., 'matches': all_matches}
        # kp1_np = kf1.reshape((1, -1, 2))
        # kp2_np = kp2_np.reshape((1, -1, 2))
        # descs1 = np.transpose(descs1 / 256.)
        # descs2 = np.transpose(descs2 / 256.)
        #
        # image = torch.from_numpy(image/255.).double()[None].cuda()
        # warped = torch.from_numpy(warped/255.).double()[None].cuda()

        return{
            'keypoints0': list(kf1),
            'keypoints1': list(kf2),
            'descriptors0': list(desc_final1),
            'descriptors1': list(desc_final2),
            'scores0': list(scores1),
            'scores1': list(scores2),
            'all_matches': list(all_matches)

        } 

