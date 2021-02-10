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


def mod_pose_graph(keypoints, num_keyframes):
    """Modify the given pose graph with ground truth transformation for training"""
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

def compute_matches(kp1, kp2):
    """Compute ground truth one-to-one matches between two localized posegraphs"""
    threshold = 0.2
    distances = cdist(kp1, kp2)
    distances[distances > threshold] == np.nan
    nn12 = np.nanargmin(distances, axis=1)
    nn21 = np.nanargmin(distances, axis=0)
    idx = np.arange(kp1.shape[0])
    valid_matches = (idx == nn21[nn12])
    matches = np.append(idx[valid_matches].reshape(-1, 1), nn12[valid_matches].reshape(-1, 1), axis=1)
    return matches


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, data_path, nfeatures):

        self.files = []
        self.files += [data_path + f for f in natsorted(os.listdir(data_path))]
        self.nfeatures = nfeatures

    def __len__(self):
        return 8

    def __getitem__(self, idx):

        idx2 = np.random.randint(8) #Randomly select second pose graph
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


        return{
            'keypoints0': list(kf1),
            'keypoints1': list(kf2),
            'descriptors0': list(desc_final1),
            'descriptors1': list(desc_final2),
            'scores0': list(scores1),
            'scores1': list(scores2),
            'all_matches': list(all_matches)

        } 

