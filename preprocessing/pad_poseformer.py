import sys
import git

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
    
# setting path
sys.path.append(homedir)

import argparse
import os
import numpy as np
import pandas as pd
import pickle
import traceback
from utils import *

def pad_poseformer(output_dir):
    
    # Get a list of all subjects
    subjects = os.listdir("{}/pretrained_models/poseformer/".format(homedir))

    # skip videos that failed to process with poseformer
    to_skip = ['1ovgasC1_copy', 'kunkun', 'woyjGsIt', 'ZAn8qhAb', 'mxKQbPdW', 'GITsdVy7', 'OuYG4U64', 'k3YTjMU4', 'yzutrWiF', 'l230IPIW', 
           'x77WJ10Q', 'XVvERXzh', 'oSFbRH4g', 'LGqwZPty', 'IMG_1623', 'JNh9mTC7', 'tFbOCnVv', '20230505_131419', 
           'ztKJoXiw', 'VID_20230505_201438080', 'gZry6Nal', 'trim.EB638E4E_B3DC_470E_883A_26253AC03B55', 'IMG_0067', 
           'VKW2KzGC', 'hsX5kAeZ', 'trim.BBF38598_A21A_4AFB_8BD6_EE7D6564DAED', 'F2FtnWVJ', 'nGyTDr5q', 'HAKCEJWo', 
           'Cw8bz9tw', 'uwtiYEpw', 'IMG_0822', 'j3iTbhie', 'IMG_8762', 'X5kmXm2t', 'JOmsdP6A', 'g1eKN8fK', 'UmopdCFg', 
           'zkO4XPXQ', 'IMG_5384', 'Wyg30nvk', 'iJsDlll8', 'IMG_6494', '2SV6hYB2', '0B2dfO4b', 'ku64Cwle', 'IMG_3013', 
           'video', 'yt2AP5Lo', 'iVGPuONf', 'KneeVideo_Seese_May8', 'eOBg4mwH', 'trim.495B207A_D816_405F_91CE_21FF0E67F907', 
           'IMG_1070', 'RJVU9j8P', '4RF13po6', 'MeABXs62', 'IMG_1202', 'BvPM57u7', '15IPa5iS', 'obDMup1i', 'g5sxqMgO', 
           'VID_20230507_175427', 'ESzlIzyO', 'IMG_0388', 'VEBURFuk', 'CWf3eyvo', 'QNEd2eGX', 'ZR3QAZmu', '0nUjlcd7', 
           'H1SDVD0X', 'IMG_4697', 'F6PuVthQ', 'qpxWRSOP', 'iqQ28Zc4', 'xWZi5URn', 'PA4bXBlr']
    
    videos = []
    for video in subjects:
        if video not in to_skip and os.path.isdir("{}/pretrained_models/poseformer/{}".format(homedir, video)): 
            videos.append(video)

    # For each video, find max frame. Store max number of frames across all videos for padding. 
    # 3d and 2d have the num of frames.
    num_joints = 17
    max_f_idx_all = 0
    max_f_idx = {}
    for video in videos:
        if not video in max_f_idx.keys():
            max_f_idx[video] = 0
        
        processed_npz_path="{}/pretrained_models/poseformer/{}/input_2D/".format(homedir, video)
        res2d = np.load("{}/keypoints.npz".format(processed_npz_path))['reconstruction']
        res2d = np.reshape(res2d, (res2d.shape[1], res2d.shape[2], res2d.shape[3]))
            
        max_f_idx[video] = res2d.shape[0]
        
        if res2d.shape[0] > max_f_idx_all:
            max_f_idx_all = res2d.shape[0]

    # check for videos with too few valid frames
    for video in max_f_idx:
        if max_f_idx[video] < 50:
            print('{} has only {} frames'.format(video, max_f_idx[video]))

    # pad all videos to the max number of frames
    joint_3d_out = np.zeros((len(max_f_idx), max_f_idx_all, num_joints, 3))
    joint_2d_out = np.zeros((len(max_f_idx), max_f_idx_all, num_joints, 2))

    # compile data into a 4 dimensional array with a 1 dimensional lookup by video name
    video_counter = 0
    for video in videos:
            
        padded = np.zeros((max_f_idx_all, num_joints, 3))
        frame_counter = 0
        processed_npy_path_3d="{}/pretrained_models/poseformer/{}/".format(homedir, video)
        res3d = np.load("{}/3d_output.npy".format(processed_npy_path_3d))
        
        for i in range(res3d.shape[0]):
            padded[frame_counter, :, :] = res3d[i]
            frame_counter += 1
        joint_3d_out[video_counter] = padded
        video_counter += 1

    video_counter = 0
    for video in videos:
            
        padded = np.zeros((max_f_idx_all, num_joints, 2))
        frame_counter = 0
        processed_npz_path_2d="{}/pretrained_models/poseformer/{}/input_2D/".format(homedir, video)
        res2d = np.load("{}/keypoints.npz".format(processed_npz_path_2d))['reconstruction']
        res2d = np.reshape(res2d, (res2d.shape[1], res2d.shape[2], res2d.shape[3]))
        
        for i in range(res2d.shape[0]):
            padded[frame_counter, :, :] = res2d[i]
            frame_counter += 1
        joint_2d_out[video_counter] = padded
        video_counter += 1

    # write outputs 
    output_npy = output_dir + 'subjects.npy'
    np.save(output_npy, videos)

    output_npy = output_dir + 'joints_2d_padded.npy'
    np.save(output_npy, joint_2d_out)

    output_npy = output_dir + 'joints_3d_padded.npy'
    np.save(output_npy, joint_3d_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='{}/preprocessing/poseformer/'.format(homedir), help='input output path')
    args = parser.parse_args()

    pad_poseformer(args.output_path)