"""
These visualization utils are adapted from https://github.com/zczcwh/poseformer_demo.
"""

import sys
import argparse
import cv2
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy

sys.path.append(os.getcwd())

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose_singleConnection(kps, img, i):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    start = map(int, kps[connections[i][0]])
    end = map(int, kps[connections[i][1]])
    start = list(start)
    end = list(end)

    cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[i] else rcolor, thickness)
    cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
    cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img

def show2Dpose(kps, img, i):
    # kps has shape (17, 2) = (number of keypoints, number of dimensions)

    # total of 17 keypoints
    # 0 middle of hip
    # 1 right hip
    # 2 right knee
    # 3 right ankle
    # 4 left hip
    # 5 left knee
    # 6 left ankle
    # 7 middle of torso
    # 8 neck
    # 9 nose
    # 10 forehead
    # 11 left shoulder
    # 12 left elbow
    # 13 left wrist
    # 14 right shoulder
    # 15 right elbow
    # 16 right wrist
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3
    # img[:, :, :] = 255

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    # vals has shape (17, 3) = (number of keypoints, number of dimensions)
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    # these are the joint connections (I contains the first joint index, J contains the second joint index)
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    # plot each connection
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    #ax.set_aspect('equal') # works fine in matplotlib==2.2.2
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    video_name = ".".join(video_path.split('/')[-1].split('.')[:-1])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 9
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/poseformer_9'
    args.n_joints, args.out_joints = 17, 17

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 3D pose...')

    keypoints3d = np.load(output_dir + '3d_output.npy', allow_pickle=True)

    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        img_size = img.shape

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        post_out = keypoints3d[i]

        input_2D_no = input_2D_no[args.pad]

        ## 2D

        # to visualize single connection
        # for i in range(16): 
        #     print("writing connections " + str(i))

        #     image = show2Dpose_singleConnection(input_2D_no, copy.deepcopy(img), i)

        #     output_dir_2D = output_dir +'pose2D/'
        #     os.makedirs(output_dir_2D, exist_ok=True)
        #     cv2.imwrite(output_dir_2D + 'keypoints_{}_2D.png'.format(i), image)
        
        # break

        image = show2Dpose(input_2D_no, copy.deepcopy(img), i)

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + '_2D.png', image)

        ## 3D
        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(post_out, ax)

        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        
    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        # edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')

