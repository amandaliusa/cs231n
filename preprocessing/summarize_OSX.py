import argparse
import os
import copy
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--sum_dict_only', type=bool, default=False)
    args = parser.parse_args()
    return args

args = parse_args()
joint_3d_num_joints = 65
joint_3d_num_dim = 3
joint_2d_num_joints = 137
joint_2d_num_dim = 2

if not args.sum_dict_only:
    # read all the obj data and extract 3d and 2d joint positions
    joint_3d = {}      # num_frames x num_joints(65) x num_dimensions (3)
    joint_2d = {}      # num_frames x num_joints(137) x num_dimensions (2)
    directories = os.scandir(args.output_folder)
    for directory in directories:
        if directory.name != 'log':
            video_name = directory.name
            objs = os.scandir(os.path.join(directory.path, "obj"))
            frame_joint_dim_3d = {}
            frame_joint_dim_2d = {}
            for obj in objs:
                joint_dim_3d = []
                joint_dim_2d = []
                frame = int(obj.name.split('.')[0])
                f = open(obj, "r")
                for line in f:
                    line_s = line[:-2].split(' ')
                    if line_s[0] == 'j3':
                        line_cleaned_3d = list(map(float, line_s[1:]))
                        joint_dim_3d.append(line_cleaned_3d)
                    if line_s[0] == 'j2':
                        line_cleaned_2d = list(map(float, line_s[1:]))
                        joint_dim_2d.append(line_cleaned_2d)
                f.close()
                frame_joint_dim_3d[frame] = copy.deepcopy(joint_dim_3d)
                frame_joint_dim_2d[frame] = copy.deepcopy(joint_dim_2d)
            objs.close()
            joint_3d[video_name] = copy.deepcopy(frame_joint_dim_3d)
            joint_2d[video_name] = copy.deepcopy(frame_joint_dim_2d)
    directories.close()

    assert(not os.path.exists(os.path.join(args.output_folder, "log", "joint_3d.p")))
    assert(not os.path.exists(os.path.join(args.output_folder, "log", "joint_2d.p")))

    pickle.dump(joint_3d, open(os.path.join(args.output_folder, "log", "joint_3d.p"), "wb"))
    pickle.dump(joint_2d, open(os.path.join(args.output_folder, "log", "joint_2d.p"), "wb"))
else:
    joint_3d = pickle.load(open(os.path.join(args.output_folder, "log", "joint_3d.p"), "rb"))
    joint_2d = pickle.load(open(os.path.join(args.output_folder, "log", "joint_2d.p"), "rb"))

# For each video, find max frame. Store max number of frames across all videos for padding. 
# 3d and 2d have the num of frames.
max_f_idx_all = 0
max_f_idx = {}
for video in joint_3d:
    if not video in max_f_idx.keys():
        max_f_idx[video] = 0
    for frame in joint_3d[video]:
        if frame > max_f_idx[video]:
            max_f_idx[video] = frame
        if frame > max_f_idx_all:
            max_f_idx_all = frame

# code used to fix errors reported below
# this video had a truncated file and was rerun
video = '2geyDTFU'
frame = 504
f = open(os.path.join(args.output_folder, video, 'obj/{:06d}.obj'.format(frame)), "r")
joint_dim_3d = []
joint_dim_2d = []
for line in f:
    line_s = line[:-2].split(' ')
    if line_s[0] == 'j3':
        line_cleaned_3d = list(map(float, line_s[1:]))
        joint_dim_3d.append(line_cleaned_3d)
    if line_s[0] == 'j2':
        line_cleaned_2d = list(map(float, line_s[1:]))
        joint_dim_2d.append(line_cleaned_2d)
f.close()
joint_3d[video][frame] = joint_dim_3d
joint_2d[video][frame] = joint_dim_2d
# drop videos with no frames
del joint_3d['g1eKN8fK']
del joint_2d['g1eKN8fK']
del joint_3d['JKDP8D2e']
del joint_2d['JKDP8D2e']

# check for videos with too few valid frames
for video in max_f_idx:
    if max_f_idx[video] < 50:
        print('{} has only {} frames'.format(video, max_f_idx[video]))

# check for frames with incomplete information
for video in joint_3d:
    for frame in joint_3d[video]:
        data = joint_3d[video][frame]
        if len(data) != joint_3d_num_joints or len(data[0]) != joint_3d_num_dim:
            print('{} {} has {} joints'.format(video, frame, len(data)))

for video in joint_2d:
    for frame in joint_2d[video]:
        data = joint_2d[video][frame]
        if len(data) != joint_2d_num_joints or len(data[0]) != joint_2d_num_dim:
            print('{} {} has {} joints'.format(video, frame, len(data)))

# check for missing frames
missing_frames = {}
for video in joint_3d:
    for i in range(0, max_f_idx[video] + 1, 3):
        if not os.path.exists(os.path.join(args.output_folder, video, 'obj', '{:06d}.obj'.format(i))):
            # print('Frame {} missing from {}'.format(i, video))
            if video not in missing_frames.keys():
                missing_frames[video] = []
            missing_frames[video].append(i)

# exclude videos with more than 10 missing frames
for video in missing_frames:
    if len(missing_frames[video]) > 10:
        del joint_3d[video]
        del joint_2d[video]

# pad all videos to the max number of frames
joint_3d_out = np.zeros((len(joint_3d), max_f_idx_all, joint_3d_num_joints, joint_3d_num_dim))
joint_2d_out = np.zeros((len(joint_2d), max_f_idx_all, joint_2d_num_joints, joint_2d_num_dim))

# compile data into a 4 dimensional array with a 1 dimensional lookup by video name
video_list_3d = []
video_counter = 0
for video in joint_3d:
    padded = np.zeros((max_f_idx_all, joint_3d_num_joints, joint_3d_num_dim))
    frame_counter = 0
    for frame in sorted(joint_3d[video].keys()):
        padded[frame_counter, :, :] = joint_3d[video][frame]
        frame_counter += 1
    joint_3d_out[video_counter] = padded
    video_counter += 1
    video_list_3d.append(video)

video_list_2d = []
video_counter = 0
for video in joint_2d:
    padded = np.zeros((max_f_idx_all, joint_2d_num_joints, joint_2d_num_dim))
    frame_counter = 0
    for frame in sorted(joint_2d[video].keys()):
        padded[frame_counter, :, :] = joint_2d[video][frame]
        frame_counter += 1
    joint_2d_out[video_counter] = padded
    video_counter += 1
    video_list_2d.append(video)

# for debug
# print(joint_3d_out[video_list_3d.index('0B2dfO4b'), 2, 1, :])
# print(joint_3d_out[video_list_3d.index('0B2dfO4b'), -1, -1, :])
# print(joint_3d_out[video_list_3d.index('0nUjlcd7'), 2, 1, :])
# print(joint_3d_out[video_list_3d.index('0nUjlcd7'), -1, -1, :])
# print(joint_2d_out[video_list_2d.index('0B2dfO4b'), 2, 1, :])
# print(joint_2d_out[video_list_2d.index('0B2dfO4b'), -1, -1, :])
# print(joint_2d_out[video_list_2d.index('0nUjlcd7'), 2, 1, :])
# print(joint_2d_out[video_list_2d.index('0nUjlcd7'), -1, -1, :])

# pickle output files. Assert statements protect against overwriting existing files.
assert(not os.path.exists(os.path.join(args.output_folder, "log", "joint_3d_out.p")))
assert(not os.path.exists(os.path.join(args.output_folder, "log", "video_list_3d.p")))
assert(not os.path.exists(os.path.join(args.output_folder, "log", "joint_2d_out.p")))
assert(not os.path.exists(os.path.join(args.output_folder, "log", "video_list_2d.p")))

pickle.dump(joint_3d_out, open(os.path.join(args.output_folder, "log", "joint_3d_out.p"), "wb"))
pickle.dump(video_list_3d, open(os.path.join(args.output_folder, "log", "video_list_3d.p"), "wb"))
pickle.dump(joint_2d_out, open(os.path.join(args.output_folder, "log", "joint_2d_out.p"), "wb"))
pickle.dump(video_list_2d, open(os.path.join(args.output_folder, "log", "video_list_2d.p"), "wb"))