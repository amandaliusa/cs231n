# Setup
import argparse
import pandas as pd
import numpy as np
import git
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
dtype=torch.float32

from barebone import *
from model_util import *
from alternativeModel import *

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_type', type=str, default='barebone')
    parser.add_argument('--hidden_size1', type=int, default=2000)
    parser.add_argument('--hidden_size2', type=int, default=1000)
    args = parser.parse_args()
    return args

args = parse_args()
    
# Load 3D joint outputs from OSX (num_videos x num_frames x num_joints (=65) x num_coordinates(=3))
# Each video is flattened to a single row of entry

pickle_path = '/home/ubuntu/OSX/output/STS_test6/log/'
joint_3d_out_o = np.array(pickle.load(open(os.path.join(pickle_path, "joint_3d_out.p"), "rb")))
video_list_3d = pickle.load(open(os.path.join(pickle_path, "video_list_3d.p"), "rb"))

num_samples = joint_3d_out_o.shape[0]
joint_3d_out = joint_3d_out_o.reshape((num_samples, -1))      # Flatten frame, joint and coordinates
joint_3d_out_pd = pd.DataFrame(joint_3d_out)
joint_3d_out_pd.insert(0, 'subjectid', video_list_3d)

# print(joint_3d_out_pd.shape)
joint_3d_out_pd

# Load survey data

df_survey = pd.read_csv(r'https://raw.githubusercontent.com/amandaliusa/cs231n/main/data/survey_data.csv')
df_survey

# Pre processing
# join the dataframes by subjectId 
df_join = joint_3d_out_pd.set_index('subjectid').join(df_survey.set_index('subjectid'), how='inner')
# print(df_join.shape)
df_join

# Data validation

# do a stratified split so that each dataset has the same proportion of OA=0 and OA=1
y = df_join['OA_check']

# set a random seed for reproducibility 
np.random.seed(42)

# split out test set from train/val
train_val_indices, test_indices = train_test_split(np.arange(len(df_join)), test_size=0.1, stratify=y)

# split out val set from train
train_ind, val_ind = train_test_split(np.arange(len(train_val_indices)), test_size=0.11111, stratify=y[train_val_indices])
train_indices = train_val_indices[train_ind]
val_indices = train_val_indices[val_ind]

# 349 examples 
NUM_TRAIN = len(train_indices)
NUM_VAL = len(val_indices)

train_data = df_join.iloc[train_indices]
val_data = df_join.iloc[val_indices]
test_data = df_join.iloc[test_indices]

# Check for number of OA positive and number of samples in the dataset
# print(df_join['OA_check'].count())  # 349
# print(df_join['OA_check'].sum())    # 21
class_counts = [df_join.iloc[:,-1].count() - df_join.iloc[:,-1].sum(), df_join.iloc[:,-1].sum()]

# Train set - This implies that if the model always predicts 0, it would have 94% training accuracy
# print(train_data['OA_check'].count())  # 279
# print(train_data['OA_check'].sum())    # 17

# Validation set - This implies that if the model always predicts 0, it would have 94% val accuracy
# print(val_data['OA_check'].count())  # 35
# print(val_data['OA_check'].sum())    # 2

# Test set - This implies that if the model always predicts 0, it would have 94% test accuracy
# print(test_data['OA_check'].count())  # 35
# print(test_data['OA_check'].sum())    # 2

# # `Norma` is not defined in the given code, so it is difficult to determine what it is doing. It is
# possible that it is a typo and was meant to be `normalize`, which is a boolean argument that
# determines whether or not to apply normalization to the data.

# Normalization
# compute mean and std of the features 
# means = []
# stds = []
# for column in train_data.iloc[:,:-1]: # only use training set, and exclude last column, which has labels
#     column_np = train_data[column].to_numpy()
#     means.append(np.mean(column_np))
#     stds.append(np.std(column_np))
    
# apply normalization
# def transform(feature):
#     print("feature", feature.shape)
#     print("mean", len(means))
#     print("std", len(stds))
#     transformed = [] * len(stds)
#     for i in range(len(stds)):
#         if stds[i] != 0:
#             transformed[i] = (feature[i] - means[i]) / stds[i]
#         else:
#             transformed[i] = feature[i]
#     return transformed

# Create data loader

# no normalization, no oversampling
train = CustomDataset(dataframe=train_data, transform=None)
loader_train = DataLoader(train, batch_size=64, 
                       sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

val = CustomDataset(dataframe=val_data, transform=None)
loader_val = DataLoader(val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))

test = CustomDataset(dataframe=test_data, transform=None)
loader_test = DataLoader(test, batch_size=64)

# oversampling
train = CustomDataset(dataframe=train_data, transform=None)
class_weights = 1./torch.tensor(class_counts, dtype=torch.float) 
labels = train_data.iloc[:,-1]
class_weights_all = class_weights[labels]

# oversample the training data 
weighted_sampler = sampler.WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)
loader_train_os = DataLoader(train, batch_size=64, sampler=weighted_sampler)

# Train
print(args.output_path)

train = CustomDataset(dataframe=train_data, transform=transform if args.normalize else None)
loader_train = DataLoader(train, batch_size=64, 
                       sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

val = CustomDataset(dataframe=val_data, transform=transform if args.normalize else None)
loader_val = DataLoader(val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))

test = CustomDataset(dataframe=test_data, transform=transform if args.normalize else None)
loader_test = DataLoader(test, batch_size=64)

input_size = 160875
hidden_size1 = args.hidden_size1
hidden_size2 = args.hidden_size2
num_classes = 1
num_samples_pos = train_data[train_data['OA_check']==1].shape[0]
num_samples_neg = train_data[train_data['OA_check']==0].shape[0]
if args.model_type == 'barebone':
    model = Barebones_model(input_size, hidden_size1, num_classes).to(device=device)
elif args.model_type == 'threeLayer':
    model = ThreeLayer_model(input_size, hidden_size1, hidden_size2, num_classes).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss, train_acc, val_acc, train_pos, val_pos, best_val, best_model = train_and_get_model(model, optimizer, loader_train, loader_val, epochs=args.epoch, \
    use_BCE_weight=True, num_samples_pos=num_samples_pos, num_samples_neg=num_samples_neg)

pickle.dump(best_model, open(args.output_path, 'wb'))

# visualize
fig, axes = plt.subplots(5, 1, figsize=(15, 15))

axes[0].set_title('Training loss')
axes[0].set_xlabel('Iteration')
axes[1].set_title('Training accuracy')
axes[1].set_xlabel('Epoch')
axes[2].set_title('Validation accuracy')
axes[2].set_xlabel('Epoch')
axes[3].set_title('Training Percent Positive Prediction')
axes[3].set_xlabel('Epoch')
axes[4].set_title('Validation Percent Positive Prediction')
axes[4].set_xlabel('Epoch')

axes[0].plot(loss, label="loss")
axes[1].plot(train_acc, label="training accuracy")
axes[2].plot(val_acc, label="validation accuracy")
axes[3].plot(train_pos, label="training % with positive predictions")
axes[4].plot(val_pos, label="validation % with positive predictions")
    
for ax in axes:
    ax.legend(loc='best', ncol=4)
    ax.grid(linestyle='--', linewidth=0.5)

plt.savefig(args.output_path + '_fig.png')

# saliency map

def compute_saliency_maps(x, y, num_samples_pos, num_samples_neg, model):
    """
    - X: Input; Tensor of shape (N, D)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained model used to generate the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, D) giving the saliency maps for the inputs.
    """
    model.eval()
    x.requires_grad_()

    saliency = None

    scores = model(x).squeeze(dim=1)
    weight = torch.as_tensor(num_samples_neg / num_samples_pos, dtype=torch.float)
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=weight)
    loss = BCEWithLogitsLoss(scores, y)
    loss.backward()
    saliency = x.grad

    return saliency

num_samples_pos = train_data[train_data['OA_check']==1].shape[0]
num_samples_neg = train_data[train_data['OA_check']==0].shape[0]
x = torch.as_tensor(train_data.iloc[:, :-1].values, dtype=torch.float).to(device=device)
y = torch.as_tensor(train_data.iloc[:, -1].values, dtype=torch.float).to(device=device)
saliency = compute_saliency_maps(x, y, num_samples_pos, num_samples_neg, best_model)

_, F, J, D = joint_3d_out_o.shape
N = saliency.shape[0]
saliency_NF = saliency.view((N, F, J, D)).mean(dim=(2, 3))
saliency_NF = nn.functional.normalize(saliency_NF, dim=1).cpu()
saliency_NJ = saliency.view((N, F, J, D)).mean(dim=(1, 3))
saliency_NJ = nn.functional.normalize(saliency_NJ, dim=1).cpu()
saliency_ND = saliency.view((N, F, J, D)).mean(dim=(1, 2))
saliency_ND = nn.functional.normalize(saliency_ND, dim=1).cpu()
saliency_FJ = saliency.view((N, F, J, D)).mean(dim=(0, 3))
saliency_FJ = nn.functional.normalize(saliency_FJ, dim=1).cpu()


plt.imshow(saliency_NF, aspect='auto')
plt.colorbar()
plt.show()
plt.imshow(saliency_NJ, aspect='auto')
plt.colorbar()
plt.show()
plt.imshow(saliency_ND, aspect='auto')
plt.colorbar()
plt.show()
plt.imshow(saliency_FJ, aspect='auto')
plt.colorbar()
plt.savefig(args.output_path + '_sal.png')
