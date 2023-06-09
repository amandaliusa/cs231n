import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

dtype=torch.float32

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# reference: https://discuss.pytorch.org/t/dataset-from-pandas-without-folder-structure/146816/4
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        
        self.dataframe = dataframe
        self.transform = transform

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[:-1]
        label = row[-1]
        
        # reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        if self.transform:
            features = self.transform(features)
            
        return features, label

    def __len__(self):
        return len(self.dataframe)

# Reference: https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []

    for i, batch in enumerate(dl):
        idxs = batch[0][:, 0].tolist()
        #classes = batch[0][:, 1]
        classes = batch[1]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            raise ValueError("More than two classes detected")

    if with_outputs:
        fig, ax = plt.subplots(figsize=(2, 15), dpi=72)

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            label=(id_to_label[0] if id_to_label is not None else "0"),
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.set_xticks(range(len(dl)))
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        plt.show()

def plot_roc_curve(model, loader, device):
    model = model.to(device)
    model.eval()
    
    y_test = []
    y_score = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float)
            scores = model(x).squeeze(1)
            
            y_test.append(y.cpu().numpy())
            y_score.append(torch.sigmoid(scores).cpu().numpy())
            
    y_test = np.concatenate(y_test)
    y_score = np.concatenate(y_score)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(true_classes, predicted_classes):
    classes = ['OA=0', 'OA=1']
    # Create a confusion matrix
    cm  = confusion_matrix(true_classes, predicted_classes)
    
    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Create a heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, cmap='Blues') 
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()
    
def evaluate_dataset(data):
    for row in range(len(data)):
        x = []
        row_data = data.iloc[row]
        y = row_data[-1] 
        x = row_data[:-1]
        x.append(x)

        output = model(torch.tensor(x).to(device))
        probs = torch.sigmoid(output)

        predicted_class = (probs > 0.5).float()

        subjectId = data.index.values[row]
        print("subjectId: {}, predicted: {}, actual: {}".format(subjectId, int(predicted_class.cpu().numpy()[0]), int(y)))