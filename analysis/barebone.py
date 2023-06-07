import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score

dtype=torch.float32

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Barebones_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores
    
def compare_scores_y(scores, y, num_correct, num_samples, num_positives):
    # Accumulates number of correct predictions, numer of samples and number of samples with prediction = 1
    preds = (scores > 0.5).float()
    num_correct += (preds == y).sum()
    num_samples += preds.size(0)
    num_positives += (preds == 1).sum()    
    return num_correct, num_samples, num_positives
    
def get_f1_score(scores, y):
    # Convert scores to binary predictions
    preds = (scores > 0.5).float()
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()

    # Calculate F1 score using sklearn function
    return f1_score(y, preds)

def train_and_get_model(model, optimizer, loader_t, loader_v, epochs=1, use_BCE_weight=False, num_samples_pos=1, num_samples_neg=1):
    model = model.to(device=device) 
    
    loss_epoch = []
    train_acc_epoch = []
    val_acc_epoch = []
    train_pos_epoch = []
    val_pos_epoch = []

    best_val = 0 
    best_model = None
    
    if use_BCE_weight:
        weight = torch.as_tensor(num_samples_neg / num_samples_pos, dtype=torch.float)
        BCEwithLogitLoss = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        BCEwithLogitLoss = nn.BCEWithLogitsLoss()
    
    for e in range(epochs):
        num_correct = 0
        num_samples = 0
        num_positives = 0

        for t, (x, y) in enumerate(loader_t):  
            model.train() 
            x = x.to(device=device, dtype=dtype)  
            y = y.to(device=device, dtype=dtype)

            scores = model(x).squeeze(1)
            loss = BCEwithLogitLoss(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct, num_samples, num_positives = \
            compare_scores_y(scores, y, num_correct, num_samples, num_positives)

        
        loss_epoch.append(loss.item())

        train_percent_pos = float(num_positives) / num_samples
        train_pos_epoch.append(100 * train_percent_pos)
        train_acc = float(num_correct) / num_samples
        train_acc_epoch.append(100 * train_acc)

        val_acc, val_percent_pos, f1_avg = check_accuracy(loader_v, model)
        val_acc_epoch.append(val_acc * 100)
        val_pos_epoch.append(val_percent_pos * 100)

        if val_acc > best_val: 
            best_val = val_acc 
            best_model = model 

        print('Epoch %d, loss = %.4f, train_acc = %.4f, val_acc = %.4f, train_pos = %.4f, val_pos = %.4f' % \
          (e, loss_epoch[-1], train_acc_epoch[-1], val_acc_epoch[-1], train_pos_epoch[-1], val_pos_epoch[-1]))

        print(f'Epoch {e}, Average Validation F1 Score: {f1_avg}')

    return loss_epoch, train_acc_epoch, val_acc_epoch, train_pos_epoch, val_pos_epoch, best_val, best_model

def train_model(model, optimizer, loader_t, loader_v, epochs=1, use_BCE_weight=False, num_samples_pos=1, num_samples_neg=1):
    loss_epoch, train_acc_epoch, val_acc_epoch, train_pos_epoch, val_pos_epoch, best_val, best_model = train_and_get_model(model, optimizer, loader_t, loader_v, epochs, use_BCE_weight, num_samples_pos, num_samples_neg)

    return loss_epoch, train_acc_epoch, val_acc_epoch, train_pos_epoch, val_pos_epoch

def check_accuracy(loader, model):  
    num_correct = 0
    num_samples = 0
    num_positives = 0
    f1_score_total = 0
    
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x).squeeze(1)
            num_correct, num_samples, num_positives = \
                compare_scores_y(scores, y, num_correct, num_samples, num_positives)
            # Compute f1 score for this batch and add it to f1_score_total
            f1_score_total += get_f1_score(scores, y)
            print(f'Updated f1_score_total: {f1_score_total}')  # Debug print

                    
        acc = float(num_correct) / num_samples
        percent_pos = float(num_positives) / num_samples
        f1_score_avg = f1_score_total / len(loader)  # Calculate the average F1 score
        
    return acc, percent_pos, f1_score_avg
