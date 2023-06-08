import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

dtype=torch.float32

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ThreeLayer_model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        scores = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return scores
    
class RNN_model(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, lin_hidden_size, num_classes, num_frames):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, lin_hidden_size)
        self.fc2 = nn.Linear(lin_hidden_size, num_classes)
        self.num_frames = num_frames
    
    def forward(self, x):
        N = x.shape[0]
        x = x.view((N, self.num_frames, -1))
        _, h_rnn = self.rnn(x)       # (num_layers * num_directions, batch, hidden_size)
        h_lin = self.fc1(F.relu(h_rnn.squeeze(dim=0)))
        scores = self.fc2(F.relu(h_lin))
        return scores
    
class LSTM_model(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lin_hidden_size, num_classes, num_frames, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(lstm_hidden_size, lin_hidden_size)
        self.fc2 = nn.Linear(lin_hidden_size, num_classes)
        self.num_frames = num_frames
    
    def forward(self, x):
        N = x.shape[0]
        x = x.view((N, self.num_frames, -1))
        _, (h_n, _) = self.lstm(x)       # (num_layers * num_directions, batch, hidden_size)
        h_lin = self.fc1(F.relu(h_n.squeeze(dim=0)))
        scores = self.fc2(F.relu(h_lin))
        return scores
