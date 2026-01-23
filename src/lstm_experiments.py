import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score


data = np.load("data/data.npz")
X = data["X"]   # (subjects, 2, 25, 5, 16)
y = data["y"]   # (subjects, 2, 25)

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3


Xn = X.copy()
for subj in range(X.shape[0]):
    X_flat = X[subj].reshape(-1, 80)
    mu = X_flat.mean(axis=0, keepdims=True)
    sd = X_flat.std(axis=0, keepdims=True) + 1e-8
    Xn[subj] = ((X_flat - mu) / sd).reshape(2,25,5,16)


def get_loso_sequences(X, y, test_subject, channels):
    """
    channels: list of channel indices
    returns:
        X_train: (N, seq_len=25, C, 16)
        y_train: (N,)
    """
    X_train, y_train, X_test, y_test = [], [], [], []

    for subj in range(X.shape[0]):
        X_subj = X[subj][:,:,channels,:]   # (2,25,C,16)
        y_subj = y[subj]                  # (2,25)

        # collapse class dimension
        X_subj = X_subj.reshape(2,25,len(channels),16)
        y_subj = y_subj.reshape(2,25)[:,0]  # one label per sequence

        if subj == test_subject:
            X_test.append(X_subj)
            y_test.append(y_subj)
        else:
            X_train.append(X_subj)
            y_train.append(y_subj)

    return (
        np.concatenate(X_train),
        np.concatenate(y_train),
        np.concatenate(X_test),
        np.concatenate(y_test),
    )


class SingleChannelLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, 25, 16)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

class MultiChannelLSTM(nn.Module):
    def __init__(self, n_channels=3, hidden=32):
        super().__init__()
        self.lstms = nn.ModuleList([
            nn.LSTM(16, hidden, batch_first=True)
            for _ in range(n_channels)
        ])
        self.fc = nn.Linear(hidden * n_channels, 1)

    def forward(self, x):
        # x: (B, 25, C, 16)
        outs = []
        for i, lstm in enumerate(self.lstms):
            _, (h, _) = lstm(x[:,:,i,:])
            outs.append(h[-1])
        h_cat = torch.cat(outs, dim=1)
        return self.fc(h_cat).squeeze()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test.cpu(), preds)
    auc = roc_auc_score(y_test.cpu(), probs)
    return acc, auc

# =========================
# EXPERIMENTS
# =========================

EXPERIMENTS = {
    "SingleChannel": [0],        # best channel from paper
    "Top3Channels": [0,1,2],     # example top-3
}

for name, channels in EXPERIMENTS.items():
    print(f"\n===== {name} LSTM =====")
    all_acc, all_auc = [], []

    for test_subject in range(27):
        X_train, y_train, X_test, y_test = get_loso_sequences(
            Xn, y, test_subject, channels
        )

        # FIX SHAPE FOR SINGLE-CHANNEL LSTM
        if len(channels) == 1:
            X_train = X_train[:, :, 0, :]   # (B, 25, 16)
            X_test  = X_test[:, :, 0, :]


        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)

        train_ds = TensorDataset(X_train, y_train)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        if len(channels) == 1:
            model = SingleChannelLSTM()
        else:
            model = MultiChannelLSTM(n_channels=len(channels))

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(EPOCHS):
            train_epoch(model, loader, optimizer, criterion)

        acc, auc = evaluate(model, X_test, y_test)
        all_acc.append(acc)
        all_auc.append(auc)

    print(f"Avg ACC: {np.mean(all_acc):.3f} | Avg AUC: {np.mean(all_auc):.3f}")
