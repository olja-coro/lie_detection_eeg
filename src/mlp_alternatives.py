import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score


data = np.load("data/data.npz")
X = data["X"]   # (subjects, 2, 25, 5, 16)
y = data["y"]   # (subjects, 2, 25)

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3


Xn = X.copy()
for subj in range(X.shape[0]):
    X_flat = X[subj].reshape(-1, 80)
    mu = X_flat.mean(axis=0, keepdims=True)
    sd = X_flat.std(axis=0, keepdims=True) + 1e-8
    Xn[subj] = ((X_flat - mu) / sd).reshape(2,25,5,16)


def get_loso_split(X, y, test_subject):
    X_train, y_train, X_test, y_test = [], [], [], []

    for subj in range(X.shape[0]):
        X_subj = X[subj].reshape(50, 5, 16)
        y_subj = y[subj].reshape(50)

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


# SIMPLE MLP
class MLP_Linear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# MLP (Linear + ReLU + Linear)
class MLP_2Layer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# MLP + LayerNorm (paper-friendly)
class MLP_LN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test.cpu(), preds)
    auc = roc_auc_score(y_test.cpu(), probs)
    return acc, auc


MODELS = {
    "Linear": MLP_Linear,
    "MLP_2Layer": MLP_2Layer,
    "MLP_LayerNorm": MLP_LN,
}

for model_name, ModelClass in MODELS.items():
    print(f"\n===== MODEL: {model_name} =====")

    all_acc, all_auc = [], []

    for test_subject in range(27):
        X_train, y_train, X_test, y_test = get_loso_split(Xn, y, test_subject)

        # MULTI-CHANNEL INPUT (80)
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test), -1)

        # shuffle train
        idx = np.random.permutation(len(y_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = ModelClass(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(EPOCHS):
            train_epoch(model, loader, optimizer, criterion)

        acc, auc = evaluate(
            model,
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )

        all_acc.append(acc)
        all_auc.append(auc)

    print(f"Avg ACC: {np.mean(all_acc):.3f} | Avg AUC: {np.mean(all_auc):.3f}")
