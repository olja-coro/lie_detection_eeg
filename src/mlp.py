import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

data = np.load("data/data.npz")
X = data["X"]
y = data["y"]

# X: subjects × classes × trials × channels × features
# y: subjects × classes × trials


#print("X:", X.shape)
#print("y:", y.shape)

RANDOM_STATE = 42
#CHANNEL_IDX = 2   
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3



#preprocessing
Xn = X.copy()
for subj in range(X.shape[0]):
    X_flat = X[subj].reshape(-1, 80)              # (50,80)
    mu = X_flat.mean(axis=0, keepdims=True)
    sd = X_flat.std(axis=0, keepdims=True) + 1e-8
    Xn[subj] = ((X_flat - mu) / sd).reshape(2,25,5,16)


'''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)
'''

class MLP_MC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)



def get_loso_split(X, y, test_subject):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for subj in range(X.shape[0]):
        # (2,25,5,16) -> (50,5,16)
        X_subj = X[subj].reshape(2*25, 5, 16)
        y_subj = y[subj].reshape(2*25)

        if subj == test_subject:
            X_test.append(X_subj)
            y_test.append(y_subj)
        else:
            X_train.append(X_subj)
            y_train.append(y_subj)

    X_train = np.concatenate(X_train, axis=0)  # (N,5,16)
    y_train = np.concatenate(y_train, axis=0)

    X_test  = np.concatenate(X_test, axis=0)
    y_test  = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test


'''
def train(model, loader, optimizer, binaryCrossEntropy):
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch).squeeze()
        loss = binaryCrossEntropy(logits, y_batch)
        loss.backward()
        optimizer.step()


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        preds = (torch.sigmoid(logits) > 0.5).int()
    return accuracy_score(y_test.cpu(), preds.cpu())
'''

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch).squeeze()
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        n += x_batch.size(0)
    return total_loss / n



def evaluate_metrics(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_test.cpu().numpy().astype(int)
        preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    return acc, auc

'''
# =========================
# STEP 2 — OVERFIT TEST
# =========================

channel_idx = 0
test_subject = 0

# LOSO split
X_train, y_train, X_test, y_test = get_loso_split(
    Xn, y, test_subject
)

# selezione canale
X_train_ch = X_train[:, channel_idx, :]  # (N,16)

# shuffle training
idx = np.random.permutation(len(y_train))
X_train_ch = X_train_ch[idx]
y_train    = y_train[idx]

# subset piccolo
n_small = 200
X_small = torch.tensor(X_train_ch[:n_small], dtype=torch.float32)
y_small = torch.tensor(y_train[:n_small], dtype=torch.float32)

small_ds = TensorDataset(X_small, y_small)
small_loader = DataLoader(small_ds, batch_size=32, shuffle=True)

# modello semplice (solo per overfit)
class MLP_overfit(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP_overfit()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()

for ep in range(300):
    loss = train_epoch(model, small_loader, optimizer, criterion)
    if ep % 50 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(model(X_small).squeeze())
            preds = (probs > 0.5).int()
            acc = (preds == y_small.int()).float().mean().item()
        print(f"ep {ep} | loss {loss:.4f} | acc {acc:.3f}")


'''
for channel_idx in range(5):
    print(f"\n=== CHANNEL {channel_idx} ===")
    all_acc = []
    all_auc = []

    for test_subject in range(27):
        X_train, y_train, X_test, y_test = get_loso_split(
            Xn, y, test_subject
        )

        print(f"Subject {test_subject} label distribution:",
                np.bincount(y_train.astype(int)))

        # 1) selezione canale
        X_train_mc = X_train.reshape(len(X_train), -1)  # (N,80)
        X_test_mc  = X_test.reshape(len(X_test), -1)


        # 2) shuffle SOLO del training set
        idx = np.random.permutation(len(y_train))
        X_train_mc = X_train_mc[idx]
        y_train    = y_train[idx]


        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()

        pos_weight = torch.tensor(
            neg / max(pos, 1),
            dtype=torch.float32
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(
            torch.tensor(X_train_mc, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )

        loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = MLP_MC()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


        for ep in range(5):
            loss = train_epoch(model, loader, optimizer, criterion)
            print(f"Epoch {ep} | train loss = {loss:.4f}")

        acc, auc = evaluate_metrics(
            model,
            torch.tensor(X_test_mc, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )

        all_acc.append(acc)
        all_auc.append(auc)


    print(f"Avg ACC: {np.mean(all_acc):.3f} | Avg AUC: {np.mean(all_auc):.3f}")


