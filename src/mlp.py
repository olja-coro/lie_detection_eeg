import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


data = np.load("data/data.npz")
X = data["X"]
y = data["y"]

# X: subjects × classes × trials × channels × features
# y: subjects × classes × trials


print("X:", X.shape)
print("y:", y.shape)

RANDOM_STATE = 42
CHANNEL_IDX = 2   
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

X_subj = X[0, :, :, CHANNEL_IDX, :]  # (2,25,16)
y_subj = y[0]                        # (2,25)

X_flat = X_subj.reshape(-1, 16)      # (50,16)
y_flat = y_subj.reshape(-1)          # (50,)

print(X_flat.shape, y_flat.shape)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

binaryCrossEntropy = nn.BCEWithLogitsLoss()




def get_loso_split(X, y, test_subject, channel_idx):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for subj in range(X.shape[0]):
        X_subj = X[subj, :, :, channel_idx, :]  # (2,25,16)
        y_subj = y[subj]                        # (2,25)

        X_flat = X_subj.reshape(-1, 16)
        y_flat = y_subj.reshape(-1)

        if subj == test_subject:
            X_test.append(X_flat)
            y_test.append(y_flat)
        else:
            X_train.append(X_flat)
            y_train.append(y_flat)

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    return X_train, y_train, X_test, y_test



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



#testing of all the code

all_acc = []

for test_subject in range(27):
    X_train, y_train, X_test, y_test = get_loso_split(
        X, y, test_subject, CHANNEL_IDX
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    binaryCrossEntropy = nn.BCEWithLogitsLoss()

    for _ in range(EPOCHS):
        train(model, train_loader, optimizer, binaryCrossEntropy)

    acc = evaluate(
        model,
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    print(f"Subject {test_subject} | Acc: {acc:.2f}")
    all_acc.append(acc)

print("Average LOSO accuracy:", np.mean(all_acc))
