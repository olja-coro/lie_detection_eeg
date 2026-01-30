import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# Reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Cross-subject LOSO split for ONE channel
def make_loso_channel_split(X, y, test_subj: int, channel_idx: int):
    """
    X: (n_subjects, 2, n_trials, n_channels, n_feat)
    y: (n_subjects, 2, n_trials)
    Returns:
      X_train_raw, y_train, X_test_raw, y_test  (raw = unscaled)
    """
    n_subjects, _, _, _, n_feat = X.shape
    train_subjects = [s for s in range(n_subjects) if s != test_subj]

    X_train_raw = X[train_subjects, :, :, channel_idx, :].reshape(-1, n_feat)
    y_train = y[train_subjects, :, :].reshape(-1).astype(np.int64)

    X_test_raw = X[test_subj, :, :, channel_idx, :].reshape(-1, n_feat)
    y_test = y[test_subj, :, :].reshape(-1).astype(np.int64)

    return X_train_raw, y_train, X_test_raw, y_test


def scale_train_test(X_train_raw, X_test_raw):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, scaler


def stratified_train_val_split(X_train, y_train, val_size=0.1, seed=42):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )
    return X_tr, X_val, y_tr, y_val



# Metrics / predictions
@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    probs = model(xb)
    return probs.detach().cpu().numpy()


def accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> float:
    y_pred = (probs >= thr).astype(np.int64)
    return float((y_pred == y_true).mean())


@torch.no_grad()
def eval_on_loader_bce(model: nn.Module, loader: DataLoader, device: str):
    """
    For BCELoss, model must output probabilities in [0,1].
    """
    model.eval()
    loss_fn = nn.BCELoss()

    losses = []
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        probs = model(xb)  # (batch,)
        loss = loss_fn(probs, yb)
        losses.append(loss.item())

        preds = (probs >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.numel()

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    acc = correct / total if total > 0 else 0.0
    return mean_loss, acc



# Generic training loop (BCELoss)
def train_one_split_bce(
    model: nn.Module,
    X_tr, y_tr, X_val, y_val,
    epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4,
    patience=25, device="cuda",
    log_every=10
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    bad = 0

    model.to(device)

#    for epoch in tqdm(range(epochs), desc=f"Training",leave=False):
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            probs = model(xb)
            loss = loss_fn(probs, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("inf")
        mean_val, val_acc = eval_on_loader_bce(model, val_loader, device=device)

        improved = mean_val < best_val - 1e-6
        if improved:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        # if (epoch % log_every == 0) or (epoch == 1) or improved or (epoch == epochs):
        #     print(
        #         f"  epoch {epoch:03d}/{epochs} | "
        #         f"train_loss={mean_train:.4f} | val_loss={mean_val:.4f} | "
        #         f"val_acc={val_acc*100:.2f}% | patience={bad}/{patience}"
        #     )

        if bad >= patience:
            #print(f"  early stop at epoch {epoch} (best val_loss={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model