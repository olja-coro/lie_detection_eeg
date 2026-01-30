import numpy as np
import torch
import torch.nn as nn
from feature_extraction import load_data
from utils import (
    set_seed, make_loso_channel_split, scale_train_test, stratified_train_val_split,
    train_one_split_bce, predict_proba, accuracy_from_probs
)

class MLP(nn.Module):
    """
    Outputs probabilities (0..1) so you can use nn.BCELoss().
    Includes an extra non-linear layer before the final output.
    """
    def __init__(self, in_dim: int, h1=64, h_pre=16, dropout=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h_pre),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(h_pre, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feat(x)
        p = self.out(x).squeeze(-1)  # (batch,)
        return p


def loso_mlp_one_channel(
    channel_idx: int,
    seed: int = 42,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    val_size: float = 0.1,
    h1: int = 64,
    h_pre: int = 16,
    dropout: float = 0.2,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = load_data()
    n_subjects, n_classes, n_trials, n_channels, n_feat = X.shape
    assert y.shape == (n_subjects, n_classes, n_trials)

    fold_accs = []

    print(f"Device: {device}")
    print(f"Data: subjects={n_subjects}, classes={n_classes}, trials={n_trials}, channels={n_channels}, feat={n_feat}")
    print(f"Channel: {channel_idx}")

    for test_subj in range(n_subjects):
        print(f"\n=== LOSO fold {test_subj+1}/{n_subjects} | test subject S{test_subj+1:02d} | channel {channel_idx} ===")

        X_train_raw, y_train, X_test_raw, y_test = make_loso_channel_split(X, y, test_subj, channel_idx)
        X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)
        X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train, y_train, val_size=val_size, seed=seed)

        model = MLP(
            in_dim=n_feat, h1=h1, h_pre=h_pre, dropout=dropout
        )

        model = train_one_split_bce(
            model,
            X_tr, y_tr, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            patience=patience, device=device, log_every=10
        )

        probs = predict_proba(model, X_test, device=device)
        acc = accuracy_from_probs(probs, y_test)
        fold_accs.append(acc)

        print(f"--> Test subject S{test_subj+1:02d}: acc={acc*100:.2f}%")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"\nLOSO mean acc (channel {channel_idx}): {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    return fold_accs


if __name__ == "__main__":
    loso_mlp_one_channel(channel_idx=1,
    h1= 32,
    h_pre= 24,
    dropout= 0.4079475822784378,
    lr= 0.0007929784638425019,
    weight_decay= 0.006282612205731881,
    batch_size= 64,
    val_size= 0.07116458232513981,
    patience= 26,
    epochs= 80)