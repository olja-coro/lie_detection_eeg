import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import csv
from feature_extraction import load_data
from utils import (
    set_seed, make_loso_channel_split, scale_train_test, stratified_train_val_split,
    train_one_split_bce, predict_proba, accuracy_from_probs
)
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../mlp_loocv_subjects_channels_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

def subject_loocv_one_channel(
    X, y,
    subject_idx: int,
    channel_idx: int,
    seed: int = 42,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    h1: int = 32,
    h_pre: int = 16,
    dropout: float = 0.1,
):
    """
    LOOCV inside one subject:
      - Each fold holds out 1 trial (out of 50 = 2*25)
      - Train on remaining 49 trials
      - Returns mean accuracy across 50 folds
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # subject data for this channel
    # X_sub: (2, 25, n_feat) -> flatten to (50, n_feat)
    n_feat = X.shape[-1]
    X_sub_raw = X[subject_idx, :, :, channel_idx, :].reshape(-1, n_feat)
    y_sub = y[subject_idx, :, :].reshape(-1).astype(np.int64)  # (50,)

    # sanity
    assert X_sub_raw.shape[0] == y_sub.shape[0]
    assert set(np.unique(y_sub)).issubset({0, 1})

    # scale per subject using ALL trials (this is OK for within-subject LOOCV
    # because scaling doesn't use labels; if you want strictness, fit scaler per fold.)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler(quantile_range=(25, 75))
    X_sub = scaler.fit_transform(X_sub_raw)

    fold_accs = []

    n_samples = X_sub.shape[0]  # 50
    for holdout in range(n_samples):
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[holdout] = True

        X_test = X_sub[test_mask]
        y_test = y_sub[test_mask]

        X_train = X_sub[~test_mask]
        y_train = y_sub[~test_mask]

        # small validation split from the 49 training trials (stratified)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train
        )

        model = MLP(in_dim=n_feat, h1=h1, h_pre=h_pre, dropout=dropout)

        model = train_one_split_bce(
            model,
            X_tr, y_tr, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            log_every=50  # quiet
        )

        probs = predict_proba(model, X_test, device=device)
        acc = accuracy_from_probs(probs, y_test)
        fold_accs.append(acc)

    return float(np.mean(fold_accs))
def loocv_all_subjects_one_channel(
    channel_idx: int,
    seed: int = 42,
    **train_kwargs
):
    X, y = load_data()
    n_subjects = X.shape[0]

    results = []
    for s in range(n_subjects):
        subj_id = f"S{s+1:02d}"
        acc = subject_loocv_one_channel(
            X, y,
            subject_idx=s,
            channel_idx=channel_idx,
            seed=seed,
            **train_kwargs
        )
        results.append((subj_id, acc))
        print(f"Channel {channel_idx} | {subj_id} LOOCV acc: {acc*100:.2f}%")

    out_path = RESULTS_DIR / f"channel_{channel_idx}_loocv.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "loocv_acc"])
        for subj, acc in results:
            w.writerow([subj, f"{acc:.4f}"])

    print(f"Saved: {out_path.resolve()}")
    return results

if __name__ == "__main__":
    for ch in range(5):
        print(f"\n=== LOOCV within-subject | channel {ch} ===")
        loocv_all_subjects_one_channel(
            channel_idx=ch,
            seed=42,
            epochs=150,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-4,
            patience=20,
            h1=64,
            h_pre=32,
            dropout=0.1,
        )