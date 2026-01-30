# optuna_search.py
import numpy as np
import optuna
from mlp import MLP
import torch
import json
from pathlib import Path
from feature_extraction import load_data
from utils import (
    set_seed,
    make_loso_channel_split, scale_train_test, stratified_train_val_split,
    train_one_split_bce, predict_proba, accuracy_from_probs
)

BASE_DIR = Path(__file__).parent
RESULTS_PATH= BASE_DIR / "../results_optuna_mlp_per_channel.json"

def objective_factory(channel_idx: int, seed: int = 42):
    X, y = load_data()
    n_subjects = X.shape[0]
    n_feat = X.shape[-1]
    device = "cpu"

    def objective(trial: optuna.Trial):
        set_seed(seed)

        # --- search space ---
        h1 = trial.suggest_int("h1", 32, 256, step=32)
        h_pre = trial.suggest_int("h_pre", 8, 64, step=8)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = 1e-4
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        val_size = 0.1
        patience = 25
        epochs = 300

        fold_accs = []
        for test_subj in range(n_subjects):
            X_train_raw, y_train, X_test_raw, y_test = make_loso_channel_split(X, y, test_subj, channel_idx)
            X_train, X_test, _ = scale_train_test(X_train_raw, X_test_raw)
            X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train, y_train, val_size=val_size, seed=seed)

            model = MLP(in_dim=n_feat, h1=h1, h_pre=h_pre, dropout=dropout)

            model = train_one_split_bce(
                model,
                X_tr, y_tr, X_val, y_val,
                epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                patience=patience, device=device, log_every=999999
            )

            probs = predict_proba(model, X_test, device=device)
            acc = accuracy_from_probs(probs, y_test)
            fold_accs.append(acc)

            # optional pruning
            trial.report(float(np.mean(fold_accs)), step=test_subj)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_accs))

    return objective


def run_optuna(channel_idx: int, n_trials: int = 30, seed: int = 42):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=f"mlp_channel_{channel_idx}_search"
    )

    study.optimize(
        objective_factory(channel_idx=channel_idx, seed=seed),
        n_trials=n_trials,
        n_jobs=1
    )

    best = {
        "channel": channel_idx,
        "best_value": float(study.best_trial.value),
        "best_params": dict(study.best_trial.params),
        "n_trials": n_trials,
        "seed": seed,
        "study_name": study.study_name,
    }

    print(f"\nBest trial channel {channel_idx}:")
    print(f"  value: {best['best_value']:.4f}")
    print("  params:")
    for k, v in best["best_params"].items():
        print(f"    {k}: {v}")

    return best


if __name__ == "__main__":

    all_results = {}

    for ch in range(5):
        print(f"\n=== OPTUNA SEARCH FOR CHANNEL {ch} ===")
        best = run_optuna(channel_idx=ch, n_trials=15)
        all_results[f"channel_{ch}"] = best

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved Optuna results to: {RESULTS_PATH.resolve()}")