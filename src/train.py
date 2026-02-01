import itertools
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from feature_extraction import load_data

# Config

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../results-ml"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]
N_SUBJECTS = 27
N_SESSIONS = 2
N_TRIALS = 25
RANDOM_STATE = 42

USE_STANDARDIZATION = False

def load_classifiers() -> Dict[str, object]:
    return {
        "SVM_Linear": SVC(kernel="linear", C=1.0),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "kNN_3": KNeighborsClassifier(n_neighbors=3),
        "kNN_5": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
    }

def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def all_channel_combos(n_channels: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for r in range(1, n_channels + 1):
        combos.extend(itertools.combinations(range(n_channels), r))
    return combos


def combo_key(combo: Tuple[int, ...]) -> str:
    # Stable, readable key like: EEG.T7+EEG.Pz
    return "+".join(CHANNELS[i] for i in combo)


def build_pipeline(clf) -> Pipeline:
    steps = []
    if USE_STANDARDIZATION:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def flatten_subject_combo(X: np.ndarray, y: np.ndarray, subj: int, combo: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    X shape: (subjects, sessions, trials, channels, features)
    y shape: (subjects, sessions, trials)

    Returns:
      X_flat: (sessions*trials, len(combo)*features)
      y_flat: (sessions*trials,)
    """
    X_sc = X[subj, :, :, combo, :]          # (sessions, trials, K, features)
    X_flat = X_sc.reshape(N_SESSIONS * N_TRIALS, -1)
    y_flat = y[subj].reshape(N_SESSIONS * N_TRIALS).astype(int)
    return X_flat, y_flat


def summarize_per_subject(per_subject: List[float]) -> Dict[str, Any]:
    arr = np.array(per_subject, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "per_subject": [float(x) for x in arr.tolist()],
    }



# Subject-dependent: evaluate all combos

def subject_dependent_all_combos() -> None:
    X, y = load_data()
    classifiers = load_classifiers()

    loo = LeaveOneOut()
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    combos = all_channel_combos(len(CHANNELS))

    std_tag = "std" if USE_STANDARDIZATION else "nostd"
    out_path = RESULTS_DIR / f"subject_dependent_allcombos_{std_tag}.json"

    # results[combo_key][clf_name] = {"LOOCV": {...}, "KFold10": {...}}
    results: Dict[str, Dict[str, Any]] = {}

    for combo in tqdm(combos, desc="Subject-dependent combos"):
        ck = combo_key(combo)
        results[ck] = {}

        for clf_name, clf in classifiers.items():
            pipe = build_pipeline(clf)

            per_subj_loo: List[float] = []
            per_subj_10f: List[float] = []

            for s in range(N_SUBJECTS):
                X_flat, y_flat = flatten_subject_combo(X, y, s, combo)

                acc_loo = cross_val_score(pipe, X_flat, y_flat, cv=loo, scoring="accuracy").mean()
                acc_10f = cross_val_score(pipe, X_flat, y_flat, cv=skf10, scoring="accuracy").mean()

                per_subj_loo.append(float(acc_loo))
                per_subj_10f.append(float(acc_10f))

            results[ck][clf_name] = {
                "LOOCV": summarize_per_subject(per_subj_loo),
                "KFold10": summarize_per_subject(per_subj_10f),
            }

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())



# Cross-subject LOSO: evaluate all combos

def cross_subject_loso_all_combos() -> None:
    X, y = load_data()
    classifiers = load_classifiers()

    combos = all_channel_combos(len(CHANNELS))

    std_tag = "std" if USE_STANDARDIZATION else "nostd"
    out_path = RESULTS_DIR / f"cross_subject_loso_allcombos_{std_tag}.json"

    # results[combo_key][clf_name] = {"mean":..., "std":..., "per_subject":[acc for each held-out subject]}
    results: Dict[str, Dict[str, Any]] = {}

    for combo in tqdm(combos, desc="Cross-subject combos"):
        ck = combo_key(combo)
        results[ck] = {}

        for clf_name, clf in classifiers.items():
            pipe = build_pipeline(clf)

            per_test_subject: List[float] = []

            for test_subject in range(N_SUBJECTS):
                X_train_list, y_train_list = [], []
                X_test, y_test = None, None

                for s in range(N_SUBJECTS):
                    X_flat, y_flat = flatten_subject_combo(X, y, s, combo)

                    if s == test_subject:
                        X_test, y_test = X_flat, y_flat
                    else:
                        X_train_list.append(X_flat)
                        y_train_list.append(y_flat)

                X_train = np.concatenate(X_train_list, axis=0)
                y_train = np.concatenate(y_train_list, axis=0)

                pipe.fit(X_train, y_train)
                acc = float(pipe.score(X_test, y_test))
                per_test_subject.append(acc)

            results[ck][clf_name] = summarize_per_subject(per_test_subject)

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())


if __name__ == "__main__":
    subject_dependent_all_combos()
    cross_subject_loso_all_combos()
