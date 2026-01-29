import numpy as np
from feature_extraction import load_data
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../results-concatenated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Best channels found by average (from subject-dependent ranking)
BEST_CHANNELS = ["EEG.T8", "EEG.Pz", "EEG.T7"]  # puoi ridurre a 2 se vuoi

# to apply standardization
USE_STANDARDIZATION = True   




CHANNELS = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]
N_SUBJECTS = 27
N_SESSIONS = 2
N_TRIALS = 25

RANDOM_STATE = 42

BEST_CH_IDX = [CHANNELS.index(ch) for ch in BEST_CHANNELS]
def load_classifiers() -> Dict[str, object]:
    # initialize classifiers
    return {
        "SVM_Linear": SVC(kernel="linear", C=1.0),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "kNN_3": KNeighborsClassifier(n_neighbors=3),
        "kNN_5": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
    }
'''
def subject_dependent_experiment():
    X, y = load_data()

    classifiers = load_classifiers()

    out_path = RESULTS_DIR / "subject_dependent_channel_results.json"

    # Cross Validation
    loo = LeaveOneOut()
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    results: List[Dict[str, Any]] = []

    #iterate over subjects and channels
    for s in range(N_SUBJECTS):
        for ch_idx, ch_name in tqdm(
            list(enumerate(CHANNELS)),
            desc=f"Channels (S{s+1})",
            leave=False
        ):
            X_sc = X[s, :, :, ch_idx, :].reshape(N_SESSIONS * N_TRIALS, -1)
            y_sc = y[s, :, :].reshape(N_SESSIONS * N_TRIALS).astype(int)

            for clf_name, clf in classifiers.items():
                # Standardize features for SVM/kNN/LDA stability
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])

                acc_loo = cross_val_score(pipe, X_sc, y_sc, cv=loo, scoring="accuracy").mean()
                acc_10f = cross_val_score(pipe, X_sc, y_sc, cv=skf10, scoring="accuracy").mean()

                results.append({
                    "subject": s + 1,
                    "channel": ch_name,
                    "classifier": clf_name,
                    "acc_loocv": acc_loo,
                    "acc_10fold": acc_10f,
                })

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())
'''
def subject_dependent_experiment():
    X, y = load_data()
    classifiers = load_classifiers()

    std_tag = "std" if USE_STANDARDIZATION else "nostd"
    out_path = RESULTS_DIR / f"subject_dependent_concat_{std_tag}.json"



    loo = LeaveOneOut()
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    results: List[Dict[str, Any]] = []

    for s in range(N_SUBJECTS):

        # select + concatenate best channels
        X_sc = X[s, :, :, BEST_CH_IDX, :]  
        # shape: (sessions, trials, K, features)

        X_sc = X_sc.reshape(N_SESSIONS * N_TRIALS, -1)
        y_sc = y[s].reshape(N_SESSIONS * N_TRIALS).astype(int)

        for clf_name, clf in classifiers.items():

            steps = []
            if USE_STANDARDIZATION:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", clf))

            pipe = Pipeline(steps)

            acc_loo = cross_val_score(pipe, X_sc, y_sc, cv=loo, scoring="accuracy").mean()
            acc_10f = cross_val_score(pipe, X_sc, y_sc, cv=skf10, scoring="accuracy").mean()

            results.append({
                "subject": s + 1,
                "channels": BEST_CHANNELS,
                "classifier": clf_name,
                "acc_loocv": acc_loo,
                "acc_10fold": acc_10f,
                "standardized": USE_STANDARDIZATION,
            })

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())


def save_json(obj: Dict[str, Any], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

'''
def cross_subject_experiment():
    X, y = load_data()
    classifiers = load_classifiers()

    out_path = RESULTS_DIR / "cross_subject_channel_results.json"

    results: List[Dict[str, Any]] = []

    # For each channel, do leave-one-subject-out evaluation
    for ch_idx, ch_name in tqdm(enumerate(CHANNELS)):
        # LOSOCV
        for test_subject in range(N_SUBJECTS):

            X_train_list: List[np.ndarray] = []
            y_train_list: List[np.ndarray] = []

            X_test = None
            y_test = None

            # Build train/test splits by subject
            for s in range(N_SUBJECTS):
                X_sc = X[s, :, :, ch_idx, :].reshape(N_SESSIONS * N_TRIALS, -1)   # (50, n_features)
                y_sc = y[s, :, :].reshape(N_SESSIONS * N_TRIALS).astype(int)     # (50,)

                if s == test_subject:
                    X_test = X_sc
                    y_test = y_sc
                else:
                    X_train_list.append(X_sc)
                    y_train_list.append(y_sc)

            if X_test is None or y_test is None:
                raise RuntimeError("Failed to create test split. Check subject indexing.")

            X_train = np.concatenate(X_train_list, axis=0)  # (26*50, n_features)
            y_train = np.concatenate(y_train_list, axis=0)  # (26*50,)

            # Fit/evaluate each classifier on this fold
            for clf_name, clf in classifiers.items():
                pipe = Pipeline([
                    #("scaler", StandardScaler()),
                    ("clf", clf),
                ])

                pipe.fit(X_train, y_train)
                acc = float(pipe.score(X_test, y_test))

                results.append({
                    "test_subject": int(test_subject + 1),  # held-out subject ID (1-based)
                    "channel": ch_name,
                    "classifier": clf_name,
                    "acc_losocv": acc,
                })

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())
'''

def cross_subject_experiment():
    X, y = load_data()
    classifiers = load_classifiers()

    std_tag = "std" if USE_STANDARDIZATION else "nostd"
    out_path = RESULTS_DIR / f"cross_subject_concat_{std_tag}.json"

    results: List[Dict[str, Any]] = []

    for test_subject in range(N_SUBJECTS):

        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        for s in range(N_SUBJECTS):

            X_sc = X[s, :, :, BEST_CH_IDX, :].reshape(N_SESSIONS * N_TRIALS, -1)
            y_sc = y[s].reshape(N_SESSIONS * N_TRIALS).astype(int)

            if s == test_subject:
                X_test, y_test = X_sc, y_sc
            else:
                X_train_list.append(X_sc)
                y_train_list.append(y_sc)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        for clf_name, clf in classifiers.items():

            steps = []
            if USE_STANDARDIZATION:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", clf))

            pipe = Pipeline(steps)

            pipe.fit(X_train, y_train)
            acc = float(pipe.score(X_test, y_test))

            results.append({
                "test_subject": test_subject + 1,
                "channels": BEST_CHANNELS,
                "classifier": clf_name,
                "acc_losocv": acc,
                "standardized": USE_STANDARDIZATION,
            })

    save_json(results, out_path)
    print("Saved JSON to:", out_path.resolve())


if __name__ == '__main__':
    subject_dependent_experiment()
    cross_subject_experiment()