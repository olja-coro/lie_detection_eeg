import numpy as np
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("../data/Preprocessing/1_BandPass_Filtered")
STIM_FILE = Path("../data/Subject_Stimuli.xlsx")
OUT_PATH = Path("../data/processed/trials.npz")

FS = 128                     
TRIAL_DURATION = 3.0          
SAMPLES_PER_TRIAL = int(FS * TRIAL_DURATION) 
N_TRIALS = 25            

CHANNELS = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]


def load_all_data():
    all_trials = []
    all_labels = []
    all_subject_ids = []
    all_session_ids = []

    meta = pd.read_excel(STIM_FILE)[['SUBJECT','SESSION','STIMULI 1','STIMULI 2','LIE/TRUTH']]
    meta["label_str"] = meta["LIE/TRUTH"].map({0: "lie", 1: "truth"})

    for _, row in meta.iterrows():
        trials, labels, subj, sess = build_trials_for_session(row, DATA_DIR)
        
        all_trials.append(trials)
        all_labels.append(labels)
        all_subject_ids.extend([subj] * len(labels))
        all_session_ids.extend([sess] * len(labels))

    # Concatenate over sessions
    X_all = np.concatenate(all_trials, axis=0)
    y_all = np.concatenate(all_labels, axis=0)

    all_subject_ids = np.array(all_subject_ids)
    all_session_ids = np.array(all_session_ids)
    
    print("Final dataset shape:", X_all.shape)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        OUT_PATH,
        X=X_all,
        y=y_all,
        subject=all_subject_ids,
        session=all_session_ids,
        channels=np.array(CHANNELS)
    )
    print("Dataset saved to:", OUT_PATH)



def build_trials_for_session(row, data_dir: Path):
    """
    row: one row from the metadata DataFrame
    returns:
        trials: np.ndarray, shape (N_TRIALS, n_samples, n_channels)
        labels: np.ndarray, shape (N_TRIALS,) with 0/1
        subject: str
        session: str
    """
    subject = row["SUBJECT"]
    session = row["SESSION"]
    label_num = int(row["LIE/TRUTH"])  # 0 = lie, 1 = truth
    
    fname = f"{subject}{session}.csv"
    csv_path = data_dir / fname
    
    df = pd.read_csv(csv_path)
    
    # Keep only the 5 EEG channels
    X = df[CHANNELS].values
    
    trials = []
    labels = []
    
    for t in range(N_TRIALS):
        start = t * SAMPLES_PER_TRIAL
        end = start + SAMPLES_PER_TRIAL
        epoch = X[start:end, :]
        trials.append(epoch)
        labels.append(label_num)
    
    trials = np.stack(trials, axis=0)
    labels = np.array(labels) 
    
    return trials, labels, subject, session


if __name__ == '__main__':
    load_all_data()