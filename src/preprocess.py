import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
from entropy_features import extract_sigle_channel_features
from functools import partial
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../data/BandPass_Filtered"
STIM_FILE = BASE_DIR / "../data/Subject_Stimuli.csv" # lie = 0, truth = 1, stimuly 1 = 1st rosary received by the subject, stimuli 2 = 2nd rosary received by the subject
OUT_PATH = BASE_DIR / "../data/data.npz"

SAMPLING_FREQUENCY = 128 # 128hz -> 128 samples points each second
EXPERIMENT_DURATION = 75 # 75 seconds
CHANNEL_POINTS = 128 * 75 # 9600 points for each experiment
PARTECIPANTS = 27
COLS = ["subject", "session", "stimuli_1", "stimuli_2", "is_truth"]
CHANNELS = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]

# Entropy values

K_MAX = 10
M = 2
N = 2
R_FACTOR = 0.15


def preprocess_data():
    """
    Reads raw EEG CSV data and metadata, structures them into tensors, and saves a compressed .npz dataset.

    Output (.npz):
        X: (27, 2, 5, Unknown) -> (Subjects, Sessions, Channels, Features)
        y: (27, 2)             -> (Subjects, Sessions)
    """
    df = pd.read_csv(STIM_FILE, names=COLS, header=0)
    df["subject"] = df["subject"].str.extract(r"(\d+)").astype(int)
    df["session"] = df["session"].str.extract(r"(\d+)").astype(int)
    
    df = df.sort_values(by=["subject", "session"]) # crucial

    expected_rows = PARTECIPANTS * 2
    assert len(df) == expected_rows, f"Reshape is not safe"

    X_list = []
    y_list = []

    preprocess_fn = partial(extract_sigle_channel_features, k_max=K_MAX, m=M, n=N, r_factor=R_FACTOR)

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        csv_path = DATA_DIR / f'S{row.subject}S{row.session}.csv'

        channel_data = pd.read_csv(
            csv_path,
            usecols=CHANNELS,
            dtype={ch: "float64" for ch in CHANNELS}
        )[CHANNELS].to_numpy().T # Shape: (5, 9600)

        # result shape: (5, N_FEATURES)
        entropy_features = np.apply_along_axis(preprocess_fn, axis=-1, arr=channel_data)
        
        X_list.append(entropy_features)
        y_list.append(row.is_truth)

    
    X_raw = np.array(X_list) 
    
    X = X_raw.reshape(PARTECIPANTS, 2, len(CHANNELS), -1)
    
    y = np.array(y_list).reshape(PARTECIPANTS, 2)

    print(f"Created dataset with shape: X={X.shape}, y={y.shape}")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        OUT_PATH,
        X=X, # Features: 27 x 2 x 5 x 9600
        y=y, # Labels: 27 x 2
    )

    print("Dataset saved to", OUT_PATH.resolve().absolute())



def load_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load the dataset from .npz file.
    
    Returns:
        tuple: (X, y)
        X shape: (PARTECIPANTS, 2, 5, 9600)
        y shape: (PARTECIPANTS, 2)
    """
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"File {OUT_PATH} doesn't exist. Exec write_npz() first.")

    data = np.load(OUT_PATH)
    
    return data['X'], data['y']



if __name__ == '__main__':
    preprocess_data()