import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
from entropy_features import extract_single_channel_features
from functools import partial
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../data/BandPass_Filtered"
STIM_FILE = BASE_DIR / "../data/Subject_Stimuli.csv" # lie = 0, truth = 1, stimuly 1 = 1st rosary received by the subject, stimuli 2 = 2nd rosary received by the subject
OUT_PATH = BASE_DIR / "../data/data.npz"

SAMPLING_FREQUENCY = 128 # 128hz -> 128 samples points each second
EXPERIMENT_DURATION = 3 # 3 seconds
CHANNEL_POINTS = 128 * 3 # 384 points for each experiment
N_EXPERIMENTS = 25
PARTECIPANTS = 27
SESSIONS = 2
COLS = ["subject", "session", "stimuli_1", "stimuli_2", "is_truth"]
CHANNELS = ["EEG.AF3", "EEG.T7", "EEG.Pz", "EEG.T8", "EEG.AF4"]

# Entropy values
K_MAX = 10
M = 2
N = 2
R_FACTOR = 0.15
FEATURES_DIM = 1 + K_MAX + 5 # 1 (FE) + 10 (TSMFE) + 5 (HMFE)

def preprocess_data():
    """
    Reads raw EEG CSV data and metadata, structures them into tensors, and saves a compressed .npz dataset.

    Output (.npz):
        X: (27, 2, 25, 5, n_features)  -> (subject, session, experiment, channel, features)
        y: (27, 2, 25)                 -> per-trial labels (each experiment inherits session label)
    """
    df = pd.read_csv(STIM_FILE, names=COLS, header=0)
    df["subject"] = df["subject"].str.extract(r"(\d+)").astype(int)
    df["session"] = df["session"].str.extract(r"(\d+)").astype(int)
    
    df = df.sort_values(by=["subject", "session"]) # crucial

    expected_rows = PARTECIPANTS * 2
    assert len(df) == expected_rows, f"Reshape is not safe"

    X = np.zeros((PARTECIPANTS, SESSIONS, N_EXPERIMENTS, len(CHANNELS), FEATURES_DIM), dtype=np.float64)
    y = np.zeros((PARTECIPANTS, SESSIONS, N_EXPERIMENTS), dtype=np.float64)

    preprocess_fn = partial(extract_single_channel_features, k_max=K_MAX, m=M, n=N, r_factor=R_FACTOR)

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        csv_path = DATA_DIR / f'S{row.subject}S{row.session}.csv'

        channel_data = pd.read_csv(
            csv_path,
            usecols=CHANNELS,
            dtype={ch: "float64" for ch in CHANNELS}
        )[CHANNELS].to_numpy().T # Shape: (5, 9600)

        # reshape into experiments: (5, 25, 384)
        experiment_data = channel_data.reshape(len(CHANNELS), N_EXPERIMENTS, CHANNEL_POINTS)

        # extract features per experiment per channel
        # result: (25, 5, n_features)
        experiment_features = np.empty((N_EXPERIMENTS, len(CHANNELS), FEATURES_DIM), dtype=np.float64)
        for t in range(N_EXPERIMENTS):
            experiment_t = experiment_data[:, t, :]  # (5, 384)
            experiment_features[t] = np.apply_along_axis(preprocess_fn, axis=-1, arr=experiment_t)

        s = row.subject - 1
        sess = row.session - 1
        X[s, sess] = experiment_features
        y[s, sess, :] = int(row.is_truth)

    print(f"Created dataset with shape: X={X.shape}, y={y.shape}")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        OUT_PATH,
        X=X, # Features: 27 x 2 x 5 x 25 x 384
        y=y, # Labels: 27 x 2 x 25
    )

    print("Dataset saved to", OUT_PATH.resolve().absolute())



def load_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load the dataset from .npz file.
    
    Returns:
        X: (27, 2, 25, 5, n_features)
        y: (27, 2, 25)
    """
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"File {OUT_PATH} doesn't exist. Run preprocess_data() first.")
    data = np.load(OUT_PATH)
    
    return data['X'], data['y']



if __name__ == '__main__':
    preprocess_data()