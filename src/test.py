import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

RESULTS_DIR = BASE_DIR / "../mlp_loco_subjects_channels_results"
OUT_FILE = RESULTS_DIR / "all_subjects.csv"

rows = []

for csv_file in sorted(RESULTS_DIR.glob("*.csv")):
    subject = csv_file.stem 

    df = pd.read_csv(csv_file)

    # map channel index → channel name
    df["channel"] = df["held_out_channel"].apply(lambda x: f"channel{x+1}")

    # make one-row dataframe
    row = (
        df.set_index("channel")["accuracy"]
          .reindex([f"channel{i}" for i in range(1, 6)])
    )

    row["subject"] = subject
    rows.append(row)

# concatenate all subjects
final_df = pd.DataFrame(rows)

# reorder columns
final_df = final_df[["subject", "channel1", "channel2", "channel3", "channel4", "channel5"]]

# save with semicolon separator
final_df.to_csv(OUT_FILE, sep=";", index=False)

print("Saved:", OUT_FILE)
