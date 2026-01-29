from pathlib import Path
import json
from collections import defaultdict
import numpy as np

# current file(src/)
BASE_DIR = Path(__file__).resolve().parent

# lie_detection_eeg/
PROJECT_DIR = BASE_DIR.parent


INPUT_JSON = PROJECT_DIR / "results-nostd-bugfix-r0.2" / "subject_dependent_channel_results.json"
OUTPUT_JSON = PROJECT_DIR / "results-nostd-bugfix-r0.2" / "channel_ranking_subject_dependent.json"

# load file
with open(INPUT_JSON, "r") as f:
    results = json.load(f)


channel_scores = defaultdict(list)

for entry in results:
    channel = entry["channel"]
    
    
    acc = entry.get("acc_loocv") or entry.get("acc_10fold")
    
    if acc is not None:
        channel_scores[channel].append(acc)

channel_avg = {
    ch: float(np.mean(scores))
    for ch, scores in channel_scores.items()
}

# from best to the worst
channel_avg_sorted = dict(
    sorted(channel_avg.items(), key=lambda x: x[1], reverse=True)
)


print("Channel ranking (by average accuracy):")
for ch, avg in channel_avg_sorted.items():
    print(f"{ch}: {avg:.4f}")

# save
with open(OUTPUT_JSON, "w") as f:
    json.dump(channel_avg_sorted, f, indent=4)

print(f"\nSaved channel ranking to {OUTPUT_JSON}")
