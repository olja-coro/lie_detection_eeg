import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

FILES = {
    "channel_wise_subject": BASE_DIR / "results-nostd-bugfix-r0.2" / "subject_dependent_channel_results.json",
    "concat_nostd_subject": BASE_DIR / "results-concatenated" / "subject_dependent_concat_nostd.json",
    "concat_std_subject": BASE_DIR / "results-concatenated" / "subject_dependent_concat_std.json",

    "channel_wise_cross": BASE_DIR / "results-nostd-bugfix-r0.2" / "cross_subject_channel_results.json",
    "concat_nostd_cross": BASE_DIR / "results-concatenated" / "cross_subject_concat_nostd.json",
    "concat_std_cross": BASE_DIR / "results-concatenated" / "cross_subject_concat_std.json",
}

def load_acc_list(path, key):
    with open(path, "r") as f:
        data = json.load(f)
    return [d[key] for d in data if key in d]

def summarize(name, accs):
    accs = np.array(accs)
    return f"{name:25s} | mean={accs.mean():.4f} ± {accs.std():.4f}"

print("\n===== SUBJECT-DEPENDENT =====")
print(summarize(
    "Channel-wise",
    load_acc_list(FILES["channel_wise_subject"], "acc_loocv")
))
print(summarize(
    "Concat (no-std)",
    load_acc_list(FILES["concat_nostd_subject"], "acc_loocv")
))
print(summarize(
    "Concat (std)",
    load_acc_list(FILES["concat_std_subject"], "acc_loocv")
))

print("\n===== CROSS-SUBJECT =====")
print(summarize(
    "Channel-wise",
    load_acc_list(FILES["channel_wise_cross"], "acc_losocv")
))
print(summarize(
    "Concat (no-std)",
    load_acc_list(FILES["concat_nostd_cross"], "acc_losocv")
))
print(summarize(
    "Concat (std)",
    load_acc_list(FILES["concat_std_cross"], "acc_losocv")
))
