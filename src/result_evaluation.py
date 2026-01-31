import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../results-nostd-bugfix-r0.2"
SUBJECT_DEPENDANT_RESULTS_PATH = RESULTS_DIR / "subject_dependent_channel_results.json"
CROSS_SUBJECT_RESULTS_PATH = RESULTS_DIR / "cross_subject_channel_results.json"

classifier_map = {
    "SVM_Linear": "C1",
    "SVM_RBF": "C2",
    "kNN_3": "C3",
    "kNN_5": "C4",
    "NB": "C5",
    "LDA": "C6"
}

def summarize_subject_dependant_results(df):
    df["C"] = df["classifier"].map(classifier_map)

    df["LOOCV"] = df["acc_loocv"] * 100
    df["KFold"] = df["acc_10fold"] * 100


    loocv_pivot = df.pivot_table(
        index="subject",
        columns="C",
        values="LOOCV",
        aggfunc="max"
    )

    kfold_pivot = df.pivot_table(
        index="subject",
        columns="C",
        values="KFold",
        aggfunc="max"
    )
    loocv_pivot.index = [f"S{int(i)}" for i in loocv_pivot.index]
    kfold_pivot.index = loocv_pivot.index

    final_table = pd.concat(
        [loocv_pivot, kfold_pivot],
        axis=1,
        keys=[
            "LOOCV Accuracy (%)",
            "10-Fold Cross-Validation Accuracy (%)"
        ]
    )

    mean_row = final_table.mean()
    std_row = final_table.std()

    final_table.loc["Mean"] = mean_row
    final_table.loc["SD"] = std_row


    final_table = final_table.round(2)
    final_table.to_csv( RESULTS_DIR / "subject_dependant_classifier_accuracy.csv")

    print("Saved: lie_detection_results_table.csv")


def summarize_best_channels(df, classifier: str):
    df_clf = df[df["classifier"] == classifier].copy()
    df_clf["LOOCV"] = df_clf["acc_loocv"] * 100
    df_clf["KFold"] = df_clf["acc_10fold"] * 100
    best_loocv = (
        df_clf.loc[df_clf.groupby("subject")["LOOCV"].idxmax()]
        .set_index("subject")[["LOOCV", "channel"]]
    )

    best_loocv.columns = ["LOOCV Accuracy (%)", "LOOCV Channel"]

    best_kfold = (
        df_clf.loc[df_clf.groupby("subject")["KFold"].idxmax()]
        .set_index("subject")[["KFold", "channel"]]
    )
    best_kfold.columns = ["10-Fold Accuracy (%)", "10-Fold Channel"]

    final_table = pd.concat([best_loocv, best_kfold], axis=1)
    final_table.index = [f"S{int(i)}" for i in final_table.index]
    final_table.index.name = "Subject"

    final_table = final_table.round(2)
    final_table.to_csv(RESULTS_DIR / f"best_channel_per_subject_{classifier}.csv")

def summarize_cross_subject_results(df):
    df["C"] = df["classifier"].map(classifier_map)

    df["LOSOCV"] = df["acc_losocv"] * 100

    losocv_pivot = df.pivot_table(
        index="test_subject",
        columns="C",
        values="LOSOCV",
        aggfunc="max"
    )

    mean_row = losocv_pivot.mean()
    std_row = losocv_pivot.std()

    losocv_pivot.loc["Mean"] = mean_row
    losocv_pivot.loc["SD"] = std_row

    losocv_pivot = losocv_pivot.round(2)
    losocv_pivot.to_csv( RESULTS_DIR / "cross_subject_classifier_accuracy.csv")

    print("Saved: cross_subject_classifier_accuracy.csv")


if __name__ == '__main__':
        df_sd = pd.read_json(SUBJECT_DEPENDANT_RESULTS_PATH)
        df_cs = pd.read_json(CROSS_SUBJECT_RESULTS_PATH)
        print(df_cs.head())
        summarize_subject_dependant_results(df_sd)
        summarize_best_channels(df_sd, "SVM_Linear")

        summarize_cross_subject_results(df_cs)