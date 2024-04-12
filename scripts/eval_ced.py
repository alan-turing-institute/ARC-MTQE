import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get directory names")

    parser.add_argument("-p", "--path", required=True, help="Path to predictions directory")

    return parser.parse_args()


def load_data(pred_dir, lp, model):
    """
    Load model predictions and gold truth labels for language pair.
    """

    main_dir = os.getcwd()

    # predictions
    pred_path = os.path.join(pred_dir, f"{lp}_{model}.csv")
    df_pred = pd.read_csv(pred_path)

    # gold labels
    data_dir = os.path.join(main_dir, "data", "mlqe-pe", "data", "catastrophic_errors_goldlabels")
    labels_path = os.path.join(data_dir, f"{lp.replace('-', '')}_majority_test_goldlabels", "goldlabels.txt")
    df_labels = pd.read_csv(labels_path, sep="\t", header=None, names=["lang_pair", "ref", "idx", "label"])

    # merge on sentence indexes
    df_results = pd.merge(df_pred, df_labels, on="idx")

    return df_results


def main():
    args = parse_args()
    model = "cometkiwi"

    language_pairs = LI_LANGUAGE_PAIRS_WMT_21_CED
    for lp in language_pairs:
        df_results = load_data(args.path, lp, model)

        # higher COMET score --> higher confidence it is NOT an error
        # However, we want the positive class to represent ERRORS
        # Therefore the labels are:  ERROR = 1, NOT = 0
        y_true = np.where(df_results["label"] == "NOT", 0, 1)
        y_pred = 1 - df_results["comet_score"]

        # RESULTS
        # 1. Precision, Recall, FPR, TPR
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)

        pr_display = PrecisionRecallDisplay(precision=prec, recall=rec).plot()
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

        min_threshold = round(y_pred.min(), 2)
        max_threshold = round(y_pred.max(), 2)
        # 2. MCC
        # tresholds for binarizing COMET output
        thresholds = np.arange(min_threshold, max_threshold, 0.01)
        # thresholds = np.arange(0.01, 0.99, 0.01)
        # thresholds = np.arange(0.05, 0.9, 0.05)
        mccs = []
        for t in thresholds:
            # the model is treated as an error detector
            # i.e., scores above threshold are "ERROR" predictions
            # y_hat = (df_results["comet_score"] >= t).astype(int)
            y_hat = (y_pred >= t).astype(int)
            mcc = matthews_corrcoef(y_true, y_hat)
            mccs.append(mcc)

        # PLOT
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        roc_display.plot(ax=ax1)
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, color="black", ls="--")
        pr_display.plot(ax=ax2)
        ax2.axhline(y=sum(y_true) / len(y_true), color="black", linestyle="--")
        ax3.legend(loc="lower right")

        idx_max = np.argmax(mccs)
        best_threshold = thresholds[idx_max]
        label = f"{best_threshold:.2f}"
        ax3.scatter(thresholds, mccs, s=10, label=label)
        ax3.set_ylabel("MCC")
        ax3.set_xlabel("Threshold")
        ax3.legend(markerscale=0, loc="upper left", title="Best threshold")

        y_pred_binary = y_pred > best_threshold
        score_precision = precision_score(y_true, y_pred_binary)
        score_recall = recall_score(y_true, y_pred_binary)
        score_f1 = f1_score(y_true, y_pred_binary)
        score_acc = accuracy_score(y_true, y_pred_binary)

        title = (
            lp
            + " | Best threshold: "
            + f"{best_threshold:.2f}"
            + " | Precision: "
            + f"{score_precision:.2f}"
            + " | Recall: "
            + f"{score_recall:.2f}"
            + " | F1: "
            + f"{score_f1:.2f}"
            + " | Acc: "
            + f"{score_acc:.2f}"
        )

        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(args.path, f"{lp}_curves.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
