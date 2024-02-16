import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get directory names")

    parser.add_argument("-m", "--model", required=True, help="Model name")
    parser.add_argument(
        "-t", "--timestamp", required=True, help="Predictions timestamp"
    )

    return parser.parse_args()


def load_data(model, timestamp, lp):
    """
    Load model predictions and gold truth labels for language pair.
    """

    main_dir = os.getcwd()

    # predictions
    predictions_dir = os.path.join(main_dir, "results", model, timestamp)
    pred_path = os.path.join(predictions_dir, f"{lp}_predictions.csv")
    df_pred = pd.read_csv(pred_path)

    # gold labels
    data_dir = os.path.join(
        main_dir, "data", "mlqe-pe", "data", "catastrophic_errors_goldlabels"
    )
    labels_path = os.path.join(
        data_dir, f"{lp}_majority_test_goldlabels", "goldlabels.txt"
    )
    df_labels = pd.read_csv(
        labels_path, sep="\t", header=None, names=["lang_pair", "ref", "idx", "label"]
    )

    # merge on sentence indexes
    df_results = pd.merge(df_pred, df_labels, on="idx")

    return df_results, predictions_dir


def main():
    args = parse_args()
    model = args.model
    timestamp = args.timestamp

    language_pairs = ["encs", "ende", "enja", "enzh"]
    for lp in language_pairs:
        df_results, results_path = load_data(model, timestamp, lp)

        # higher COMET score --> higher confidence it is NOT an error
        # labels:  ERROR = 0, NOT = 1
        y_true = np.where(df_results["label"] == "NOT", 1, 0)
        y_pred = df_results["comet_score"]

        # RESULTS
        # 1. Precision, Recall, FPR, TPR
        prec, rec, _ = precision_recall_curve(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)

        pr_display = PrecisionRecallDisplay(precision=prec, recall=rec).plot()
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

        # 2. MCC
        # tresholds for binarizing COMET output
        thresholds = np.arange(0.05, 0.9, 0.05)
        mccs = []
        for t in thresholds:
            # the model is treated as "NOT an error" detector
            # i.e., scores above threshold are "NOT an error" predictions
            y_hat = (df_results["comet_score"] >= t).astype(int)
            mcc = matthews_corrcoef(y_true, y_hat)
            mccs.append(mcc)

        # PLOT
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        roc_display.plot(ax=ax1)
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, color="black", ls="--")
        pr_display.plot(ax=ax2)
        ax2.axhline(y=sum(y_true) / len(y_true), color="black", linestyle="--")

        idx_max = np.argmax(mccs)
        label = f"{thresholds[idx_max]:.2f}"
        ax3.scatter(thresholds, mccs, s=10, label=label)
        ax3.set_ylabel("MCC")
        ax3.set_xlabel("Threshold")
        ax3.legend(markerscale=0, loc="upper left", title="Best threshold")

        fig.suptitle(lp, fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(results_path, f"{lp}_curves.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
