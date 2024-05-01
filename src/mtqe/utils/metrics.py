from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, Precision, Recall


def calculate_threshold(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    This function uses a set of predictions and targets to determine the
    threshold value that provides the highest MCC value. The predictions are
    expected in a one-dimensional tensor, i.e. from a model using binary
    cross entropy

    Parameters
    ----------
    preds: torch.Tensor
        One dimensional tensor of prediction values
    targets: torch.Tensor
        Tensor of true values (either 0 or 1)

    Returns
    -------
    best_threshold: float
        A float value representing the threshold that gives the highest MCC value

    """
    min_threshold = round(float(preds.min()), 2)
    max_threshold = round(float(preds.max()), 2)

    thresholds = np.arange(min_threshold, max_threshold, 0.01)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcc = MatthewsCorrCoef(task="binary", num_classes=2).to(device)
    mccs = []
    for t in thresholds:
        preds_temp = (preds >= t).int()
        mcc_val = mcc(targets, preds_temp)
        mccs.append(mcc_val.item())

    idx_max = np.argmax(mccs)
    best_threshold = thresholds[idx_max]

    return best_threshold


# Expect we will want to call this function outside of instances of ClassificationMetrics
# e.g., when calculating metrics on test data or calculating metrics from LLM output.
def calculate_metrics(
    prefix: str,
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: int,
    num_classes: int = 2,
    max_vals: dict = None,
    vals_at_max_mcc: dict = None,
) -> Dict:
    """
    Calculates and returns classification metrics given the true values and predictions.
    Currently calculates:
    - Matthew's Correlation Coefficient
    - F1 Score
    - Accuracy
    - Precision
    - Recall
    NOTE: Expecting the positive class to mean there is an ERROR

    Parameters
    ----------
    prefix: str
        Text to be prefixed to the metric names
    preds: torch.Tensor
        Tensor of predictions
    targets: torch.Tensor
        Tensor of true values (either 0 or 1)
    threshold: int
        The threshold used to calculate the predictions
    num_classes: int
        The number of classes
    max_vals: dict or None
        A dictionary that stores the maximum values achieved for all the metrics, can be
        None in which case the maximum values won't get calculated
    vals_at_max_mcc: dict or None
        A dictionary that stores the values of the metrics at the maximum MCC value, can
        be None in which case the maximum values won't get calculated

    Returns
    ----------
    report: dict
        Dictionary containing the classification metrics for the predictions and target values
    max_vals: dict or None
        Dictionary containing the maximum values achieved over all epochs - None is returned if
        the max_vals parameter is passed to the function as None
    vals_at_max_mcc: dict
        Dictionary containing the values of the metrics at the point the maximum MCC was achieved - None
        is returned if the vals_at_max_mcc parameter is passed to the function as None

    """

    # would be better if this was set once (in train_ced.py) and passed
    # to functions when needed - currently also set in comet.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create metrics objects and score predictions
    mcc = MatthewsCorrCoef(task="binary", num_classes=num_classes).to(device)
    mcc_val = mcc(targets, preds)
    score_precision = Precision(task="binary").to(device)
    precision_val = score_precision(targets, preds)
    score_recall = Recall(task="binary").to(device)
    recall_val = score_recall(targets, preds)
    score_f1 = F1Score(task="binary").to(device)
    f1_val = score_f1(targets, preds)
    score_acc = Accuracy(task="binary").to(device)
    acc_val = score_acc(targets, preds)

    # create dictionary with metric values
    report = {
        prefix + "_threshold": threshold,
        prefix + "_MCC": mcc_val,
        prefix + "_precision": precision_val,
        prefix + "_recall": recall_val,
        prefix + "_f1": f1_val,
        prefix + "_acc": acc_val,
    }

    if max_vals is not None:
        assert vals_at_max_mcc is not None, "If passing `max_vals` dict, expect `vals_at_max_mcc` too."
        if mcc_val > max_vals[prefix + "_max_MCC"]:
            max_vals[prefix + "_max_MCC"] = mcc_val
            vals_at_max_mcc[prefix + "_at_max_mcc_threshold"] = threshold
            vals_at_max_mcc[prefix + "_at_max_mcc_precision"] = precision_val
            vals_at_max_mcc[prefix + "_at_max_mcc_recall"] = recall_val
            vals_at_max_mcc[prefix + "_at_max_mcc_f1"] = f1_val
            vals_at_max_mcc[prefix + "_at_max_mcc_acc"] = acc_val
        if precision_val > max_vals[prefix + "_max_precision"]:
            max_vals[prefix + "_max_precision"] = precision_val
        if recall_val > max_vals[prefix + "_max_recall"]:
            max_vals[prefix + "_max_recall"] = recall_val
        if f1_val > max_vals[prefix + "_max_f1"]:
            max_vals[prefix + "_max_f1"] = f1_val
        if acc_val > max_vals[prefix + "_max_acc"]:
            max_vals[prefix + "_max_acc"] = acc_val

        return report, max_vals, vals_at_max_mcc
    else:
        return report


def create_plots(plot_name: str, preds: np.array, targets: np.array, plots_path: str):
    """
    WIP function to create plots of the evaluation metrics
    Might want to consider using the pytorch metrics instead of sklearn - but
    would be worth comparing the two to make sure they return the same values.
    """
    # higher COMET score --> higher confidence it is NOT an error
    # However, we want the positive class to represent ERRORS
    # Therefore the labels are:  ERROR = 1, NOT = 0
    preds = 1 - preds
    targets = 1 - targets

    # RESULTS
    # 1. Precision, Recall, FPR, TPR
    prec, rec, _ = precision_recall_curve(targets, preds)
    fpr, tpr, _ = roc_curve(targets, preds)
    cm = confusion_matrix(targets, preds >= 0.5)

    pr_display = PrecisionRecallDisplay(precision=prec, recall=rec).plot()
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    cm_display = ConfusionMatrixDisplay(cm)

    # 2. MCC
    # tresholds for binarizing COMET output
    thresholds = np.arange(0.01, 0.99, 0.01)
    mccs = []
    for t in thresholds:
        # the model is treated as an error detector
        # i.e., scores above threshold are "ERROR" predictions
        # y_hat = (df_results["comet_score"] >= t).astype(int)
        y_hat = (preds >= t).astype(int)
        mcc = matthews_corrcoef(targets, y_hat)
        mccs.append(mcc)

    # PLOT
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))

    roc_display.plot(ax=ax1)
    ax1.plot([0, 1], [0, 1], transform=ax1.transAxes, color="black", ls="--")
    pr_display.plot(ax=ax2)
    ax2.axhline(y=sum(targets) / len(targets), color="black", linestyle="--")
    ax3.legend(loc="lower right")
    cm_display.plot(ax=ax4)

    idx_max = np.argmax(mccs)
    best_threshold = thresholds[idx_max]
    label = f"{best_threshold:.2f}"
    ax3.scatter(thresholds, mccs, s=10, label=label)
    ax3.set_ylabel("MCC")
    ax3.set_xlabel("Threshold")
    ax3.legend(markerscale=0, loc="upper left", title="Best threshold")

    y_pred_binary = preds > best_threshold
    score_precision = precision_score(targets, y_pred_binary)
    score_recall = recall_score(targets, y_pred_binary)
    score_f1 = f1_score(targets, y_pred_binary)
    score_acc = accuracy_score(targets, y_pred_binary)

    title = (
        plot_name
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
    fig.savefig(plots_path + "/" + plot_name + "_plot.png", bbox_inches="tight")
