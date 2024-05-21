from typing import Dict

import numpy as np
import torch
from scipy import stats
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
    threshold: float = 0.5,
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
    threshold: float
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
    mcc_val = mcc(preds, targets)
    score_precision = Precision(task="binary").to(device)
    precision_val = score_precision(preds, targets)
    score_recall = Recall(task="binary").to(device)
    recall_val = score_recall(preds, targets)
    score_f1 = F1Score(task="binary").to(device)
    f1_val = score_f1(preds, targets)
    score_acc = Accuracy(task="binary").to(device)
    acc_val = score_acc(preds, targets)

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


def williams_test(
    human_metric_a_corr: float, human_metric_b_corr: float, metric_a_metric_b_corr: float, n: int = 1000
) -> float:
    """
    William's significance test for whether metric_a_corr is the same as
    the metric_b_corr (one-sided).

    NOTE: the method expects that r12 is bigger than r13. The values can be
    swapped, this is merely to ensure the

    From: https://github.com/inmoonlight/nlp-williams/blob/master/williams.py

    Parameters
    ----------
    human_metric_a_corr: float
        Correlation between human scores and metric A.
    human_metric_a_corr: float
        Correlation between human scores and metric B.
    metric_a_metric_b_corr: float
        Correlation between metric A and metric B.
    n: int
        The number of

    Returns
    ----------
    float
        The p-value
    """

    # this is to make sure that the t-value is positive
    # it does not affect the result in any way
    if human_metric_a_corr > human_metric_b_corr:
        r12 = human_metric_a_corr
        r13 = human_metric_b_corr
    else:
        r12 = human_metric_b_corr
        r13 = human_metric_a_corr
    r23 = metric_a_metric_b_corr

    K = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    denominator = np.sqrt(2 * K * (n - 1) / (n - 3) + (((r12 + r13) ** 2) / 4) * ((1 - r23) ** 3))
    numerator = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    t = numerator / denominator
    p = 1 - stats.t.cdf(abs(t), df=n - 3)

    return p
