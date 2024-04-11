from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from comet.models.metrics import RegressionMetrics
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, Precision, Recall


class ClassificationMetrics(RegressionMetrics):
    """
    New class to calculate classification metrics for the COMET CED model.
    This is similar to the RegressionMetrics class in the COMET repo
    NOTE: a higher value is assumed to be a better value for all calculated metrics.

    Attributes
    ----------
    To do

    Methods
    -------
    To do
    """

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
        binary=True,
        num_classes=2,
        calc_threshold=False,
    ) -> None:
        super().__init__(
            prefix=prefix, dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn
        )
        self.binary_loss = binary
        self.num_classes = num_classes
        self.calc_threshold = calc_threshold
        self.max_vals = {
            prefix + "_max_MCC": -1,
            prefix + "_max_precision": 0,
            prefix + "_max_recall": 0,
            prefix + "_max_f1": 0,
            prefix + "_max_acc": 0,
        }
        self.vals_at_max_mcc = {
            prefix + "_at_max_mcc_threshold": 0.5,
            prefix + "_at_max_mcc_precision": 0,
            prefix + "_at_max_mcc_recall": 0,
            prefix + "_at_max_mcc_f1": 0,
            prefix + "_at_max_mcc_acc": 0,
        }

    def compute(self, threshold: float = 0.5) -> torch.Tensor:
        """Computes classification metrics."""
        try:
            preds = torch.cat(self.preds, dim=0)
            targets = torch.cat(self.target, dim=0)
        except TypeError:
            preds = self.preds
            targets = self.target

        if self.binary_loss:
            if self.calc_threshold:
                threshold = calculate_threshold(preds, targets)
            else:
                threshold = threshold
            # make the predictions
            preds = preds > threshold
            preds = preds.long()
        else:
            _, preds = torch.max(preds, 1)

        # higher COMET score --> higher confidence it is NOT an error
        # However, we want the positive class to represent ERRORS
        # Therefore we change the labels to:  ERROR = 1, NOT = 0
        preds = 1 - preds
        targets = 1 - targets

        report, max_vals, vals_at_max_mcc = calculate_metrics(
            self.prefix, preds, targets, threshold, self.num_classes, self.max_vals, self.vals_at_max_mcc
        )

        self.max_vals = max_vals
        self.vals_at_max_mcc = vals_at_max_mcc

        return report, max_vals, vals_at_max_mcc, threshold


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
        A float value representing the threshold that gives the highest
        MCC value

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
    num_classes: int,
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

    Returns
    ----------
    report: dict
        Dictionary containing the classification metrics for the predictions and target values
    max_vals: dict
        Dictionary containing the maximum values achieved over all epochs
    vals_at_max_mcc: dict
        Dictionary containing the values of the metrics at the point the maximum MCC was achieved.

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
