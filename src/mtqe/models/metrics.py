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
            prefix + "_val_at_max_mcc_threshold": 0.5,
            prefix + "_val_at_max_mcc_precision": 0,
            prefix + "_val_at_max_mcc_recall": 0,
            prefix + "_val_at_max_mcc_f1": 0,
            prefix + "_val_at_max_mcc_acc": 0,
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

        report = calculate_metrics(self.prefix, preds, targets, threshold, self.num_classes)

        return report, threshold


def calculate_threshold(preds: torch.Tensor, targets: torch.Tensor) -> float:
    min_threshold = round(float(preds.min()), 2)
    max_threshold = round(float(preds.max()), 2)

    thresholds = np.arange(min_threshold, max_threshold, 0.01)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcc = MatthewsCorrCoef(task="binary", num_classes=2).to(device)
    mccs = []
    for t in thresholds:
        preds_temp = (preds >= t).int()
        mcc_val = mcc(targets, preds_temp)
        mccs.append(mcc_val)

    idx_max = np.argmax(mccs)
    best_threshold = thresholds[idx_max]

    return best_threshold


# Expect we will want to call this function outside of instances of ClassificationMetrics
# e.g., when calculating metrics on test data or calculating metrics from LLM output.
def calculate_metrics(
    prefix: str, preds: torch.Tensor, targets: torch.Tensor, threshold: int, num_classes: int
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
        Tensor of sigmoid predictions (between 0 and 1)

    targets: torch.Tensor
        Tensor of true values (either 0 or 1)

    Returns
    ----------
    dict
        Dictionary containing the classification metrics for the predictions and target values
    """

    # would be better if this was set once (in train_ced.py) and passed
    # to functions when needed - currently also set in comet.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create metrics objects
    mcc = MatthewsCorrCoef(task="binary", num_classes=2).to(device)
    score_precision = Precision(task="binary").to(device)
    score_recall = Recall(task="binary").to(device)
    score_f1 = F1Score(task="binary").to(device)
    score_acc = Accuracy(task="binary").to(device)

    # create dictionary with metric values
    report = {
        prefix + "_threshold": threshold,
        prefix + "_MCC": mcc(targets, preds),
        prefix + "_precision": score_precision(targets, preds),
        prefix + "_recall": score_recall(targets, preds),
        prefix + "_f1": score_f1(targets, preds),
        prefix + "_acc": score_acc(targets, preds),
    }

    return report
