from typing import Dict

import torch
from comet.models.metrics import RegressionMetrics
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, Precision, Recall


class ClassificationMetrics(RegressionMetrics):
    """
    New class to calculate classification metrics for the COMET CED model.
    This is similar to the RegressionMetrics class in the COMET repo
    NOTE: a higher value is assumed to be a better value for all calculated metrics.
    """

    def compute(self) -> torch.Tensor:
        """Computes classification metrics."""
        try:
            preds = torch.cat(self.preds, dim=0)
            targets = torch.cat(self.target, dim=0)
        except TypeError:
            preds = self.preds
            targets = self.target

        report = calculate_metrics(self.prefix, preds, targets)

        return report


# Expect we will want to call this function outside of instances of ClassificationMetrics
# e.g., when calculating metrics on test data or calculating metrics from LLM output.
def calculate_metrics(prefix: str, preds: torch.Tensor, targets: torch.Tensor) -> Dict:
    """
    Calculates and returns classification metrics given the true values and predictions.
    Currently calculates:
    - Matthew's Correlation Coefficient
    - F1 Score
    - Accuracy
    - Precision
    - Recall
    NOTE: This code changes the positive class to mean there is an ERROR

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
    # higher COMET score --> higher confidence it is NOT an error
    # However, we want the positive class to represent ERRORS
    # Therefore we change the labels to:  ERROR = 1, NOT = 0
    preds = 1 - preds
    targets = 1 - targets

    # Threshold currently fixed, might want to experiment at some point
    # when analysing the results with identifying the 'best' threshold
    # according to some metric
    threshold = 0.5

    # make the predictions
    preds_binary = preds > threshold

    # would be better if this was set once (in train_ced.py) and passed
    # to functions when needed - currently also set in comet.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create metrics objects
    mcc = MatthewsCorrCoef(num_classes=2).to(device)
    score_precision = Precision().to(device)
    score_recall = Recall().to(device)
    score_f1 = F1Score().to(device)
    score_acc = Accuracy().to(device)

    # create dictionary with metric values
    report = {
        prefix + "_threshold": threshold,
        prefix + "_MCC": mcc(targets, preds_binary),
        prefix + "_precision": score_precision(targets, preds_binary),
        prefix + "_recall": score_recall(targets, preds_binary),
        prefix + "_f1": score_f1(targets, preds_binary),
        prefix + "_acc": score_acc(targets, preds_binary),
    }

    return report
