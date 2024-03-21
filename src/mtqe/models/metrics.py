from typing import Any, Callable, List, Optional

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torchmetrics import Metric


class ClassificationMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("systems", default=[], dist_reduce_fx=None)
        self.prefix = prefix

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        systems: Optional[List[str]] = None,
    ) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model
            target (torch.Tensor): Ground truth values
        """
        self.preds.append(preds)
        self.target.append(target)

        if systems:
            self.systems += systems

    def compute(self) -> torch.Tensor:
        """Computes spearmans correlation coefficient."""
        try:
            preds = torch.cat(self.preds, dim=0)
            target = torch.cat(self.target, dim=0)
        except TypeError:
            preds = self.preds
            target = self.target

        report = calculate_metrics(self.prefix, preds, target)

        return report


def calculate_metrics(prefix: str, preds: torch.Tensor, target: torch.Tensor) -> dict:
    # higher COMET score --> higher confidence it is NOT an error
    # However, we want the positive class to represent ERRORS
    # Therefore we change the labels to:  ERROR = 1, NOT = 0
    preds = 1 - preds
    target = 1 - target

    # thresholds = np.arange(0, 1, 0.01)

    # mccs = []
    # for t in thresholds:
    #     # the model is treated as an error detector
    #     # i.e., scores above threshold are "ERROR" predictions
    #     # y_hat = (df_results["comet_score"] >= t).astype(int)
    #     y_hat = (preds >= t).type(torch.uint8)
    #     mcc = matthews_corrcoef(target, y_hat)
    #     mccs.append(mcc)

    # idx_max = np.argmax(mccs)
    # best_threshold = thresholds[idx_max]

    best_threshold = 0.5

    y_pred_binary = preds > best_threshold
    mcc = matthews_corrcoef(target, y_pred_binary)
    score_precision = precision_score(target, y_pred_binary)
    score_recall = recall_score(target, y_pred_binary)
    score_f1 = f1_score(target, y_pred_binary)
    score_acc = accuracy_score(target, y_pred_binary)

    report = {
        prefix + "_threshold": best_threshold,
        # prefix + "_MCC": mccs[idx_max],
        prefix + "_MCC": mcc,
        prefix + "_precision": score_precision,
        prefix + "_recall": score_recall,
        prefix + "_f1": score_f1,
        prefix + "_acc": score_acc,
    }

    return report
