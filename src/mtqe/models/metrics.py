from typing import Any, Callable, Dict, Optional, Tuple

import torch
from comet.models.metrics import RegressionMetrics

from mtqe.utils.metrics import calculate_metrics, calculate_threshold


class ClassificationMetrics(RegressionMetrics):
    """
    New class to calculate classification metrics for the COMET CED model.
    This is similar to the RegressionMetrics class in the COMET repo
    NOTE: a higher value is assumed to be a better value for all calculated metrics.
    NOTE: current implementation only allows for `num_classes=2`, but the code could
    be extended to allow for multiple classes when the loss function is cross entropy

    Attributes
    ----------
    To do - inherited attributes
    binary_loss: bool
        Is set to `True` if the loss function is binary cross entropy, `False` otherwise
        (assumed cross entropy)
    num_classes: int
        Number of classes that the predictions are made over
    calc_threshold: bool
        Is set to `False` if the threshold will be fixed, and `True` if it is to be
        calculated to find the best threshold for a given set of predictions
    activation_fn: Optional[Callable]
        The function to be applied to the predictions from the model, default is None
        Allowed values are `softmax` or `sigmoid`
    activation_fn_args: dict
        Dictionary of arguments to be passed to the activation function. If no args are
        required then an empty dictionary should be used.

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
        binary_loss: bool = True,
        num_classes: int = 2,
        calc_threshold: bool = False,
        activation_fn: Optional[Callable] = None,
        activation_fn_args: dict = {},
    ) -> None:
        super().__init__(
            prefix=prefix, dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn
        )
        self.binary_loss = binary_loss
        self.num_classes = num_classes
        self.calc_threshold = calc_threshold
        self.activation_fn = activation_fn
        self.activation_fn_args = activation_fn_args
        self.max_vals = {
            prefix + "_max_MCC": -1,
            prefix + "_max_precision": 0,
            prefix + "_max_recall": 0,
            prefix + "_max_f1": 0,
            prefix + "_max_acc": 0,
        }
        self.vals_at_max_mcc = {}

    def compute(self, threshold: float = 0.5) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
        """
        Computes classification metrics.

        Parameters
        ----------
        threshold: float
            The threshold value used to make predictions (for binary cross entropy loss func. only)

        Returns
        -------
        report: dict
            Dictionary containing the classification metrics for the predictions and target values
        max_vals: dict
            Dictionary containing the maximum values achieved over all epochs - only returned if
        vals_at_max_mcc: dict
            Dictionary containing the values of the metrics at the point the maximum MCC was achieved.
        threshold: float
            The threshold value used to calculate the metrics
        """
        try:
            preds = torch.cat(self.preds, dim=0)
            targets = torch.cat(self.target, dim=0)
        except TypeError:
            preds = self.preds
            targets = self.target

        # Apply final activation function to the predictions, if one exists
        if self.activation_fn is not None:
            preds = self.activation_fn(preds, **self.activation_fn_args)

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
