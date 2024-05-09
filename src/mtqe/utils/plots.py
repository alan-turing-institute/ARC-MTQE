import matplotlib.pyplot as plt
import numpy as np
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


def historgram_scores_plot():
    """
    Plots a histogram of the [0, 1] predictions made by a supervised model.
    """
    pass


def create_confusion_matrix_plot(fig_name: str, plot_names: list, preds: list, targets: list):
    """ """

    assert len(preds) == len(targets), (
        "Expecting the same number of predictions and targets, got " + len(preds) + " and " + len(targets)
    )

    num_plots = len(preds)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5))

    for ind, _ in enumerate(preds):
        cm = confusion_matrix(targets[ind], preds[ind])
        cm_display = ConfusionMatrixDisplay(cm)
        cm_display.plot(ax=axs[ind])

    fig.suptitle(fig_name, fontsize=16)
    fig.tight_layout()

    return fig


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
