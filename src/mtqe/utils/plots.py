import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5), sharey="row")

    for ind, _ in enumerate(preds):
        cm = confusion_matrix(targets[ind], preds[ind])
        cm_display = sns.heatmap(
            cm,
            fmt="0.0f",
            cmap="gray_r",
            annot=True,
            vmin=0,
            vmax=0,
            cbar=False,
            linecolor="k",
            linewidths=0.5,
            square=True,
            yticklabels=["No Critical Error", "Critical Error"],
            xticklabels=["No Critical Error", "Critical Error"],
            ax=axs[ind],
        )
        cm_display.set_title(plot_names[ind])
        cm_display.set_xlabel("")
        if ind != 0:
            cm_display.set_ylabel("")
        else:
            cm_display.set_ylabel("True label", fontsize=16)
        sns.despine(left=False, right=False, top=False, bottom=False)

    fig.text(0.5, 0.1, "Predicted label", fontsize=16, ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    fig.text(0.5, 1.02, fig_name, fontsize=20, ha="center")

    return fig


def create_histogram_plot(fig_name: str, plot_names: list, preds: list):
    """
    Plots histogram of output scores of a supervised model.
    """

    num_plots = len(preds)
    fig, axs = plt.subplots(int(num_plots / 2), 2, figsize=(20, 20), sharey="all", sharex="col")

    for ind, _ in enumerate(preds):
        row = math.floor(ind / 2)
        col = 0 if ind % 2 < 1 else 1

        hist = sns.histplot(data=preds[ind], binwidth=0.1, binrange=[0, 1], ax=axs[row, col])
        hist.set_title(plot_names[ind], fontsize=16)

    fig.text(0.5, 0.9, fig_name, fontsize=20, ha="center")

    return fig


def create_calib_plot(fig_name: str, plot_names: list, x_vals: list, y_vals):
    """
    Calibration plots - scatter plots - for each language pair
    """

    num_plots = len(y_vals)
    fig, axs = plt.subplots(int(num_plots / 2), 2, figsize=(20, 20), sharey="all", sharex="col")

    for ind, _ in enumerate(y_vals):
        row = math.floor(ind / 2)
        col = 0 if ind % 2 < 1 else 1

        data = {"x": x_vals[ind], "y": y_vals[ind]}

        scatter = sns.scatterplot(data, x="x", y="y", ax=axs[row, col])

        scatter.set_title(plot_names[ind], fontsize=16)

        line = mlines.Line2D([0, 1], [0, 1], color="grey")
        line.set_linestyle("dashed")
        scatter.add_line(line)

        axs[row, col].set_xlim(0, 1)
        axs[row, col].set_ylim(0, 1)

    fig.text(0.5, 0.9, fig_name, fontsize=20, ha="center")

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


def create_annotator_cm_plot(fig_name: str, plot_names: list, preds: list, targets: list):
    """
    Plots a confusion matrix for each language pair
    """

    assert len(preds) == len(targets), (
        "Expecting the same number of predictions and targets, got " + len(preds) + " and " + len(targets)
    )

    num_plots = len(preds)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5), sharey="row")

    for ind, _ in enumerate(preds):
        cm = confusion_matrix(targets[ind], preds[ind])
        cm = cm[:, :2]
        cm_display = sns.heatmap(
            cm,
            fmt="0.0f",
            cmap="gray_r",
            annot=True,
            vmin=0,
            vmax=0,
            cbar=False,
            linecolor="k",
            linewidths=0.5,
            square=False,
            xticklabels=["No Critical Error", "Critical Error"],
            ax=axs[ind],
        )
        cm_display.set_title(plot_names[ind])
        cm_display.set_xlim(0, 2)
        cm_display.set_xlabel("")
        if ind != 0:
            cm_display.set_ylabel("")
        else:
            cm_display.set_ylabel("True label", fontsize=16)
        sns.despine(left=False, right=False, top=False, bottom=False)

    fig.text(0.5, -0.1, "Predicted label", fontsize=16, ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    fig.text(0.5, 1.02, fig_name, fontsize=20, ha="center")

    return fig


def create_precision_recall_curve(fig_name: str, plot_names: list, preds: list, targets: list):
    """
    Plots a precision-recall curve for each language pair
    """
    assert len(preds) == len(targets), (
        "Expecting the same number of predictions and targets, got " + len(preds) + " and " + len(targets)
    )

    num_plots = len(preds)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5), sharey="row")

    for ind, _ in enumerate(preds):
        _ = PrecisionRecallDisplay.from_predictions(targets[ind], preds[ind], ax=axs[ind], plot_chance_level=True)
        axs[ind].legend(loc=1)
        axs[ind].set_title(plot_names[ind])
        axs[ind].set_xlabel("Recall")
        if ind > 0:
            axs[ind].set_ylabel("")
        else:
            axs[ind].set_ylabel("Precision")

    # fig.text(0.5, -0.1, "Recall", fontsize=16, ha="center")
    fig.text(0.5, 1, fig_name, fontsize=20, ha="center")

    return fig
