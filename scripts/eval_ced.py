import argparse
import os

import pandas as pd
import torch

from mtqe.data.loaders import load_ced_data
from mtqe.models.metrics import ClassificationMetrics
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import EVAL_DIR, PREDICTIONS_DIR


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get directory names")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name to evaluate")

    return parser.parse_args()


def evaluate(
    experiment_group_name: str,
    pred_dir: str = PREDICTIONS_DIR,
    eval_dir: str = EVAL_DIR,
    language_pairs: list = LI_LANGUAGE_PAIRS_WMT_21_CED,
):
    """
    Evaluate predictions for a given experiment group using gold truth labels per language pair.
    """

    group_dir = os.path.join(pred_dir, "ced_data", experiment_group_name)

    assert os.path.exists(group_dir), "Path for predictions for " + experiment_group_name + " does not exist"

    results_path = os.path.join(eval_dir, experiment_group_name)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    group_results = []

    for file in os.listdir(group_dir):
        file_words = file.split("_")
        lp = file_words[0]
        split = file_words[1]

        assert lp in language_pairs, "Unexpected language pair found in predictions: " + lp
        assert split in ["dev", "test"], "Unexpected split: " + split + ". Expecting either 'dev' or 'test."

        metrics = ClassificationMetrics(prefix="")

        # load predictions
        df_preds = pd.read_csv(os.path.join(group_dir, file))

        # load true target scores
        df_targets = load_ced_data(split, lp)

        # convert scores to Tensors
        preds = torch.Tensor(df_preds["score"])
        targets = torch.Tensor(df_targets["score"])

        # results = calculate_metrics(prefix = lp + "_" + split_name, preds = preds, targets=targets, threshold=0.5)

        metrics.update(preds, targets)

        results = metrics.compute(threshold=0.5, train=False)
        results["language_pair"] = lp
        results["split"] = split
        group_results.append(results)

    df = pd.DataFrame.from_dict(group_results)
    try:
        df.rename(
            columns={
                "_threshold": "threshold",
                "_MCC": "MCC",
                "_precision": "precision",
                "_recall": "recall",
                "_f1": "f1",
                "_acc": "accuracy",
            },
            inplace=True,
        )
    except Exception:
        pass

    df.to_csv(results_path + "/" + experiment_group_name + "_results.csv")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.group)
