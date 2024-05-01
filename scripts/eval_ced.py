import argparse
import os

import pandas as pd
import torch

from mtqe.data.loaders import load_ced_data
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.metrics import calculate_metrics
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
    pos_class_error: bool = False,
    pred_dir: str = PREDICTIONS_DIR,
    eval_dir: str = EVAL_DIR,
    language_pairs: list = LI_LANGUAGE_PAIRS_WMT_21_CED,
):
    """
    Evaluate predictions for a given experiment group using gold truth labels per language pair.
    Saves the metrics as a csv file

    Parameters
    ----------
    experiment_group_name: str
        The name of the experiment group to be evaluated
    pos_class_error: bool
        Whether the positive class of the predictions represents an ERROR
    pred_dir: str
        The directory where the predictions are stored
    eval_dir: str
        The directory where the evaluations are to be saved
    language_pairs: list
        A list of the possible language pairs
    """

    group_dir = os.path.join(pred_dir, "ced_data", experiment_group_name)
    assert os.path.exists(group_dir), "Path for predictions for " + experiment_group_name + " does not exist"

    results_path = os.path.join(eval_dir, experiment_group_name)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    # list to store results per file
    individual_results = []
    ensemble_results = []

    for lp in language_pairs:
        for split in ["dev", "test"]:
            filenames = [filename for filename in os.listdir(group_dir) if filename.startswith(lp + "_" + split)]
            num_files = len(filenames)

            # load true target scores
            df_targets = load_ced_data(split, lp)
            targets = torch.Tensor(df_targets["score"])
            # If pos_class_error = False then the positive class is NOT and error
            # and this will be reverse so that the positive class represents ERRORs
            if not pos_class_error:
                targets = 1 - targets

            cumulative_preds = torch.zeros(len(targets))

            for file in filenames:
                file_words = file.split("_")
                seed = file_words[2]  # Note that for non-supervised model this may not represent a random seed

                # load predictions
                df_preds = pd.read_csv(os.path.join(group_dir, file))

                # convert scores to Tensors
                preds = torch.Tensor(df_preds["score"])

                if not pos_class_error:
                    preds = 1 - preds

                # binarise the predictions - threshold of 0.5 is currently hard-coded
                threshold = 0.5
                preds = preds > threshold
                preds = preds.long()
                cumulative_preds += preds

                results = get_results(preds=preds, targets=targets, threshold=0.5, lp=lp, split=split)
                if seed.isdigit():
                    results["seed"] = int(seed)
                else:
                    results["seed"] = seed
                individual_results.append(results)

            # Only ensemble if there was more than one file (i.e, seed) for each lp and split
            if num_files > 1:  # Might want to check that num_files is an odd number?
                majority_preds = cumulative_preds > (num_files / 2)
                majority_preds = majority_preds.long()

                results = get_results(preds=majority_preds, targets=targets, threshold=0.5, lp=lp, split=split)
                results["seed"] = "ensemble"
                ensemble_results.append(results)

    # Convert list of results to a dataframe
    df = create_results_df(individual_results)
    # Save results
    df.to_csv(results_path + "/" + experiment_group_name + "_results.csv")
    # Log the max MCC for each lp / split combination
    max_inds = list(df.groupby(["language_pair", "split"]).idxmax()["MCC"].values)
    df_max = df.loc[max_inds]
    df_max.to_csv(results_path + "/" + experiment_group_name + "_max_results.csv")
    # Log the min MCC for each lp / split combination
    min_inds = list(df.groupby(["language_pair", "split"]).idxmin()["MCC"].values)
    df_min = df.loc[min_inds]
    df_min.to_csv(results_path + "/" + experiment_group_name + "_min_results.csv")

    df_ensemble = create_results_df(ensemble_results)
    # Save results
    df_ensemble.to_csv(results_path + "/" + experiment_group_name + "_ensemble_results.csv")


def get_results(preds: torch.Tensor, targets: torch.Tensor, threshold: float, lp: str, split: str) -> dict:
    results = calculate_metrics(prefix="", preds=preds, targets=targets, threshold=threshold)
    # Convert results from Tensor to float
    for k in results:
        if type(results[k]) is torch.Tensor:
            results[k] = results[k].item()
    # Record the language pair and data split
    results["language_pair"] = lp
    results["split"] = split
    return results


def create_results_df(results: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(results)
    # Re-names the columns if they contain a "_" as a prefix
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
    return df


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.group)
