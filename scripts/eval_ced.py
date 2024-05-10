import argparse
import os

import pandas as pd
import torch

from mtqe.data.loaders import load_ced_data
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.metrics import calculate_metrics, calculate_threshold
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
    Saves the metrics as a csv file.
    If multiple predictions exist for a given language pair and data split (i.e., there are
    multiple random seeds), then an ensemble prediction will be made and analysed

    Parameters
    ----------
    experiment_group_name: str
        The name of the experiment group to be evaluated
    pos_class_error: bool
        Whether the positive class value in the predictions represents an ERROR.
        Defaults to `False`. This means positive labels in the predictions (1)  indicate
        NOT an error while negative labels (0) indicate an ERROR.
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

    ensemble_path = os.path.join(group_dir, "ensemble_preds")
    if not os.path.isdir(ensemble_path):
        os.mkdir(ensemble_path)
    # list to store results per file
    individual_results = []
    ensemble_results = []

    for lp in language_pairs:
        best_thresholds = {}  # Stores the best threshold on the dev dataset for each model
        for split in ["dev", "test"]:
            filenames = [
                filename
                for filename in os.listdir(group_dir)
                if (filename.startswith(lp + "_" + split) and filename.endswith(".csv"))
            ]
            num_files = len(filenames)

            # load true target scores
            df_targets = load_ced_data(split, lp)
            targets = torch.Tensor(df_targets["score"])

            # If pos_class_error = False then the positive class is NOT an error
            # and this will be reversed so that the positive class represents ERRORs
            if not pos_class_error:
                targets = 1 - targets

            # Dictionary to record the cumulative predictions for each threshold strategy
            # (used later to calculate ensemble predictions using majority voting)
            preds_by_threshold = {
                "default": {"threshold": 0.5, "cumulative_preds": torch.zeros(len(targets))},
                "best": {"cumulative_preds": torch.zeros(len(targets))},
                "extreme": {"threshold": 0.1, "cumulative_preds": torch.zeros(len(targets))},
            }

            for file in filenames:
                file_words = file.split("_")
                if file_words[2].isdigit():
                    # file_words[2] will represent a random seed for supervised models only
                    seed = file_words[2]
                    model_type = "supervised"
                else:
                    seed = "-"
                    model_type = file_words[2]

                # load predictions
                df_preds = pd.read_csv(os.path.join(group_dir, file))

                # convert scores to Tensors
                preds = torch.Tensor(df_preds["score"])

                if not pos_class_error:
                    preds = 1 - preds

                if split == "dev":
                    # Calculate the 'best' threshold for MCC on the dev data
                    best_threshold = calculate_threshold(preds, targets)
                    best_thresholds[seed + "_" + model_type] = best_threshold
                else:  # is test set
                    # use the threshold calculated on the same model using the dev dataset
                    best_threshold = best_thresholds[seed + "_" + model_type]
                preds_by_threshold["best"]["threshold"] = best_threshold

                for key in preds_by_threshold:
                    threshold = preds_by_threshold[key]["threshold"]
                    binary_preds = preds > threshold
                    binary_preds = binary_preds.long()

                    preds_by_threshold[key]["cumulative_preds"] += binary_preds

                    results = get_results(
                        experiment_group=experiment_group_name,
                        preds=binary_preds,
                        targets=targets,
                        threshold=threshold,
                        threshold_strategy=key,
                        lp=lp,
                        split=split,
                        seed=seed,
                        model_type=model_type,
                    )

                    individual_results.append(results)

            # Calculate an ensemble prediction per threshold strategy.
            for key in preds_by_threshold:
                # Only ensemble if there was more than one file (i.e, seed) for each lp and split
                # And if the number of files is an odd number as these ensembles use majority voting
                if (num_files > 1) and ((num_files % 2) == 1):
                    if key == "best":
                        # When selecting the 'best' threshold, this can be different for each model
                        # so just going to record a threshold of 0 - the 'threshold_strategy' feature
                        # will record that the strategy is selecting the 'best' threshold.
                        threshold = 0
                    else:
                        threshold = preds_by_threshold[key]["threshold"]

                    # Calculate the majority prediction over all the random seeds
                    majority_preds = preds_by_threshold[key]["cumulative_preds"] > (num_files / 2)
                    majority_preds = majority_preds.long()

                    # Save majority predictions
                    majority_preds_np = majority_preds.numpy()
                    df_majority_preds = pd.DataFrame(majority_preds_np)

                    df_majority_preds.to_csv(
                        ensemble_path
                        + "/"
                        + lp
                        + "_"
                        + split
                        + "_"
                        + experiment_group_name
                        + "_ensemble_majority_preds_"
                        + key
                        + ".csv"
                    )

                    results = get_results(
                        experiment_group=experiment_group_name,
                        preds=majority_preds,
                        targets=targets,
                        threshold=threshold,
                        threshold_strategy=key,
                        lp=lp,
                        split=split,
                        seed="-",
                        model_type="majority_ensemble",
                    )

                    ensemble_results.append(results)

    # Convert list of results to a dataframe
    df = create_results_df(individual_results)
    # Save results
    df.to_csv(results_path + "/" + experiment_group_name + "_results.csv")
    # Log the max MCC for each lp / split / threshold strategy combination
    max_inds = list(df.groupby(["language_pair", "split", "threshold_strategy"]).idxmax()["MCC"].values)
    df_max = df.loc[max_inds]
    df_max.to_csv(results_path + "/" + experiment_group_name + "_max_results.csv")
    # Log the min MCC for each lp / split / threshold strategy combination
    min_inds = list(df.groupby(["language_pair", "split", "threshold_strategy"]).idxmin()["MCC"].values)
    df_min = df.loc[min_inds]
    df_min.to_csv(results_path + "/" + experiment_group_name + "_min_results.csv")

    # Log the median MCC for each lp / split / threshold strategy combination
    df_median = (
        df.groupby(["language_pair", "split", "threshold_strategy", "exp_group"])
        .agg(
            {
                "MCC": "median",
                "seed": "first",
                "threshold": "first",
                "precision": "first",
                "recall": "first",
                "f1": "first",
                "accuracy": "first",
            }
        )
        .reset_index()
    )
    df_median.to_csv(results_path + "/" + experiment_group_name + "_median_results.csv")

    # Log the mean values by lp / split / threshold strategy combination
    df_mean = df.groupby(["language_pair", "split", "threshold_strategy", "exp_group"]).mean("MCC").reset_index()
    df_mean.to_csv(results_path + "/" + experiment_group_name + "_mean_results.csv")

    df_ensemble = create_results_df(ensemble_results)
    # Save ensemble results
    df_ensemble.to_csv(results_path + "/" + experiment_group_name + "_ensemble_results.csv")


def get_results(
    experiment_group: str,
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    threshold_strategy: str,
    lp: str,
    split: str,
    seed: str,
    model_type: str,
) -> dict:
    """
    Parameters
    ----------
    experiment_group: str
        The name of the experiment group that the results are from
    preds: torch.Tensor
        Tensor of predictions
    targets: torch.Tensor
        Tensor of true target values
    threshold: float
        Threshold value used to calculate the metrics - cast as a string
    threshold_strategy: str
        A given name for selecting the binarisation threshold (e.g., 'default' for 0.5, 'best' for best selction
        using dev dataset)
    lp: str
        ISO code for language pair
    split: str
        Data split, expecting 'dev' or 'test'
    seed: str
        The random seed used to train the model - cast as a string
    model_type: str
        A name for the model type

    Returns
    -------
    results: dict
        Dictionary of the metrics for the given preds and targets
    """
    results = calculate_metrics(prefix="", preds=preds, targets=targets)
    # Convert results from Tensor to float
    for k in results:
        if type(results[k]) is torch.Tensor:
            results[k] = results[k].item()
    # Record additional information in the results dictionary
    results["_threshold"] = threshold
    results["threshold_strategy"] = threshold_strategy
    results["language_pair"] = lp
    results["split"] = split
    results["seed"] = seed
    results["model_type"] = model_type
    results["exp_group"] = experiment_group
    return results


def create_results_df(results: dict) -> pd.DataFrame:
    """
    Takes a dictionary of metrics for one or more experiments and returns a
    pandas DataFrame with some columns re-named

    Parameters
    ----------
    results: dict
        Dictionary of metrics

    Returns
    -------
    df: pd.DataFrame
        DataFrame of the data in the dictionary
    """
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
