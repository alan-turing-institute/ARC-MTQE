import argparse
import os
from typing import Tuple

import pandas as pd

from mtqe.utils.paths import EVAL_DIR
from mtqe.utils.tables import create_latex_table


def parse_args():
    """
    Construct argument parser.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "-data", required=True, help="The data split, expecting either 'dev' or 'test")

    return parser.parse_args()


def get_df_results(file_path: str, data_split: str, train_type: str, experiment_group: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0)
    df = df[df["split"] == data_split]
    df["train_type"] = train_type
    df["experiment_group"] = experiment_group
    return df


def collate_results(data_split: str, eval_dir: str = EVAL_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folders = os.listdir(eval_dir)
    results_ensemble = []
    results_max = []
    for folder in folders:
        if folder.endswith(".csv"):
            continue
        elif folder.startswith("second"):
            train_type = "Two-step"
        elif folder.startswith("train"):
            train_type = "One-step"
        else:  # assuming is LLM results
            train_type = "-"
        files = os.listdir(os.path.join(eval_dir, folder))
        for file in files:
            file_path = os.path.join(eval_dir, folder, file)
            if file.endswith("max_results.csv"):
                experiment_group = file[:-16]
                df = get_df_results(file_path, data_split, train_type, experiment_group)
                results_max.append(df)
            elif file.endswith("ensemble_results.csv"):
                experiment_group = file[:-21]
                df = get_df_results(file_path, data_split, train_type, experiment_group)
                results_ensemble.append(df)
    df_max = pd.concat(results_max)
    df_max = get_mcc_results(df_max)
    df_ensemble = pd.concat(results_ensemble)
    df_ensemble = get_mcc_results(df_ensemble)

    return df_max, df_ensemble


def get_mcc_results(df: pd.DataFrame) -> pd.DataFrame:
    df_pivot = df.pivot(index=["train_type", "experiment_group"], columns="language_pair", values="MCC")
    df_pivot.reset_index(inplace=True)
    return df_pivot


def print_table(df: pd.DataFrame) -> str:
    col_names = df.columns.values
    di_results = df.to_dict()

    latex_table = create_latex_table(col_names, di_results)

    return latex_table


if __name__ == "__main__":
    df_max, df_ensemble = collate_results("test")
    table_max = print_table(df_max)
    df_max.to_csv("results_max_test.csv", float_format="%.3f")
    table_ensemble = print_table(df_ensemble)
    print("finished")
