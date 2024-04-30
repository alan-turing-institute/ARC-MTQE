import os

import pandas as pd

from mtqe.data.loaders import load_ced_data, load_wmt22_ced_data
from mtqe.utils.data import get_token_length_columns
from mtqe.utils.paths import PROCESSED_DATA_DIR


def main():
    """
    Process WMT 2022 En-De synthetic data (train and dev).
    - Add token lengths to WMT22 train dataset.
    - Combine with 2021 data
    """

    print("Processing WMT 2022 En-De data....")

    lp = "en-de"

    df_wmt22_dev = load_wmt22_ced_data("dev", lp)
    df_wmt22_dev[["mt", "src", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, f"wmt22_{lp}_dev.csv"))

    df_wmt22_train = load_wmt22_ced_data("train", lp)
    df_wmt22_train[["mt", "src", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, f"wmt22_{lp}_train.csv"))

    print("Computing token lengths...")

    # this takes a bit of time
    df_wmt22_train_expanded = get_token_length_columns(df_wmt22_train)
    df_wmt22_train_expanded[["mt", "src", "score", "src_token_len", "mt_token_len", "token_lengths"]].to_csv(
        os.path.join(PROCESSED_DATA_DIR, f"wmt22_{lp}_train_expanded.csv")
    )

    # create a smaller dataset of 40k records from the train data after filtering for short
    # sequence lengths (longest authentic En-De token length in WMT 2021 data is 124)
    df_wmt22_train_short = df_wmt22_train_expanded[df_wmt22_train_expanded["token_lengths"] < 125]
    df_wmt22_errors = df_wmt22_train_short[df_wmt22_train_short["score"] == 0]
    df_wmt22_good = df_wmt22_train_short[df_wmt22_train_short["score"] == 1]
    n_errors = df_wmt22_errors.shape[0]
    n_to_add = 40000 - n_errors
    df_wmt22_reduced = pd.concat([df_wmt22_errors, df_wmt22_good.iloc[:n_to_add]])
    assert df_wmt22_reduced.shape[0] == 40000
    df_wmt22_reduced[["mt", "src", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, f"wmt22_{lp}_train_reduced.csv"))

    # save half of the reduced dataset with same ERR/NOT split
    n_errors = int(n_errors / 2)
    n_to_add = 20000 - n_errors
    df_wmt22_small = pd.concat([df_wmt22_errors.iloc[:n_errors], df_wmt22_good.iloc[:n_to_add]])
    assert df_wmt22_small.shape[0] == 20000
    df_wmt22_small[["mt", "src", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, f"wmt22_{lp}_train_small.csv"))

    # combine with the authentic En-De data from WMT 2021 to create a balanced dataset
    df_wmt21_data = load_ced_data("train", "en-de")
    n_bad = df_wmt21_data[df_wmt21_data["score"] == 0].shape[0]
    n_good = df_wmt21_data[df_wmt21_data["score"] == 1].shape[0]
    n_bad_missing = n_good - n_bad
    df_wmt22_errors_subset = df_wmt22_errors.iloc[:n_bad_missing]
    balanced_df = pd.concat([df_wmt21_data[["src", "mt", "score"]], df_wmt22_errors_subset[["src", "mt", "score"]]])
    assert balanced_df.shape[0] == 5674 * 2
    balanced_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "balanced_ende.csv"))


if __name__ == "__main__":
    main()
