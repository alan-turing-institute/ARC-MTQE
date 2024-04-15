import os
from collections import defaultdict

import pandas as pd

from mtqe.data.loaders import load_ced_data
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import PROCESSED_DATA_DIR


def main():
    """
    The COMET code expects train and dev data to be in CSV files.

    This script creates the necessary files to train:
        - one model per language pair using just that pair's training data
        - one model per language pair using all the authentic training data
        - a single multilingual model that has seen data from all language pairs

    """

    # keep track of source sentences in dev and test sets for each language pair
    # will exclude these when making the multilingual datasets
    all_src_to_exclude = defaultdict(list)

    # ==================================================================
    # 1: save each WMT21 train and dev file as is
    # ==================================================================
    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        for data_split in ["train", "dev"]:
            df_data = load_ced_data(data_split, lp)
            df_data[["src", "mt", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, "{lp}_majority_{data_split}.csv"))

            if data_split == "dev":
                all_src_to_exclude[lp].extend(df_data["src"])

        # add source sentences in the test sets
        df_test = load_ced_data("test", lp)
        all_src_to_exclude[lp].extend(df_test["src"])

    # ==================================================================
    # 2. create multilingual training dataset combining all lps
    # ==================================================================

    # exclude the dev/test source sentences from all training datasets
    # then combine all the training data in a single CSV file
    all_dfs = []
    all_src_to_exclude_flat = [x for xs in all_src_to_exclude.values() for x in xs]
    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        df_train = load_ced_data("train", lp)
        df_train_all_reduced = df_train[~df_train["src"].isin(all_src_to_exclude_flat)]
        all_dfs.append(df_train_all_reduced[["src", "mt", "score"]])

    df_multilingual_train = pd.concat(all_dfs)
    df_multilingual_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "all_multilingual_train.csv"))

    # ==================================================================
    # 2. create multilingual training dataset tailored for each lp
    # ==================================================================

    # this involves double loop
    # for each LP - loop through all the OTHER lps and remove that LPS
    # test and dev set --> then concat dfs and save
    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        df_train = load_ced_data("train", lp)
        dfs = [df_train]
        for other_lps in LI_LANGUAGE_PAIRS_WMT_21_CED:
            if other_lps != lp:
                df_train_other = load_ced_data("train", other_lps)
                df_train_other_reduced = df_train_other[~df_train_other["src"].isin(all_src_to_exclude[lp])]
                dfs.append(df_train_other_reduced[["src", "mt", "score"]])
        df_multilingual_lp_train = pd.concat(dfs)
        df_multilingual_lp_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "{lp}_multilingual_train.csv"))


if __name__ == "__main__":
    main()
