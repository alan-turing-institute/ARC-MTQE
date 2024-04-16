import os
from collections import defaultdict

import numpy as np
import pandas as pd

from mtqe.data.loaders import load_ced_data, score_ced
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import DEMETR_DIR, PROCESSED_DATA_DIR


def main():
    """
    The COMET code expects train and dev data to be in CSV files. This script creates
    the necessary files for WMT21 CED data to train:
        - one model per language pair using just that pair's training data
        - one model per language pair using all the authentic training data
        - a single multilingual model that has seen data from all language pairs

    For the latter two data sets we make sure that the training data does not
    contain any source sentences that are in the dev and test sets (for the given
    language pair or for all the language pairs).

    This script also creates a single file for DEMETR data.
    """

    # keep track of source sentences in dev and test sets for each language pair
    # and exclude these when making the multilingual datasets
    all_src_to_exclude = defaultdict(list)

    # ==================================================================
    # 1: save each WMT21 train and dev file as is
    # ==================================================================

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        for data_split in ["train", "dev"]:
            df_data = load_ced_data(data_split, lp)
            df_data[["src", "mt", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, f"{lp}_majority_{data_split}.csv"))

            if data_split == "dev":
                all_src_to_exclude[lp].extend(df_data["src"])

        # add source sentences in the test sets to the to exclude lists
        df_test = load_ced_data("test", lp)
        all_src_to_exclude[lp].extend(df_test["src"])

    # ==================================================================
    # 2. create multilingual training dataset combining all lps
    #   - exclude ALL dev/test source sentences from each training set
    #   - combine filtered training data for all lps in a single CSV file
    # ==================================================================

    # training data dfs cleared of dev/test sentences across ALL lps
    all_dfs = []
    # flatten dictionary values to get all source sentences to exclude
    all_src_to_exclude_flat = [x for xs in all_src_to_exclude.values() for x in xs]
    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        df_train = load_ced_data("train", lp)
        df_train_all_reduced = df_train[~df_train["src"].isin(all_src_to_exclude_flat)]
        all_dfs.append(df_train_all_reduced[["src", "mt", "score"]])

    df_train_all_multilingual = pd.concat(all_dfs)
    df_train_all_multilingual.to_csv(os.path.join(PROCESSED_DATA_DIR, "all_multilingual_train.csv"))

    # ==================================================================
    # 3. create multilingual training dataset tailored for each lp
    #   - for each lp, loop through all the OTHER lps and get train data
    #   - for the other lps train data, remove any dev/test sentences for
    #     the given lp
    # ==================================================================

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        df_train = load_ced_data("train", lp)
        lp_dfs = [df_train]
        for other_lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
            if other_lp != lp:
                df_train_other = load_ced_data("train", other_lp)
                df_train_other_reduced = df_train_other[~df_train_other["src"].isin(all_src_to_exclude[lp])]
                lp_dfs.append(df_train_other_reduced[["src", "mt", "score"]])
        df_train_lp_multilingual = pd.concat(lp_dfs)
        df_train_lp_multilingual.to_csv(os.path.join(PROCESSED_DATA_DIR, f"{lp}_multilingual_train.csv"))

        # ==================================================================
        # 4. DEMETR data
        # ==================================================================

        dfs = []
        for f in os.listdir(DEMETR_DIR):
            df = pd.read_json(os.path.join(DEMETR_DIR, f))
            dfs.append(df)
        df_all_demetr = pd.concat(dfs)

        # the perturbations are scored as base, minor, major or critical
        df_all_demetr["label"] = np.where(df_all_demetr["severity"] == "critical", "ERR", "NOT")
        # two of the base categories are critical errors:
        #   - 'unrelated translation - baseline' and 'empty string (full stop only)'
        # and the third is a perfect translation -> rescore the critical errors
        df_all_demetr.loc[
            (df_all_demetr["severity"] == "base")
            & (df_all_demetr["pert_desc"] != "reference as translation - baseline"),
            "label",
        ] = "ERR"

        # use labels followed by score_ced function to ensure consistency with WMT data
        df_all_demetr["score"] = score_ced(df_all_demetr["label"])
        df_all_demetr = df_all_demetr.rename(columns={"src_sent": "src", "pert_sent": "mt"})
        df_all_demetr[["src", "mt", "score"]].to_csv(os.path.join(PROCESSED_DATA_DIR, "demetr.csv"))


if __name__ == "__main__":
    main()
