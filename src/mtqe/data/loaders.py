import os
import typing

import numpy as np
import pandas as pd

from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import MLQE_PE_DIR, WMT_QE_22_DIR, WMT_QE_23_DIR


def comet_format(data: pd.DataFrame) -> typing.List[typing.Dict[str, str]]:
    """
    Format source and machine translated sentence pairs into COMET format.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame of source and translated text in "src" and "mt" columns.

    Returns
    ----------
    list[dict[str, str]]
        [{"src":"...", "mt":"..."}, {...}]
    """

    return [{"src": src, "mt": mt} for src, mt in zip(data["src"], data["mt"])]


def load_da_test_data(
    lp: str, year: str, wmt22_dir: str = WMT_QE_22_DIR, wmt23_dir: str = WMT_QE_23_DIR
) -> pd.DataFrame:
    """
    Load labelled WMT Direct Assessment test data for given language pair and WMT year.

    Parameters
    ----------
    lp: str
        The langauge pair, passed as IOS code (e.g., "en-cs").
    year: str
        The WMT year ("2022" or "2023").
    wmt22_dir: str
        Path to clone of the WMT-QE-TASK/wmt-qe-2022-data GitHub repistory.
    wmt23_dir: str
        Path to clone of the WMT-QE-TASK/wmt-qe-2023-data GitHub repistory.

    Returns
    ----------
    pd.DataFrame
        DataFrame with "src", "mt" and "score" columns.
    """

    assert year in ["2022", "2023"], f"Invalid year {year}, valid input is either '2022' or '2023'..."

    if year == "2022":
        WMT_DA_22_TEST_DIR = os.path.join(wmt22_dir, "test_data-gold_labels", "task1_da")

        with open(os.path.join(WMT_DA_22_TEST_DIR, lp, "test.2022.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(WMT_DA_22_TEST_DIR, lp, "test.2022.src")) as f:
            src_data = f.read().splitlines()

        # NOTE: inconsistency in 2022 file names
        if os.path.exists(os.path.join(WMT_DA_22_TEST_DIR, lp, f"test.2022.{lp}.da_score")):
            with open(os.path.join(WMT_DA_22_TEST_DIR, lp, f"test.2022.{lp}.da_score")) as f:
                labels = [float(score) for score in f.read().splitlines()]
        else:
            with open(os.path.join(WMT_DA_22_TEST_DIR, lp, f"test2022.{lp}.da_score")) as f:
                labels = [float(score) for score in f.read().splitlines()]

    elif year == "2023":
        WMT_DA_23_TEST_DIR = os.path.join(wmt23_dir, "test_data_2023", "task1_sentence_level")

        with open(os.path.join(WMT_DA_23_TEST_DIR, lp, f"test.{lp.replace('-', '')}.final.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(WMT_DA_23_TEST_DIR, lp, f"test.{lp.replace('-', '')}.final.src")) as f:
            src_data = f.read().splitlines()

        # 2023 has a single goldlabels file for all language pairs
        labels_path = os.path.join(wmt23_dir, "goldlabels", "hallucinations_gold_T1s_header.tsv")
        df_labels = pd.read_csv(labels_path, sep="\t")
        labels = df_labels[df_labels["lp"] == lp]["score"]

    return pd.DataFrame({"src": src_data, "mt": mt_data, "score": labels})


def load_ced_test_data(lp: str, mlqepe_dir: str = MLQE_PE_DIR) -> pd.DataFrame:
    """
    Load labelled WMT 2021 Critical Error Detection test data for given language pair.

    Parameters
    ----------
    lp: str
        The langauge pair, passed as IOS code (e.g., "en-cs").
    mlqepe_dir: str
        Path to clone of the sheffieldnlp/mlqe-pe GitHub repistory.

    Returns
    ----------
    pd.DataFrame
        DataFrame with "src", "mt" and "score" columns.
    """

    WMT_QE_21_CED_DIR = os.path.join(mlqepe_dir, "catastrophic_errors")
    data_path = os.path.join(WMT_QE_21_CED_DIR, f"{lp}_majority_test_blind.tsv")
    df_data = pd.read_csv(data_path, sep="\t", header=None, names=["idx", "src", "mt"])

    WMT_QE_21_CED_LABELS_DIR = os.path.join(mlqepe_dir, "catastrophic_errors_goldlabels")
    labels_path = os.path.join(WMT_QE_21_CED_LABELS_DIR, f"{lp}_majority_test_goldlabels", "goldlabels.txt")
    df_labels = pd.read_csv(labels_path, sep="\t", header=None, names=["lang_pair", "ref", "idx", "score"])

    df_full = pd.merge(df_data, df_labels, on="idx")

    return df_full[["src", "mt", "score"]]


def save_ced_data_to_csv(data_split: str, lp: str, mlqepe_dir: str = MLQE_PE_DIR):
    """
    Save WMT 2021 Critical Error Detection train or dev data for given language pair to CSV file.

    Parameters
    ----------
    data_split: str
        One of "train" or "dev".
    lp: str
        The langauge pair, passed as IOS code (e.g., "en-cs").
    """

    path_data = os.path.join(mlqepe_dir, "catastrophic_errors", f"{lp.replace('-', '')}_majority_{data_split}.tsv")
    df_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

    # NOT en error = 1, CRITICAL ERROR = 0
    df_data["score"] = np.where(df_data["error"] == "NOT", 1, 0)

    # save as csv file
    df_data[["src", "mt", "score"]].to_csv(path_data.replace("tsv", "csv"))


def load_ced_data_paths(
    data_split: str, lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED, mlqepe_dir: str = MLQE_PE_DIR
) -> typing.List[str]:
    """
    Get paths to WMT 2021 Critical Error Detection train or dev data CSV files for given language pairs.
    NOTE: creates the CSV file if it does not already exist.

    Parameters
    ----------
    data_split: str
        One of "train" or "dev".
    lps: list[str]
        List of language pairs to return CED data for (passed as IOS codes, such as ["en-cs"]).
    mlqepe_dir: str
        Path to clone of the sheffieldnlp/mlqe-pe GitHub repistory.

    Returns
    ----------
    list[str]
        List of CSV file paths.
    """

    assert data_split in ["train", "dev"], f"Invalid data_split {data_split}, valid input is either 'train' or 'dev'..."

    file_paths = []
    for lp in lps:
        fp = os.path.join(mlqepe_dir, "catastrophic_errors", f"{lp.replace('-', '')}_majority_{data_split}.csv")
        if not os.path.exists(fp):
            save_ced_data_to_csv(data_split, lp, mlqepe_dir)
        file_paths.append(fp)
    return file_paths
