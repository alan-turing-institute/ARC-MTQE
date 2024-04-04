import os
import typing

import numpy as np
import pandas as pd

from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import (
    MLQE_PE_DIR,
    WMT_QE_22_DIR,
    WMT_QE_23_DIR,
    get_ced_data_path,
)


def score_ced(ced_data: typing.Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Rescore ERR/NOT CED labels into binary indicators: ERR = 0, NOT = 1.

    Parameters
    ----------
    ced_data: Union[pd.Series, np.ndarray]
        Array of ERR/NOT labels.

    Returns
    ----------
    np.ndarray
    """

    return np.where(ced_data == "NOT", 1, 0)


def comet_format(data: pd.DataFrame) -> typing.List[typing.Dict[str, str]]:
    """
    Format source and machine translated sentence pairs into COMET format:
        - [{"src": "<sentence>", "mt": "<translation>"}, {...}].

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame of source and translated text in "src" and "mt" columns.

    Returns
    ----------
    list[dict[str, str]]
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
        DataFrame composed of the following columns:
            - "src": source text
            - "mt": machine translated text
            - "score": Direct Assessment score
    """

    assert year in ["2022", "2023"], f"Invalid year {year}, valid input is either '2022' or '2023'..."

    if year == "2022":
        WMT_DA_22_TEST_DIR = os.path.join(wmt22_dir, "test_data-gold_labels", "task1_da")

        with open(os.path.join(WMT_DA_22_TEST_DIR, lp, "test.2022.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(WMT_DA_22_TEST_DIR, lp, "test.2022.src")) as f:
            src_data = f.read().splitlines()

        # there's an inconsistency in 2022 file names
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
        labels_path = os.path.join(wmt23_dir, "gold_labels", "hallucinations_gold_T1s_header.tsv")
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
        Path to the `data/` directory in clone of the sheffieldnlp/mlqe-pe GitHub repository.

    Returns
    ----------
    pd.DataFrame
        DataFrame composed of the following columns:
            - "idx": unique identifier
            - "src": source text
            - "mt": machine translated text
            - "score": whether the translation contains a critical error (0) or not (1)
    """

    data_path, labels_path = get_ced_data_path("test", lp, mlqepe_dir)
    df_data = pd.read_csv(data_path, sep="\t", header=None, names=["idx", "src", "mt"])
    df_labels = pd.read_csv(labels_path, sep="\t", header=None, names=["lang_pair", "ref", "idx", "error"])

    # NOT en error = 1, CRITICAL ERROR = 0
    df_labels["score"] = score_ced(df_labels["error"])

    df_full = pd.merge(df_data, df_labels, on="idx")

    return df_full[["idx", "src", "mt", "score"]]


def load_ced_data(data_split: str, lp: str, mlqepe_dir: str = MLQE_PE_DIR) -> pd.DataFrame:
    """
    Load WMT 2021 Critical Error Detection train, dev or test data for given language pair.

    Parameters
    ----------
    data_split: str
        One of "train", "dev" or "test".
    lp: str
        The langauge pair, passed as IOS code (e.g., "en-cs").
    mlqepe_dir: str
        Path to the `data/` directory in clone of the sheffieldnlp/mlqe-pe GitHub repository.

     Returns
    ----------
    pd.DataFrame
        DataFrame composed of the following columns:
            - "idx": unique identifier
            - "src": source text
            - "mt": machine translated text
            - "score": whether the translation contains a critical error (0) or not (1)
    """

    if data_split == "test":
        df_data = load_ced_test_data(lp, mlqepe_dir)

    else:
        path_data = get_ced_data_path(data_split, lp, mlqepe_dir)
        df_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

        # NOT en error = 1, CRITICAL ERROR = 0
        df_data["score"] = score_ced(df_data["error"])

    return df_data[["idx", "src", "mt", "score"]]


def save_ced_data_to_csv(data_split: str, lp: str, mlqepe_dir: str = MLQE_PE_DIR):
    """
    Save WMT 2021 Critical Error Detection train or dev data for given language pair to CSV file.

    Parameters
    ----------
    data_split: str
        One of "train" or "dev".
    lp: str
        The langauge pair, passed as IOS code (e.g., "en-cs").
    mlqepe_dir: str
        Path to the `data/` directory in clone of the sheffieldnlp/mlqe-pe GitHub repository.
    """

    path_data = get_ced_data_path(data_split, lp, mlqepe_dir)
    df_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

    # NOT en error = 1, CRITICAL ERROR = 0
    df_data["score"] = score_ced(df_data["error"])

    # save as csv file
    df_data[["src", "mt", "score"]].to_csv(path_data.replace("tsv", "csv"))


def get_ced_data_paths(
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
        Path to the `data/` directory in clone of the sheffieldnlp/mlqe-pe GitHub repository.

    Returns
    ----------
    list[str]
        List of CSV file paths.
    """

    assert data_split in ["train", "dev"], f"Invalid data_split {data_split}, valid input is either 'train' or 'dev'..."

    file_paths = []
    for lp in lps:
        fp = get_ced_data_path(data_split, lp, mlqepe_dir)
        fp_csv = fp.replace("tsv", "csv")
        file_paths.append(fp_csv)

        # the CSV files get created when `make data` is called so this should not be necessary
        if not os.path.exists(fp):
            save_ced_data_to_csv(data_split, lp, mlqepe_dir)

    return file_paths
