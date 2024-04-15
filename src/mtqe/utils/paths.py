import os
import typing

# This file contains paths relative to the project root that can be imported.
# If this file moves relative to the project root, find_repo_root() must be changed.


def find_project_root() -> str:
    """
    Returns root folder for the project. Current file is assumed to be at:
        "ARC-MTQE/src/mtqe/utils/paths.py"

    Returns
    ----------
    str
        Path to the root ARC-MTQE directory.
    """

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# path to ARC-MTQE directory
ROOT_DIR = find_project_root()

# DATA
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MLQE_PE_DIR = os.path.join(DATA_DIR, "mlqe-pe", "data")
WMT_QE_21_CED_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors")
WMT_QE_21_CED_GOLDLABELS_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors_goldlabels")
WMT_QE_22_DIR = os.path.join(DATA_DIR, "wmt-qe-2022-data")
WMT_QE_23_DIR = os.path.join(DATA_DIR, "wmt-qe-2023-data")


def get_mlqepe_catastrophic_errors_data_paths(
    data_split: str, lp: str, mlqepe_dir: str = MLQE_PE_DIR
) -> typing.Union[str, typing.List[str]]:
    """
    Get path(s) to the original CED data file(s) for the given data split and language pair.
    This is either a single file or two files (the first containing the sentences and the second
    containg the gold labels).a

    Parameters
    ----------
    data_split: str
        One of "train", "dev" or "test".
    lps: list[str]
        List of language pairs to return CED data for (passed as IOS codes, such as ["en-cs"]).
    mlqepe_dir: str
        Path to the `data/` directory in clone of the `sheffieldnlp/mlqe-pe/` GitHub repository.

    Returns
    ----------
    Union[str, list]
        For "train" and "dev", a single path is returned. For "test" data, a path to the src/mt
        text is returned first followed by path to the gold labels.
    """

    if data_split == "test":
        text_data_path = os.path.join(
            mlqepe_dir, "catastrophic_errors", f"{lp.replace('-', '')}_majority_test_blind.tsv"
        )
        labels_path = os.path.join(
            mlqepe_dir,
            "catastrophic_errors_goldlabels",
            f"{lp.replace('-', '')}_majority_test_goldlabels",
            "goldlabels.txt",
        )
        return [text_data_path, labels_path]
    else:
        return os.path.join(mlqepe_dir, "catastrophic_errors", f"{lp.replace('-', '')}_majority_{data_split}.tsv")


def get_processed_ced_data_paths(
    data_split: str, lps: typing.List[str], mlqepe_dir: str = MLQE_PE_DIR
) -> typing.List[str]:
    """
    Get paths to processed data files
    WMT 2021 Critical Error Detection train or dev data CSV files for given language pairs.
    These are then passed to the CEDModel.

    Parameters
    ----------
    data_split: str
        One of "train" or "dev".
    lps: list[str]
        List of language pairs to return CED data for (passed as IOS codes, such as ["en-cs"]).
    mlqepe_dir: str
        Path to the `data/` directory in clone of the `sheffieldnlp/mlqe-pe` GitHub repository.

    Returns
    ----------
    list[str]
        List of CSV file paths.
    """

    assert data_split in ["train", "dev"], f"Invalid data_split {data_split}, valid input is either 'train' or 'dev'..."

    file_paths = []
    for lp in lps:
        fp = get_mlqepe_catastrophic_errors_data_paths(data_split, lp, mlqepe_dir)
        fp_csv = fp.replace("tsv", "csv")
        file_paths.append(fp_csv)

    return file_paths


# CONFIG
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

# CHECKPOINTS
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# SLURM SCRIPTS FOR TRAINING
SLURM_DIR = os.path.join(ROOT_DIR, "scripts", "slurm_scripts")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "scripts", "templates")

# MODELS (LOCAL)
COMET_QE_21 = os.path.join(ROOT_DIR, "models", "wmt21-comet-qe-da", "checkpoints", "model.ckpt")

# OUTPUTS
PREDICTIONS_DIR = os.path.join(ROOT_DIR, "predictions")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
