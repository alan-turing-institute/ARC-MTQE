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
MLQE_PE_DIR = os.path.join(DATA_DIR, "mlqe-pe", "data")
WMT_QE_21_CED_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors")
WMT_QE_21_CED_GOLDLABELS_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors_goldlabels")
WMT_QE_22_DIR = os.path.join(DATA_DIR, "wmt-qe-2022-data")
WMT_QE_23_DIR = os.path.join(DATA_DIR, "wmt-qe-2023-data")


def get_ced_data_path(data_split: str, lp: str, mlqepe_dir: str = MLQE_PE_DIR) -> typing.Union[str, typing.List[str]]:
    """
    Get path(s) to CED data file(s) for the given data split and language pair.

    Parameters
    ----------
    data_split: str
        One of "train", "dev" or "test".
    lps: list[str]
        List of language pairs to return CED data for (passed as IOS codes, such as ["en-cs"]).
    mlqepe_dir: str
        Path to the `data/` directory in clone of the sheffieldnlp/mlqe-pe GitHub repository.

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


# MODELS (LOCAL)
COMET_QE_21 = os.path.join(ROOT_DIR, "models", "wmt21-comet-qe-da", "checkpoints", "model.ckpt")

# OUTPUTS
PREDICTIONS_DIR = os.path.join(ROOT_DIR, "predictions")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
