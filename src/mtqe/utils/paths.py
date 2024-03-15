import os

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
DATA_MLQE_PE_DIR = os.path.join(DATA_DIR, "mlqe-pe", "data")
DATA_CED_21_DIR = os.path.join(DATA_MLQE_PE_DIR, "catastrophic_errors")
DATA_DA_22_DIR = os.path.join(DATA_DIR, "wmt-qe-2022-data", "test_data-gold_labels", "task1_da")
DATA_DA_23_DIR = os.path.join(DATA_DIR, "wmt-qe-2023-data", "test_data_2023", "task1_sentence_level")

# MODELS - LOCAL
COMET_QE_21 = os.path.join(ROOT_DIR, "models", "wmt21-comet-qe-da", "checkpoints", "model.ckpt")

# OUTPUTS
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
