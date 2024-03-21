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
MLQE_PE_DIR = os.path.join(DATA_DIR, "mlqe-pe", "data")
WMT_QE_21_CED_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors")
WMT_QE_21_CED_GOLDLABELS_DIR = os.path.join(MLQE_PE_DIR, "catastrophic_errors_goldlabels")
WMT_QE_22_DIR = os.path.join(DATA_DIR, "wmt-qe-2022-data")
WMT_QE_23_DIR = os.path.join(DATA_DIR, "wmt-qe-2023-data")

# CONFIG
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")

# CHECKPOINTS
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# MODELS (LOCAL)
COMET_QE_21 = os.path.join(ROOT_DIR, "models", "wmt21-comet-qe-da", "checkpoints", "model.ckpt")

# OUTPUTS
PREDICTIONS_DIR = os.path.join(ROOT_DIR, "predictions")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
