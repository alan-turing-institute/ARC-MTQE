import os
from pathlib import Path

import torch
from comet import download_model, load_from_checkpoint
from pytorch_lightning import LightningModule

from mtqe.models.comet import CEDModel
from mtqe.utils.paths import COMET_QE_21, PROCESSED_DATA_DIR


def load_comet_model(model_name: str = "cometkiwi_22", comet_qe_21: str = COMET_QE_21):
    """
    Return one of the COMET models (COMETKiwi22 by default).
    """

    assert model_name in [
        "comet_qe_20",
        "comet_qe_21",
        "cometkiwi_22",
        "cometkiwi_23_xl",
    ], f"Invalid model_name {model_name}, ...."

    if model_name == "comet_qe_20":
        model_path = download_model("Unbabel/wmt20-comet-qe-da")
    elif model_name == "comet_qe_21":
        model_path = comet_qe_21
    elif model_name == "cometkiwi_22":
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    elif model_name == "cometkiwi_23_xl":
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")

    model = load_from_checkpoint(model_path)

    return model


def load_qe_model_from_checkpoint(
    checkpoint_path: str,
    train_model: bool,
    paths_train_data: list = None,
    paths_dev_data: list = None,
    strict: bool = False,
    **kwargs,
) -> CEDModel:
    """
    This code has been updated from the load_from_checkpoint function imported
    from the comet repo - the difference is that the class is hard-coded
    to be CEDModel and the device is set to cuda, if available.

    Parameters
    ----------
    checkpoint_path: str
        Path to a model checkpoint
    train_model: bool
        Set to `True` if the model is going to be trained, `False` otherwise
    paths_train_data: list
        List of paths to training datasets
    paths_val_data: list
        List of paths to validation datasets
    strict: bool
        Strictly enforce that the keys in checkpoint_path match the
        keys returned by this module's state dict. Defaults to False

    Returns
    -------
    model: CEDModel
        An instance of class CEDModel, loaded from the given checkpoint

    """
    checkpoint_path = Path(checkpoint_path)
    # would be better if this was set once (in train_ced.py) and passed
    # to functions when needed - currently also set in metrics.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CEDModel.load_from_checkpoint(
        checkpoint_path,
        load_pretrained_weights=False,
        map_location=torch.device(device),
        strict=strict,
        train_data=paths_train_data,
        validation_data=paths_dev_data,
        **kwargs,
    )
    if train_model:
        model.update_estimator()
    return model


def load_model_from_file(config: dict, experiment_name: str, train_model: bool) -> LightningModule:
    """
    Gets paths to train and dev data from specification in config file
    Loads model using hparams from config file
    NOTE: the checkpoint to load the model is currently hard-coded

    Parameters
    ----------
    config: dict
        Dictionary containing the config needed to load the model
    experiment_name: str
        The name of the experiment
    train_model: bool
        Set to `True` if the model is to be loaded for training, and `False` otherwise (just making predictions)

    Returns
    -------
    LightningModule
        A QE model inherited from CometKiwi, repurposed for clasification
    """
    # set data paths
    train_paths = []
    dev_paths = []
    exp_setup = config["experiments"][experiment_name]
    train_data = exp_setup["train_data"]
    dev_data = exp_setup["dev_data"]
    for dataset in train_data:
        if "all" in train_data[dataset]["language_pairs"] and train_data[dataset]["dataset_name"] == "multilingual_ced":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "all_multilingual_train.csv"))
        elif train_data[dataset]["dataset_name"] == "demetr":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "demetr_train.csv"))
        elif train_data[dataset]["dataset_name"] == "all_multilingual_demetr":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "all_multilingual_with_demetr_train.csv"))
        elif train_data[dataset]["dataset_name"] == "wmt22_ende_ced":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "wmt22_en-de_train.csv"))
        elif train_data[dataset]["dataset_name"] == "wmt22_ende_ced_reduced":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "wmt22_en-de_train_reduced.csv"))
        elif train_data[dataset]["dataset_name"] == "wmt22_ende_ced_small":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "wmt22_en-de_train_small.csv"))
        elif train_data[dataset]["dataset_name"] == "balanced_ende":
            train_paths.append(os.path.join(PROCESSED_DATA_DIR, "balanced_ende.csv"))
        else:
            for lp in train_data[dataset]["language_pairs"]:
                if train_data[dataset]["dataset_name"] == "ced":
                    train_paths.append(os.path.join(PROCESSED_DATA_DIR, f"{lp}_majority_train.csv"))
                elif train_data[dataset]["dataset_name"] == "demetr_ced":
                    train_paths.append(os.path.join(PROCESSED_DATA_DIR, f"{lp}_train_with_demetr.csv"))
                elif train_data[dataset]["dataset_name"] == "multilingual_ced":
                    train_paths.append(os.path.join(PROCESSED_DATA_DIR, f"{lp}_multilingual_train.csv"))
    for dataset in dev_data:
        if (
            dev_data[dataset]["dataset_name"] == "wmt22_ende_ced"
            or dev_data[dataset]["dataset_name"] == "wmt22_ende_ced_reduced"
        ):
            dev_paths.append(os.path.join(PROCESSED_DATA_DIR, "wmt22_en-de_dev.csv"))
        elif dev_data[dataset]["dataset_name"] == "demetr":
            dev_paths.append(os.path.join(PROCESSED_DATA_DIR, "demetr_dev.csv"))
        else:
            # in most scenarios, want to use the authentic validation data from WMT21
            for lp in dev_data[dataset]["language_pairs"]:
                dev_paths.append(os.path.join(PROCESSED_DATA_DIR, f"{lp}_majority_dev.csv"))
            if dev_data[dataset]["dataset_name"] == "all_multilingual_demetr":
                dev_paths.append(os.path.join(PROCESSED_DATA_DIR, "demetr_dev.csv"))

    model_params = config["hparams"]  # these don't change between experiments
    if "hparams" in exp_setup:
        # add any experiment-specific params
        model_params = {**model_params, **exp_setup["hparams"]}

    if "model_path" in config:
        model_path = config["model_path"]["path"]
    else:
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    # reload_hparams hard-coded to False, but might want to modify this in future - this would force params
    # to be loaded from a file, but we want to pass them through as arguments here.
    model = load_qe_model_from_checkpoint(
        model_path,
        train_model=train_model,
        paths_train_data=train_paths,
        paths_dev_data=dev_paths,
        reload_hparams=False,
        **model_params,
    )

    return model
