import argparse
import os
from datetime import datetime

import wandb
import yaml
from comet import download_model
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from torch import cuda

from mtqe.data.loaders import get_ced_data_paths
from mtqe.models.comet import load_qe_model_from_checkpoint
from mtqe.utils.paths import CHECKPOINT_DIR, CONFIG_DIR


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get experiment config settings")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name")
    parser.add_argument("-e", "--exp", required=True, help="Experiment name")
    parser.add_argument("-s", "--seed", required=True, help="Seed")

    return parser.parse_args()


def load_model_from_file(config: dict, experiment_name: str) -> LightningModule:
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
        lps = train_data[dataset]["language_pairs"]
        train_paths.extend(get_ced_data_paths("train", lps))
    for dataset in dev_data:
        lps = dev_data[dataset]["language_pairs"]
        dev_paths.extend(get_ced_data_paths("dev", lps))

    model_params = config["hparams"]  # these don't change between experiments
    if "hparams" in exp_setup:
        # add any experiment-specific params
        model_params = {**model_params, **exp_setup["hparams"]}

    # checkpoint path is currently hard-coded below - I think this should also
    # be in the config so we can load any checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    # reload_hparams hard-coded to False, but might want to modify this in future to be able to load from checkpoint
    model = load_qe_model_from_checkpoint(model_path, train_paths, dev_paths, reload_hparams=False, **model_params)

    return model


def create_model_name(experiment_group_name: str, experiment_name: str, seed: int) -> str:
    """
    Creates (as good as unique) model name using the current datetime stamp

    Parameters
    ----------
    experiment_group_name: str
        The name of the group of experiments
    experiment_name: str
        The name of the experiment
    seed: int
        The initial random seed value

    Returns
    ----------
    str
        A model name
    """
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    model_name = experiment_group_name + "__" + experiment_name + "__" + str(seed) + "__" + now_str
    return model_name


def get_callbacks(
    config: dict,
    model_name: str,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> list:
    """
    Creates the callbacks to be used by the trainer
    The trainer will always have a ModelCheckpoint callback, which will be stored in the first index of the list
    The trainer may or may not have an EarlyStopping callback

    Parameters
    ----------
    config: dict
        Dictionary holding the config for the trainer
    model_name: str
        The name of the model
    checkpoint_dir: str
        The directory where the checkpoints will be stored

    Returns
    -------
    list[ModelCheckpoint]
        A list of callbacks that will be used.
    """

    if "early_stopping" in config:
        early_stopping_params = config["early_stopping"]
        # callback for early stopping
        early_stopping_callback = EarlyStopping(**early_stopping_params)
        # callback to log model checkpoints locally
        # also needs to monitor the same metric as the early stopping callback so that we can work out
        # which is the best checkpoint for that metric, mode currently hard-coded to 'max'
        checkpoint_callback = ModelCheckpoint(
            checkpoint_dir + "/" + model_name + "/", monitor=early_stopping_params["monitor"], mode="max"
        )
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        checkpoint_callback = ModelCheckpoint(checkpoint_dir + "/" + model_name + "/")
        callbacks = [checkpoint_callback]

    return callbacks


def train_model(
    experiment_group_name: str,
    experiment_name: str,
    seed: int,
    config_dir: str = CONFIG_DIR,
):
    """
    Takes the name of an experiment group (and config file), an experiment name and random seed.
    Opens config file
    Checks that the named experiment and random seed are in the file
    Creates a model using the data from the config file
    Creates a trainer object using data from the config file
    Fits the model and results are logged to wandb and checkpoint saved locally

    Parameters
    ----------
    experiment_group_name: str
        The name of the group of experiments
    experiment_name: str
        The name of the experiment
    seed: int
        The initial random seed value
    config_dir: str
        The directory where the config files are stored

    """

    with open(os.path.join(config_dir, experiment_group_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)

    # Check that the experiment name is in the yaml file - the load won't work otherwise.
    assert experiment_name in config["experiments"], (
        experiment_name + " does not exist in " + experiment_group_name + ".yaml"
    )
    assert int(seed) in config["seeds"], "seed " + str(seed) + " does not exist in " + experiment_group_name + ".yaml"

    # Create model
    model = load_model_from_file(config, experiment_name)
    # Name for this model / experiment
    model_name = create_model_name(experiment_group_name, experiment_name, seed)

    # Initialise random seed
    seed_everything(seed, workers=True)

    # Create wandb logger
    wandb_params = config["wandb"]
    wandb_logger = WandbLogger(
        entity=wandb_params["entity"],
        project=wandb_params["project"],
        name=model_name,
        mode="online",
        log_model=False,  # Takes too long to log the checkpoint in wandb, so keep false
    )

    # would be better if this was set earlier and then passed to functions as required
    # to functions when needed - currently also set in metrics.py and comet.py
    device = "cuda" if cuda.is_available() else "cpu"
    # create the callbacks for the trainer
    callbacks = get_callbacks(config, model_name)
    # create new trainer object
    trainer_params = config["trainer_config"]
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, accelerator=device, **trainer_params)

    trainer.fit(model)

    # Recording the 'best' checkpoint according to the metric we are monitoring
    # so that we can load it again - do we want to save any other checkpoints?
    # The ModelCheckpoint is identified as the first callback to be stored in the
    # callbacks list - I don't particularly like this
    wandb.config["best_checkpoint_path"] = callbacks[0].best_model_path

    wandb.finish()


def main():
    args = parse_args()
    experiment_group_name = args.group
    experiment_name = args.exp
    seed = args.seed

    train_model(experiment_group_name, experiment_name, seed)


if __name__ == "__main__":
    main()
