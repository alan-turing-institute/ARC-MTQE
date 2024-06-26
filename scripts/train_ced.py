import argparse
import os

import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from torch import cuda

import wandb
from mtqe.models.loaders import load_model_from_file
from mtqe.utils.models import get_model_name
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


def get_callbacks(
    config: dict,
    model_name: str,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> list:
    """
    Creates the callbacks to be used by the trainer
    The trainer will always have a ModelCheckpoint callback, which will be stored in the first index of the list
    The trainer may or may not have an EarlyStopping callback
    Config for both callbacks can be provided in the config files, but if not provided default values will be
    used for the model checkpoint callback.

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
    checkpoint_path = checkpoint_dir + "/" + model_name + "/"
    if "early_stopping" in config:
        early_stopping_params = config["early_stopping"]
        # callback for early stopping
        early_stopping_callback = EarlyStopping(**early_stopping_params)
        # callback to log model checkpoints locally
        if "model_checkpoint" in config:
            model_checkpoint_params = config["model_checkpoint"]
            checkpoint_callback = ModelCheckpoint(checkpoint_path, **model_checkpoint_params)
        else:
            # If checkpoint config is not provided then set this with the same metric and mode as the early stopping
            # callback so that it can save the best checkpoint for the monitored metric
            checkpoint_callback = ModelCheckpoint(
                checkpoint_path, monitor=early_stopping_params["monitor"], mode=early_stopping_params["mode"]
            )
        callbacks = [checkpoint_callback, early_stopping_callback]
    elif "model_checkpoint" in config:
        model_checkpoint_params = config["model_checkpoint"]
        checkpoint_callback = ModelCheckpoint(checkpoint_path, **model_checkpoint_params)
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = ModelCheckpoint(checkpoint_path)
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

    # Initialise random seed
    seed_everything(seed, workers=True)

    # Create model
    model = load_model_from_file(config, experiment_name, train_model=True)
    # Name for this model / experiment
    model_name = get_model_name(experiment_group_name, experiment_name, seed)

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
