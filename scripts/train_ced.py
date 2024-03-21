import os
from datetime import datetime

import yaml
from comet import download_model
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

import wandb
from mtqe.data.loaders import get_ced_data_paths
from mtqe.models.comet import load_qe_model_from_checkpoint
from mtqe.utils.paths import CHECKPOINT_DIR, CONFIG_DIR


def create_trainer(experiment_name: str, wandb_params: dict, checkpoint_dir: str = CHECKPOINT_DIR, **kwargs) -> Trainer:
    """
    Creates a trainer with a WandB logger, sets local directory to log checkpoint - have not yet
    tested loading from the local checkpoint

    Returns a Trainer object
    """

    wandb_logger = WandbLogger(
        entity=wandb_params["entity"],
        project=wandb_params["project"],
        name=experiment_name,
        mode="online",
        log_model=False,  # Takes too long to log the checkpoint in wandb, so keep false
    )

    # early_stopping_callback = EarlyStopping(monitor="val_precision", patience=1, verbose=False, strict=True)
    # callback to log model checkpoints locally
    checkpoint_callback = ModelCheckpoint(checkpoint_dir + "/" + experiment_name + "/")
    # create new trainer object
    # trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping_callback], **kwargs)
    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], **kwargs)

    return trainer


def load_model_from_file(config: dict, experiment_name: str) -> LightningModule:
    """
    Gets paths to train and dev data from specification in config file
    Loads model using hparams from config file
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


def create_model_name(experiment_group_name: str, experiment_name: str, seed: int):
    """
    Creates (as good as unique) model name using the current datetime stamp
    """
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H:%M:%S")
    model_name = experiment_group_name + "__" + experiment_name + "__" + str(seed) + "__" + now_str
    return model_name


def train_model(experiment_group_name: str, experiment_name: str, seed: int):
    """
    Takes the name of an experiment group (and config file), an experiment name and random seed.
    Opens config file
    Checks that the named experiment and random seed are in the file
    Creates a model using the data from the config file
    Creates a trainer object using data from the config file
    Fits the model and results are logged to wandb and checkpoint saved locally
    """
    with open(os.path.join(CONFIG_DIR, experiment_group_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)

    assert experiment_name in config["experiments"], (
        experiment_name + " does not exist in " + experiment_group_name + ".yaml"
    )
    assert seed in config["seeds"], "seed " + str(seed) + " does not exist in " + experiment_group_name + ".yaml"

    # Create model
    model = load_model_from_file(config, experiment_name)

    seed_everything(seed, workers=True)

    # Create trainer
    model_name = create_model_name(experiment_group_name, experiment_name, seed)
    trainer_params = config["trainer_config"]
    trainer = create_trainer(model_name, config["wandb"], **trainer_params)

    trainer.fit(model)

    # If we want to log other metrics or config after the run, we could use the following code:
    # wandb.log({"test_metric": 0.5})
    # wandb.config["test_config"] = "abcd"

    wandb.finish()


if __name__ == "__main__":
    train_model("experiment_group_1", "en-cs_frozen", 12)
