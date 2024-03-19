import os
from datetime import datetime

import pandas as pd
from comet import download_model
from comet.models import UnifiedMetric
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

import wandb
from mtqe.data.loaders import get_ced_data_paths, load_ced_test_data
from mtqe.models.comet import load_qe_model_from_checkpoint
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import OUTPUTS_DIR


def evaluate_model(lp: str, model: UnifiedMetric, out_dir: str):
    """
    Evaluate the model using the test data for one language pair and
    save the results to the out_dir
    """
    # load test data
    df_test_data = load_ced_test_data(lp)

    test_data = df_test_data.to_dict("records")

    # predict
    model_output = model.predict(test_data, batch_size=8, gpus=0)

    # save output
    out_file_name = os.path.join(out_dir, f"{lp}_cometkiwi.csv")
    df_results = pd.DataFrame({"idx": df_test_data["idx"], "comet_score": model_output.scores})
    df_results.to_csv(out_file_name, index=False)


def make_output_folder(folder_name: str, out_dir: str = OUTPUTS_DIR):
    new_dir = os.path.join(out_dir, folder_name)
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def train_comet(model: UnifiedMetric, paths_train_data: list, paths_dev_data: list, experiment_name: str):
    """
    Trains the given model using the processes developed in the comet
    code base
    """

    # Set paths for training and val data
    model.hparams.train_data = paths_train_data
    model.hparams.validation_data = paths_dev_data
    model.hparams.batch_size = 8

    wandb_logger = WandbLogger(
        entity="turing-arc",
        project="MTQE",
        name=experiment_name,
        mode="online",
        log_model=False,
    )

    # create new trainer object
    trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=wandb_logger)

    trainer.fit(model)

    wandb.log({"test_metric": 0.5})
    wandb.config["test_config"] = "abcd"

    wandb_logger.experiment.config.update({"test_config_2": "efgh"})

    wandb.finish()

    return model


def train_ced_model_class(language_pairs: list = LI_LANGUAGE_PAIRS_WMT_21_CED, freeze_encoder: bool = True):
    """
    This uses COMET-KIWI for classification rather than regression
    Current process is to only use training data from one language pair - will want to be able
    to configure this so that multiple files of training data can be used.
    """
    train_paths = get_ced_data_paths("train")
    dev_paths = get_ced_data_paths("dev")

    for idx, lp in enumerate(language_pairs):
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_qe_model_from_checkpoint(model_path, freeze_encoder=freeze_encoder, final_activation="sigmoid")

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H:%M:%S")
        model_name = "refactor1_cls_" + lp + "_" + now_str
        model = train_comet(model, [train_paths[idx]], [dev_paths[idx]], model_name)

        # Create output folder specific to this training approach
        out_dir = make_output_folder("trained_model_classification/" + model_name)
        evaluate_model(lp, model, out_dir)

        break


if __name__ == "__main__":
    train_ced_model_class()
