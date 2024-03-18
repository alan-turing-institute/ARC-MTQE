import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from comet import download_model
from comet.models import UnifiedMetric
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

import wandb

LI_LANGUAGE_PAIRS = ["encs", "ende", "enja", "enzh"]
# path to ARC-MTQE directory
MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
# WMT 2021 critical error test data
DATA_DIR = os.path.join(MAIN_DIR, "data", "mlqe-pe", "data", "catastrophic_errors")
# save results here
OUT_DIR = os.path.join(MAIN_DIR, "predictions", "ced_test_data")
os.makedirs(OUT_DIR, exist_ok=True)


class CEDModel(UnifiedMetric):
    """
    New class created that inherits from the UnifiedMetric class from COMET
    This class overrides the val_dataloader and train_dataloader functions
    to overcome an error obtianed when running the first training approach
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.9,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 3.0e-06,
        learning_rate: float = 3.0e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "microsoft/infoxlm-large",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = True,
        word_layer: int = 24,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_segments: List[str] = ["mt", "src", "ref"],
        word_level_training: bool = False,
        loss_lambda: float = 0.65,
        error_labels: List[str] = ["minor", "major"],
        cross_entropy_weights: Optional[List[float]] = None,
        load_pretrained_weights: bool = True,
    ):

        super().__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            sent_layer=sent_layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            word_layer=word_layer,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            hidden_sizes=hidden_sizes,
            activations=activations,
            final_activation=final_activation,
            input_segments=input_segments,
            word_level_training=word_level_training,
            loss_lambda=loss_lambda,
            # error_labels = error_labels,
            # cross_entropy_weights = cross_entropy_weights,
            load_pretrained_weights=load_pretrained_weights,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation sets."""
        val_data = [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                num_workers=0,
            )
        ]
        for validation_set in self.validation_sets:
            val_data.append(
                DataLoader(
                    dataset=validation_set,
                    batch_size=self.hparams.batch_size,
                    collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                    num_workers=0,
                )
            )
        return val_data

    def train_dataloader(self) -> DataLoader:
        """Method that loads the train dataloader. Can be called every epoch to load a
        different trainset if `reload_dataloaders_every_n_epochs=1` in Lightning
        Trainer.
        """
        data_path = self.hparams.train_data[self.current_epoch % len(self.hparams.train_data)]
        train_dataset = self.read_training_data(data_path)

        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=0,
        )

    def init_losses(self) -> None:
        """Initializes Loss functions to be used."""
        self.sentloss = nn.CrossEntropyLoss()
        # self.sentloss = nn.MSELoss()
        if self.word_level:
            if self.hparams.cross_entropy_weights:
                assert len(self.hparams.cross_entropy_weights) == self.num_classes
                loss_weights = torch.tensor(self.hparams.cross_entropy_weights)
            else:
                loss_weights = None

            self.wordloss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1, weight=loss_weights)


def load_qe_model_from_checkpoint(
    checkpoint_path: str,
    reload_hparams: bool = False,
    strict: bool = False,
    freeze_encoder: bool = True,
    final_activation: str = "tanh",
) -> UnifiedMetric:
    """
    This code is copied from the load_from_checkpoint function imported
    from the comet repo - the difference is that the class is hard-coded
    to be CEDModel
    """
    checkpoint_path = Path(checkpoint_path)
    parent_folder = checkpoint_path.parents[1]
    hparams_file = parent_folder / "hparams.yaml"

    if hparams_file.is_file():
        model = CEDModel.load_from_checkpoint(
            checkpoint_path,
            load_pretrained_weights=False,
            hparams_file=hparams_file if reload_hparams else None,
            map_location=torch.device("cpu"),
            strict=strict,
            keep_embeddings_frozen=freeze_encoder,
            final_activation=final_activation,
            batch_size=16,
        )
        return model


def score_data(row):
    if row["error"] == "NOT":
        return 1
    else:
        return 0


def get_ced_train_dev_data(
    return_paths: bool = True, language_pairs: list = LI_LANGUAGE_PAIRS, data_dir: str = DATA_DIR
):
    """
    Saves CED training and dev data in CSV format and returns lists of file paths
    """
    train_paths = []
    dev_paths = []

    train_data = []
    dev_data = []

    for lp in language_pairs:
        # load train data
        path_data = os.path.join(data_dir, f"{lp}_majority_train.tsv")
        df_train_data = pd.read_csv(
            path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"]
        )

        df_train_data["score"] = df_train_data.apply(score_data, axis=1).astype("int32")
        # NOTE: LIMITING TRAINING DATA TO 1000 RECORDS FOR TESTING ONLY
        df_train_data = df_train_data[:100]
        # Save to csv format
        path_train_data = os.path.join(data_dir, f"{lp}_majority_train.csv")
        df_train_data[["src", "mt", "score"]].to_csv(path_train_data)

        train_paths.append(path_train_data)
        train_data.append(df_train_data)

        # load dev data
        path_data = os.path.join(data_dir, f"{lp}_majority_dev.tsv")
        df_dev_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])
        # NOTE: LIMITING DEV DATA TO 20 RECORDS FOR TESTING ONLY
        df_dev_data = df_train_data[:20]
        df_dev_data["score"] = df_dev_data.apply(score_data, axis=1).astype("int32")
        # Save to csv format
        path_dev_data = os.path.join(data_dir, f"{lp}_majority_dev.csv")
        df_dev_data[["src", "mt", "score"]].to_csv(path_dev_data)

        dev_paths.append(path_dev_data)
        dev_data.append(df_dev_data)

    if return_paths:
        return train_paths, dev_paths
    else:
        return train_data, dev_data


def evaluate_model(lp: str, model: UnifiedMetric, out_dir: str, data_dir: str = DATA_DIR):
    """
    Evaluate the model using the test data for one language pair and
    save the results to the out_dir
    """
    # load test data
    path_data = os.path.join(data_dir, f"{lp}_majority_test_blind.tsv")
    df_test_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt"])

    test_data = df_test_data.to_dict("records")

    # predict
    model_output = model.predict(test_data, batch_size=8, gpus=0)

    # save output
    out_file_name = os.path.join(out_dir, f"{lp}_cometkiwi.csv")
    df_results = pd.DataFrame({"idx": df_test_data["idx"], "comet_score": model_output.scores})
    df_results.to_csv(out_file_name, index=False)


def make_output_folder(folder_name: str, out_dir: str = OUT_DIR):
    new_dir = os.path.join(out_dir, folder_name)
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def train_comet(model: UnifiedMetric, lp_id: int, experiment_name: str):
    """
    Trains the given model using the processes developed in the comet
    code base
    """

    train_paths, dev_paths = get_ced_train_dev_data()
    # for now just take the first language pair (en-cs)
    path_train_data = train_paths[lp_id]
    path_dev_data = dev_paths[lp_id]
    # Set paths for training and val data
    model.hparams.train_data = [path_train_data]
    model.hparams.validation_data = [path_dev_data]
    model.hparams.batch_size = 8

    wandb_logger = WandbLogger(
        entity="turing-arc",
        project="MTQE",
        name=experiment_name,
        mode="online",
        log_model=False,
    )

    # create new trainer object
    trainer = Trainer(max_epochs=4, accelerator="cpu", devices=1, logger=wandb_logger)

    trainer.fit(model)

    wandb.log({"test_metric": 0.5})
    wandb.config["test_config"] = "abcd"

    wandb_logger.experiment.config.update({"test_config_2": "efgh"})

    wandb.finish()

    return model


def train_ced_model_reg(language_pairs: list = LI_LANGUAGE_PAIRS, freeze_encoder: bool = True):
    """
    This approach uses a new class of CEDModel with the methods that caused
    an error (specific to my Mac?) overwritten. This runs slowly the first time when
    something(?) downloads.
    Getting an error when I run this on mps (my gpu) - it doesn't like float64 data types,
    and I can't find a way to fix it.
    """

    for idx, lp in enumerate(language_pairs):
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_qe_model_from_checkpoint(model_path, freeze_encoder)

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H:%M:%S")
        model = train_comet(model, idx, "test_experiment_reg_" + lp + "_" + now_str)

        # Create output folder specific to this training approach
        out_dir = make_output_folder("trained_model_regression")
        evaluate_model(lp, model, out_dir)

        break


def train_ced_model_class(language_pairs: list = LI_LANGUAGE_PAIRS, freeze_encoder: bool = True):
    """
    This is similar to train_ced_model_reg, but is attempting to use COMET-KIWI for classification
    rather than regression
    """

    for idx, lp in enumerate(language_pairs):
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_qe_model_from_checkpoint(model_path, freeze_encoder=freeze_encoder, final_activation="sigmoid")

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H:%M:%S")
        model = train_comet(model, idx, "test_experiment_cls_" + lp + "_" + now_str)

        # Create output folder specific to this training approach
        out_dir = make_output_folder("trained_model_classification")
        evaluate_model(lp, model, out_dir)

        break


if __name__ == "__main__":
    train_ced_model_class()
