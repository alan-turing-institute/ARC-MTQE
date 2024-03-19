from pathlib import Path
from typing import List, Optional, Union

import torch
from comet.models import UnifiedMetric
from torch import nn
from torch.utils.data import DataLoader, RandomSampler


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
