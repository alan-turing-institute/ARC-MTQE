from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from comet.models import UnifiedMetric
from comet.models.utils import Prediction, Target
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from mtqe.models.metrics import ClassificationMetrics


class CEDModel(UnifiedMetric):
    """
    New class created that inherits from the UnifiedMetric class from COMET
    This class overrides the val_dataloader and train_dataloader functions
    to overcome an error obtianed when running the first training approach
    Three new hparams have been added: `oversample_minority`, `exclude_outliers`
    and `error_weight`

    Attributes
    ----------
    TO DO

    oversample_minority:bool
        Boolean value indicating whether or not to oversample the minority class
        in the training data. If this is set to True, then it is expected that
        `reload_dataloaders_every_n_epochs` is set to 1 in the Trainer config
    exclude_outliers:int
        If set to a value greater than zero, then any records where the target
        (machine translated) text is longer than this value is excluded from
        the training dataset.
    error_weight:int
        If set to a value greater than 1, then it is the weight applied to
        all samples classed as a critical error. All samples that are not a
        critical error will always have a weight of 1.

    Methods
    -------
    TO DO
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
        load_pretrained_weights: bool = False,
        oversample_minority=False,
        exclude_outliers=0,
        error_weight=1,
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
            load_pretrained_weights=load_pretrained_weights,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Method that loads the validation sets.
        NOTE: this is overriden from the parent class because of an error when running
        locally on a Macbook. The num_workers variables were changed from 2 to 0.
        NOTE: A subset of training data is loaded for evaluation but is not mixed into
        the validation dataset and the score on the train subset is recorded separately
        to the score on the validation dataset
        NOTE: While removing the train subset would save some computational cost, it
        would require additional changes elsewhere in the COMET code, which would
        require more testing.

        Returns
        -------
        torch.utils.data.DataLoader
        """
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
        """
        Method that loads the train dataloader. Can be called every epoch to load a
        different trainset if `reload_dataloaders_every_n_epochs=1` in Lightning
        Trainer.
        NOTE: this is overriden from the parent class because of an error when running
        locally on a Macbook. The num_workers variables were changed from 2 to 0.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        data_path = self.hparams.train_data[self.current_epoch % len(self.hparams.train_data)]
        train_dataset = self.read_training_data(data_path)

        # If `exclude_outliers` is a value greater than zero, then any records where the
        # machine translation is longer than this value is excluded from the training
        # dataset
        if self.hparams.exclude_outliers > 0:
            df_train_dataset = pd.DataFrame(train_dataset)
            df_train_dataset = df_train_dataset[df_train_dataset["mt"].str.len() < self.hparams.exclude_outliers]
            train_dataset = df_train_dataset.to_dict("records")

        # If `oversample_minority` is set to True, then a WeightedRandomSampler is used
        # to oversample the minority class when constructing the training dataset.
        # Having replacement = True means that the size of the training dataset can be
        # kept at the same size as the original training dataset, but it does mean that
        # samples in the minority class will be repeated in the training data. Also note
        # there is no guarantee how many epochs will be required for the model to see all
        # of the samples in the majority class.
        if self.hparams.oversample_minority:
            df_train_dataset = pd.DataFrame(train_dataset)
            class_counts = df_train_dataset.score.value_counts()
            sample_weights = [1 / class_counts[i] for i in df_train_dataset.score.values]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=df_train_dataset.shape[0], replacement=True
            )
        else:
            sampler = RandomSampler(train_dataset)

        return DataLoader(
            dataset=train_dataset,
            sampler=sampler,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=0,
        )

    def init_losses(self) -> None:
        """
        Initializes Loss functions to be used.
        This overrides the method in the UnifiedMetric class to set the loss function to binary cross entropy
        """
        # if self.hparams.error_weight > 0:
        #     loss_weights = torch.tensor(self.hparams.cross_entropy_weights)
        # else:
        #     loss_weights = None
        if self.hparams.error_weight > 1:
            # The reduction of `mean` will be calculated in method `compute_loss` using the weights
            # so set to `none` here
            self.sentloss = nn.BCELoss(reduction="none")
        else:
            self.sentloss = nn.BCELoss(reduction="mean")

    def init_metrics(self) -> None:
        """
        Initializes training and validation classification metrics
        This overrides the method in UnifiedMetric class to use the ClassificationMetrics class instead of
        RegressionMetrics
        """
        self.train_corr = ClassificationMetrics(prefix="train")
        self.val_corr = nn.ModuleList([ClassificationMetrics(prefix=d) for d in self.hparams.validation_data])

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """
        Receives model batch prediction and respective targets and computes
        a loss value.
        This overrides the method in UnifiedMetric class to apply class weights

        Parameters
        ----------
        prediction: Prediction
            Batch prediction
        target: Target
            Batch targets

        Returns
        -------
        torch.Tensor
        """
        sentence_loss = self.sentloss(prediction.score, target.score)
        if self.hparams.error_weight > 1:
            # The weight for samples that don't contain a critical error will always be 1
            # The weight for samples with a critical error is contained in `error_weight`
            # As samples without a critical error have a score of 1 we need to first get
            # a tensor with a 1 for every critical error `(1 - target.score)`. Then multiply
            # by the error weight - 1 and add 1 to everything. This will give a weight of
            # 1 to all samples without a critical error, and weight of `error_weight` to
            # those with a critical error.
            weights = (1 - target.score) * (self.hparams.error_weight - 1) + 1
            # Multiple `weights` by `sentence_loss` and take the mean value
            sentence_loss = torch.mean(weights * sentence_loss)
        return sentence_loss


def load_qe_model_from_checkpoint(
    checkpoint_path: str,
    paths_train_data: list,
    paths_dev_data: list,
    strict: bool = False,
    reload_hparams=False,
    **kwargs
) -> UnifiedMetric:
    """
    This code is copied from the load_from_checkpoint function imported
    from the comet repo - the difference is that the class is hard-coded
    to be CEDModel and the device is set to cuda, if available.
    """
    checkpoint_path = Path(checkpoint_path)
    parent_folder = checkpoint_path.parents[1]
    hparams_file = parent_folder / "hparams.yaml"
    # would be better if this was set once (in train_ced.py) and passed
    # to functions when needed - currently also set in metrics.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if hparams_file.is_file():
        model = CEDModel.load_from_checkpoint(
            checkpoint_path,
            load_pretrained_weights=False,
            hparams_file=hparams_file if reload_hparams else None,
            map_location=torch.device(device),
            strict=strict,
            train_data=paths_train_data,
            validation_data=paths_dev_data,
            **kwargs
        )
        return model
