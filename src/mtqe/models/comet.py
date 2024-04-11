from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from comet.models import UnifiedMetric
from comet.models.utils import Prediction, Target
from comet.modules import FeedForward
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset, WeightedRandomSampler

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
    error_weight:float
        If set to a value greater than 1, then it is the weight applied to
        all samples classed as a critical error. All samples that are not a
        critical error will always have a weight of 1.
    out_dim:int
        The number of outputs in the model
    random_weights:bool
        A boolean value that determines whether the weights of the feed forward
        head network are randomly initialised (`True`) or whether the default
        weights from the checkpoint are used (`False`)
    calc_threshold:bool
        A boolean value that determines whether the value for setting the threshold
        for determining the class is calculated using the training data (`True`)
        or whether it is fixed at 0.5 (`False`) - will only take effect if the
        loss function is binary cross entropy.
    train_subset_size:int
        The size of the training subset used for validation (and be extension also
        used to determine the threshold if `calc_threshol` is `True`)

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
        out_dim=1,
        random_weights=False,
        calc_threshold=False,
        train_subset_size=1000,
        train_subset_replace=True,
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
        # Check the validity of some combinations of parameters
        self._check_param_combinations()

    def _check_param_combinations(self) -> None:
        """
        Method to check that valid combinations of hyperparameters have been provided.
        The code may not function or may not have been tested on other combinations
        NOTE: These checks might not be exhaustive and may want to add others
        """
        assert self.hparams.loss in ["cross_entropy", "binary_cross_entropy_with_logits"], (
            "Unexpected loss function, expecting `cross_entropy`, or "
            + "`binary_cross_entropy_with_logits` and got: "
            + self.hparams.loss
        )
        if self.hparams.loss == "cross_entropy":
            assert self.hparams.out_dim > 1, "Cross entropy loss must have at least two class outputs"
            assert (
                self.hparams.final_activation is None
            ), "Final activation for cross entropy loss not valid - expecting `None`"
        if self.hparams.loss == "binary_cross_entropy_with_logits":
            assert self.hparams.out_dim == 1, (
                "Only one sentence class expected for binary cross entropy, " + self.hparams.out_dim + " provided"
            )
            assert (
                self.hparams.final_activation is None
            ), "Final activation for cross entropy loss not valid - expecting `None`"

    def update_estimator(self) -> None:
        """
        Method that makes changes to the feed-forward head of the model if
        prescribed by the hyperparameters. The changes that can be
        made are:
         - Randomly initialising the weights, this is controlled by hparam `random_weights`
         - Updating the number of output nodes (if the loss function
        is cross entropy), this is controlled by the hparam `out_dim`
        """
        if self.hparams.random_weights:
            self.estimator = FeedForward(
                in_dim=self.encoder.output_units,
                hidden_sizes=self.hparams.hidden_sizes,
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=self.hparams.final_activation,
            )
        if self.hparams.out_dim > 1:
            final_layer = nn.Linear(self.hparams.hidden_sizes[-1], self.hparams.out_dim)
            self.estimator.ff = nn.Sequential(*self.estimator.ff[:-1], final_layer)

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "fit"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        This method tokenizes input data and prepares targets for training.
        This is overriden from the COMET code to amend the format of the
        scores and targets when there are multiple classes (in the case
        of cross entropy loss function)

        Parameters
        ----------
        sample: List[Dict[str, Union[str, float]]]
            Mini-batch
        stage: str
            Model stage ('train' or 'predict'). Defaults to "fit".

        Returns
        -------
        model_inputs["inputs"]: Tuple[Dict[str, torch.Tensor]]
            Tokenised input sequence
        targets: Dict[str, torch.Tensor]
            Dictionary containing the target values - only returned if the stage
            is not `predict`
        """
        inputs = {k: [d[k] for d in sample] for k in sample[0]}
        input_sequences = [
            self.encoder.prepare_sample(inputs["mt"], self.word_level, None),
        ]

        src_input, ref_input = False, False
        if ("src" in inputs) and ("src" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["src"]))
            src_input = True

        if ("ref" in inputs) and ("ref" in self.hparams.input_segments):
            input_sequences.append(self.encoder.prepare_sample(inputs["ref"]))
            ref_input = True

        unified_input = src_input and ref_input
        model_inputs = self.concat_inputs(input_sequences, unified_input)
        if stage == "predict":
            return model_inputs["inputs"]

        if self.hparams.out_dim > 1:
            scores = inputs["score"]
            targets = Target(score=torch.tensor(scores, dtype=torch.long))
        else:
            scores = [float(s) for s in inputs["score"]]
            targets = Target(score=torch.tensor(scores, dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        if self.word_level:
            # Labels will be the same accross all inputs because we are only
            # doing sequence tagging on the MT. We will only use the mask corresponding
            # to the MT segment.
            seq_len = model_inputs["mt_length"].max()
            targets["mt_length"] = model_inputs["mt_length"]
            targets["labels"] = model_inputs["inputs"][0]["label_ids"][:, :seq_len]

        return model_inputs["inputs"], targets

    def setup(self, stage: str) -> None:
        """
        Data preparation function called before training by Lightning.
        Overriden from COMET code to allow for configuring the size of the
        training subset used for validation. Controlled with hparams
        `train_subset_size` and `train_subset_replace`

        Parameters
        ----------
        stage: str
            either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in (None, "fit"):
            train_dataset = self.read_training_data(self.hparams.train_data[0])

            self.validation_sets = [self.read_validation_data(d) for d in self.hparams.validation_data]

            self.first_epoch_total_steps = len(train_dataset) // (
                self.hparams.batch_size * max(1, self.trainer.num_devices)
            )
            # Always validate the model with part of training.
            train_subset = np.random.choice(
                a=len(train_dataset), size=self.hparams.train_subset_size, replace=self.hparams.train_subset_replace
            )
            self.train_subset = Subset(train_dataset, train_subset)

    def val_dataloader(self) -> DataLoader:
        """
        Method that loads the validation sets.
        NOTE: this is overriden from the parent class because of an error when running
        locally on a Macbook. The num_workers variables were changed from 2 to 0.
        NOTE: A subset of training data is loaded for evaluation but is not mixed into
        the validation dataset and the score on the train subset is recorded separately
        to the score on the validation dataset

        Returns
        -------
        val_data: torch.utils.data.DataLoader
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
        Trainer. This is overriden from the COMET code to provide some additional
        functionality:
         - Records with text over a given length can be excluded from the dataset, this
           is controlled with the hparam `exclude_outliers`
         - Allows for the minority class to be oversampled in the dataset, this is
           controlled with the hparam `oversample_minority`

        NOTE: this also overrides from the parent class because of an error when running
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
        This overrides the method in the COMET code to set the loss function for classification
        Also determins the reduction of the loss function based on whether class weights are
        applied. If class weights are applied then the reduction is carried out when computing
        the loss and the rediction here is set to `none`. Otherwise a `mean` reduction is used.
        """
        if self.hparams.error_weight > 1:
            # The reduction of `mean` will be calculated in method `compute_loss` using the weights
            # so set to `none` here
            reduction = "none"
        else:
            reduction = "mean"

        if self.hparams.loss == "cross_entropy":
            self.sentloss = nn.CrossEntropyLoss(reduction=reduction)
        elif self.hparams.loss == "binary_cross_entropy_with_logits":
            self.sentloss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            raise Exception(
                "Expecting loss function of `cross_entropy`, " + "or `binary_cross_entropy_with_logits`, instead got:",
                self.hparams.loss,
            )

    def init_metrics(self) -> None:
        """
        Initializes training and validation classification metrics
        This overrides the method in the COMET code to use the ClassificationMetrics class instead of
        RegressionMetrics
        NOTE: the names of the objects that store the classification metrics have not been overriden
        so still read `train_corr` and `val_corr` even though they are not just representing
        correlations
        """
        # Set params used for calculating metrics
        if self.hparams.loss == "binary_cross_entropy_with_logits":
            binary = True
            num_classes = 2
        else:
            binary = False
            num_classes = self.hparams.out_dim
        #
        self.train_corr = ClassificationMetrics(
            prefix="train", binary=binary, num_classes=num_classes, calc_threshold=self.hparams.calc_threshold
        )
        self.val_corr = nn.ModuleList(
            [
                ClassificationMetrics(prefix=d, binary=binary, num_classes=num_classes)
                for d in self.hparams.validation_data
            ]
        )

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """
        Receives model batch prediction and respective targets and computes a loss value.
        This overrides the method in the COMET code to apply class weights if the hparam
        `error_weight` is set to a value greater than 1.
        NOTE: the word-level loss function is not included here at all

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
            # The weight for samples with a critical error is the value `error_weight`
            # As samples without a critical error have a score of 1 we need to first get
            # a tensor with a 1 for every critical error `(1 - target.score)`. Then multiply
            # by the `(error_weight - 1)` and add 1 to everything. This will give a weight of
            # 1 to all samples without a critical error, and weight of `error_weight` to
            # those with a critical error.
            weights = (1 - target.score) * (self.hparams.error_weight - 1) + 1
            # Multiple `weights` by `sentence_loss` and take the mean value
            sentence_loss = torch.mean(weights * sentence_loss)
        return sentence_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward function.
        Overriden from the COMET code to change the dimensions of the output
        if `out_dim` greater than 1 (i.e., if cross-entropy is
        being employed).

        Parameters
        ----------
        input_ids: torch.Tensor
            Input sequence.
        attention_mask: torch.Tensor
            Attention mask.
        token_type_ids: Optional[torch.Tensor]
            Token type ids for BERT-like models. Defaults to None.

        Returns
        -------
        Dict[str, torch.Tensor]
            Sentence scores and word-level logits (if word_level_training = True)

        Raises
        ------
        Exception
            Invalid model word/sent layer if self.{word/sent}_layer are not valid encoder model layers.
        """
        encoder_out = self.encoder(input_ids, attention_mask, token_type_ids=token_type_ids)

        # Word embeddings used for the word-level classification task
        if self.word_level:
            if isinstance(self.hparams.word_layer, int) and 0 <= self.hparams.word_layer < self.encoder.num_layers:
                wordemb = encoder_out["all_layers"][self.hparams.word_layer]
            else:
                raise Exception("Invalid model word layer {}.".format(self.hparams.word_layer))

        # embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(encoder_out["all_layers"], attention_mask)
        elif isinstance(self.hparams.sent_layer, int) and 0 <= self.hparams.sent_layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.hparams.sent_layer]
        else:
            raise Exception("Invalid model sent layer {}.".format(self.hparams.word_layer))
        sentemb = embeddings[:, 0, :]  # We take the CLS token as sentence-embedding

        if self.word_level:
            sentence_output = self.estimator(sentemb)
            word_output = self.hidden2tag(wordemb)
            return Prediction(score=sentence_output.view(-1), logits=word_output)

        if self.hparams.out_dim > 1:
            return Prediction(score=self.estimator(sentemb).view(-1, self.hparams.out_dim))
        else:
            return Prediction(score=self.estimator(sentemb).view(-1))

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """
        Computes and logs metrics
        This overrides the COMET code to log additional metrics
        """
        train_dict, train_max_metric_vals, train_at_max_mcc_vals, threshold = self.train_corr.compute()
        self.log_dict(train_dict, prog_bar=False, sync_dist=True)
        self.log_dict(train_max_metric_vals, prog_bar=False, sync_dist=True)
        self.log_dict(train_at_max_mcc_vals, prog_bar=False, sync_dist=True)
        self.train_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False, sync_dist=True)
            self.train_mcc.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            corr_metrics, val_max_metric_vals, val_at_max_mcc_vals, _ = self.val_corr[i].compute(threshold=threshold)
            self.log_dict(val_max_metric_vals, prog_bar=False, sync_dist=True)
            self.log_dict(val_at_max_mcc_vals, prog_bar=False, sync_dist=True)
            self.val_corr[i].reset()
            if self.word_level:
                cls_metric = self.val_mcc[i].compute()
                self.val_mcc[i].reset()
                results = {**corr_metrics, **cls_metric}
            else:
                results = corr_metrics

            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False, sync_dist=True)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()},
            prog_bar=True,
            sync_dist=True,
        )


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
        model.update_estimator()
        return model
