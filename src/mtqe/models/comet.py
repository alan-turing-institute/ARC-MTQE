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
    A binary critical error classifier that inherits from COMET's UnifiedMetric class.

    This class overrides the following methods:
    - prepare_sample
    - setup
    - val_dataloader
    - train_dataloader
    - init_losses
    - init_metrics
    - compute_loss
    - forward
    - on_validation_epoch_end

    Additionally, new methods have been added:
    - _check_param_combinations
    - update_estimator

    See UnifiedMetrics for description of all inherited methods:
    https://github.com/Unbabel/COMET/blob/master/comet/models/multitask/unified_metric.py

    Parameters
    ----------
    nr_frozen_epochs: Union[float, int]
        Number of epochs OR % of epoch that the encoder is frozen before unfreezing it. If
        the value is greater than one, then the encoder is frozen for that number of epochs.
        If the value is between 0 and 1, then the encoder is frozen for that percentage of
        the first epoch. If the value is 0 then it is always froezen. Defaults to 0.9.
    keep_embeddings_frozen: bool
        Keeps the embedding layer of the encoder frozen during training. If `nf_frozen_epochs`
        is greater than 0 then the encoder will be unfrozen during training while this allows
        for the embedding layer to always remain frozen. Defaults to `True`.
    optimizer: str
        Optimizer used during training. Defaults to 'AdamW'.
    warmup_steps: int
        Warmup steps for LR scheduler. Defaults to 0.
    encoder_learning_rate: float
        Learning rate used to fine-tune the encoder model. Defaults to 3.0e-06.
    learning_rate: float
        Learning rate used to fine-tune the top layers. Defaults to 3.0e-05.
    layerwise_decay: float
        Learning rate % decay from top-to-bottom encoder layers. Defaults to 0.95.
    encoder_model: str
        Encoder model architecture to be used. Defaults to 'XLM-RoBERTa'.
    pretrained_model: str
        Pretrained encoder model weights to load from Hugging Face. This won't be applied
        if set `load_pretrained_weights` to `False`. Defaults to 'microsoft/infoxlm-large'.
    sent_layer: Union[str, int]
        Which encoder layer to use as input for the sentence level task ('mix' for pooling
        info from all layers). Defaults to 'mix'.
    layer_transformation: str
        Transformation applied when pooling info from all layers of the encoder. Defaults
        to 'sparsemax'.
    layer_norm: bool
        Apply layer normalization. Defaults to `True`.
    word_layer: int
        Encoder layer to be used for word-level classification. Defaults to 24.
    loss: str
        This parameter isn't actually used in the UnifiedMetrics class. Also, it is
        hardcoded in the base COMET class to 'mse' irrespective of what is passed to it.
    dropout: float
        Dropout used in the top-layers. Defaults to 0.1.
    batch_size: int
         Batch size used during training. Defaults to 4.
    train_data: Optional[List[str]]
         List of paths to training data. Each file is loaded consecutively for each epoch.
         Expects csv files with "src", "mt" and "score" columns. Defaults to `None`.
    validation_data: Optional[List[str]] = None
        List of paths to validation data. Validation results are averaged across validation
        set. Expects csv files with "src", "mt" and "score" columns. Defaults to `None`.
    hidden_sizes: List[int]
        Size of hidden layers used in the classification head. Defaults to [3072, 1024].
    activations: str
        Activation function used within the classification head. Defaults to 'Tanh'.
    final_activation: Optional[str]
        Activation function used in the last layer of the regression head. For this model,
        the only accepted value is `None`. Defaults to `None`.
    input_segments: List[str]
        List with input segment names to be used. Defaults to ["mt", "src", "ref"]. Can be
        restricted to ["mt", "src"].
    word_level_training: bool
        If `True`, the model is trained with multitask (i.e., combined sentence and word level)
        objective. We have not tried this objective with this model. Defaults to `False`.
    loss_lambda: float
         Weight assigned to the word-level loss. Defaults to 0.65.
    load_pretrained_weights: bool
        If set to `False` it avoids loading the weights of the pretrained model (e.g.
        XLM-R) before it loads the COMET checkpoint. Unless training from scratch, this is
        always desirable. Defaults to `False`.
    oversample_minority: bool
        Whether or not to oversample the minority clas in the training data.
        If set to `True`, then it is expected that `reload_dataloaders_every_n_epochs`
        is set to 1 in the Trainer config. Defaults to `False`.
    exclude_outliers: int
        If set to a value greater than zero, then any records where the target
        (machine translated) text is longer than this value is excluded from
        the training dataset. Defaults to 0.
    error_weight: Union[float, int]
        The weight applied to all samples classed as a critical error. All samples
        that are not a critical error will always have a weight of 1. Defaults to 1.
    out_dim: int
        The number of outputs in the model. Defaults to 1.
    random_weights: bool
        Determines whether the weights of the classification head (feed forward network)
        are randomly initialised (`True`) or whether the default weights from the checkpoint
        are used (`False`). Defaults to `False`.
    calc_threshold: bool
        Indicates whether the threshold for binarising predictions is fixed at 0.5 (`False`)
        or whether it is determined using prediction performance on the training data (`True`).
        Will only take effect if the loss function is binary cross entropy. Defaults to `False`.
    train_subset_size: int
        The size of the training subset used for validation (and by extension used to determine
        the binarisation threshold if `calc_threshol` is set to `True`). Defaults to 1000.
    train_subset_replace: bool
        Determines whether the training subset is sampled with replacement or not.
        Defaults to `False`.
    """

    def __init__(
        self,
        # inherited UnifiedMetrics parameters
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
        # CEDModel specific parameters below
        oversample_minority: bool = False,
        exclude_outliers: int = 0,
        error_weight: Union[float, int] = 1,
        out_dim: int = 1,
        random_weights: bool = False,
        calc_threshold: bool = False,
        train_subset_size: int = 1000,
        train_subset_replace: bool = True,
    ):
        # The UnifiedMetric.__init__() method calls LightningModule.save_hyperparameters(),
        # which means all the provided arguments above are saved and accessible in self.hparams
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

    def _check_param_combinations(self):
        """
        Method to check that valid hyperparameters (and their combinations) have been provided.
        NOTE: These checks are not exhaustive.
        """
        assert self.hparams.final_activation is None, "Final activation not valid - expecting `None`"
        assert self.hparams.loss in ["cross_entropy", "binary_cross_entropy_with_logits"], (
            "Unexpected loss function, expecting `cross_entropy`, or "
            + "`binary_cross_entropy_with_logits` and got: "
            + self.hparams.loss
        )
        if self.hparams.loss == "cross_entropy":
            assert self.hparams.out_dim > 1, "Cross entropy loss must have at least two class outputs"
        if self.hparams.loss == "binary_cross_entropy_with_logits":
            assert self.hparams.out_dim == 1, (
                "Only one sentence class expected for binary cross entropy, " + self.hparams.out_dim + " provided"
            )
        assert self.hparams.out_dim <= 2, (
            "Not tested the implementation for greater than two outputs, there will also be some code that needs "
            + "updating for handling more than two outputs, such as setting error weights."
        )
        assert not (self.hparams.oversample_minority and self.hparams.error_weight > 1), (
            "Oversampling the minority class and setting weights for the error class are not expected to be used "
            + "simultaneously."
        )

    def update_estimator(self):
        """
        Update the feed-forward head of the model. The changes that can be made are:
         - Randomly initialising the weights, controlled by hparam `random_weights`
         - Updating number of output nodes (if the loss function is cross entropy),
           controlled by the hparam `out_dim`
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
        Updates the COMET method to amend format of the scores and targets
        when there are multiple classes (for cross entropy loss function).

        Parameters
        ----------
        sample: List[Dict[str, Union[str, float]]]
            Mini-batch.
        stage: str
            Model stage ('fit', 'train', 'validate' or 'predict').
            Defaults to 'fit'.

        Returns
        -------
        model_inputs["inputs"]: Tuple[Dict[str, torch.Tensor]]
            Tokenised input sequence.
        targets: Dict[str, torch.Tensor]
            Dictionary containing the target values - only returned if the stage
            is not 'predict'.
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

    def setup(self, stage: str):
        """
        Data preparation function called before training by Lightning.
        Overriden from COMET code to allow for configuring the size of the
        training subset used for validation. Controlled with hparams
        `train_subset_size` and `train_subset_replace`

        Parameters
        ----------
        stage: str
            Either 'fit', 'validate', 'test', or 'predict'.
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
        to the score on the validation dataset.

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

    def init_losses(self):
        """
        Initializes Loss functions to be used.
        This overrides the method in the COMET code to set the loss function for classification
        Also determins the reduction of the loss function based on whether class weights are
        applied. If class weights are applied then the reduction is carried out when computing
        the loss and the reduction here is set to `None`. Otherwise a `mean` reduction is used.
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

    def init_metrics(self):
        """
        Initializes training and validation classification metrics.
        This overrides the method in the COMET code to use the ClassificationMetrics class instead of
        RegressionMetrics.
        NOTE: the names of the objects that store the classification metrics have not been overriden
        so still read `train_corr` and `val_corr` even though they are not just representing
        correlations.
        """
        # Set params used for calculating metrics
        if self.hparams.loss == "binary_cross_entropy_with_logits":
            binary_loss = True
            activation_fn = torch.sigmoid
            activation_fn_args = {}
        elif self.hparams.loss == "cross_entropy":
            binary_loss = False
            activation_fn = torch.softmax
            activation_fn_args = {"dim": 1}
        else:
            raise NotImplementedError("Loss function not implemented:" + self.hparams.loss)

        self.train_corr = ClassificationMetrics(
            prefix="train",
            binary_loss=binary_loss,
            calc_threshold=self.hparams.calc_threshold,
            activation_fn=activation_fn,
            activation_fn_args=activation_fn_args,
        )
        # Validation datasets will never be used to calculate a binarisation threshold, so this is set to `False`
        self.val_corr = nn.ModuleList(
            [
                ClassificationMetrics(
                    prefix=d,
                    binary_loss=binary_loss,
                    calc_threshold=False,
                    activation_fn=activation_fn,
                    activation_fn_args=activation_fn_args,
                )
                for d in self.hparams.validation_data
            ]
        )

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """
        Receives model batch prediction and respective targets and computes a loss value.
        This overrides the method in the COMET code to apply class weights if the hparam
        `error_weight` is set to a value greater than 1.
        NOTE: the word-level loss function is not included here at all.
        NOTE: this would need updating if the code were extended to allow for more than
        two classes.

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
            # Multiply `weights` by `sentence_loss` and take the mean value
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
        if `out_dim` greater than 1 (i.e., if cross-entropy is being employed).

        Parameters
        ----------
        input_ids: torch.Tensor
            Input sequence.
        attention_mask: torch.Tensor
            Attention mask.
        token_type_ids: Optional[torch.Tensor]
            Token type ids for BERT-like models. Defaults to `None`.

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

    def on_validation_epoch_end(self, *args, **kwargs):
        """
        Computes and logs metrics.
        This overrides the COMET code to log additional metrics.
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
