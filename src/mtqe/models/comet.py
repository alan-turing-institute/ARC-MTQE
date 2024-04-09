from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    error_weight:float
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
        num_sentence_classes=1,
        random_weights=False,
        initializer_range=0.2,
        calc_threshold=False,
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

    def update_estimator(self):
        if self.hparams.num_sentence_classes > 1:
            assert self.hparams.final_activation is None
            assert self.hparams.loss == "cross_entropy"
            final_layer = nn.Linear(self.hparams.hidden_sizes[-1], self.hparams.num_sentence_classes)
            self.estimator.ff = nn.Sequential(*self.estimator.ff[:-1], final_layer)
        if self.hparams.random_weights:
            self.estimator.ff.apply(self._init_weights)
            print("hello")
        # if self.hparams.random_weights:
        #     for layer in self.estimator.ff:
        #         if isinstance(layer, nn.Linear):
        #             print(layer.weight[0])
        #             nn.init.uniform_(layer.weight)
        #             print(layer.weight[0])
        #             layer.bias.data.fill_()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def read_training_data(self, path: str) -> List[dict]:
        """Reads a csv file with training data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of training examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        data = self._read_data(df, columns)
        return data

    def read_validation_data(self, path: str) -> List[dict]:
        """Reads a csv file with validation data.

        Args:
            path (str): Path to the csv file to be loaded.

        Returns:
            List[dict]: Returns a list of validation examples.
        """
        df = pd.read_csv(path)
        # Deep copy input segments
        columns = self.hparams.input_segments[:]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
        data = self._read_data(df, columns)
        return data

    def _read_data(self, df: pd.DataFrame, columns: list) -> List[dict]:
        # Make sure everything except score is str type
        for col in columns:
            df[col] = df[col].astype(str)
        columns.append("score")
        df["score"] = df["score"].astype("float16")
        # if self.hparams.num_sentence_classes == 2:
        #     df["score"] = df.apply(update_score_two_cols, axis=1)
        # else:
        #     assert self.hparams.num_sentence_classes == 1, "Number of sentence classes should be 1 or 2"
        df = df[columns]
        return df.to_dict("records")

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "fit"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Tokenizes input data and prepares targets for training.

        Args:
            sample (List[Dict[str, Union[str, float]]]): Mini-batch
            stage (str, optional): Model stage ('train' or 'predict'). Defaults to "fit".

        Returns:
            Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: Model input
                and targets.
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

        if self.hparams.num_sentence_classes > 1:
            scores = [s for s in inputs["score"]]
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
        if self.hparams.error_weight > 1:
            # The reduction of `mean` will be calculated in method `compute_loss` using the weights
            # so set to `none` here
            reduction = "none"
        else:
            reduction = "mean"

        if self.hparams.loss == "cross_entropy":
            self.sentloss = nn.CrossEntropyLoss(reduction=reduction)
        elif self.hparams.loss == "binary_cross_entropy":
            self.sentloss = nn.BCELoss(reduction=reduction)
        elif self.hparams.loss == "binary_cross_entropy_with_logits":
            self.sentloss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            raise Exception(
                "Expecting loss function of 'cross_entropy' or 'binary_cross_entropy', instead got:", self.hparams.loss
            )

    def init_metrics(self) -> None:
        """
        Initializes training and validation classification metrics
        This overrides the method in UnifiedMetric class to use the ClassificationMetrics class instead of
        RegressionMetrics
        """
        if self.hparams.loss in ["binary_cross_entropy", "binary_cross_entropy_with_logits"]:
            binary = True
            num_classes = 2
        else:
            binary = False
            num_classes = self.hparams.num_sentence_classes
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
        """Forward function.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
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

        if self.hparams.num_sentence_classes > 1:
            return Prediction(score=self.estimator(sentemb).view(-1, self.hparams.num_sentence_classes))
        else:
            return Prediction(score=self.estimator(sentemb).view(-1))

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics - overriding COMET code"""
        train_dict, threshold = self.train_corr.compute()
        self.log_dict(train_dict, prog_bar=False, sync_dist=True)
        self.train_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False, sync_dist=True)
            self.train_mcc.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            corr_metrics, _ = self.val_corr[i].compute(threshold=threshold)
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
