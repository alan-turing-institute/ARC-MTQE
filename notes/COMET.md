## Model

The below are notes on the hyperparameters listed in the `hparams.yaml` file that is downloaded from [HuggingFace with the COMETKiwi DA 2022 model](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).

COMETKiwi is an instance of the [`UnifiedMetric`](https://github.com/Unbabel/COMET/blob/master/comet/models/multitask/unified_metric.py) [COMET](https://github.com/Unbabel/COMET/tree/master) model class, which is an instance of a Pytorch Lightning module.

As indicated in the notes, some of the parameters are harcoded in the model implementation and so get ignored.

|Parameter|Default|Notes|
|---|---|---|
|activations|Tanh|Activation function used inside the regression head (not the `final_activation`).|
|batch_size|4||
|class_identifier|unified_metric|We override this when creating our model.|
|dropout|0.1|Dropout in the regression head.|
|encoder_learning_rate|1.0e-06|Learning rate used to fine-tune the encoder model.|
|encoder_model|XLM-RoBERTa||
|final_activation|null|The activation function used in the last layer of the regression head. |
|hidden_sizes|3072, 1024|Size of hidden layers used in the regression head.|
|input_segments|mt, src|This is the names of the two model inputs.|
|keep_embeddings_frozen|true|If `True` then keeps the encoder frozen during training.|
|layer|mix|Encoder layer to be used for regression ('mix' for pooling info from all layers). Although this appears in the hparams file for the Huggingface model, I don't think this is actually being used. In the code it looks like `sent_layer` is being used to set the `layer` argument.|
|layer_norm|false|Apply layer normalization to encoder.|
|layer_transformation|sparsemax|Transformation applied when pooling info from all layers of the encoder. This [issue](https://github.com/Unbabel/COMET/issues/195) suggests this doesn't actually get applied. |
|layerwise_decay|0.95|Learning rate % decay from top-to-bottom encoder layers.|
|loss|mse|This isn't actually used in the UnifiedMetric, it is hardcoded within the class.  |
|loss_lambda|0.65|The weight assigned to the word-level loss compared to sentence-level loss (if doing word-level training). |
|nr_frozen_epochs|0.3|Number of epochs OR % of epoch that the encoder is frozen before unfreezing it. If the value is greater than one, then the encoder is frozen for that number of epochs. If the value is between 0 and 1, then the encoder is frozen for that percentage of the first epoch. This [issue](https://github.com/Unbabel/COMET/issues/158) reports worse performance when fine-tuning the encoder is left until later|
|optimizer|AdamW||
|pool|avg|When a `UnifiedMetric` object is created this isn't a parameter that can be set on initialisation. There is a default value in the base `CometMetric` class, which is `avg`, but this is only used in methods that the `UnifiedMetric` ignores. Instead, they [hardcode the sentence embedding to be the `CLS` token](https://github.com/Unbabel/COMET/blob/74ef71547f3f411e1403368101a035a22502f72a/comet/models/multitask/unified_metric.py#L473).|
|pretrained_model|microsoft/infoxlm-large|The pre-trained model (i.e. checkpoint) to use from Huggingface. Won't actually be applied if set `load_pretrained_weights` to `False`.|
|sent_layer|mix| Which encoder layer to use as input for the sentence level task (default `mix` indicates to pool across layers). |
|train_data||List of paths to training data files (in csv format)|
|validation_data||List of paths to validation data files|
|word_layer|24| See `sent_layer`.|
|word_level_training|false|Determines whether the model will do word-level training as well as sentence-level training.|
|word_weights|0.15, 0.85|Can't actually find where this is used, doesn't seem to be an argument that the `UnifiedMetric` class is expecting.|

Other `UnifiedMetric` parameters:

|Parameter|Default|Notes|
|---|---|---|
|load_pretrained_weights||If set to False it avoids loading the weights of the pretrained model (e.g. XLM-R) before it loads the COMET checkpoint. Presumably this is always the desired behaviour. |

## Training

The COMET repository also contains a Pytorch Lightning [training config](https://github.com/Unbabel/COMET/blob/master/configs/trainer.yaml).

Values for most of the parameters are just the PyTorch Lightning Trainer defaults. Specifically: benchmark, check_val_every_n_epoch, fast_dev_run, max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches, limit_predict_batches, overfit_batches, val_check_interval, log_every_n_steps, enable_progress_bar, enable_model_summary, gradient_clip_algorithm, benchmark, inference_mode, use_distributed_sampler, profiler, detect_anomaly, barebones, sync_batchnorm, reload_dataloaders_every_n_epochs.
