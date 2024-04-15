# COMET notes

This document contains notes on the [COMET](https://github.com/Unbabel/COMET/tree/master) codebase that the `CEDModel` is built on.

## Model

` COMETKiwi` (the model we finetune) is an instance of the [`UnifiedMetrics`](https://github.com/Unbabel/COMET/blob/master/comet/models/multitask/unified_metric.py) class, which in turn inherits from the base [`CometModel`](https://github.com/Unbabel/COMET/blob/master/comet/models/base.py) class, which is an instance of a Pytorch Lightning module.

There are parameters in the `UnifiedMetrics` class which are leftovers from the base class and not actually used. A number of parameters are also passed when the class is instantiated but then hardcoded to some value within the class itself with the input ignored.

 The below are notes on the hyperparameters listed in the `hparams.yaml` file that is downloaded from [HuggingFace with the COMETKiwi DA 2022 model](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) that are not otherwise described in the [`CEDModel`](../src/mtqe/models/comet.py) docstring.

|Parameter|Default|Notes|
|class_identifier|unified_metric|We override this when creating our model.|
|layer|mix|Encoder layer to be used for regression ('mix' for pooling info from all layers). Although this appears in the hparams file for the Huggingface model, I don't think this is actually being used. In the code it looks like `sent_layer` is being used to set the `layer` argument.|
|pool|avg|When a `UnifiedMetric` object is created this isn't a parameter that can be set on initialisation. There is a default value in the base `CometMetric` class, which is `avg`, but this is only used in methods that the `UnifiedMetric` ignores. Instead, they [hardcode the sentence embedding to be the `CLS` token](https://github.com/Unbabel/COMET/blob/74ef71547f3f411e1403368101a035a22502f72a/comet/models/multitask/unified_metric.py#L473).]
|word_weights|0.15, 0.85|Can't actually find where this is used, doesn't seem to be an argument that the `UnifiedMetric` class is expecting.|

## Training

 The COMET repository also contains a Pytorch Lightning [training config](https://github.com/Unbabel/COMET/blob/master/configs/trainer.yaml).

 Values for most of the parameters are just the PyTorch Lightning Trainer defaults. Specifically: benchmark, check_val_every_n_epoch, fast_dev_run, max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches, limit_predict_batches, overfit_batches, val_check_interval, log_every_n_steps, enable_progress_bar, enable_model_summary, gradient_clip_algorithm, benchmark, inference_mode, use_distributed_sampler, profiler, detect_anomaly, barebones, sync_batchnorm, reload_dataloaders_every_n_epochs.
