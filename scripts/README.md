# Scripts

Scripts to train models, generate predictions and evaluate performance.

## Table of contents

- [Train models](#train-models)
- [Predictions](#predictions)
- [Evaluation](#evaluation)

## Train models

### Config files

YAML config files are used to control the parameters for training a model, such as the model hyperparameters and parameters for saving model checkpoints. Examples of these are found in the [configs/](../configs/) directory. Each config file will contain a `group` of experiments with each experiment in the group reperesenting a different language pair (or group of language pairs in the multilingual setting).

Some notes on the different sections of the config files can be found [here](../notes/configs.md).

### Train a single model

Models can be trained using the `train_ced.py` script. Either the COMETKiwi-22 model weights are used a starting point, or model weights saved from a previous run (in the `checkpoints` folder which is created when training models) - this is controlled in the config file.

The script arguments are:
- experiment group (this is the name of the config file, without the `.yaml` extension)
- experiment name (which must match the one listed in the config file)
- the initial random seed (which, again, must match one listed in the config file).

For example:
```bash
poetry run python train_ced.py -g train_monolingual_auth_data -e en_cs -s 42
```

See the [notes/](../notes/models.md) directory for an overview of available training strategies and the corresponding config files.

### Generate slurm scripts

Slurm scripts for training models can be generated per experiment group (i.e., per config file) using the `generate_train_scripts.py` script. For this all that is needed is the experiment group name (i.e., the config file name). For example:

```bash
poetry run python generate_train_scripts.py -g train_monolingual_auth_data
```

This will generate one slurm script per model to train. That is, one slurm script per experiment and random seed listed in the config file. These will be saved into the `scripts/slurm_scripts/<experiment_group_name>` folder.

## Predictions

### Overview

The predictions from the trained models and LLM prompts are stored in the [ARC-MTQE/predictions/ced](../predictions/ced_data/) directory with a different folder for each experiment group. In the case of the trained models the experiment group is the name of the config file used to run the experiment and for the LLM prompts it is either `prompt_basic`, `prompt_GEMBA` or `wmt21_annotator`.

### Baseline

To make baseline COMETKiwi-22 predictions for the  CED test and dev data:

```bash
poetry run python predict_ced.py -g baseline -p cometkiwi_22 -d test -l all
poetry run python predict_ced.py -g baseline -p cometkiwi_22 -d dev -l all
```

### Trained models

Once a model has been trained, to make predictions use the `predict_ced.py` script.

The script arguments are:
- the experiment group (the name of the config file that was used to train the model)
- the experiment name (matching one in the config file)
- the initial random seed (again, matching one in the config file)
- the path of the checkpoint containing the weights of the model
- the data split to make predictions for (either `dev` or `test`)
- the language pair to make predictions for (either `en-cs`, `en-de`, `en-ja`, `en-zh` or `all`)

For example:

```bash
poetry run python predict_ced.py -g train_monolingual_auth_data -e en_cs -s 42 -p checkpoints/train_monolingual_auth_data__en-cs__42__20240416_174404/epoch=0-step=468.ckpt -d test -lp en-cs
```

### LLM prompts

To use the OpenAI API to make critical error predictions run the following script.

The script arguments are:
- prompt (`basic`, `GEMBA` or `wmt21_annotator`)
- GPT model to use (e.g., `gpt-3.5-turbo` or `gpt-4-turbo`)
- how many data rows to make predictions for (test data has 1000 rows)
- the language pair to make predictions for (e.g., `en-cs` but can also be `all`)
- the data split to make predictions for (`train`, `dev` or `test`)

For example:

```bash
poetry run python llm_ced.py -n 1000 -p GEMBA -l all -d test -m gpt-4-turbo
```

## Evaluation

### Baseline

Once predictions have been made for both the test and dev data, run:

```bash
poetry run python baseline_eval.py
```

### Trained models and LLMs

The predictions from the trained models and the LLM prompts can be evaluated using the `eval_ced.py` script. The only parameter that needs to be passed is the name of the experiment group. The script will then evaluate all the predictions that have been made for that experiment group. In the case of the trained models the experiment group is the name of the config file used to run the experiment and for the LLM prompts it is either `prompt_basic`, `prompt_GEMBA` or `wmt21_annotator`. For example:

```bash
poetry run python eval_ced.py -g train_monolingual_auth_data
```

Each model is evaluated using three thresholds:
- The default threshold of 0.5
- An 'extreme' threshold of 0.1
- And the 'best' threshold which is calculated as the threshold that achieves the highest MCC on the dev data

All the results for the given experiment group will be stored in the file with the suffix `_results.csv`. The files with the suffix `_max_results.csv`, `_mean_results.csv`, `_median_results.csv` and `_min_results.csv` will respectively show the maximum, mean, median and minimum MCC (and the corresponding metrics such as precision and recall) achieved over all the random seeds that were run. The file with the suffix `_ensemble_results.csv` will show the MCC values when majority voting takes place over all the random seeds.

### Plots

The notebook `metrics.ipynb` provides some code to plot confusion matrices and precision-recall curves for the predictions. It is necessary to have run the evaluation script before using this notebook as the evaluation script identifies the random seed that achieved, for example, the median MCC value which can then be used to make the plots.

### Latex tables of results

The notebook `create_tables.ipynb` provides some code to plot a metric or metrics (such as MCC or precision and recall) in latex tables.
