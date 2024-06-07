# Scripts

## Train models

### Config files
YAML config files are used to control the parameters for training a model, such as the model hyperparameters and parameters for saving model checkpoints. Examples of these are found in the `configs` folder in the repo. Each config file will contain a 'group' of experiments with each experiment in the group reperesenting a different language pair (or group of language pairs in the multilingual setting).

Some notes on the different sections of the config files can be found [here](notes/configs.md).

### Train a single model

Models can be trained using the `train_ced.py` script. Either the COMETKiwi-22 model weights are used a starting point, or model weights saved from a previous run (in the `checkpoints` folder) - this is controlled in the config file.

The parameters to use when running the script to train a model are the experiment group (this is the name of the config file, without the `.yaml` extension), the experiment name (which must match one listed in the config file) and the initial random seed (which, again, must match one listed in the config file).

For example:
```bash
python scripts/train_ced.py -g train_monolingual_auth_data -e en_cs -s 42
```

### Generate slurm scripts

Slurm scripts for training models can be generated per experiment group (i.e., per config file) using the `generate_train_scripts.py` script. For this all that is needed is the experiment group name (i.e., the config file name). For example:

```bash
python scripts/generate_train_scripts.py -g train_monolingual_auth_data
```

This will generate one slurm script per model to be trained. That is, one slurm script per experiment and random seed listed in the config file. These will be saved into the folder `scripts/slurm_scripts/<experiment_group_name>`

## Make predictions

To make predictions for WMT 2023 DA test data using the COMET-QE 2020 and 2021 models and the COMETKiwi 2022 model:

```bash
make analyse_da
```

It is also possible to make predictions with the COMETKiwi-XL 2023 model but note that this can take couple of hours per language pair (there are 5 in the 2023 DA test set):

```bash
make analyse_da_xl
```

To make COMETKiwi-22 predictions for the  CED test and dev data:

```bash
make baseline_predict
```

To use the OpenAI API to make critical error predictions run the following script. The parameters passed to the script indicate, which prompt (`basic`, `GEMBA` or `wmt21_annotator`) and GPT model (e.g., `gpt-3.5-turbo` or `gpt-4-turbo`) to use and how many translations, which language pair (e.g., `en-cs` but can also be `all`) and which data split (`train`, `dev` or `test`) to make predictions for. For example:

```bash
poetry run python scripts/llm_ced.py -n 5 -p GEMBA -l all -d test -m gpt-4-turbo
```

## Evaluation

To evaluate the baseline predictions:

```bash
poetry run python scripts/baseline_eval.py
```

To create a latex table in the outputs directory with performance scores of the different COMET models on the WMT 2023 DA data:

```bash
make eval_da
```
