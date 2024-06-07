# Scripts

## Train models

### Config files
YAML config files are used to control the parameters for training a model. Examples of these are found in the `configs` folder in the repo.

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
