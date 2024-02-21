# MTQE

Machine Translation Quality Estimation.

## Set up

Clone this repository and change the current working directory.

```bash
git clone https://github.com/alan-turing-institute/ARC-MTQE.git
cd ARC-MTQE
```

Install dependencies and pre-commit hooks with Poetry:

```bash
make setup
```

Download datasets:

```bash
make data
```

To use COMETKiwi, you need a HuggingFace account and access token (they're under https://huggingface.co/settings/tokens in your account settings). Log in to the HuggingFace CLI which will request the token:

```bash
poetry run huggingface-cli login
```

## Structure of this repository

Each model has its own directory and within it each run of the model has a timestamped directory. Predictions are saved in a separate csv file for each language pair (`encs`, `ende`, `enja`, `enzh`).

```
├── data/
│   ├── demetr/
│   ├── mlqe-pe/
│   ├── unbabel/
│   ├── wmt-qe-2022-data/
│   ├── wmt-qe-2023-data/
├── notebooks/
│   ├── ...
├── results/
│   ├── <model name>
│   ├──   ├── <timestamp>
│   ├──   ├──   ├── <language pair>_predictions.csv
│   ├──   ├──   ├── ...
├── scripts/
│   ├── ...
```

## Links

- [Overview of available COMET models](https://github.com/Unbabel/COMET/blob/master/MODELS.md)

## COMETKiwi

Run the `comet_kiwi.py` script to make predictions for the test data.

```bash
poetry run python scripts/comet_kiwi.py
```

The evaluation script produces summary statistics and plots given a model and timestamp within the same results directory.

```bash
poetry run python scripts/eval.py -m <model name> -t <timestamp>
```
