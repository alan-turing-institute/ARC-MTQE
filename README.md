# MTQE

Machine Translation Quality Estimation.

## Pre-requisites

Huggingface account with an access token to log into Huggingface hub. 

## Set up

Update and install Poetry dependencies:

```bash
poetry update
poetry install
```

Download WMT 2021 critical error data:

```bash
git clone https://github.com/sheffieldnlp/mlqe-pe.git
```

Extract gold labels for test data (for all 4 language pairs in the directory):

```bash
cd mlqe-pe/data/catastrophic_errors_goldlabels
for a in *.tar.gz; do tar -xvzf "$a"; done
```

## Structure of this repository

The `results` directory is created when running a model for the first time. Each model has its own directory and within it each run of the model has a timestamped directory. Predictions are saved in a separate csv file for each language pair (`encs`, `ende`, `enja`, `enzh`).

```
├── mlqe-pe/
│   ├── data/
│   ├──   ├── .../
├── notebooks/
│   ├── ...
├── results/
│   ├── <model name>
│   ├──   ├── <timestamp>
│   ├──   ├──   ├── <language pair>_predictions.csv
│   ├──   ├──   ├── ...
├── scripts/
│   ├── eval.py
│   ├── ...
```

## Evaluation

The evaluation script produces summary statistics and plots given a model and timestamp within the results directory.

```bash
poetry run python scripts/eval.py -m <model name> -t <timestamp>
```