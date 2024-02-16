# MTQE

Machine Translation Quality Estimation.

## Set up

Clone this repository and change the current working directory.

```bash
git clone https://github.com/alan-turing-institute/ARC-MTQE.git
cd ARC-MTQE
```

To use COMETKiwi, you need a HuggingFace account and access token (they're under https://huggingface.co/settings/tokens in your account settings). Log in to the HuggingFace CLI which will request the token: 

```bash
huggingface-cli login
```

Update and install Poetry dependencies:

```bash
make setup
```

Download WMT 2021 critical error data:

```bash
make data
```

## Structure of this repository

Each model has its own directory and within it each run of the model has a timestamped directory. Predictions are saved in a separate csv file for each language pair (`encs`, `ende`, `enja`, `enzh`).

```
├── data/
│   ├── mlqe-pe/
│   ├──   ├── .../
│   ├── unbabel/
│   ├──   ...
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

## COMETKiwi

Run the `comet_kiwi.py` script to make predictions for the test data.

```bash
poetry run python scripts/comet_kiwi.py
```

The evaluation script produces summary statistics and plots given a model and timestamp within the results directory.

```bash
poetry run python scripts/eval.py -m <model name> -t <timestamp>
```

## Development

To run linters:

```bash
poetry run black .
poetry run isort .
poetry run flake8
```