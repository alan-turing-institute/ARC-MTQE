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

To use the COMET models, you must also acknowledge the license for their latest models on the HuggingFace page:
- [COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
- [COMETKiwi-23-XL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl)
- [COMETKiwi-23-XXL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl)

## Structure of this repository

```
├── data/
│   ├── demetr/
│   ├── mlqe-pe/
│   ├── unbabel/
│   ├── wmt-qe-2022-data/
│   ├── wmt-qe-2023-data/
├── notebooks/
│   ├── ...
├── predictions/
│   ├── ced_test_data/
│   ├── da_test_data/
├── scripts/
│   ├── ...
```

## Links

- [Overview of available COMET models](https://github.com/Unbabel/COMET/blob/master/MODELS.md)

## Make predictions

### DA test data

To make predictions for WMT 2022 and 2023 DA test data using COMET-QE and COMETKiwi-22:

```bash
make predict_da
```

To make COMETKiwi-22 predictions for the  CED test data:

```bash
make predict_ced
```
