# Critical Error Detection for Machine Translation

Code to train and evaluate models for detecting critical errors in machine translations using only the original source text and the machine translated text.

## Table of contents

- [Background](#background)
- [Approaches](#approaches)
- [Structure of this repository](#structure-of-this-repository)
- [Set up](#set-up)
- [Useful links and notes](#useful-links-and-notes)

## Background

The goal of critical error detection (CED) is to identify translated text that deviates in meaning from the original text. CED was introduced at the Conference on Machine Translation (WMT) 2021 quality estimation (QE) subtask ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)), which also released a unique dataset of authentic critical error annotations in translations of Wikipedia comments. See [Knight et al. (2024)](https://doi.org/10.5281/zenodo.10931558) for a literature review on machine translation quality estimation (MTQE) including CED.

## Approaches

### Trained models

We used [COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) ([Rei et al., 2022](https://aclanthology.org/2022.wmt-1.60/)), which outputs quality scores between 0 and 1 (1=perfect translation).

For the baseline, we picked a binarisation threshold using the WMT dev data and used it to binarise COMETKiwi-22 predictions on the test data.

We also adapted COMETKiwi-22 for binary classification in the [CEDModel](src/mtqe/models/comet.py) class. Broadly, we tried two main training strategies:
- Fine-tune `CEDModel` with the WMT released authentic training data
- Pre-train the `CEDModel` with syntethic data from the DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)) and then fine-tune  with the WMT authentic data

See the [notes/](notes/) directory for an overview of the [different training strategies](notes/models.md) and the [scripts/README](scripts/README.md) file on how to train models.

### LLM prompts

- We tried three LLM prompts:
    - The [basic](src/mtqe/llms/query.py) prompt asks if the translation has the same meaning as the original text
    - [GEMBA-MQM](src/mtqe/llms/gemba.py) from [Kocmi and Federmann (2024)](https://arxiv.org/abs/2310.13988)
    - Using the original [WMT annotator guidelines](src/mtqe/llms/annotator_guidelines.py) from [Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)

## Structure of this repository

```
├── configs/                   -- configs used for training experiments
├── data/
│   ├── ...                    -- downloaded data files
│   ├── preprocessed/          -- preprocessed data used in experiments
├── notes/                     -- includes overview of training strategies
│   ├── ...
├── notebooks/
│   ├── ...
├── predictions/ced_data/      -- predictions on the test (and dev) data
│   ├── ...
├── scripts/                   -- training, prediction and evaluation code
│   ├── ...
├── src/                       -- model and prompt implementations
│   ├── mtqe/
│   │   ├── data/
│   │   ├── llms/
│   │   ├── models/
│   │   ├── utils/
```

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

Download and preprocess datasets:

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

We use [WandB](https://wandb.ai/) to track experiments. It is necessary to login first (you should only need to do this once). The below code will prompt you for an API key, which you can find in the [User Settings](https://wandb.ai/settings):

```python
import wandb
wandb.login()
```

To make predictions using GPT, you need an OpenAI API key saved as an environment variable named OPENAI_API_KEY. To do this in a Mac terminal:

```
export OPENAI_API_KEY="your_api_key"
```

## Useful links and files

- [Instructions for making and evaluating predictions](scripts/README.md) on the WMT test data
- [Overview of available COMET models](https://github.com/Unbabel/COMET/blob/master/MODELS.md).
- [Notes on the COMET codebase](notes/COMET.md) that our trained `CEDModel` inherits from.
- [Instructions for using Baskerville's Tier 2 HPC service](notes/Baskerville.md) to train models.

## Development

The code base could be updated to use models other than [COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da). This would require an update to the [load_model_from_file](src/mtqe/models/loaders.py) which is currently hard-coded to download COMETKiwi-22:

```python
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
```

This could be updated to allow for the pre-trained QE model to be changed to, for example, [COMETKiwi-23-XL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) or [COMETKiwi-23-XXL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl).

This would also require updating the encoder related hyperparameters in the config file (e.g., `encoder_model: XLM-RoBERTa-XL`).
