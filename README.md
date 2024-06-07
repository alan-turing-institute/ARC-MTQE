# Critical Error Detection for Machine Translation

Code to train and evaluate models for detecting critical errors in machine translations using only the original source text and the translated text.

## Background

The goal of critical error detection (CED) is to identify translated text that deviates in meaning from the original text. CED was introduced at the Conference on Machine Translation (WMT) 2021 quality estimation (QE) subtask ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)), which also released a unique dataset of authentic critical error annotations in translations of Wikipedia comments. See [Knight et al. (2024)](https://doi.org/10.5281/zenodo.10931558) for a literature review on machine translation quality estimation (MTQE) including CED.

This project investigated CED using ([COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)) as the starting point. We used COMETKiwi-22 as our baseline and evaluated its performance on the WMT CED test data using a binarisation threshold. We also tried a number of fine-tuning strategies with the WMT 2021 authentic CED data ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) as well as synthetic data from the DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)). Additionally, we also investigated three prompting strategies with large language models (LLMs) given their emergence at WMT 2023 ([Blain et al.,2023](https://doi.org/10.18653/v1/2023.wmt-1.52); [Freitag et al., 2023](https://doi.org/10.18653/v1/2023.wmt-1.51)), although they are not the main focus of this project.

## Structure of this repository

```
├── configs/                   -- configs used for training experiments
├── data/
│   ├── ...                    -- downloaded data files
│   ├── preprocessed/          -- preprocessed data used in experiments
├── notes/
│   ├── ...
├── notebooks/
│   ├── ...
├── predictions/ced_data/      -- predictions on the test (and dev) data
│   ├── ...
├── scripts/                   -- training, prediction and evaluation code
│   ├── ...
├── src/                       -- model implementation
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

## Useful links

- [Overview of available COMET models](https://github.com/Unbabel/COMET/blob/master/MODELS.md)
- [Notes on the COMET codebase](notes/COMET.md)
