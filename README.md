# MTQE

⚠️ This is a work in progress ⚠️

Machine translation (MT) systems have improved significantly in recent years but they are not immune to
errors. Quality estimation (QE) is the task of predicting the quality of a translation given only the source and the
target translated text without a gold standard reference translation for comparison. See  [Knight et al. (2024)](https://doi.org/10.5281/zenodo.10931558) for a literature review on MTQE.

The goal of critical error detection (CED) is to identify translated text that deviates in meaning from the source
text. CED was introduced at the Conference on Machine Translation (WMT) 2021 QE subtask ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) but has not been held since, and the submitted models from 2021 are not publicly available. As part of the subtask, WMT released a unique dataset of authentic critical error annotations in translations of Wikipedia comments.

This project investigates CED using one of the highest performing QE models from WMT 2022 ([COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)) as our starting point. We used the COMETKiwi-22 as our baseline and evaluated its performance on the CED task using a binarisation threshold. We also tried a number of fine-tuning strategies with the WMT 2021 authentic CED data ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) as well as synthetic data from the DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)). Additionally, we also investigated large language models (LLMs) given their emergence at WMT 2023 ([Blain et al.,2023](https://doi.org/10.18653/v1/2023.wmt-1.52); [Freitag et al., 2023](https://doi.org/10.18653/v1/2023.wmt-1.51)), although they are not the main focus of this project.

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

Download COMET-QE 2021:

```bash
make models
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

To make predictions using GPT, you will need to have access to the OpenAI API. The API key will need to be saved as an environment variable named OPENAI_API_KEY. To do this in a Mac terminal:

```
export OPENAI_API_KEY="your_api_key"
```

## Structure of this repository

```

├── data/
│   ├── demetr/
│   ├── mlqe-pe/
│   ├── unbabel/
│   ├── wmt-qe-2022-data/
│   ├── wmt-qe-2023-data/
│   ├── preprocessed/
├── models/
│   ├── ...
├── notebooks/
│   ├── ...
│   outputs/
│   ├── ...
├── predictions/
│   ├── ced_test_data/
│   ├── da_test_data/
├── scripts/
│   ├── ...
```

## Links

- [Overview of available COMET models](https://github.com/Unbabel/COMET/blob/master/MODELS.md)
- [Baskerville instructions](notes/BASKERVILLE.md)
- [Notes on COMET codebase](notes/COMET.md)

## Make predictions

To make predictions for WMT 2023 DA test data using the COMET-QE 2020 and 2021 models and the COMETKiwi 2022 model:

```bash
make analyse_da
```

It is also possible to make predictions with the COMETKiwi-XL 2023 model but note that this can take couple of hours per language pair (there are 5 in the 2023 DA test set):

```bash
make analyse_da_xl
```

To make COMETKiwi-22 predictions for the  CED test data:

```bash
make baseline_predict
```

To use the OpenAI API to make critical error predictions run the following script. The parameters passed to the script indicate, which prompt (`basic`, `GEMBA` or `wmt21_annotator`) and GPT model (e.g., `gpt-3.5-turbo` or `gpt-4-turbo`) to use and how many translations, which language pair (e.g., `en-cs` but can also be `all`) and which data split (`train`, `dev` or `test`) to make predictions for. For example:

```bash
poetry run python scripts/llm_ced.py -n 5 -p GEMBA -l all -d test -m gpt-4-turbo
```

## Evaluation

To create a latex table in the outputs directory with performance scores of the different COMET models on the WMT 2023 DA data:

```bash
make eval_da
```
