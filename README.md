# MTQE

⚠️ This is a work in progress ⚠️

Machine translation (MT) systems have improved significantly in recent years but they are not immune to
errors. Quality estimation (QE) is the task of predicting the quality of a translation given only the source and the
target translated text without a gold standard reference translation for comparison. See  [Knight et al. (2024)](https://doi.org/10.5281/zenodo.10931558) for a literature review on MTQE.

The goal of critical error detection (CED) is to identify translated text that deviates in meaning from the source
text. CED was introduced at the Conference on Machine Translation (WMT) 2021 QE subtask ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) but has not been held since, and the submitted models from 2021 are not publicly available. As part of the subtask, WMT released a unique dataset of authentic critical error annotations in translations of Wikipedia comments.

This project investigates CED using one of the highest performing QE models from WMT 2022 ([COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)) as our starting point. We used the COMETKiwi-22 as our baseline and evaluated its performance on the CED task using a binarisation threshold. We also tried a number of fine-tuning strategies with the WMT 2021 authentic CED data ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) as well as synthetic data from the DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)). Additionally, we also investigated large language models (LLMs) given their emergence at WMT 2023 ([Blain et al.,2023](https://doi.org/10.18653/v1/2023.wmt-1.52); [Freitag et al., 2023](https://doi.org/10.18653/v1/2023.wmt-1.51)), although they are not the main focus of this project.

From a practical perspective, accuracy/efficiency trade-offs are a key consideration ([Shterionov et al., 2019](https://aclanthology.org/W19-6738/)). We therefore avoided using large QE models such as COMET-KIWI-XL, COMET-KIWI-XXL or xCOMET ([Rei et al., 2023](https://doi.org/10.18653/v1/2023.wmt-1.73)) or prioritising model ensembles.

## QE Models

As well as the CED models that are developed in this project, this repo also provides the functionality to download and make predictions using direct assessment (DA) data with the [COMET-QE](https://aclanthology.org/2020.wmt-1.101/) model from 2020. This model has often been used as a baseline in WMT subtasks. The repo also has the functionality to make predictions using DA data and the [COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) model. This is not essential for the main CED work, but it allows the user to benchmark how these models work 'out of the box' at the task they were trained for, without any further fine-tuning.

The code base could also be updated to use models other than [COMETKiwi-22](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) for CED. This would require an update to the `load_model_from_file` function in `src/mtqe/models/loaders.py` which is currently hard-coded to download COMETKiwi-22:

`model_path = download_model("Unbabel/wmt22-cometkiwi-da")`

This could be updated to allow for the pre-trained QE model to be changed to, for example, [COMETKiwi-23-XL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) or [COMETKiwi-23-XXL](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl).

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

Download COMET-QE 2021 (optional if you only want to develop CED models):

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
├── checkpoints/
├── configs/
├── data/
│   ├── demetr/
│   ├── mlqe-pe/
│   ├── processed/
│   ├── unbabel/
│   ├── wmt-qe-2022-data/
│   ├── wmt-qe-2023-data/
│   ├── preprocessed/
├── evaluations/
│   ├── ...
├── notes/
├── notebooks/
│   ├── ...
│   outputs/
│   ├── ...
├── predictions/
│   ├── ...
├── scripts/
│   ├── ...
├── src/
│   ├── mtqe/
│   │   ├── data/
│   │   ├── llms/
│   │   ├── models/
│   │   ├── utils/
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
