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
make analyse_ced
```

## Evaluation

To create a latex table in the outputs directory with performance scores of the different COMET models on the WMT 2023 DA data:

```bash
make eval_da
```
