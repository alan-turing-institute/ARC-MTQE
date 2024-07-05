# Trained models

Training strategies differ primarily in what data were used for training. Additionally, there are two distinct groups of training strategies depending on whether the model was trained in a single step or whether the data was used in two steps, using subset of the data for pre-training followed by a fine-tuning step.

## Data

We used two datasets:
- WMT 2021 released critical error annotations of translations of English Wikipedia comments into Czech, German, Japanese and Chinese ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/))
- The DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)) of synthetically created critical errors in news data (10 language pairs)

The WMT 2021 authentic data was used in two ways:
- `monolingual`: the language pairs were treated independently and a model was trained on each separately
- `multilingual`: the data for all 4 language pairs was combined into a single dataset

## Training strategies

In the `one-step` group, we primarily used the authentic WMT data. In the monolingual case, a model was
trained using the original monolingual data - one model for each language pair (English-Czech, English-German, English-Japanese and English-Chinese). For the multilingual case,
one multilingual model was trained on the combined multilingual authentic data and then used to make
predictions for all the language pairs.

In the `two-step` group, we first pre-trained the model using either:
- the multilingual authentic data
- the synthetic DEMETR data
- the multilingual authentic data and DEMETR data
Each of the three base models were then fine-tuned on each of the four sets of authentic monolingual data.

Each of the five experiments provides a model for each of the 4 language pairs, giving 20 models in total. For all experiments, we repeat training over five random seeds. For each training run, of 100 epochs, we selected the epoch that achieved the highest MCC on the development dataset.

## Config files

The tables below gives an overview of the trained models and the corresponding config file used to train them.

For the two-step experiments, we first pre-trained a new base model. All of the base models are defined in the same config file (`base_models`) but each with a different experiment name:

| Experiment name | Base model config file experiment name |
|----------------------------|-----------------------------|
| Two-step > Multilingual auth data | `base_auth` |
| Two-step > Synthetic | `base_demetr` |
| Two-step > Multilingual auth + synthetic data | `base_demetr_auth` |

For models that were trained on monolingual data, the En-Ja language pair was run in a separate config group as it required a smaller batch size of 32 compared to other language pairs, which all could run with a batch size of 64.

|Experiment name |Experiment description|Config file|
|-------------------------|----------------------|-----------|
|One-step > Monolingual auth data|A model was trained using the WMT 2021 monolingual data - one model for each language pair.|`train_monolingual_auth_data` and <br>`train_monolingual_auth_data_enja`|
|One-step > Multilingual auth data|One model was trained on the combined multilingual authentic data|`train_multilingual_auth_data_all`|
|Two-step > Multilingual auth data|A model was first pre-trained on all multilingual authentic data, then fine-tuned on each of the four sets of authentic monolingual data|`second_step_base_auth_data` and `second_step_base_auth_data_enja`|
|Two-step > Synthetic| A model was first pre-trained on all multilingual synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|`second_step_base_demetr_data` and `second_step_base_demetr_data_enja`|
|Two-step > Multilingual auth + synthetic data|A model was first pre-trained on all multilingual authentic and synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|`second_step_base_demetr_auth_data` and `second_step_base_demetr_auth_data`|
