# Trained models

## Overview of training strategies

There are two key datasets:
- WMT released critical error annotations of translations of English Wikipedia comments into Czech, German, Japanese and Chinese
- The DEMETR dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)) of synthetically created critical errors in news data (10 language pairs)

Training strategies differ primarily in what data were used for training. Additionally, there are two
distinct groups of training strategies depending on whether the model was trained in a single step or whether the data was used in two steps, using subset of the data for pre-training followed by a fine-tuning step. There are five experiments with trained models, each defined by a training strategy.

In the one-step group, we only used the authentic WMT data. In the monolingual case, a model was
trained using the original monolingual data - one model for each language pair (English-Czech, English-German, English-Japanese and English-Chinese). For the multilingual case,
one multilingual model was trained on the combined multilingual authentic data and then used to make
predictions for all the language pairs.

In the two-step group, we first pre-trained the model using either the multilingual authentic data,
the synthetic data or both. Each of the three base models were then fine-tuned on each of the four sets of authentic monolingual data.

Each of the five experiments provides a model for each of the 4 language pairs, giving 20 models in total. For all experiments, we repeat training over five random seeds. For each training run, of 100 epochs, we selected the epoch that achieved the highest MCC on the development dataset.

## Config files

The table below gives an overview of the trained models and the corresponding config file used to train it.

For models that were trained on monolingual data, the en-ja language pair was run in a separate config group as it required a smaller batch size of 32 compared to other language pairs, which all could run with a batch size of 64.

|Experiment name |Experiment description|Config file|
|-------------------------|----------------------|-----------|
|One-step > Monolingual auth data|A model was trained using the original monolingual data - one model for each language pair.|`train_monolingual_auth_data`<br>`train_monolingual_auth_data_enja`|
|One-step > Multilingual auth data|One multilingual model was trained on the combined multilingual authentic data|`train_multilingual_auth_data_all`|
|Two-step > Multilingual auth data|Model was first pre-trained on all multilingual authentic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_auth`)<br>Second step: `second_step_base_auth_data` and `second_step_base_auth_data_enja`|
|Two-step > Synthetic|Model was first pre-trained on all multilingual synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_demetr`)<br>Second step: `second_step_base_demetr_data` and `second_step_base_demetr_data_enja`|
|Two-step > Multilingual auth + synthetic data|Model was first pre-trained on all multilingual authentic and synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_demetr_auth`)<br>Second step: `second_step_base_demetr_auth_data` and `second_step_base_demetr_auth_data`|
