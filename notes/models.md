This is a table showing the models in our report (cite report) and what config file in this repo they correspond to.
For models that were trained on monolingual data, the en-ja language pair was run in a separate config group as it required a smaller batch size of 32 compared to other language pairs, which all could run with a batch size of 64.

|Experiment name in report|Experiment description|Config file|
|-------------------------|----------------------|-----------|
|One-step > Monolingual auth data|A model was trained using the original monolingual data - one model for each language pair.|`train_monolingual_auth_data`<br>`train_monolingual_auth_data_enja`|
|One-step > Multilingual auth data|One multilingual model was trained on the combined multilingual authentic data|`train_multilingual_auth_data_all`|
|Two-step > Multilingual auth data|Model was first pre-trained on all multilingual authentic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_auth`)<br>Second step: `second_step_base_auth_data` and `second_step_base_auth_data_enja`|
|Two-step > Synthetic|Model was first pre-trained on all multilingual synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_demetr`)<br>Second step: `second_step_base_demetr_data` and `second_step_base_demetr_data_enja`|
|Two-step > Multilingual auth + synthetic data|Model was first pre-trained on all multilingual authentic and synthetic data, then fine-tuned on each of the four sets of authentic monolingual data|First step: `base_models` (experiment name `base_demetr_auth`)<br>Second step: `second_step_base_demetr_auth_data` and `second_step_base_demetr_auth_data`|
