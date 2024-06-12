# Data

This file lists the datasets that are downloaded and the licenses under which they are distributed when the command `make data` is run. Datasets that have been labelled as 'optional' are not required to re-train the models as described in [these notes](models.md).

## MLQE-PE data
This is the Multilingual Quality Estimation and Automatic Post-editing Dataset and it is downloaded from the [github repo](https://github.com/sheffieldnlp/mlqe-pe) where more information about the dataset can be found. This dataset includes the WMT QE CED 2021 data ([Specia et al.,2021](https://aclanthology.org/2021.wmt-1.71/)) which is required for training and testing our CED models. It is distributed under the [CC0-1.0 license](https://github.com/sheffieldnlp/mlqe-pe?tab=CC0-1.0-1-ov-file#readme).

## DEMETR data

The Diagnosing Evaluation Metrics for Translation (DEMETR) dataset ([Karpinska et al., 2022](https://doi.org/10.18653/v1/2022.emnlp-main.649)) is downloaded from the [github repo](https://github.com/marzenakrp/demetr) where more information about the dataset can be found. This dataset consists of synthetic CED data and was used in some of the training strategies for training the CED models. It is distributed under the [MIT license](https://github.com/marzenakrp/demetr?tab=MIT-1-ov-file#readme).

## WMT QE 2022

This is the dataset distributed for the QE shared task at WMT 2022 and is downloaded from the [github repo](https://github.com/WMT-QE-Task/wmt-qe-2022-data) where more information about the dataset can be found. The dataset contains some synthetic English-German critical error data which was used for some of our training strategies. There is no license associated with this dataset.

## WMT QE 2023 [Optional]

This is the dataset distributed for the QE shared task at WMT2023 and is downloaded from the [github repo](https://github.com/WMT-QE-Task/wmt-qe-2023-data.git) where more information about the dataset can be found. These data were not used in our training strategies and there is no license associated with this dataset.

## Unbabel 2022 MQM data [Optional]
This is Unbabel's 2022 MQM data from English to Russian, although it is downloaded from Google's [github repo](https://github.com/google/wmt-mqm-human-evaluation/tree/main) where other MQM datasets can also be found. This English-Russian MQM dataset differs from the others as it contains a critical error category (as well as minor and major). This is distributed under the [Apache-2.0 license](https://github.com/google/wmt-mqm-human-evaluation?tab=Apache-2.0-1-ov-file#readme).
