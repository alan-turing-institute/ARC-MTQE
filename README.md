# MTQE

## Pre-requisites

Huggingface account with an access token to log into Huggingface hub. 

## Set up

Update and install Poetry dependencies.

```bash
poetry update
poetry install
```

Download WMT 2021 critical error data and 

```bash
git clone https://github.com/sheffieldnlp/mlqe-pe.git
```

extract gold labels for test data (for all 4 language pairs in the directory).

```bash
cd mlqe-pe/data/catastrophic_errors_goldlabels
for a in *.tar.gz; do tar -xvzf "$a"; done
```