#==================================#
# DOWNLOADED DATA DIRECTORY NAMES  #
#==================================#

WMT21_LANGUAGE_PAIRS=encs ende enja enzh
TRAIN_DEV=train dev

setup:
	poetry update
	poetry install

	poetry run pre-commit autoupdate
	poetry run pre-commit install --install-hooks

	# make directories specified in `src/mtqe/utils/paths.py`
	mkdir -p predictions
	mkdir -p outputs

models:
	mkdir -p models

	# COMET-QE-21 DA
	cd ./models && wget https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-da.tar.gz
	cd ./models && tar -xf wmt21-comet-qe-da.tar.gz

data: download_data preprocess_data

download_data:
	mkdir -p data

	#===============================================#
	# MLQE-PE data (includes WMT QE CED 2021 data)  #
	#===============================================#

	cd data && git clone https://github.com/sheffieldnlp/mlqe-pe.git

	cd ./data/mlqe-pe/data/catastrophic_errors_goldlabels && \
	for lp in $(WMT21_LANGUAGE_PAIRS) ; do \
		tar -xvzf $${lp}_majority_test_goldlabels.tar.gz ; \
	done

	#=======================#
	# Unbabel 2022 MQM data #
	#=======================#

	cd data && \
	mkdir -p unbabel && \
	cd unbabel && \
	wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/enru/mqm_generalMT2022_enru.tsv

	#================#
	# DEMETR dataset #
	#================#

	cd data && git clone https://github.com/marzenakrp/demetr.git

preprocess_data:
	cd data/ && mkdir -p processed
	poetry run python scripts/data_preprocess.py

analyse_da:
	poetry run python scripts/predict_da.py -m comet_qe_20 -y 2023
	poetry run python scripts/predict_da.py -m comet_qe_21 -y 2023
	poetry run python scripts/predict_da.py -m cometkiwi_22 -y 2023

analyse_da_xl:
	poetry run python scripts/predict_da.py -m cometkiwi_23_xl -y 2023

eval_da:
	poetry run python scripts/eval_da.py

baseline_predict:
	poetry run python scripts/predict_ced.py -g baseline -p cometkiwi_22 -d test -l all
	poetry run python scripts/predict_ced.py -g baseline -p cometkiwi_22 -d dev -l all
