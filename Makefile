setup:
	poetry update
	poetry install

	poetry run pre-commit autoupdate
	poetry run pre-commit install --install-hooks

data:
	mkdir -p data

	#===============================================#
	# MLQE-PE data (includes WMT QE CED 2021 data)  #
	#===============================================#

	cd data && git clone https://github.com/sheffieldnlp/mlqe-pe.git

	cd ./data/mlqe-pe/data/catastrophic_errors_goldlabels && \
	tar -xvzf encs_majority_test_goldlabels.tar.gz && \
	tar -xvzf ende_majority_test_goldlabels.tar.gz && \
	tar -xvzf enja_majority_test_goldlabels.tar.gz && \
	tar -xvzf enzh_majority_test_goldlabels.tar.gz

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

	#=============#
	# WMT QE 2022 #
	#=============#

	cd data && git clone https://github.com/WMT-QE-Task/wmt-qe-2022-data.git

	cd ./data/wmt-qe-2022-data/train-dev_data/task3_ced/dev/en-de && tar -xvzf en-de-dev.tar.gz
	cd ./data/wmt-qe-2022-data/train-dev_data/task3_ced/dev/pt-en && tar -xvzf pt-en-dev.tar.gz
	cd ./data/wmt-qe-2022-data/train-dev_data/task3_ced/train/en-de && tar -xvzf en-de-train.tar.gz
	cd ./data/wmt-qe-2022-data/train-dev_data/task3_ced/train/pt-en && tar -xvzf pt-en-train.tar.gz
