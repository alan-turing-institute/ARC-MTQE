#==================================#
# DOWNLOADED DATA DIRECTORY NAMES  #
#==================================#

WMT21_LANGUAGE_PAIRS=encs ende enja enzh
WMT22_LANGUAGE_PAIRS=en-de pt-en
TRAIN_DEV=train dev

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

	#=============#
	# WMT QE 2022 #
	#=============#

	cd data && git clone https://github.com/WMT-QE-Task/wmt-qe-2022-data.git

	cd ./data/wmt-qe-2022-data/train-dev_data/task3_ced && \
	for lp in ${WMT22_LANGUAGE_PAIRS} ; do \
		for d in ${TRAIN_DEV} ; do \
			cd $${d}/$${lp} ; \
			tar -xvzf $${lp}-$${d}.tar.gz ; \
			cd ../../ ; \
		done \
	done

	#=============#
	# WMT QE 2023 #
	#=============#

	cd data && git clone https://github.com/WMT-QE-Task/wmt-qe-2023-data.git

analyse_da_data:
	poetry run python scripts/comets_compare.py

analyse_ced_test_data:
	poetry run python scripts/comet_kiwi.py
	poetry run python scripts/eval.py -p ./predictions/ced_test_data/
