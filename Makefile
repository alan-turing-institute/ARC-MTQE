setup:
	mkdir -p results

	poetry update
	poetry install

data:
	mkdir -p data

	#================#
	# MLQE-PE DATA   #
	#================#

	cd data && git clone https://github.com/sheffieldnlp/mlqe-pe.git

	cd ./data/mlqe-pe/data/catastrophic_errors_goldlabels && \
	tar -xvzf encs_majority_test_goldlabels.tar.gz && \
	tar -xvzf ende_majority_test_goldlabels.tar.gz && \
	tar -xvzf enja_majority_test_goldlabels.tar.gz && \
	tar -xvzf enzh_majority_test_goldlabels.tar.gz 

	#=======================#
	# Unbabel 2022 MQM data #
	#=======================#

	cd data/ && \
	mkdir -p unbabel && \
	cd unbabel && \
	wget https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/generalMT2022/enru/mqm_generalMT2022_enru.tsv