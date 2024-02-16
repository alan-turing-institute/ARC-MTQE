setup:
	mkdir -p results
	
	poetry update
	poetry install

data:
	git clone https://github.com/sheffieldnlp/mlqe-pe.git

	cd ./mlqe-pe/data/catastrophic_errors_goldlabels && \
	tar -xvzf encs_majority_test_goldlabels.tar.gz && \
	tar -xvzf ende_majority_test_goldlabels.tar.gz && \
	tar -xvzf enja_majority_test_goldlabels.tar.gz && \
	tar -xvzf enzh_majority_test_goldlabels.tar.gz 
