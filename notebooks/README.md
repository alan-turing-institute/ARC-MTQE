## Plots

The notebook `metrics.ipynb` provides some code to plot confusion matrices and precision-recall curves for the predictions. It is necessary to have run the evaluation script before using this notebook as the evaluation script identifies the random seed that achieved, for example, the median MCC value which can then be used to make the plots. See the [scripts/README](../scripts/README.md) for instructions on how to run the evaluation script.

## Latex tables of results

The notebook `create_tables.ipynb` provides some code to plot a metric or metrics (such as MCC or precision and recall) in latex tables.
