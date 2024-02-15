import argparse
import os

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Get directory names"
    )

    parser.add_argument("-m", "--model", required=True,
                        help="Model name")
    parser.add_argument("-t", "--timestamp", required=True,
                        help="Predictions timestamp")
    
    return parser.parse_args()


def load_data(model, timestamp, lp):
    """
    Load model predictions and gold truth labels for language pair.
    """

    main_dir = os.getcwd()
    
    # predictions
    predictions_dir = os.path.join(main_dir, "results", model, timestamp)
    pred_path = os.path.join(predictions_dir, f"{lp}_predictions.csv")
    df_pred = pd.read_csv(pred_path)

    # gold labels
    data_dir = os.path.join(main_dir, "mlqe-pe", "data", "catastrophic_errors_goldlabels")
    labels_path = os.path.join(data_dir, f"{lp}_majority_test_goldlabels", "goldlabels.txt")
    df_labels = pd.read_csv(labels_path, sep='\t', header=None, names=["lang_pair", "ref", "idx", "label"])

    # merge on sentence indexes
    df_results = pd.merge(df_pred, df_labels, on="idx")

    return df_results


def main():
    args = parse_args()
    model = args.model
    timestamp = args.timestamp

    language_pairs = ["encs", "ende", "enja", "enzh"]
    for lp in language_pairs:

        df_results = load_data(model, timestamp, lp)

        # higher COMET score --> higher confidence it is NOT an error
        # labels:  ERROR = 0, NOT = 1
        y_true =  np.where(df_results["label"]=="NOT", 1, 0)

        # tresholds for binarizing COMET output
        thresholds = np.arange(0.1, 1, 0.1)
        for t in thresholds:
            print("\n", lp)

            # the model is treated as "NOT an error" detector
            # i.e., scores above threshold are "NOT an error" predictions
            y_hat = (df_results['comet_score'] >= t).astype(int)
            print(t, matthews_corrcoef(y_true, y_hat))

if __name__ == "__main__":
    main()