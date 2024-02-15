import argparse
import os

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef


def parse_args():
    """"""
    parser = argparse.ArgumentParser(
        description="Get directory names"
    )

    parser.add_argument("-m", "--model", required=True,
                        help="Model name")
    parser.add_argument("-t", "--timestamp", required=True,
                        help="Predictions timestamp")
    
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    timestamp = args.timestamp

    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, "mlqe-pe", "data", "catastrophic_errors_goldlabels")
    predictions_dir = os.path.join(main_dir, "results", model, timestamp)

    language_pairs = ["encs", "ende", "enja", "enzh"]
    for lp in language_pairs:
        # predictions
        pred_path = os.path.join(predictions_dir, f"{lp}_predictions.csv")
        df_pred = pd.read_csv(pred_path)

        # gold labels
        labels_path = os.path.join(data_dir, f"{lp}_majority_test_goldlabels", "goldlabels.txt")
        df_labels = pd.read_csv(labels_path, sep='\t', header=None, names=["lang_pair", "ref", "idx", "label"])

        # merge labels with predictions on sentence indexes
        df_results = pd.merge(df_pred, df_labels, on="idx")

        # labels:  ERROR = 0, NOT = 1
        # higher COMET score --> higher confidence it is NOT an error
        y_true =  np.where(df_labels["label"]=="NOT", 1, 0)

        # tresholds for binarizing COMET output
        thresholds = np.arange(0.1, 1, 0.1)
        for t in thresholds:
            print("\n", lp)

            # scores above threshold are treated as NOT an error predictions
            y_hat = (df_results['comet_score'] >= t).astype(int)
            print(t, matthews_corrcoef(y_true, y_hat))

if __name__ == "__main__":
    main()