import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from mtqe.data.loaders import load_ced_data
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import OUTPUTS_DIR, PREDICTIONS_DIR
from mtqe.utils.tables import create_latex_table


def main():
    """
    Evaluate MCC performance on CED test data given best binarisation threshold chosen on the:
        - dev data
        - test data
    """

    language_pairs = LI_LANGUAGE_PAIRS_WMT_21_CED
    results = {"COMET-KIWI": [], "ceiling COMET-KIWI": []}
    for lp in language_pairs:

        # NOTES:
        # - higher COMET score --> higher confidence it is NOT an error
        # - true labels:  ERROR = 0, NOT = 1

        # get test data and baseline predictions
        df_test = load_ced_data("test", lp)
        pred_test_path = os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_test_cometkiwi.csv")
        df_pred_test = pd.read_csv(pred_test_path)
        df_test_results = pd.merge(df_pred_test[["idx", "comet_score"]], df_test[["idx", "score"]], on="idx")

        # get validation data and baseline predictions
        df_dev = load_ced_data("dev", lp)
        pred_dev_path = os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_dev_cometkiwi.csv")
        df_pred_dev = pd.read_csv(pred_dev_path)
        df_dev_results = pd.merge(df_pred_dev[["idx", "comet_score"]], df_dev[["idx", "score"]], on="idx")

        # evaluate MCC on the test and dev sets over a range of binarisation thresholds
        thresholds = np.arange(0, 1, 0.01)
        mccs_dev = []
        mccs_test = []
        for t in thresholds:
            y_hat = (df_test_results["comet_score"] >= t).astype(int)
            mcc_test = matthews_corrcoef(df_test_results["score"], y_hat)
            mccs_test.append(mcc_test)

            y_hat = (df_dev_results["comet_score"] >= t).astype(int)
            mcc_dev = matthews_corrcoef(df_dev_results["score"], y_hat)
            mccs_dev.append(mcc_dev)

        # index of best threshold on the validation data
        idx_dev_max = np.argmax(mccs_dev)

        # MCC on test data given the best threshold on the validation data
        results["COMET-KIWI"].append(mccs_test[idx_dev_max])
        # the best MCC on test data from all thesholds
        results["ceiling COMET-KIWI"].append(max(mccs_test))

    tex_full = create_latex_table(LI_LANGUAGE_PAIRS_WMT_21_CED, results)

    with open(os.path.join(OUTPUTS_DIR, "cometkiwi_baseline.tex"), "w") as f:
        for line in tex_full:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
