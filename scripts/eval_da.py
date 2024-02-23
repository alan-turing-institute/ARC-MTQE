import os
import pickle

from scipy.stats import spearmanr

# lists include DA not MQM data
LANGUAGE_PAIRS_22 = ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"]
LANGUAGE_PAIRS_23 = ["en-gu", "en-hi", "en-mr", "en-ta", "he-en"]

root_dir = os.getcwd()

data_dir_22 = os.path.join(root_dir, "data", "wmt-qe-2022-data", "test_data-gold_labels", "task1_da")

for lp in LANGUAGE_PAIRS_22:
    out_dir = os.path.join(root_dir, "predictions", "da_test_data")

    with open(os.path.join(out_dir, f"2022_{lp}_comet_qe"), "rb") as f:
        qe_scores = pickle.load(f)

    with open(os.path.join(out_dir, f"2022_{lp}_cometkiwi_22"), "rb") as f:
        kiwi_scores = pickle.load(f)

    # INCONSISTENCY IN 2022 FILE NAMES
    if os.path.exists(os.path.join(data_dir_22, lp, f"test.2022.{lp}.da_score")):
        with open(os.path.join(data_dir_22, lp, f"test.2022.{lp}.da_score")) as f:
            labels = [float(score) for score in f.read().splitlines()]
    else:
        with open(os.path.join(data_dir_22, lp, f"test2022.{lp}.da_score")) as f:
            labels = [float(score) for score in f.read().splitlines()]

    print("COMET-QE", f"{spearmanr(qe_scores, labels).statistic:.2f}")
    print("COMETKiwi", f"{spearmanr(kiwi_scores, labels).statistic:.2f}")
