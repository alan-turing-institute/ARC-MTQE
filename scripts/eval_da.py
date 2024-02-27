import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# lists include DA not MQM data
LANGUAGE_PAIRS_22 = ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"]
LANGUAGE_PAIRS_23 = ["en-gu", "en-hi", "en-mr", "en-ta", "en-te"]

root_dir = os.getcwd()


def create_latex_table(col_names, results):
    """
    Each row is a model and column a language pair.
    """
    textabular = f"c|{'c'*len(col_names)}"
    texheader = " & " + " & ".join(map(lambda x: x.upper(), col_names)) + "\\\\"
    texdata = "\\hline\n"
    for label, values in results.items():
        texdata += f"{label} & {' & '.join(map(lambda n: '%.3f'%n, values))} \\\\\n"
    texdata += "\\hline"
    tex_full = [
        "\\begin{table}",
        "\\centering",
        "\\begin{tabular}{" + textabular + "}",
        texheader,
        texdata,
        "\\end{tabular}",
        "\\end{table}",
    ]

    return tex_full


print("\n")
print("WMT 2022 DA")
print("================")

data_dir_22 = os.path.join(root_dir, "data", "wmt-qe-2022-data", "test_data-gold_labels", "task1_da")

for lp in LANGUAGE_PAIRS_22:
    print(lp)

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
    print("----------------")


print("\n")
print("WMT 2023 DA")
print("================")

labels_path = os.path.join(root_dir, "data", "wmt-qe-2023-data", "gold_labels", "hallucinations_gold_T1s_header.tsv")
df_labels = pd.read_csv(labels_path, sep="\t")
results = {"COMET-QE": [], "COMETKiwi": []}

for lp in LANGUAGE_PAIRS_23:
    print(lp)
    out_dir = os.path.join(root_dir, "predictions", "da_test_data")

    with open(os.path.join(out_dir, f"2023_{lp}_comet_qe"), "rb") as f:
        qe_scores = pickle.load(f)

    with open(os.path.join(out_dir, f"2023_{lp}_cometkiwi_22"), "rb") as f:
        kiwi_scores = pickle.load(f)

    lp_scores = df_labels[df_labels["lp"] == lp]["score"]

    # remove hallucinations, this was also done at WMT 2023:
    # https://github.com/WMT-QE-Task/qe-eval-scripts/blob/main/wmt23/task1_sentence_evaluate.py
    hallucination_idx = [i for i, x in enumerate(lp_scores) if x == "hallucination"]
    labels = [float(score) for score in lp_scores if score != "hallucination"]
    qe_scores_clean = np.delete(qe_scores, hallucination_idx)
    kiwi_scores_clean = np.delete(kiwi_scores, hallucination_idx)

    qe_corr = spearmanr(qe_scores_clean, labels).statistic
    kiwi_corr = spearmanr(kiwi_scores_clean, labels).statistic
    print("COMET-QE", f"{qe_corr:.3f}")
    print("COMETKiwi", f"{kiwi_corr:.3f}")
    print("----------------")

    results["COMET-QE"].append(qe_corr)
    results["COMETKiwi"].append(kiwi_corr)


tex_full = create_latex_table(LANGUAGE_PAIRS_23, results)
with open("comets_compare.tex", "w") as f:
    for line in tex_full:
        f.write(f"{line}\n")
