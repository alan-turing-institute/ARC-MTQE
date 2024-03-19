import os
import pickle
from collections import defaultdict

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


def load_labels(lp, year):

    if year == "2022":
        data_dir_22 = os.path.join(root_dir, "data", "wmt-qe-2022-data", "test_data-gold_labels", "task1_da")

        # INCONSISTENCY IN 2022 FILE NAMES
        if os.path.exists(os.path.join(data_dir_22, lp, f"test.2022.{lp}.da_score")):
            with open(os.path.join(data_dir_22, lp, f"test.2022.{lp}.da_score")) as f:
                labels = [float(score) for score in f.read().splitlines()]
        else:
            with open(os.path.join(data_dir_22, lp, f"test2022.{lp}.da_score")) as f:
                labels = [float(score) for score in f.read().splitlines()]

    elif year == "2023":
        labels_path = os.path.join(
            root_dir, "data", "wmt-qe-2023-data", "gold_labels", "hallucinations_gold_T1s_header.tsv"
        )
        df_labels = pd.read_csv(labels_path, sep="\t")
        labels = df_labels[df_labels["lp"] == lp]["score"]

    return labels


def load_predictions(lp, year, model):

    out_dir = os.path.join(root_dir, "predictions", "da_test_data")

    with open(os.path.join(out_dir, f"{year}_{lp}_{model}"), "rb") as f:
        predictions = pickle.load(f)

    return predictions


def score_predictions(year):

    results = defaultdict(list)
    lps = LANGUAGE_PAIRS_22 if year == "2022" else LANGUAGE_PAIRS_23
    model_names = {
        "comet_qe": "COMET-QE 2020",
        "comet_qe_21": "COMET-QE 2021",
        "cometkiwi_22": "COMETKiwi 2022",
        "cometkiwi_23_xl": "COMETKiwi-XL 2023",
    }

    for lp in lps:

        labels = load_labels(lp, year)

        for model in model_names:

            preds = load_predictions(lp, year, model)

            if year == "2023":
                # remove hallucinations, this was also done at WMT 2023:
                # https://github.com/WMT-QE-Task/qe-eval-scripts/blob/main/wmt23/task1_sentence_evaluate.py
                hallucination_idx = [i for i, x in enumerate(labels) if x == "hallucination"]
                labels_clean = [float(score) for score in labels if score != "hallucination"]
                preds_clean = np.delete(preds, hallucination_idx)
                corr = spearmanr(preds_clean, labels_clean).statistic
            else:
                corr = spearmanr(preds, labels).statistic

            results[model_names[model]].append(corr)

    return results


def main():
    out_dir = os.path.join(root_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    results_23 = score_predictions("2023")
    tex_full = create_latex_table(LANGUAGE_PAIRS_23, results_23)

    with open(os.path.join(out_dir, "comets_compare_2023.tex"), "w") as f:
        for line in tex_full:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
