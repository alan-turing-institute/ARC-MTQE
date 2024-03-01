import os

import pandas as pd

# path to the ARC-MTQE directory
MAIN_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(MAIN_DIR, "arc-mtqe/data/mlqe-pe/data")
DI_LANGUAGE_PAIRS = {
    "encs": "English-Czech",
    "ende": "English-German",
    "enja": "English-Japanese",
    "enzh": "English-Chinese",
}
# Values indicate whether the dataset has scores by annotator or not
DI_DATASET_ANNOTATIONS = {"train": True, "dev": True, "test": False}
# List of names for each row of the latex table that will be created
LI_ROW_NAMES = [
    "Num rows",
    "Num rows with agreement",  # Number of rows where all 3 annotators agree on the score
    "Percentage total rows with agreement",  # Percentage of all rows where all 3 annotators agree on the score
    "Num critical errors",  # Number of critical errors in the dataset
    "Percentage critical errors",  # Percentage of all rows that are critical errors
    "Num critical errors with agreement",  # Num of critical errors where all 3 annotators scored the translation as 1
    "Percentage critical errors with agreement",  # Percentage of all critical errors where all 3 annotators agree
    "Num critical errors - Annotator 1",  # Number of records that annotator 1 scored as a critical error
    "Num critical errors - Annotator 2",  # Number of records that annotator 2 scored as a critical error
    "Num critical errors - Annotator 3",  # Number of records that annotator 3 scored as a critical error
    "Median source segment length",  # Length is calculated by number of characters (not num of words)
    "Mean source segment length",
    "Median target segment length",
    "Mean target segment length",
]


def extract_annotations(df):
    """
    Takes a dataframe of CED data an extracts the individual annotator scores
    into new columns - for these data there are three annotators per translation.
    Note that the annotations in the new columns remain as strings, rather than
    integers which is deliberate as we don't want to sum the values for any analysis
    as the possible values for annotations are:
    '0' - no error
    '1' - critical error as defined in https://aclanthology.org/2021.wmt-1.71.pdf
    '2' - source text unintelligible
    '3' - translated text contains too many errors to be annotated
    """
    df["annotations"] = df["annotations"].str.replace("[", "")
    df["annotations"] = df["annotations"].str.replace("]", "")
    df[["ann_1", "ann_2", "ann_3"]] = df["annotations"].str.split(",", expand=True)
    for annotator in ["ann_1", "ann_2", "ann_3"]:
        df[annotator] = df[annotator].str.strip()
    return df


def define_agreement(row):
    """
    Returns 1 if all three annotators give a record the same score
    """
    if row["ann_1"] == row["ann_2"] == row["ann_3"]:
        return 1
    else:
        return 0


def create_latex_table(summary):
    """
    Create Latex table given some summary data
    """
    num_columns = len(DI_DATASET_ANNOTATIONS) * len(DI_LANGUAGE_PAIRS)
    textabular = f"c|{'c'*num_columns}"
    texheader1 = ""
    texheader2 = ""
    texdata = LI_ROW_NAMES.copy()
    for lp in DI_LANGUAGE_PAIRS:
        texheader1 += " & \\multicolumn{3}{c}{" + DI_LANGUAGE_PAIRS[lp] + "}"
        for dataset in DI_DATASET_ANNOTATIONS:
            texheader2 += " & " + dataset
            data = summary[lp][dataset]
            for ind in range(len(data)):
                texdata[ind] += " & " + data[ind]
    for ind in range(len(texdata)):
        texdata[ind] += "\\\\\n"
    texdata[-1] += "\\hline"
    texheader1 += "\\\\\n"
    texheader2 += "\\\\\\hline\n"

    tex_full = [
        "\\begin{table}",
        "\\centering",
        "\\begin{adjustbox}{width=1\\textwidth}",
        "\\begin{tabular}{" + textabular + "}",
        texheader1,
        texheader2,
    ]

    tex_full.extend(texdata)

    tex_full.extend(
        [
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\end{table}",
        ]
    )

    return tex_full


di_ced_data = {}  # dictionary to store the dataframes for each dataset

# Load data by language pair
for lp in DI_LANGUAGE_PAIRS:
    n_rows = 0
    n_segments = 0
    n_critical_errors = 0

    path_dev = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_dev.tsv")
    path_train = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_train.tsv")
    path_test = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_test_blind.tsv")
    path_goldlabels = os.path.join(
        DATA_DIR, "catastrophic_errors_goldlabels", f"{lp}_majority_test_goldlabels", "goldlabels.txt"
    )

    df_dev = pd.read_csv(path_dev, sep="\t", header=None, names=["idx", "source", "target", "annotations", "label"])
    df_train = pd.read_csv(path_train, sep="\t", header=None, names=["idx", "source", "target", "annotations", "label"])
    df_test = pd.read_csv(path_test, sep="\t", header=None, names=["idx", "source", "target"])
    df_labels = pd.read_csv(path_goldlabels, sep="\t", header=None, names=["lang_pair", "ref", "idx", "label"])
    df_test_labelled = pd.merge(df_test, df_labels, on="idx")
    di_ced_data[lp] = {"train": df_train, "dev": df_dev, "test": df_test_labelled}

di_data_summary = {}  # dictionary for recording summary metrics

for lp in DI_LANGUAGE_PAIRS:
    di_data_summary[lp] = {}
    for dataset in DI_DATASET_ANNOTATIONS:
        df = di_ced_data[lp][dataset]

        # Calculate values for all datasets
        n_rows = df.shape[0]
        n_median_source_length = "{:.2f}".format(df["source"].str.len().median())
        n_mean_source_length = "{:.2f}".format(df["source"].str.len().mean())
        n_median_target_length = "{:.2f}".format(df["target"].str.len().median())
        n_mean_target_length = "{:.2f}".format(df["target"].str.len().mean())
        n_critical_errors = df[df["label"] == "ERR"].shape[0]
        p_critical_errors = "{:.2f}".format(100 * n_critical_errors / n_rows)

        if DI_DATASET_ANNOTATIONS[dataset]:
            # These values are only calculated if there are scores by annotator
            df = extract_annotations(df)
            df["agree"] = df.apply(define_agreement, axis=1)
            df["error_agree"] = (df["agree"] == 1) & (df["label"] == "ERR")
            n_rows_agreement = df[df["agree"] == 1].shape[0]
            p_rows_agreement = "{:.2f}".format(100 * n_rows_agreement / n_rows)
            n_critical_errors_agreement = df[df["error_agree"] == 1].shape[0]
            p_critical_errors_agreement = "{:.2f}".format(100 * n_critical_errors_agreement / n_critical_errors)
            n_critical_errors1 = df[df["ann_1"] == "1"].shape[0]
            n_critical_errors2 = df[df["ann_2"] == "1"].shape[0]
            n_critical_errors3 = df[df["ann_3"] == "1"].shape[0]
        else:
            n_rows_agreement = "-"
            p_rows_agreement = "-"
            n_critical_errors_agreement = "-"
            p_critical_errors_agreement = "-"
            n_critical_errors1 = "-"
            n_critical_errors2 = "-"
            n_critical_errors3 = "-"

        # A temp list to store the summary data
        summary = []
        # Note that the order the data are added to the summary list must match
        # the order of the metrics listed in LI_ROW_NAMES otherwise the data
        # will appear in the wrong row in the latex table
        summary.append(str(n_rows))
        summary.append(str(n_rows_agreement))
        summary.append(p_rows_agreement)
        summary.append(str(n_critical_errors))
        summary.append(p_critical_errors)
        summary.append(str(n_critical_errors_agreement))
        summary.append(p_critical_errors_agreement)
        summary.append(str(n_critical_errors1))
        summary.append(str(n_critical_errors2))
        summary.append(str(n_critical_errors3))
        summary.append(n_median_source_length)
        summary.append(n_mean_source_length)
        summary.append(n_median_target_length)
        summary.append(n_mean_target_length)
        di_data_summary[lp][dataset] = summary

# create table
latex_table = create_latex_table(di_data_summary)

# save table to file
with open("eda_summary.tex", "w") as f:
    for line in latex_table:
        f.write(f"{line}\n")
