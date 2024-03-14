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
    "Min source segment length",  # Length is calculated by number of characters (not num of words)
    "Median source segment length",
    "Mean source segment length",
    "Max source segment length",
    "Min target segment length",
    "Median target segment length",
    "Mean target segment length",
    "Max target segment length",
]


def extract_annotations(df: pd.DataFrame) -> pd.DataFrame:
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

    Parameters:
    df - a dataframe with the column named 'annotations', the data in the column is expected to
         be in the format [a, b, c] where a, b and c are the scores from three annotators

    Returns:
    df - the dataframe with three new columns containing the scores for the three annotators
    """
    df["annotations"] = df["annotations"].str.replace("[", "")
    df["annotations"] = df["annotations"].str.replace("]", "")
    df[["ann_1", "ann_2", "ann_3"]] = df["annotations"].str.split(",", expand=True)
    for annotator in ["ann_1", "ann_2", "ann_3"]:
        df[annotator] = df[annotator].str.strip()
    return df


def define_agreement(row: pd.Series) -> int:
    """
    Takes a row CED data and returns 1 if all three annotators give a record the same score

    Parameters:
    row - a row of the CED dataframe

    Returns:
    1 if the three annotations are the same, 0 otherwise
    """
    if row["ann_1"] == row["ann_2"] == row["ann_3"]:
        return 1
    else:
        return 0


def create_latex_table(
    summary: dict,
    dataset_annotations: dict = DI_DATASET_ANNOTATIONS,
    language_pairs: list = DI_LANGUAGE_PAIRS,
    row_names: list = LI_ROW_NAMES,
) -> list:
    """
    Create Latex table given some summary data

    Parameters:
    summary - a dictionary where each key is a language pair, each value is itself a dictionary
              where the keys are the names of the datasets (e.g., train, dev, test) and the values
              correspond to metrics listed in order of row_names
    dataset_annotations - a dictionary where the keys are the names of the datasets (e.g., train)
              and the values are boolean with TRUE meaning there are scores recorded by individual annotator
    language_pairs - a dictionary where the keys are short names for each language pair (e.g., en-cs)
              and the values are the long names (e.g., English-Czech)
    row_names - a list of the names of the rows to add to the table

    Returns:
    tex_full - a list containing the content of a latex table with one entry per row of the table

    """
    num_columns = len(dataset_annotations) * len(language_pairs)
    textabular = f"c|{'c'*num_columns}"
    texheader1 = ""
    texheader2 = ""
    texdata = row_names.copy()
    for lp in language_pairs:
        texheader1 += " & \\multicolumn{3}{c}{" + language_pairs[lp] + "}"
        for dataset in dataset_annotations:
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


def load_data(language_pairs: dict = DI_LANGUAGE_PAIRS) -> dict:
    """
    Placeholder function for loading data - will be replaced

    Parameters:
    language_pairs - a dictionary where the keys are short names for each language pair (e.g., en-cs) and the
                     values are the long names (e.g., English-Czech)

    Returns:
    ced_data - a dictionary containing the dataframes for each dataset
    """

    ced_data = {}  # dictionary to store the dataframes for each dataset

    # Load data by language pair
    for lp in language_pairs:

        path_dev = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_dev.tsv")
        path_train = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_train.tsv")
        path_test = os.path.join(DATA_DIR, "catastrophic_errors", f"{lp}_majority_test_blind.tsv")
        path_goldlabels = os.path.join(
            DATA_DIR, "catastrophic_errors_goldlabels", f"{lp}_majority_test_goldlabels", "goldlabels.txt"
        )

        df_dev = pd.read_csv(path_dev, sep="\t", header=None, names=["idx", "source", "target", "annotations", "label"])
        df_train = pd.read_csv(
            path_train, sep="\t", header=None, names=["idx", "source", "target", "annotations", "label"]
        )
        df_test = pd.read_csv(path_test, sep="\t", header=None, names=["idx", "source", "target"])
        df_labels = pd.read_csv(path_goldlabels, sep="\t", header=None, names=["lang_pair", "ref", "idx", "label"])
        df_test_labelled = pd.merge(df_test, df_labels, on="idx")
        ced_data[lp] = {"train": df_train, "dev": df_dev, "test": df_test_labelled}

    return ced_data


def get_sentence_stats(column: pd.Series) -> tuple[str, str, str, str]:
    """
    Takes a column of a dataframe and returns basic statistics about the length of text in the column

    Parameters:
    column - column (series) of a dataframe

    Returns:
    median_len - the median length of text in the column over all rows formatted as a string to 2dp
    mean_len - the mean length of text in the column over all rows formatted as a string to 2dp
    min_len - the minimum length of text in the column over all rows formatted as a string to 2dp
    max_len - the maximum length of text in the column over all rows formatted as a string to 2dp
    """

    column_len = column.str.len()

    median_len = "{:.2f}".format(column_len.median())
    mean_len = "{:.2f}".format(column_len.mean())
    min_len = "{:.2f}".format(column_len.min())
    max_len = "{:.2f}".format(column_len.max())

    return median_len, mean_len, min_len, max_len


def summarise_data(
    ced_data: dict, dataset_annotations: dict = DI_DATASET_ANNOTATIONS, language_pairs: list = DI_LANGUAGE_PAIRS
) -> dict:
    """
    Function that collects the data statistics by language pair and dataset type

    Parameters:
    ced_data - dictionary of the CED data where the keys are the langague pairs and the values are themselves
               dictionaries where the keys are the names of the datasets (e.g., train) and the values are dataframes
    dataset_annotations - a dictionary where the keys are the names of the datasets (e.g., train)
              and the values are boolean with TRUE meaning there are scores recorded by individual annotator
    language_pairs - a dictionary where the keys are short names for each language pair (e.g., en-cs)
              and the values are the long names (e.g., English-Czech)

    Returns:
    data_summary - dictionary of the calculated metrics where the keys are the langague pairs and the values
                   are themselves dictionaries where the keys are the names of the datasets (e.g., train) and
                   the values are summarised metrics

    """
    data_summary = {}  # dictionary for recording summary metrics

    for lp in language_pairs:
        data_summary[lp] = {}
        for dataset in dataset_annotations:
            df = ced_data[lp][dataset]

            # Calculate values for all datasets
            n_rows = df.shape[0]
            n_median_source_length, n_mean_source_length, n_min_source_length, n_max_source_length = get_sentence_stats(
                df["source"]
            )
            n_median_target_length, n_mean_target_length, n_min_target_length, n_max_target_length = get_sentence_stats(
                df["target"]
            )
            n_critical_errors = df[df["label"] == "ERR"].shape[0]
            p_critical_errors = "{:.2f}".format(100 * n_critical_errors / n_rows)

            if dataset_annotations[dataset]:
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
            summary.append(n_min_source_length)
            summary.append(n_median_source_length)
            summary.append(n_mean_source_length)
            summary.append(n_max_source_length)
            summary.append(n_min_target_length)
            summary.append(n_median_target_length)
            summary.append(n_mean_target_length)
            summary.append(n_max_target_length)
            data_summary[lp][dataset] = summary

    return data_summary


def perform_eda():
    """
    Loads the data, calculates summary metrics then writes data to file in the format of a latex table
    """

    ced_data = load_data()

    data_summary = summarise_data(ced_data)

    # create table
    latex_table = create_latex_table(data_summary)

    # save table to file
    with open("eda_summary.tex", "w") as f:
        for line in latex_table:
            f.write(f"{line}\n")


if __name__ == "__main__":
    perform_eda()
