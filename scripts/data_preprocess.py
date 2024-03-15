import os

import numpy as np
import pandas as pd

from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_CED_21
from mtqe.utils.paths import DATA_CED_21_DIR


def main():
    """
    Save CED data in csv files.
    """

    for lp in LI_LANGUAGE_PAIRS_CED_21:

        for data_set in ["train", "dev"]:

            # load data
            path_data = os.path.join(DATA_CED_21_DIR, f"{lp.replace('-', '')}_majority_{data_set}.tsv")
            df_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

            # NOT en error = 1, CRITICAL ERROR = 0
            df_data["score"] = np.where(df_data["error"] == "NOT", 1, 0)

            # save to file
            path_train_data = os.path.join(DATA_CED_21_DIR, f"{lp.replace('-', '')}_majority_{data_set}.csv")
            df_data[["src", "mt", "score"]].to_csv(path_train_data)


if __name__ == "__main__":
    main()
