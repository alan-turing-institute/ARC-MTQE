from mtqe.data.loaders import save_ced_data_to_csv
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED


def main():
    """
    Save Critical Error Detection train and dev data to CSV files.
    """

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        for data_split in ["train", "dev"]:
            save_ced_data_to_csv(data_split, lp)


if __name__ == "__main__":
    main()
