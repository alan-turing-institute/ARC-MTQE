from mtqe.data.loaders import load_ced_data
from mtqe.utils.data import get_token_length_columns
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED


def main():

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:

        df = load_ced_data("train", lp)
        df_ = get_token_length_columns(df)

        print(lp, df_["token_lengths"].min(), df_["token_lengths"].max())


if __name__ == "__main__":
    main()
