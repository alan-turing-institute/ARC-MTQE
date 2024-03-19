import argparse
import os
import pickle

from mtqe.data.loaders import comet_format, load_da_test_data
from mtqe.models.loaders import load_comet_model
from mtqe.utils.language_pairs import (
    LI_LANGUAGE_PAIRS_WMT_22_DA,
    LI_LANGUAGE_PAIRS_WMT_23_DA,
)
from mtqe.utils.paths import PREDICTIONS_DIR

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get directory names")

    parser.add_argument("-m", "--model", required=True, help="Model")
    parser.add_argument("-y", "--year", required=True, help="Year")

    return parser.parse_args()


def create_output_dir(pred_dir: str = PREDICTIONS_DIR) -> str:
    """
    Create directory for results and return path.
    """
    out_dir = os.path.join(pred_dir, "da_test_data")
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def main():
    """
    Make predictions for DA test data in the given WMT year using one of the models in the COMET family.
    """

    args = parse_args()

    model_name = args.model
    model = load_comet_model(model_name)

    year = args.year
    lps = LI_LANGUAGE_PAIRS_WMT_22_DA if year == "2022" else LI_LANGUAGE_PAIRS_WMT_23_DA

    out_dir = create_output_dir()

    for lp in lps:
        print(f"{model_name} predictions for WMT {year} {lp}")

        out_file_name = os.path.join(out_dir, f"{year}_{lp}_{model_name}")
        if os.path.exists(out_file_name):
            print(f"{out_file_name} already exists, skipping...")
            continue

        data = load_da_test_data(lp, year)
        comet_data = comet_format(data)
        model_output = model.predict(comet_data, batch_size=8, gpus=0)
        with open(out_file_name, "wb") as f:
            pickle.dump(model_output.scores, f)


if __name__ == "__main__":
    main()
