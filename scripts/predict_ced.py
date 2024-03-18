import os

import pandas as pd

from mtqe.data.loaders import comet_format, load_ced_test_data
from mtqe.models.loaders import load_comet_model
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import PREDICTIONS_DIR


def main():
    """
    Make predictions for WMT 2021 CED test data using COMETKiwi 2022.
    """

    # COMETKiwi 2022
    model = load_comet_model()

    # save results here
    out_dir = os.path.join(PREDICTIONS_DIR, "ced_test_data")
    os.makedirs(out_dir, exist_ok=True)

    # make predictions for all language pairs listed here
    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        out_file_name = os.path.join(out_dir, f"{lp}_cometkiwi.csv")
        if os.path.exists(out_file_name):
            print(f"{out_file_name} already exists, skipping...")
            continue

        # load data
        df_data = load_ced_test_data(lp)
        comet_data = comet_format(df_data)

        # predict
        model_output = model.predict(comet_data, batch_size=8, gpus=0)

        # save output
        df_results = pd.DataFrame({"idx": df_data["idx"], "comet_score": model_output.scores})
        df_results.to_csv(out_file_name, index=False)


if __name__ == "__main__":
    main()
