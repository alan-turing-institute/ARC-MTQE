import os

import pandas as pd
from comet import download_model, load_from_checkpoint


def main():
    # COMETKiwi 2022
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    # path to ARC-MTQE directory
    main_dir = os.path.dirname(os.path.dirname(__file__))

    # WMT 2021 critical error test data
    data_dir = os.path.join(main_dir, "data", "mlqe-pe", "data", "catastrophic_errors")

    # save results here
    out_dir = os.path.join(main_dir, "predictions", "ced_test_data")
    os.makedirs(out_dir, exist_ok=True)

    # make predictions for all language pairs listed here
    language_pairs = ["encs", "ende", "enja", "enzh"]
    for lp in language_pairs:

        out_file_name = os.path.join(out_dir, f"{lp}_cometkiwi.csv")
        if os.path.exists(out_file_name):
            print(f"{out_file_name} already exists, skipping...")
            continue

        # load data
        path_data = os.path.join(data_dir, f"{lp}_majority_test_blind.tsv")
        df_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "source", "target"])

        # format for COMETKiwi input: [{"src":"...", "mt":"..."}, {...}]
        comet_data = []
        for i, row in df_data.iterrows():
            comet_data.append({"src": row["source"], "mt": row["target"]})

        # predict
        model_output = model.predict(comet_data, batch_size=8, gpus=0)

        # save output
        df_results = pd.DataFrame({"idx": df_data["idx"], "comet_score": model_output.scores})
        df_results.to_csv(out_file_name, index=False)


if __name__ == "__main__":
    main()
