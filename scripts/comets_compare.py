import os
import pickle
import typing

from comet import download_model, load_from_checkpoint

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# lists include DA not MQM data
LANGUAGE_PAIRS_22 = ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"]
LANGUAGE_PAIRS_23 = ["en-gu", "en-hi", "en-mr", "en-ta", "he-en"]


def create_output_dir(root_dir: str = ROOT_DIR) -> str:
    """
    Create directory for results and return path.
    """
    out_dir = os.path.join(root_dir, "predictions", "da_test_data")
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def load_test_data(lp: str, year: str, root_dir: str = ROOT_DIR) -> typing.Dict[str, typing.List[str]]:
    """
    Load 2023 data for language pair and year formatted as COMET input.
    """

    if year == "2022":
        data_dir = os.path.join(root_dir, "data", "wmt-qe-2022-data", "test_data-gold_labels", "task1_da")

        with open(os.path.join(data_dir, lp, "test.2022.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(data_dir, lp, "test.2022.src")) as f:
            src_data = f.read().splitlines()

    elif year == "2023":
        data_dir = os.path.join(root_dir, "data", "wmt-qe-2023-data", "test_data_2023", "task1_sentence_level")

        with open(os.path.join(data_dir, lp, f"test.{lp.replace('-', '')}.final.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(data_dir, lp, f"test.{lp.replace('-', '')}.final.src")) as f:
            src_data = f.read().splitlines()

    return [{"src": src, "mt": mt} for src, mt in zip(src_data, mt_data)]


def load_model(model_name: str = "cometkiwi_22"):
    """
    Return one of the COMET models (COMETKiwi22 by default).
    """
    if model_name == "comet_qe":
        model_path = download_model("Unbabel/wmt20-comet-qe-da")
    elif model_name == "cometkiwi_22":
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    # elif model_name == "cometkiwi_23_xl":
    #     model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    return model


def main():
    """ """

    out_dir = create_output_dir()

    for model_name in ["comet_qe", "cometkiwi_22"]:
        model = load_model(model_name)
        for year in ["2022", "2023"]:
            lps = LANGUAGE_PAIRS_22 if year == "2022" else LANGUAGE_PAIRS_23
            for lp in lps:
                print(f"{model_name} predictions for WMT {year} {lp}")

                out_file_name = os.path.join(out_dir, f"{year}_{lp}_{model_name}")
                if os.path.exists(out_file_name):
                    print(f"{out_file_name} already exists, skipping...")
                    continue

                comet_data = load_test_data(lp, year)
                model_output = model.predict(comet_data, batch_size=8, gpus=0)
                with open(out_file_name, "wb") as f:
                    pickle.dump(model_output.scores, f)


if __name__ == "__main__":
    main()
