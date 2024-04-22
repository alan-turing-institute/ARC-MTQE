import argparse
import os
import typing
from pathlib import Path

import pandas as pd
import yaml

from mtqe.data.loaders import comet_format, load_ced_data
from mtqe.models.loaders import load_model_from_file
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.models import get_model_name
from mtqe.utils.paths import CONFIG_DIR, PREDICTIONS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Get experiment config settings")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name")
    parser.add_argument("-e", "--exp", required=True, help="Experiment name")
    parser.add_argument("-s", "--seed", required=True, help="Seed")
    parser.add_argument("-p", "--path", required=True, help="The path of the checkpoint")
    parser.add_argument("-d", "--data", required=True, help="Data split to make predictions for ('dev' or 'test').")
    parser.add_argument(
        "-l", "--lp", required=True, help="Language pair to make predictions for (e.g., 'en-cs'), can also be 'all'."
    )

    return parser.parse_args()


def supervised_predict(
    experiment_group_name: str,
    experiment_name: str,
    seed: int,
    checkpoint_path: str,
    config_dir: str = CONFIG_DIR,
    data_split: str = "dev",
    lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED,
) -> None:
    """
    Make predictions for dev or test data.

    Parameters
    ----------
    experiment_group_name: str
        The name of the group of experiments
    experiment_name: str
        The name of the experiment
    seed: int
        The initial random seed value
    checkpoint_path: str
        The path of the checkpoint to be loaded to make predictions
    config_dir: str
        The directory where the config files are stored
    data_split: str
        Whether to make predictions for train, dev or test data. Defaults to 'dev'".
    lps: list[str]
        List of WMT21 language-pairs to make predictions for. Defaults to all.
    """

    model_name = get_model_name(experiment_group_name, experiment_name, seed)

    checkpoint_path = Path(checkpoint_path)

    assert model_name[:-15] == os.path.split(os.path.dirname(checkpoint_path))[-1][:-15], (
        "Error" + model_name + os.path.dirname(checkpoint_path)
    )

    with open(os.path.join(config_dir, experiment_group_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)

    config["model_path"] = {"path": checkpoint_path}

    # Check that the experiment name is in the yaml file - the load won't work otherwise.
    assert experiment_name in config["experiments"], (
        experiment_name + " does not exist in " + experiment_group_name + ".yaml"
    )
    assert int(seed) in config["seeds"], "seed " + str(seed) + " does not exist in " + experiment_group_name + ".yaml"

    model = load_model_from_file(config, experiment_name, train_model=False)

    # save results here
    out_dir = os.path.join(PREDICTIONS_DIR, "ced_test_data")
    os.makedirs(out_dir, exist_ok=True)

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        out_file_name = os.path.join(
            out_dir, f"{lp}_" + data_split + "_" + os.path.split(os.path.dirname(checkpoint_path))[-1] + ".csv"
        )

        # load data
        df_data = load_ced_data(data_split, lp)
        comet_data = comet_format(df_data)

        # predict
        model_output = model.predict(comet_data, batch_size=8, gpus=0)

        # save output
        df_results = pd.DataFrame({"idx": df_data["idx"], "comet_score": model_output.scores})
        df_results.to_csv(out_file_name, index=False)

    return model


# def main():
#     """
#     Make predictions for WMT 2021 CED test data using COMETKiwi 2022.
#     """

#     # COMETKiwi 2022
#     model = load_comet_model()

#     # save results here
#     out_dir = os.path.join(PREDICTIONS_DIR, "ced_test_data")
#     os.makedirs(out_dir, exist_ok=True)

#     # make predictions for all language pairs listed here
#     for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
#         out_file_name = os.path.join(out_dir, f"{lp}_cometkiwi.csv")
#         if os.path.exists(out_file_name):
#             print(f"{out_file_name} already exists, skipping...")
#             continue

#         # load data
#         df_data = load_ced_test_data(lp)
#         comet_data = comet_format(df_data)

#         # predict
#         model_output = model.predict(comet_data, batch_size=8, gpus=0)

#         # save output
#         df_results = pd.DataFrame({"idx": df_data["idx"], "comet_score": model_output.scores})
#         df_results.to_csv(out_file_name, index=False)


def main():

    os.makedirs(os.path.join(PREDICTIONS_DIR, "ced_data"), exist_ok=True)

    args = parse_args()
    experiment_group_name = args.group
    experiment_name = args.exp
    seed = args.seed
    checkpoint_path = args.path
    data_split = args.data
    if args.lp == "all":
        lps = LI_LANGUAGE_PAIRS_WMT_21_CED
    else:
        lps = [args.lp]

    supervised_predict(
        experiment_group_name,
        experiment_name,
        seed,
        checkpoint_path=checkpoint_path,
        data_split=data_split,
        lps=lps,
    )


if __name__ == "__main__":
    main()
