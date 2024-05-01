import argparse
import os
import typing
from pathlib import Path

import pandas as pd
import yaml
from torch import cuda

from mtqe.data.loaders import comet_format, load_ced_data
from mtqe.models.loaders import load_comet_model, load_model_from_file
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.models import get_model_name
from mtqe.utils.paths import CONFIG_DIR, PREDICTIONS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Get experiment config settings")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name")
    parser.add_argument("-e", "--exp", help="Experiment name")
    parser.add_argument("-s", "--seed", help="Seed")
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

    if experiment_group_name == "baseline":
        model = load_comet_model(checkpoint_path)
    else:
        # Get the model name given the experiment group name, experiment name, and seed
        model_name = get_model_name(experiment_group_name, experiment_name, seed)
        # Set the path to the checkpoint
        checkpoint_path = Path(checkpoint_path)
        # Make sure that the model name matches the checkpoint model name
        assert model_name[:-15] == os.path.split(os.path.dirname(checkpoint_path))[-1][:-15], (
            "Error, model name "
            + model_name
            + " does not match the checkpoint model name"
            + os.path.dirname(checkpoint_path)
        )
        # Load the config file - want to load the model with the same hparams as used for training
        # NOTE: There is no guarantee the config file has not been changed since training... this
        # could be made more robust by saving the hparams or config with the checkpoint?
        with open(os.path.join(config_dir, experiment_group_name + ".yaml")) as stream:
            config = yaml.safe_load(stream)
        # Set or overwrite the model path in the config file
        config["model_path"] = {"path": checkpoint_path}

        # Load the model from the config file, setting the `train_model` param to False
        model = load_model_from_file(config, experiment_name, train_model=False)

    # save results here
    out_dir = os.path.join(PREDICTIONS_DIR, "ced_data", experiment_group_name)
    os.makedirs(out_dir, exist_ok=True)

    for lp in lps:
        out_file_name = os.path.join(
            out_dir,
            f"{lp}_"
            + data_split
            + "_"
            + os.path.split(os.path.dirname(checkpoint_path))[-1]
            + "_"
            + os.path.basename(checkpoint_path)[:-5]
            + ".csv",
        )

        # load data
        df_data = load_ced_data(data_split, lp)
        comet_data = comet_format(df_data)

        # set number of gpus - is there a better way of doing this?
        if cuda.is_available():
            gpus = 1
        else:
            gpus = 0
        # predict
        model_output = model.predict(comet_data, batch_size=8, gpus=gpus)

        # save output
        if experiment_group_name == "baseline":
            df_results = pd.DataFrame({"idx": df_data["idx"], "score": model_output.scores})
        else:
            # NOTE: the sigmoid function has not been applied to this output.
            df_results = pd.DataFrame({"idx": df_data["idx"], "logits": model_output.scores})
        df_results.to_csv(out_file_name, index=False)

    return model


def main():

    args = parse_args()
    experiment_group_name = args.group
    checkpoint_path = args.path
    data_split = args.data

    if experiment_group_name == "baseline":
        experiment_name = None
        seed = None
    else:
        experiment_name = args.exp
        seed = args.seed

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
