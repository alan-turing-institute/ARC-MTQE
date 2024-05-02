import argparse
import os

import yaml
from jinja2 import Environment, FileSystemLoader

from mtqe.utils.models import get_model_name
from mtqe.utils.paths import (
    CHECKPOINT_DIR,
    CONFIG_DIR,
    ROOT_DIR,
    SLURM_DIR,
    TEMPLATES_DIR,
)


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get experiment group name")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name")
    parser.add_argument("-e", "--exp", required=True, help="Experiment name")
    parser.add_argument("-d", "--data", required=True, help="Data split to make predictions for ('dev' or 'test').")
    parser.add_argument(
        "-l", "--lp", required=True, help="Language pair to make predictions for (e.g., 'en-cs'), can also be 'all'."
    )

    return parser.parse_args()


def write_slurm_script(
    account_name: str,
    slurm_config: dict,
    template_name: str = "slurm_pred_template.sh",
    template_dir: str = TEMPLATES_DIR,
):
    """
    Given config values, writes a slurm file to a specified location
    NOTE: the template name is currently given a default value of the only prediction template

    Parameters
    ----------
    account_name: str
        The name of the account for the HPC we are using
    slurm_config: dict
        A dictionary of config values for use in the slurm file
    template_name: str
        The name of the slurm template file to be used
    template_dir: str
        The path where the template can be found
    """
    environment = Environment(loader=FileSystemLoader(template_dir))
    template = environment.get_template(template_name)

    for config in slurm_config:
        python_call = ""
        for call in config["python_calls"]:
            python_call = python_call + call + "\n"

        script_content = template.render(
            account_name=account_name,
            time=config["time"],
            memory=config["memory"],
            experiment_name=config["experiment_name"],
            python_call=python_call,
        )
        with open(config["script_name"], "w") as f:
            f.write(script_content)


def generate_scripts(
    experiment_group_name: str,
    experiment_name: str,
    data_split: str,
    lp: str,
    config_dir: str = CONFIG_DIR,
    slurm_dir: str = SLURM_DIR,
    root_dir: str = ROOT_DIR,
):
    """
    This function generates slurm scripts that can then be run on an HPC cluster
    One script is generated for each experiment group with multiple predictions per script

    Parameters
    ----------
    experiment_group_name: str
        The name of the experiment group (should match a config yaml file name)
    experiment_name: str
        The name of the experiment name to be used
    data_split: str
        The data split to be used (expecting either 'dev' or 'test')
    lp: str
        The ISO code of the language pair to be used
    config_dir: str
        The path where the config files are stored
    slurm_dir: str
        The path where the generated slurm files will be stored
    root_dir: str
        The root directory for the project
    """
    # Load config
    with open(os.path.join(config_dir, experiment_group_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)

    # path where the slurm scripts will be stored
    scripts_path = os.path.join(slurm_dir, "pred_scripts", experiment_group_name)
    log_path = os.path.join(scripts_path, "slurm_pred_logs")
    # make the directory, if it doesn't already exist
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    # Account name for our HPC (Baskerville)
    account_name = config["slurm"]["account"]
    # Generate config for the slurm file for each model that will run
    slurm_config = [
        {
            "python_calls": [
                "python "
                + root_dir
                + "/scripts/predict_ced.py "
                + "--group "
                + experiment_group_name
                + " "
                + f"--exp {experiment_name} "
                + f"--seed {seed} "
                + "--path "
                + get_checkpoint_path(experiment_group_name, experiment_name, seed)
                + " "
                + f"--data {data_split} "
                + f"--lp {lp}"
                for seed in config["seeds"]
            ],
            "experiment_name": experiment_name + "__" + lp,
            "script_name": os.path.join(scripts_path, experiment_name + "__" + lp + "__pred.sh"),
            "time": "01:00:00",  # hard coded to one hour
            "memory": config["experiments"][experiment_name]["slurm"]["memory"],
        }
    ]

    # Don't send through a template_name, just take the default of the only template
    # Will want to make the template_name a config value if we end up with more templates
    write_slurm_script(account_name, slurm_config)


def get_checkpoint_path(
    experiment_group_name: str, experiment_name: str, seed: str, checkpoint_dir: str = CHECKPOINT_DIR
) -> str:
    """
    Returns the checkpoint path for a given exp group, exp name and seed combination

    Parameters
    ----------
    experiment_group_name: str
        The name of the experiment group
    experiment_name: str
        The name of the experiment
    seed: str
        The random seed (as a string type)
    checkpoint_dir:
        The directory where checkpoints are stored

    Returns
    -------
    str:
        The path to the checkpoint
    """
    folder_name_prefix = get_model_name(
        experiment_group_name=experiment_group_name, experiment_name=experiment_name, seed=seed
    )
    folder_name_prefix = folder_name_prefix[:-15]

    folders = [folder for folder in os.listdir(checkpoint_dir) if folder.startswith(folder_name_prefix)]

    assert len(folders) == 1, "More than one checkpoint folder exists for " + folder_name_prefix

    folder = folders[0]

    checkpoint_path = os.path.join(checkpoint_dir, folder)

    checkpoints = os.listdir(checkpoint_path)

    assert len(checkpoints) == 1, "More than one checkpoint exists in " + folder

    checkpoint = checkpoints[0]

    return os.path.join(checkpoint_path, checkpoint)


def main():
    args = parse_args()
    experiment_group_name = args.group
    experiment_name = args.exp
    data_split = args.data
    lp = args.lp
    generate_scripts(experiment_group_name, experiment_name, data_split, lp)


if __name__ == "__main__":
    main()
