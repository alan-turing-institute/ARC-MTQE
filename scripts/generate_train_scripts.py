import argparse
import os

import yaml
from jinja2 import Environment, FileSystemLoader

from mtqe.utils.paths import CONFIG_DIR, SLURM_DIR, TEMPLATES_DIR


def parse_args():
    """
    Construct argument parser.
    """
    parser = argparse.ArgumentParser(description="Get experiment group name")

    parser.add_argument("-g", "--group", required=True, help="Experiment group name")

    return parser.parse_args()


def write_slurm_script(
    account_name: str,
    slurm_config: dict,
    template_name: str = "slurm_train_template.sh",
    template_dir: str = TEMPLATES_DIR,
):
    """
    Given config values, writes a slurm file to a specified location
    NOTE: the template name is currently given a default value of the only template

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
        python_call = config["python_call"]
        script_content = template.render(
            account_name=account_name,
            time=config["time"],
            memory=config["memory"],
            experiment_name=config["experiment_name"],
            python_call=python_call,
        )
        with open(config["script_name"], "w") as f:
            f.write(script_content)


def generate_scripts(experiment_group_name: str, config_dir: str = CONFIG_DIR, slurm_dir: str = SLURM_DIR):
    """
    This function generates slurm scripts that can then be run on an HPC cluster
    One script is generated for each model that will be trained
    One model is created for each experiment and seed value listed in the config file

    Parameters
    ----------
    experiment_group_name: str
        The name of the experiment group (should match a config yaml file name)
    config_dir: str
        The path where the config files are stored
    slurm_dir: str
        The path where the generated slurm files will be stored
    """
    # Load config
    with open(os.path.join(config_dir, experiment_group_name + ".yaml")) as stream:
        config = yaml.safe_load(stream)

    # path where the slurm scripts will be stored
    scripts_path = os.path.join(slurm_dir, experiment_group_name)
    # make the directory, if it doesn't already exist
    if not os.path.isdir(scripts_path):
        os.mkdir(scripts_path)
    # Account name for our HPC (Baskerville)
    account_name = config["slurm"]["account"]
    # Generate config for the slurm file for each model that will run
    slurm_config = [
        {
            "python_call": "python scripts/train_ced.py "
            + "--group "
            + experiment_group_name
            + f"--exp {experiment_name} "
            + f"--seed {seed}",
            "experiment_name": experiment_name + "__" + str(seed),
            "script_name": os.path.join(scripts_path, experiment_name + "__" + str(seed) + "__train.sh"),
            "time": config["experiments"][experiment_name]["slurm"]["time"],
            "memory": config["experiments"][experiment_name]["slurm"]["memory"],
        }
        for seed in config["seeds"]
        for experiment_name in config["experiments"]
    ]

    # Don't send through a template_name, just take the default of the only template
    # Will want to make the template_name a config value if we end up with more templates
    write_slurm_script(account_name, slurm_config)


def main():
    args = parse_args()
    experiment_group_name = args.group
    generate_scripts(experiment_group_name)


if __name__ == "__main__":
    main()
