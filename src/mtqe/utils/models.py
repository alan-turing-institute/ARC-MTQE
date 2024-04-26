from datetime import datetime


def get_model_name(experiment_group_name: str, experiment_name: str, seed: int) -> str:
    """
    Creates (as good as unique) model name using the current datetime stamp

    Parameters
    ----------
    experiment_group_name: str
        The name of the group of experiments
    experiment_name: str
        The name of the experiment
    seed: int
        The initial random seed value

    Returns
    ----------
    str
        A model name
    """
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    model_name = experiment_group_name + "__" + experiment_name + "__" + str(seed) + "__" + now_str
    return model_name
