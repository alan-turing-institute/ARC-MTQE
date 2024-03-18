from comet import download_model, load_from_checkpoint

from mtqe.utils.paths import COMET_QE_21


def load_comet_model(model_name: str = "cometkiwi_22", comet_qe_21: str = COMET_QE_21):
    """
    Return one of the COMET models (COMETKiwi22 by default).
    """
    if model_name == "comet_qe":
        model_path = download_model("Unbabel/wmt20-comet-qe-da")
    elif model_name == "comet_qe_21":
        model_path = comet_qe_21
    elif model_name == "cometkiwi_22":
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    elif model_name == "cometkiwi_23_xl":
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")

    model = load_from_checkpoint(model_path)

    return model
