import os
import pickle
from datetime import datetime

from comet import download_model, load_from_checkpoint


def main():
    # the output score is noisy, not bounded, has no clear interpretation
    # it is recommended for ranking systems or translations of the same source
    model_path = download_model("Unbabel/wmt20-comet-qe-da")
    cometqe = load_from_checkpoint(model_path)

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    cometkiwi = load_from_checkpoint(model_path)

    # TAKES 35 MINUTES TO DONWLOAD
    # GOT AN ERROR TRYING TO RUN IT ON THE DATA
    # model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    # cometkiwi_xl = load_from_checkpoint(model_path)

    # path to ARC-MTQE directory
    main_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(
        main_dir, "data", "wmt-qe-2022-data", "test_data-gold_labels", "task1_da"
    )

    now = datetime.today().strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join(main_dir, "results", "comets_compare", now)
    os.makedirs(out_dir, exist_ok=True)

    language_pairs = ["en-cs", "en-ja", "en-mr"]

    for lp in language_pairs:
        print(lp)

        with open(os.path.join(data_dir, lp, "test.2022.mt")) as f:
            mt_data = f.read().splitlines()
        with open(os.path.join(data_dir, lp, "test.2022.src")) as f:
            src_data = f.read().splitlines()
        comet_data = [{"src": src, "mt": mt} for src, mt in zip(src_data, mt_data)]

        cometqe_output = cometqe.predict(comet_data, batch_size=8, gpus=0)
        with open(os.path.join(out_dir, f"cometqe_{lp}"), "wb") as f:
            pickle.dump(cometqe_output.scores, f)

        cometkiwi_output = cometkiwi.predict(comet_data, batch_size=8, gpus=0)
        with open(os.path.join(out_dir, f"cometkiwi_{lp}"), "wb") as f:
            pickle.dump(cometkiwi_output.scores, f)

        # cometkiwi_xl_output = cometkiwi_xl.predict(comet_data, batch_size=8, gpus=0)
        # with open(os.path.join(out_dir, f"cometkiwi_xl_{lp}"), "wb") as f:
        #     pickle.dump(cometkiwi_xl_output.scores, f)


if __name__ == "__main__":
    main()
