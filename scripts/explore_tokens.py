import os

from mtqe.models.comet import CEDModel
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import PROCESSED_DATA_DIR


def main():

    for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
        train_paths = [os.path.join(PROCESSED_DATA_DIR, f"{lp}_majority_train.csv")]
        dev_paths = [os.path.join(PROCESSED_DATA_DIR, f"{lp}_majority_dev.csv")]

        model = CEDModel(
            batch_size=1,
            train_data=train_paths,
            validation_data=dev_paths,
            loss="cross_entropy",
            out_dim=2,
            input_segments=["mt", "src"],
        )

        lengths = []
        train_data_loader = model.train_dataloader()
        for b in train_data_loader:
            token_ids = b[0][0]["input_ids"][0]
            lengths.append(len(token_ids))

        lengths.sort()
        print(lp, min(lengths), max(lengths), lengths[-10:])


if __name__ == "__main__":
    main()
