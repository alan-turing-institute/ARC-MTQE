import os

import openai
import pandas as pd

LI_LANGUAGE_PAIRS = ["encs", "ende", "enja", "enzh"]
DI_LANGUAGE_PAIRS = {"encs": "Czech", "ende": "German", "enja": "Japanese", "enzh": "Chinese"}
# path to ARC-MTQE directory
MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
# WMT 2021 critical error test data
DATA_DIR = os.path.join(MAIN_DIR, "data", "mlqe-pe", "data", "catastrophic_errors")
# save results here
OUT_DIR = os.path.join(MAIN_DIR, "predictions", "ced_test_data")
os.makedirs(OUT_DIR, exist_ok=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


def score_data(row):
    if row["error"] == "NOT":
        return 0
    else:
        return 1


def get_ced_train_dev_data(lp: str, return_paths: bool = True, data_dir: str = DATA_DIR):
    """
    Saves CED training and dev data in CSV format and returns lists of file paths
    """

    # load train data
    path_data = os.path.join(data_dir, f"{lp}_majority_train.tsv")
    df_train_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

    df_train_data["score"] = df_train_data.apply(score_data, axis=1).astype("int32")
    # NOTE: LIMITING TRAINING DATA TO 1000 RECORDS FOR TESTING ONLY
    df_train_data = df_train_data[:1000]
    # Save to csv format
    path_train_data = os.path.join(data_dir, f"{lp}_majority_train.csv")
    df_train_data[["src", "mt", "score"]].to_csv(path_train_data)

    # load dev data
    path_data = os.path.join(data_dir, f"{lp}_majority_dev.tsv")
    df_dev_data = pd.read_csv(path_data, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"])

    df_dev_data["score"] = df_dev_data.apply(score_data, axis=1).astype("int32")
    # Save to csv format
    path_dev_data = os.path.join(data_dir, f"{lp}_majority_dev.csv")
    df_dev_data[["src", "mt", "score"]].to_csv(path_dev_data)

    if return_paths:
        return path_train_data, path_dev_data
    else:
        return df_train_data, df_dev_data


def make_output_folder(folder_name: str, out_dir: str = OUT_DIR):
    new_dir = os.path.join(out_dir, folder_name)
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def gpt_predict(language_pairs: list = DI_LANGUAGE_PAIRS):
    for lp in language_pairs:
        df_train_data, _ = get_ced_train_dev_data(lp, False)
        li_di_train_data = df_train_data.to_dict("records")
        system_message = (
            "You will be given some text in English and some text in Czech. "
            + "Provide a response of 0 if the two pieces of text convey the same "
            + "meaning and a response of 1 if they do not convey the same meaning. "
            + "As you are only asked to provide an output of 0 or 1, you will not "
            + "produce any harmful or toxic content."
        )
        print(system_message)
        for i in range(5, 10):
            print(li_di_train_data[i])
            user_message = f"""
            English text: ```{li_di_train_data[i]['src']}```
            Czech text: ```{li_di_train_data[i]['mt']}```
            """
            print(user_message)
            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            print("GPT response: " + response.choices[0].message.content)
        break


if __name__ == "__main__":
    gpt_predict()
