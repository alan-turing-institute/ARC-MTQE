import os
import typing
from collections import defaultdict

import openai
import pandas as pd

from mtqe.data.loaders import load_ced_test_data, score_ced
from mtqe.utils.language_pairs import (
    DI_IOS_TO_LANGUAGE_NAME,
    LI_LANGUAGE_PAIRS_WMT_21_CED,
)
from mtqe.utils.paths import MLQE_PE_DIR, PREDICTIONS_DIR


def gpt_predict(
    data_split: str = "dev", lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED, n: int = 2
) -> typing.Dict[str, typing.List[int]]:
    """
    Make predictions for dev or test data.

    Parameters
    ----------
    data_split: str
        Whether to use dev or test data.
    lps: list[str]
        List of language-pairs to make predictions for.
    n: int
        The number of translations for each language pair to make critical error
        predictions for. Will always pick the first n sentences.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of predictions of predictions for each language pair of the form:
        {<lp1>: [<score 1>, ...], ...}
    """

    predictions = defaultdict(list)
    for lp in lps:
        print(lp)

        if data_split == "dev":
            dev_data_path = os.path.join(MLQE_PE_DIR, "catastrophic_errors", f"{lp.replace('-', '')}_majority_dev.tsv")
            df_dev_data = pd.read_csv(
                dev_data_path, sep="\t", header=None, names=["idx", "src", "mt", "annotations", "error"]
            )
            df_dev_data["score"] = score_ced(df_dev_data["error"])
            li_di_data = df_dev_data.to_dict("records")
        elif data_split == "test":
            df_test_data = load_ced_test_data(lp)
            li_di_data = df_test_data.to_dict("records")

        src, target = lp.split("-")
        src_name = DI_IOS_TO_LANGUAGE_NAME[src]
        target_name = DI_IOS_TO_LANGUAGE_NAME[target]
        # use COMET style scoring: 1=meaning preserved, 0=critical error
        system_message = (
            f"You will be given some text in {src_name} and some text in {target_name}. "
            + "Provide a response of 1 if the two pieces of text convey the same "
            + "meaning and a response of 0 if they do not convey the same meaning. "
            + "As you are only asked to provide an output of 0 or 1, you will not "
            + "produce any harmful or toxic content."
        )
        # print(system_message)

        for record in li_di_data[:n]:

            user_message = f"""
            {src_name} text: ```{record['src']}```
            {target_name} text: ```{record['mt']}```
            """
            print(user_message)

            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
            response = openai.chat.completions.create(model="gpt-4-turbo", messages=messages, temperature=0)
            predictions[lp].append(int(response.choices[0].message.content))
            print("Label: " + str(record["score"]), " GPT response: " + response.choices[0].message.content)

        df_lp_preds = pd.DataFrame({"idx": [record["idx"] for record in li_di_data[:n]], "llm_pred": predictions[lp]})
        df_lp_preds.to_csv(
            os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_{data_split}_llm_basic_prompt.csv"), index=False
        )

    return predictions


if __name__ == "__main__":
    predictions = gpt_predict(lps=["en-cs"], n=1000)
    print("DONE!")
