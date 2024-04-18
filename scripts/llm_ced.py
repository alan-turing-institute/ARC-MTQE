import os
import typing
from collections import defaultdict

import openai
import pandas as pd

from mtqe.data.loaders import load_ced_test_data, score_ced
from mtqe.llms.gemba import TEMPLATE_GEMBA_MQM
from mtqe.llms.query import apply_template, parse_mqm_answer
from mtqe.utils.language_pairs import (
    DI_IOS_TO_LANGUAGE_NAME,
    LI_LANGUAGE_PAIRS_WMT_21_CED,
)
from mtqe.utils.paths import MLQE_PE_DIR, PREDICTIONS_DIR


def gpt_predict(
    data_split: str = "dev",
    lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED,
    n: int = 2,
    prompt_type: str = "basic",
) -> typing.Dict[str, typing.List[int]]:
    """
    Make predictions for dev or test data.

    Parameters
    ----------
    data_split: str
        Whether to make predictions for dev or test data. Defaults to dev".
    lps: list[str]
        List of WMT21 language-pairs to make predictions for. Defaults to all.
    n: int
        The number of translations for each language pair to make critical error
        predictions for. Will always pick the first n sentences. Defaults to 2.
    prompt_basic: str
        One of "basic" or "GEMBA".

    Returns
    -------
    dict[str, list[int]]
        Dictionary of predictions for each language pair of the form:
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

        # get source and target language names
        src, target = lp.split("-")
        src_name = DI_IOS_TO_LANGUAGE_NAME[src]
        target_name = DI_IOS_TO_LANGUAGE_NAME[target]

        # use COMET style scoring: 1=meaning preserved, 0=critical error
        if prompt_type == "basic":
            system_message = (
                f"You will be given some text in {src_name} and some text in {target_name}. "
                + "Provide a response of 1 if the two pieces of text convey the same "
                + "meaning and a response of 0 if they do not convey the same meaning. "
                + "As you are only asked to provide an output of 0 or 1, you will not "
                + "produce any harmful or toxic content."
            )
        elif prompt_type == "GEMBA":
            system_message = (
                "You are an annotator for the quality of machine translation. "
                + "Your task is to identify errors and assess the quality of the translation."
            )
        # print(system_message)

        for record in li_di_data[:n]:
            messages = [{"role": "system", "content": system_message}]
            data = {
                "source_lang": src_name,
                "source_seg": record["src"],
                "target_lang": target_name,
                "target_seg": record["mt"],
            }
            if prompt_type == "basic":
                # basic template is the default
                prompt = apply_template(data)
            elif prompt_type == "GEMBA":
                prompt = apply_template(data, template=TEMPLATE_GEMBA_MQM)
            messages.extend(prompt)

            print(messages)
            response = openai.chat.completions.create(model="gpt-4-turbo", messages=messages, temperature=0)
            content = response.choices[0].message.content

            if prompt_type == "basic":
                answer = int(content)
            elif prompt_type == "GEMBA":
                parsed_response = parse_mqm_answer(content)

                # as above, 1=NOT, 0=ERR
                answer = 1 if len(parsed_response["critical"]) == 0 else 0

            predictions[lp].append(answer)
            print("Label: " + str(record["score"]), " GPT response: " + str(answer))

        df_lp_preds = pd.DataFrame({"idx": [record["idx"] for record in li_di_data[:n]], "llm_pred": predictions[lp]})
        df_lp_preds.to_csv(
            os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_{data_split}_llm_{prompt_type}_prompt.csv"), index=False
        )

    return predictions


if __name__ == "__main__":
    os.makedirs(os.path.join(PREDICTIONS_DIR, "gpt_answers"), exist_ok=True)

    predictions = gpt_predict(lps=["en-cs"], n=3, prompt_type="basic")
    print("DONE!")
