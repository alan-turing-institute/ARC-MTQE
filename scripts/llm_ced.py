import argparse
import os
import pickle
import typing

import openai
import pandas as pd

from mtqe.data.loaders import load_ced_data
from mtqe.llms.gemba import TEMPLATE_GEMBA_MQM
from mtqe.llms.query import apply_template, parse_mqm_answer
from mtqe.utils.format import create_now_str
from mtqe.utils.language_pairs import (
    DI_IOS_TO_LANGUAGE_NAME,
    LI_LANGUAGE_PAIRS_WMT_21_CED,
)
from mtqe.utils.paths import PREDICTIONS_DIR


def parse_args():
    """
    Construct argument parser.
    """

    parser = argparse.ArgumentParser(description="Get experiment config settings")

    parser.add_argument("-p", "--prompt", required=True, help="Prompt type ('basic' or 'GEMBA').")
    parser.add_argument("-d", "--data", required=True, help="Data split to make predictions for ('dev' or 'test').")
    parser.add_argument(
        "-l", "--lp", required=True, help="Language pair to make predictions for (e.g., 'en-cs'), can also be 'all'."
    )
    parser.add_argument(
        "-n", "--number", required=True, help="Number of translations in the dataset to make predictions for."
    )
    parser.add_argument(
        "-m", "--model", required=True, help="The name of the OpenAI model to be used, e.g., 'gpt-3.5-turbo'"
    )

    return parser.parse_args()


def gpt_predict(
    data_split: str = "dev",
    lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED,
    n: int = 2,
    prompt_type: str = "basic",
    openai_model: str = "gpt-3.5-turbo",
) -> typing.Dict[str, typing.List[int]]:
    """
    Make predictions for dev or test data.

    Parameters
    ----------
    data_split: str
        Whether to make predictions for train, dev or test data. Defaults to 'dev'".
    lps: list[str]
        List of WMT21 language-pairs to make predictions for. Defaults to all.
    n: int
        The number of translations for each language pair to make critical error
        predictions for. Will always pick the first n sentences. Defaults to 2.
    prompt_basic: str
        One of "basic" or "GEMBA".
    openai_model: str
        The name of the OpenAI model to be used. Defaults to 'gpt-3.5-turbo'.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of predictions for each language pair of the form:
        {<lp1>: [<score 1>, ...], ...}
    """

    assert prompt_type in [
        "basic",
        "GEMBA",
    ], f"Invalid prompt_type {prompt_type} provided, must be one of 'basic' or 'GEMBA'..."
    assert data_split in [
        "train",
        "dev",
        "test",
    ], f"Invalid data_split {data_split} provided, must be one of 'train', 'dev' or 'test'..."
    assert "OPENAI_API_KEY" in os.environ, "Environment variable `OPENAI_API_KEY` has not been set."

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # use to create directory name to save full GPT answers in
    now_str = create_now_str()

    for lp in lps:
        print(lp)
        # get source and target language names
        src, target = lp.split("-")
        src_name = DI_IOS_TO_LANGUAGE_NAME[src]
        target_name = DI_IOS_TO_LANGUAGE_NAME[target]

        # save full GPT answer as well as ERROR (0)/NOT ERROR (1) predictions made by the model in a CSV file
        predictions = []
        responses_dir = os.path.join(PREDICTIONS_DIR, "gpt_answers", data_split, prompt_type, lp)
        os.makedirs(responses_dir, exist_ok=True)

        df_data = load_ced_data(data_split, lp)
        for _, row in df_data[:n].iterrows():
            data = {
                "source_lang": src_name,
                "source_seg": row["src"],
                "target_lang": target_name,
                "target_seg": row["mt"],
            }
            if prompt_type == "basic":
                # basic template is the default
                messages = apply_template(data)
            elif prompt_type == "GEMBA":
                messages = apply_template(data, template=TEMPLATE_GEMBA_MQM)
            # print(messages)

            response = openai.chat.completions.create(model=openai_model, messages=messages, temperature=0, seed=1234)
            with open(os.path.join(responses_dir, f'{now_str}_{row["idx"]}.obj'), "wb") as fp:
                pickle.dump(response, fp)

            content = response.choices[0].message.content
            if prompt_type == "basic":
                answer = int(content)
            elif prompt_type == "GEMBA":
                parsed_response = parse_mqm_answer(content)
                # use COMET style scoring: 1=meaning preserved, 0=critical error
                answer = 1 if len(parsed_response["critical"]) == 0 else 0
            predictions.append(answer)
            print("Label: " + str(row["score"]), " GPT response: " + str(answer))

        df_lp_preds = pd.DataFrame({"idx": df_data["idx"][:n], "llm_pred": predictions})
        df_lp_preds.to_csv(
            os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_{data_split}_llm_{prompt_type}_prompt.csv"), index=False
        )


def main():

    os.makedirs(os.path.join(PREDICTIONS_DIR, "ced_data"), exist_ok=True)

    args = parse_args()
    if args.lp == "all":
        lps = LI_LANGUAGE_PAIRS_WMT_21_CED
    else:
        lps = [args.lp]

    gpt_predict(data_split=args.data, lps=lps, n=int(args.number), prompt_type=args.prompt, openai_model=args.model)

    print("DONE!")


if __name__ == "__main__":
    main()
