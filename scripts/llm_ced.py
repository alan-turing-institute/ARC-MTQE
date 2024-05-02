import argparse
import os
import pickle
import typing

import openai
import pandas as pd

from mtqe.data.loaders import load_ced_data
from mtqe.llms.annotator_guidelines import create_wmt21_template
from mtqe.llms.gemba import TEMPLATE_GEMBA_MQM
from mtqe.llms.query import apply_template, parse_mqm_answer
from mtqe.utils.format import create_now_str
from mtqe.utils.language_pairs import (
    DI_IOS_TO_LANGUAGE_NAME,
    LI_LANGUAGE_PAIRS_WMT_21_CED,
)
from mtqe.utils.paths import PREDICTIONS_DIR

ERROR_CATEGORIES = ["tox", "saf", "nam", "sen", "num"]


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


def wmt21_prompt(
    data: typing.Dict[str, str],
    idx: str,
    responses_dir: str,
    now_str: str,
    openai_model: str,
    err_categories: typing.List[str] = ERROR_CATEGORIES,
) -> typing.Tuple[str, typing.Dict[str, str]]:
    """
    Use WMT 2021 CED subtask annotator guidelines as prompts. Iteratively ask to identify
    each of the 5 critical error categories (stop if a critical error has been found).

    Parameters
    ----------
    data: dict[str, str]
        A dictionary with the following keys:
            - source_lang
            - source_seg
            - target_lang
            - target_seg
    idx: str
        A unique identifier.
    responses_dir: str
        Path to directory where to store GPT responses.
    now_str: str
        A datetime string.
    openai_model: str
        The name of the OpenAI model to be used (e.g., 'gpt-3.5-turbo').
    err_categories: list[str]
        List of critical error categories to query.

    Returns
    -------
    tuple(str, dict[str, str])
        Content of the last GPT response ("0" or "1") and a dictionary with data on responses
        to all the queried prompts.
    """

    response_data = {"idx": idx}

    for err_cat in err_categories:
        template = create_wmt21_template(err_cat)
        messages = apply_template(data, template)
        response = openai.chat.completions.create(model=openai_model, messages=messages, temperature=0, seed=1234)
        response_data[err_cat] = err_cat
        with open(os.path.join(responses_dir, f"{now_str}_{idx}_{err_cat}.obj"), "wb") as fp:
            pickle.dump(response, fp)

        content = response.choices[0].message.content
        response_data[f"created_{err_cat}"] = response.created
        response_data[f"model_{err_cat}"] = response.model
        response_data[f"finish_reason_{err_cat}"] = response.choices[0].finish_reason
        response_data[f"role_{err_cat}"] = response.choices[0].message.role
        response_data[f"content_{err_cat}"] = content

        # stop if have found a critical error
        if content == "0":
            response_data["error_cat"] = err_cat
            response_data["score"] = int(content)
            return response_data
        # check that nothing unexpected is happening
        if content != "1":
            print(f"Invalid response for {idx}: ", content)

    # no critical error found
    response_data["error_cat"] = "none"
    response_data["score"] = int(content)
    return response_data


def gpt_predict(
    data_split: str = "dev",
    lps: typing.List[str] = LI_LANGUAGE_PAIRS_WMT_21_CED,
    n: int = 2,
    prompt_type: str = "basic",
    openai_model: str = "gpt-3.5-turbo",
    err_categories: typing.List[str] = ERROR_CATEGORIES,
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
        One of "basic", "GEMBA" or "wmt21_annotator".
    openai_model: str
        The name of the OpenAI model to be used. Defaults to 'gpt-3.5-turbo'.
    err_categories: list[str]
        List of critical error categories to query.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of predictions for each language pair of the form:
        {<lp1>: [<score 1>, ...], ...}
    """

    assert prompt_type in [
        "basic",
        "GEMBA",
        "wmt21_annotator",
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

        responses_dir = os.path.join(PREDICTIONS_DIR, "gpt_answers", data_split, prompt_type, lp)
        os.makedirs(responses_dir, exist_ok=True)
        predictions_dir = os.path.join(PREDICTIONS_DIR, "ced_data", f"prompt_{prompt_type}")
        os.makedirs(predictions_dir, exist_ok=True)

        df_data = load_ced_data(data_split, lp)
        # have slightly different keys for GEMBA and basic vs wmt21_annotator prompts data
        RESPONSE_KEYS = ["created", "model", "finish_reason", "role", "content"]
        response_data = {"idx": [], "score": []}
        if prompt_type == "wmt21_annotator":
            response_data["eror_cat"] = []
            for key in RESPONSE_KEYS:
                for err_cat in err_categories:
                    response_data[f"{key}_{err_cat}"] = []
        else:
            for key in RESPONSE_KEYS:
                response_data[key] = []

        for _, row in df_data[:n].iterrows():
            data = {
                "source_lang": src_name,
                "source_seg": row["src"],
                "target_lang": target_name,
                "target_seg": row["mt"],
            }

            if prompt_type == "wmt21_annotator":
                response_row = wmt21_prompt(data, row["idx"], responses_dir, now_str, openai_model)
                # save response data
                for key in response_data.keys():
                    # the wmt21_annotator prompt exits early if a critical error is found, leaving
                    # some columns empty
                    if key in response_row:
                        response_data[key].append(response_row[key])
                    else:
                        response_data[key].append("")
            else:
                if prompt_type == "basic":
                    # basic template is the default
                    messages = apply_template(data)
                elif prompt_type == "GEMBA":
                    messages = apply_template(data, template=TEMPLATE_GEMBA_MQM)

                response = openai.chat.completions.create(
                    model=openai_model, messages=messages, temperature=0, seed=1234
                )
                with open(os.path.join(responses_dir, f'{now_str}_{row["idx"]}.obj'), "wb") as fp:
                    pickle.dump(response, fp)
                content = response.choices[0].message.content

                # save response data
                response_data["idx"].append(row["idx"])
                response_data["created"].append(response.created)
                response_data["model"].append(response.model)
                response_data["finish_reason"].append(response.choices[0].finish_reason)
                response_data["role"].append(response.choices[0].message.role)
                response_data["content"].append(content)

                if prompt_type == "GEMBA":
                    parsed_response = parse_mqm_answer(content)
                    # use COMET style scoring: 1=meaning preserved, 0=critical error
                    score = 1 if len(parsed_response["critical"]) == 0 else 0
                else:
                    # both basic and wmt21_annotator prompts return 0/1 respoonses
                    score = int(content)
                response_data["score"].append(score)

        df_lp_preds = pd.DataFrame(response_data)
        df_lp_preds.to_csv(
            os.path.join(
                PREDICTIONS_DIR, "ced_data", prompt_type, f"{lp}_{data_split}_llm_{prompt_type}_prompt_full_data.csv"
            ),
            index=False,
        )


def main():

    args = parse_args()
    if args.lp == "all":
        lps = LI_LANGUAGE_PAIRS_WMT_21_CED
    else:
        lps = [args.lp]

    os.makedirs(os.path.join(PREDICTIONS_DIR, "ced_data", args.prompt), exist_ok=True)
    gpt_predict(data_split=args.data, lps=lps, n=int(args.number), prompt_type=args.prompt, openai_model=args.model)

    print("DONE!")


if __name__ == "__main__":
    main()
