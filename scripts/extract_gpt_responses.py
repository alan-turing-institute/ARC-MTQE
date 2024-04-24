import os
import pickle

# import openai
import pandas as pd

from mtqe.llms.query import parse_mqm_answer
from mtqe.utils.language_pairs import LI_LANGUAGE_PAIRS_WMT_21_CED
from mtqe.utils.paths import PREDICTIONS_DIR

PROMPTS = ["basic", "GEMBA"]
TEST_ANSWERS_DIR = os.path.join(PREDICTIONS_DIR, "gpt_answers", "test")
# make sure we are loading the right files
# - the datetime stamps indicate when a run started
# - for the basic prompt, we run all language pairs at once
# - for the GEMBA prompt, we run this one prompt at a time
TIMESTAMPS = {
    "basic": {
        "en-cs": "20240418_185515",
        "en-de": "20240418_185515",
        "en-zh": "20240418_185515",
        "en-ja": "20240418_185515",
    },
    "GEMBA": {
        "en-cs": "20240419_133646",
        "en-de": "20240419_152230",
        "en-zh": "20240422_082655",
        "en-ja": "20240422_102626",
    },
}


def main():
    """
    Save key info from GPT responses to test CED data as CSV files.
    """

    for prompt_type in PROMPTS:
        for lp in LI_LANGUAGE_PAIRS_WMT_21_CED:
            RESPONSES_DIR = os.path.join(TEST_ANSWERS_DIR, prompt_type, lp)
            all_files = os.listdir(RESPONSES_DIR)
            data = {
                "idx": [],
                "created": [],
                "model": [],
                "finish_reason": [],
                "role": [],
                "content": [],
                "llm_pred": [],
            }
            for f in all_files:
                if TIMESTAMPS[prompt_type][lp] in f:
                    fp = os.path.join(RESPONSES_DIR, f)
                    # print(fp)
                    with open(fp, "rb") as obj:
                        response = pickle.load(obj)
                    content = response.choices[0].message.content

                    data["idx"].append(f.split("_")[-1].split(".")[0])
                    data["created"].append(response.created)
                    data["model"].append(response.model)
                    data["finish_reason"].append(response.choices[0].finish_reason)
                    data["role"].append(response.choices[0].message.role)
                    data["content"].append(content)

                    if prompt_type == "basic":
                        score = int(content)
                    elif prompt_type == "GEMBA":
                        parsed_response = parse_mqm_answer(content)
                        # use COMET style scoring: 1=meaning preserved, 0=critical error
                        score = 1 if len(parsed_response["critical"]) == 0 else 0
                    data["llm_pred"].append(score)

            df = pd.DataFrame(data)
            df.to_csv(os.path.join(PREDICTIONS_DIR, "ced_data", f"{lp}_test_llm_{prompt_type}_prompt_full_data.csv"))


if __name__ == "__main__":
    main()
