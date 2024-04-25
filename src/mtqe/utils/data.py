import pandas as pd
from transformers import XLMRobertaTokenizerFast

XLMRL_TOKENIZER = XLMRobertaTokenizerFast.from_pretrained("microsoft/infoxlm-large")


def compute_token_length(text: str, tokenizer: XLMRobertaTokenizerFast = XLMRL_TOKENIZER) -> int:
    """
    Compute length of embedding tokens.

    Parameters
    ----------
    text: str
        Text to encode.
    tokenizer: XLMRobertaTokenizerFast
        Tokenizer.

    Returns
    -------
    int
        The number of tokens.
    """

    return len(tokenizer(text)["input_ids"])


def get_token_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute source and MT token lengths, add columns to dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with `src` and `mt` columns.

    Returns
    -------
    pd.DataFrame
        A dataframe with 3 new columns:
            - `src_token_len`
            - `mt_token_len`
            - `token_lengths`
    """

    df["src_token_len"] = df["src"].apply(compute_token_length)
    df["mt_token_len"] = df["mt"].apply(compute_token_length)
    df["token_lengths"] = df["src_token_len"] + df["mt_token_len"]

    return df
