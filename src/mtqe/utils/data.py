import pandas as pd
from transformers import XLMRobertaTokenizerFast

XLMRL_TOKENIZER = XLMRobertaTokenizerFast.from_pretrained("microsoft/infoxlm-large")


def compute_token_length(text: str, tokenizer: XLMRobertaTokenizerFast = XLMRL_TOKENIZER) -> int:
    """
    Tokenize text and return token count.

    Parameters
    ----------
    text: str
        Text to encode.
    tokenizer: XLMRobertaTokenizerFast
        Tokenizer.

    Returns
    -------
    int
        Number of tokens.
    """

    return len(tokenizer(text)["input_ids"])


def get_token_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenize source and MT text and compute number of tokens.
    Add token counts as columns to dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with `src` and `mt` columns.

    Returns
    -------
    pd.DataFrame
        The input dataframe with 3 new columns:
            - `src_token_len`: number of source text tokens
            - `mt_token_len`: number of MT text tokens
            - `token_lengths`: sum of source and MT tokens
    """

    df["src_token_len"] = df["src"].apply(compute_token_length)
    df["mt_token_len"] = df["mt"].apply(compute_token_length)
    df["token_lengths"] = df["src_token_len"] + df["mt_token_len"]

    return df
