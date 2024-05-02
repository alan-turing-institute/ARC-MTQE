import hashlib
import os
import typing

import git
import pandas as pd
from pandas.util import hash_pandas_object

from mtqe.utils.paths import ROOT_DIR


def hash_df(df: pd.DataFrame) -> typing.List[str]:
    """
    Hash contents of a DataFrame by row.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame to hash.

    Returns
    -------
    list[str]
        Hash of the DataFrame object by row.
    """

    return list(hash_pandas_object(df))


def hash_file(filepath: str) -> str:
    """
    Hash contents of a file.

    Parameters
    ----------
    filepath : str
        A string pointing to the file you want to hash

    Returns
    -------
    str
        Hash of the file.
    """

    assert os.path.exists(filepath), "Path {} does not exist".format(filepath)

    m = hashlib.sha512()

    with open(filepath, "rb") as f:
        # The following construction lets us read f in chunks,
        # instead of loading an arbitrary file in all at once.
        while True:
            b = f.read(2**10)
            if not b:
                break
            m.update(b)

    return m.hexdigest()


def get_git_commit_hash(repo_path: str = ROOT_DIR) -> str:
    """
    Get commit digest for current HEAD commit. If the current working directory
    contains uncommitted changes, raises `RepositoryDirtyError`.

    Parameters
    ----------
    repo_path: str
        Path to a git repository.

    Returns
    -------
    str
        Git commit digest for the current HEAD commit of the git repository.
    """

    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise git.InvalidGitRepositoryError("provided code directory is not a valid git repository")

    if repo.is_dirty():
        raise git.RepositoryDirtyError(repo, "git repository contains uncommitted changes")

    return repo.head.commit.hexsha
