import hashlib
import os

import git

from mtqe.utils.paths import ROOT_DIR


def hash_file(filepath: str):
    """
    Hash the contents of a file.

    Parameters
    ----------
    filepath : str
        A string pointing to the file you want to hash
    m : hashlib hash object, optional (default is None to create a new object)
        hash_file updates m with the contents of filepath and returns m

    Returns
    -------
    hashlib hash object
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


def get_git_commit_hash(repo_path: str = ROOT_DIR):
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
