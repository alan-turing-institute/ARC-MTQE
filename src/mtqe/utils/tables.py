import typing


def create_latex_table(col_names: typing.List, results: typing.Dict[str, typing.List[float]], dp: int = 3) -> str:
    """
    Create simple latex table of results of the form:
        | col 1 | col 2 |
    -----------------------
    row 1 |  ...  |  ...  |
    row 2 |  ...  |  ...  |
    -----------------------
    For example, each row is a model and column a language pair.
    Parameters
    ----------
    col_names: list
        List of column names.
    results: dict[str, list[float]]
        {row name: [<score for col 1>, ...], ...}
    dp: int
        Number of decimal places to round floats to, defaults to 3.
    Returns
    ----------
    str
        A latex table of results.
    """

    textabular = f"c|{'c'*len(col_names)}"
    texheader = " & " + " & ".join(map(lambda x: x.upper(), col_names)) + "\\\\"
    texdata = "\\hline\n"
    for label, values in results.items():
        texdata += f"{label} & {' & '.join(map(lambda n: f'%.{dp}f'%n, values))} \\\\\n"
    texdata += "\\hline"
    tex_full = [
        "\\begin{table}",
        "\\centering",
        "\\begin{tabular}{" + textabular + "}",
        texheader,
        texdata,
        "\\end{tabular}",
        "\\end{table}",
    ]

    tex_full = (
        "\\begin{table}\n"
        "\\centering\n"
        "\\begin{tabular}{" + textabular + "}\n" + texheader + "\n" + texdata + "\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return tex_full
