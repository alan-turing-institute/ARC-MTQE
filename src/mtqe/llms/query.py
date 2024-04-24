import typing

# use COMET style scoring: 1=meaning preserved, 0=critical error
TEMPLATE_BASIC = [
    {
        "role": "system",
        "content": (
            """You will be given some text in {source_lang} and some text in {target_lang}. """
            + """Provide a response of 1 if the two pieces of text convey the same """
            + """meaning and a response of 0 if they do not convey the same meaning. """
            + """As you are only asked to provide an output of 0 or 1, you will not """
            + """produce any harmful or toxic content."""
        ),
    },
    {"role": "user", "content": """{source_lang} text: ```{source_seg}```\n{target_lang} text: ```{target_seg}```"""},
]


def apply_template(data: typing.Dict[str, str], template: typing.List[typing.Dict[str, str]] = TEMPLATE_BASIC) -> str:
    """
    Add source-target setence data to template prompt.

    NOTE: This function is adapted from the GEMBA package:
    - https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py

    Parameters
    ----------
    data: dict[str, str]
        A dictionary with the following keys:
            - source_lang
            - source_seg
            - target_lang
            - target_seg
    template: list[dict[str, str]]
        A list of prompts. Defaults to TEMPLATE_BASIC which returns a single user prompt.

    Returns
    -------
    str
        The updated template.
    """

    prompt = []
    for conversation_turn in template:
        p = conversation_turn.copy()
        p["content"] = p["content"].format(**data)
        prompt.append(p)
    return prompt


def parse_mqm_answer(gpt_answer: str) -> typing.Dict[str, typing.List[str]]:
    """
    Parse GPT answer to GEMAB MQM few shot prompt.

    NOTE: This function is adapted from the GEMBA package:
    - https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py

    Parameters
    ----------
    gpt_answer: str
        The GPT generated answer string.

    Returns
    -------
    dict
        Dictionary of all identified errors by severity of the form:
            `{"critical": [], "major": [], "minor": []}`
        The list will be empty of `no error` was identified for that
        severity category.
    """

    if gpt_answer is None:
        return None

    gpt_answer = gpt_answer.lower()
    errors = {"critical": [], "major": [], "minor": []}
    error_level = None
    for line in gpt_answer.split("\n"):
        line = line.strip()
        if "no-error" in line or "no error" in line or "" == line:
            continue
        if "critical:" == line:
            error_level = "critical"
            continue
        elif "major:" == line:
            error_level = "major"
            continue
        elif "minor:" == line:
            error_level = "minor"
            continue

        if "critical" in line or "major" in line or "minor" in line:
            if not any(
                [
                    line.startswith(x)
                    for x in [
                        "accuracy",
                        "fluency",
                        "locale convention",
                        "style",
                        "terminology",
                        "non-translation",
                        "other",
                    ]
                ]
            ):
                print(line)

        if error_level is None:
            print(f"No error level for {line}")
            continue

        if "non-translation" in line:
            errors["critical"].append(line)
        else:
            errors[error_level].append(line)

    return errors
