"""
The code in this script is taken from the GEMBA package. The original can be found at:
- https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/gemba_mqm_utils.py

Use of the GEMBA-MQM prompting strategy must be attributed by citing:
@inproceedings{kocmi-federmann-2023-gemba-mqm,
    title = {GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4},
    author = {Kocmi, Tom  and Federmann, Christian},
    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
"""

import typing

FEW_SHOTS = {
    "ende": {
        "source_lang": "English",
        "source_seg": (
            "I do apologise about this, we must gain permission from the account holder to discuss "
            + "an order with another person, I apologise if this was done previously, however, I would"
            + "not be able to discuss this with yourself without the account holders permission."
        ),
        "target_lang": "German",
        "target_seg": (
            "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer "
            + "anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber "
            + "ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement."
        ),
        "answer": """Critical:
                no-error
                Major:
                accuracy/mistranslation - "involvement"
                accuracy/omission - "the account holder"
                Minor:
                fluency/grammar - "wäre"
                fluency/register - "dir"
                """,
    },
    "encs": {
        "source_lang": "English",
        "source_seg": (
            "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to "
            + "gauge the prospects of success after the latest exchanges in the stop-start negotiations."
        ),
        "target_lang": "Czech",
        "target_seg": (
            "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se "
            "snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních."
        ),
        "answer": """Critical:
                no-error
                Major:
                accuracy/addition - "ve Vídni"
                accuracy/omission - "the stop-start"
                Minor:
                terminology/inappropriate for context - "partaje"
                """,
    },
    "zhen": {
        "source_lang": "Chinese",
        "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
        "target_lang": "English",
        "target_seg": (
            "Urumqi Home Furnishing Store Channel provides you with the latest business information such as "
            + "the address, telephone number, business hours, etc., of high-speed rail, and find a "
            + "decoration company, and go to the reviews."
        ),
        "answer": """Critical:
                accuracy/addition - "of high-speed rail"
                Major:
                accuracy/mistranslation - "go to the reviews"
                Minor:
                style/awkward - "etc.,"
                """,
    },
}


def mqm_fewshot(few_shots: typing.List[typing.Dict[str, str]]) -> typing.List[typing.Dict[str, str]]:
    """
    Create GEMBA fewshot prompt template.

    Parameters
    ----------
    few_shots: dict
        A list of few shot examples, each a dictionary with the following keys:
            - source_lang
            - source_seg
            - target_lang
            - target_seg
            - answer

    Returns
    -------
    list[dict[str, str]]
        A list of GEMBA MQM template prompts including few shot examples.
    """

    prompts = [
        {
            "role": "system",
            "content": (
                "You are an annotator for the quality of machine translation. Your task is to identify errors "
                + "and assess the quality of the translation."
            ),
        }
    ]

    template = (
        """{source_lang} source:\n```{source_seg}```\n{target_lang} translation:\n```{target_seg}```\n\n"""
        + """Based on the source segment and machine translation surrounded with triple backticks, identify """
        + """error types in the translation and classify them. The categories of errors are: accuracy (addition, """
        + """mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, """
        + """punctuation, register, spelling), style (awkward), terminology (inappropriate for context, """
        + """inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three """
        + """categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors """
        + """disrupt the flow, but what the text is trying to say is still understandable. Minor errors are """
        + """technically errors, but do not disrupt the flow or hinder comprehension."""
    )

    for shot in few_shots:
        prompts.append({"role": "user", "content": template.format(**shot)})
        answer = shot["answer"]

        prompts.append({"role": "assistant", "content": answer})

    prompts.append({"role": "user", "content": template})

    return prompts


TEMPLATE_GEMBA_MQM = mqm_fewshot([FEW_SHOTS["ende"], FEW_SHOTS["encs"], FEW_SHOTS["zhen"]])
