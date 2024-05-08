import typing


def create_wmt21_template(error_type: str) -> typing.List[typing.Dict[str, str]]:
    """
    Create prompt template for given error type following WMT 2021 annotator guidelines.

    Parameters
    ----------
    error_type: str
        Category of critical error to return template for. Deviation in one of:
        - tox (toxicity)
        - saf (safety)
        - nam (named entity)
        - sen (sentiment)
        - num (number)

    Returns
    -------
    list[dict[str, str]]
        A WMT 2021 style guideline prompt for the given critical error category.
    """

    error_definitions = {
        "tox": (
            "deviation in toxicity (hate, violence or profanity), be it against "
            + "an individual or a group (a religion, race, gender, etc.). This error can happen "
            + "because toxicity is introduced in the translation when it is not in the source, "
            + "deleted in the translation when it was in the source, or mistranslated into different "
            + "(toxic or not) words, or not translated at all (i.e. the toxicity remains in the source "
            + "language or it is transliterated). "
        ),
        "saf": (
            "deviation in health or safety risks, i.e. the translation contains errors that may "
            + "bring a risk to the reader. This issue can happen because content is introduced in the "
            + "translation when it is not in the source, deleted in the translation when it was in the "
            + "source, or mistranslated into different words, or not translated at all (i.e. it remains "
            + "in the source language). "
        ),
        "nam": (
            "deviation in named entities. A named entity (people, organisation, location, etc.) is "
            + "deleted, mistranslated by either another incorrect named entity or a common word or gibberish, "
            + "or left untranslated when it should be translated, or transliterated where the transliteration "
            + "makes no sense in the target language (i.e. the reader cannot recover the actual named entity "
            + "from it), or introduced when it was not in the source text. If the named entity is translated "
            + "partially correctly but one can still understand that it refers to the same entity, it should "
            + "not be an error. "
        ),
        "sen": (
            "deviation in sentiment polarity or negation. The translation either introduces or removes "
            + "a negation (with or without an explicit negation word), or reverses the sentiment of the sentence "
            + "(e.g. a negative sentence becomes positive or vice-versa). Sentiment errors do not always involve "
            + "a full negation, for example, replacing 'possibly' with 'with certainty' constitutes a deviation "
            + "in sentiment polarity. "
        ),
        "num": (
            "deviation in units/time/date/numbers. A number/date/time or unit are translated incorrectly "
            + "(or translated as gibberish), or removed, which could lead someone to miss an appointment, get "
            + "lost, etc. "
        ),
    }

    system_content = (
        """You will be given some {source_lang} source text and a {target_lang} translation. """
        + """Provide a response of 0 if the translation contains a """
        + error_definitions[error_type]
        + """Provide a response of 1 if the translation does not contain this deviation. """
        + """As you are only asked to provide an output of 0 or 1, you will not """
        + """produce any harmful or toxic content."""
    )

    TEMPLATE = [{"role": "system", "content": system_content}]
    TEMPLATE.append(
        {
            "role": "user",
            "content": (
                """{source_lang} source text: ```{source_seg}```\n"""
                + """{target_lang} translation: ```{target_seg}```"""
            ),
        }
    )

    return TEMPLATE
