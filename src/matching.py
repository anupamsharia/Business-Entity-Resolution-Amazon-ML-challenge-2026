import re
from rapidfuzz import fuzz


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def token_set(text):
    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


def name_similarity(name1, name2):
    name1 = normalize_text(name1)
    name2 = normalize_text(name2)

    if not name1 or not name2:
        return 0.0

    return fuzz.WRatio(name1, name2)


def address_similarity(address1, address2):
    address1 = normalize_text(address1)
    address2 = normalize_text(address2)

    if not address1 or not address2:
        return 0.0

    return fuzz.WRatio(address1, address2)


def token_similarity(text1, text2):
    tokens1 = token_set(text1)
    tokens2 = token_set(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    if union == 0:
        return 0.0

    return 100.0 * intersection / union


def combined_score(
    name1,
    name2,
    address1,
    address2,
    country1,
    country2
):

    name_score = name_similarity(
        name1,
        name2
    )

    address_score = address_similarity(
        address1,
        address2
    )

    token_score = token_similarity(
        address1,
        address2
    )

    country_score = (
        100.0
        if (
            country1
            and country2
            and str(country1).strip().lower()
            == str(country2).strip().lower()
        )
        else 0.0
    )

    if address1 and address2:
        final_score = (
            0.65 * name_score
            + 0.20 * address_score
            + 0.10 * token_score
            + 0.05 * country_score
        )
    else:
        final_score = (
            0.80 * name_score
            + 0.20 * country_score
        )

    return round(final_score, 2)

