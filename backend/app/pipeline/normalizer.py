import re
import unicodedata

_POSSESSIVE_SUFFIX_PATTERN = re.compile(r"['\u2019]s\b")

_TRAILING_APOSTROPHE_PATTERN = re.compile(r"(?<=s)['\u2019]$")

_WHITESPACE_PATTERN = re.compile(r"\s+")

_LEADING_NON_WORD_PATTERN = re.compile(r"^[^\w]+", re.UNICODE)
_TRAILING_NON_WORD_PATTERN = re.compile(r"[^\w]+$", re.UNICODE)


def _strip_possessive(token: str) -> str:
    token = _POSSESSIVE_SUFFIX_PATTERN.sub("", token)
    token = _TRAILING_APOSTROPHE_PATTERN.sub("", token)
    return token


def normalize_for_matching(phrase: str) -> str:
    if not isinstance(phrase, str):
        raise TypeError(f"normalize_for_matching expects a string, got {type(phrase).__name__}")

    text = unicodedata.normalize("NFKC", phrase)
    text = text.casefold()

    tokens = [_strip_possessive(tok) for tok in text.split()]
    text = " ".join(tokens)

    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _LEADING_NON_WORD_PATTERN.sub("", text)
    text = _TRAILING_NON_WORD_PATTERN.sub("", text)

    return text