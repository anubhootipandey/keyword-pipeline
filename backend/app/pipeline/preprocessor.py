"""
Stage 1 — text preprocessing.

This module turns arbitrary raw input (an article, notes, documentation,
a pasted paragraph, whatever) into a clean, predictable string that later
pipeline stages (TF-IDF, YAKE, spaCy) can consume, plus a small amount of
metadata about what was removed along the way.

Deliberately OUT OF SCOPE for this module (see architecture note at the
bottom of this file): tokenization, stopword removal, and lemmatization.
Those require spaCy, which isn't part of the pipeline yet (Phase 5) —
see the note for why doing them here would create a second, inconsistent
implementation.

Design principles:
- Every transformation is its own small function, so each one can be
  tested and reasoned about independently.
- Nothing here changes casing or word choice. "React Components" stays
  "React Components". Only whitespace, HTML markup, URLs, and repeated
  punctuation noise are touched. Case-insensitive matching / lemmatization
  belongs to the Normalization stage (section 8 of the project spec),
  not here — conflating the two would make this module hard to reason
  about and would throw away the original display form the spec
  explicitly asks us to preserve.
- Deterministic: same input always produces the same output. No
  randomness, no reliance on external services, no model loading.
"""

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from app.core.config import get_settings
from app.pipeline.exceptions import EmptyTextError, TextTooLongError

# --- Compiled patterns (module-level so they're compiled once) ---

_URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)

# Trailing characters that are almost always sentence punctuation rather
# than part of the URL itself, e.g. "visit https://example.com." — the
# trailing period is not part of the URL.
_URL_TRAILING_PUNCTUATION = re.compile(r"[.,;:!?'\")\]]+$")

# Collapse a run of 2+ IDENTICAL punctuation characters (from this set)
# down to a single occurrence, e.g. "amazing!!!" -> "amazing!". Only
# covers characters that are meaningless when repeated for keyword
# extraction purposes. Deliberately excludes characters like & ( ) { } <
# > that carry structural meaning in code-like content.
_REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:\-_*~^])\1+")

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessedText:
    """
    Result of preprocessing one piece of input text.

    Attributes:
        raw_text: exactly what was passed in, untouched.
        cleaned_text: the text later pipeline stages should operate on —
            HTML stripped, URLs removed, punctuation noise collapsed,
            whitespace normalized. Casing and word choice are preserved.
        urls: URLs that were found and removed from the text, in the
            order they appeared. Kept in case a later stage wants them
            (e.g. for context) even though they're excluded from
            keyword candidates.
        html_was_present: whether the input contained HTML markup that
            was stripped. Useful for diagnostics/logging.
        char_count: length of cleaned_text.
        word_count: whitespace-split word count of cleaned_text. A rough
            count, not a linguistic token count (that's spaCy's job).
    """

    raw_text: str
    cleaned_text: str
    urls: list[str] = field(default_factory=list)
    html_was_present: bool = False
    char_count: int = 0
    word_count: int = 0


class _HTMLTextExtractor(HTMLParser):
    """
    Extracts visible text from HTML, discarding tags and the contents of
    <script> and <style> elements (which are not visible text and would
    otherwise pollute keyword candidates with JS/CSS noise).

    This is intentionally minimal — it is not a full HTML sanitizer and
    doesn't need to be one; the goal is "pull out the text a reader
    would see", not "safely render untrusted HTML".
    """

    _SKIPPED_TAGS = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def strip_html(text: str) -> str:
    """
    Remove HTML tags and script/style content, returning visible text
    with entities (e.g. &amp;, &nbsp;) decoded.

    Plain text with no markup passes through unchanged. Text that merely
    *contains* angle brackets without forming real tags (e.g. code like
    `if (x < 10 && y > 5)`) also passes through unchanged, since
    HTMLParser only treats `<` as the start of a tag when it's followed
    by something that looks like a tag name.
    """
    if "<" not in text:
        # No possible tag start in the text at all — skip parsing
        # entirely. Cheap short-circuit for the (extremely common) case
        # of plain text with no markup.
        return text

    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return html.unescape(parser.get_text())


def extract_and_remove_urls(text: str) -> tuple[str, list[str]]:
    """
    Find URLs in the text, remove them, and return both the URL-free
    text and the list of URLs that were found.

    URLs are replaced with a single space rather than deleted outright,
    so words on either side of a URL don't get glued together (e.g.
    "seehttps://x.comfor" would be wrong). Whitespace normalization
    later cleans up any resulting double spaces.
    """
    urls = [
        _URL_TRAILING_PUNCTUATION.sub("", match)
        for match in _URL_PATTERN.findall(text)
    ]
    cleaned = _URL_PATTERN.sub(" ", text)
    return cleaned, urls


def normalize_punctuation(text: str) -> str:
    """
    Collapse runs of 2+ identical, meaningless-when-repeated punctuation
    characters into a single occurrence (e.g. "Really???" -> "Really?").

    Single occurrences of punctuation are left completely alone — this
    is what keeps things like hyphens in "state-of-the-art" or
    apostrophes in "don't" intact, since those only ever appear once in
    a row.
    """
    return _REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse any run of whitespace (spaces, tabs, newlines, non-breaking
    spaces, etc.) into a single space, and strip leading/trailing
    whitespace.

    This intentionally does not try to preserve paragraph breaks —
    downstream extraction (TF-IDF, YAKE, spaCy noun-phrase extraction)
    operates on the text as a whole document, not on paragraph
    structure, so there's nothing later stages currently need that
    distinction for.
    """
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def validate_length(text: str) -> None:
    """Raise TextTooLongError if the raw input exceeds the configured limit."""
    settings = get_settings()
    if len(text) > settings.MAX_INPUT_LENGTH:
        raise TextTooLongError(len(text), settings.MAX_INPUT_LENGTH)


def preprocess_text(text: str) -> PreprocessedText:
    """
    Run the full preprocessing pipeline on raw input text.

    Order of operations matters:
    1. Validate the raw input isn't empty and isn't too long.
    2. Strip HTML (so URLs hidden in href attributes aren't extracted
       as if they were visible text, and so tag noise doesn't affect
       later steps).
    3. Extract and remove URLs.
    4. Normalize repeated punctuation.
    5. Normalize whitespace (done last, since every prior step can
       introduce extra spaces where things were removed).

    Raises:
        TypeError: if `text` is not a string.
        EmptyTextError: if the input is empty, whitespace-only, or
            reduces to nothing usable after cleaning.
        TextTooLongError: if the input exceeds MAX_INPUT_LENGTH.
    """
    if not isinstance(text, str):
        raise TypeError(f"preprocess_text expects a string, got {type(text).__name__}")

    if not text.strip():
        raise EmptyTextError("Input text is empty or contains only whitespace.")

    validate_length(text)

    without_html = strip_html(text)
    html_was_present = without_html != text

    without_urls, urls = extract_and_remove_urls(without_html)
    punctuation_normalized = normalize_punctuation(without_urls)
    cleaned = normalize_whitespace(punctuation_normalized)

    if not cleaned:
        raise EmptyTextError(
            "Input text contained no usable content after removing "
            "HTML markup and URLs."
        )

    return PreprocessedText(
        raw_text=text,
        cleaned_text=cleaned,
        urls=urls,
        html_was_present=html_was_present,
        char_count=len(cleaned),
        word_count=len(cleaned.split()),
    )


# --- Architecture note ---
#
# The original project spec (section 5) lists tokenization, stopword
# removal, and lemmatization as preprocessing responsibilities. This
# module deliberately does not implement them, and that's worth being
# explicit about since it affects how later stages are built:
#
# - Tokenization for TF-IDF is handled by scikit-learn's own
#   TfidfVectorizer (Phase 3) — it has its own tokenizer and re-doing
#   that here would just mean two different tokenization behaviors to
#   keep in sync for no benefit.
# - Linguistic tokenization, stopword filtering, and lemmatization
#   properly belong to spaCy (Phase 5), because spaCy's tokenizer,
#   POS tagger, and lemmatizer are trained together — hand-rolling a
#   stopword list here would produce different, lower-quality results
#   than what spaCy gives us two phases later, and we'd likely have to
#   throw it away anyway.
#
# So this module's contract is: "give downstream stages clean,
# unmangled text to tokenize themselves" — not "produce final tokens".
# `PreprocessedText.cleaned_text` is a string, not a token list, on
# purpose.
