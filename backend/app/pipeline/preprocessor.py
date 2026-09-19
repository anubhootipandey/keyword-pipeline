import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from app.core.config import get_settings
from app.pipeline.exceptions import EmptyTextError, TextTooLongError

_URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)

_URL_TRAILING_PUNCTUATION = re.compile(r"[.,;:!?'\")\]]+$")

_REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:\-_*~^])\1+")

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessedText:

    raw_text: str
    cleaned_text: str
    urls: list[str] = field(default_factory=list)
    html_was_present: bool = False
    char_count: int = 0
    word_count: int = 0


class _HTMLTextExtractor(HTMLParser):
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
    if "<" not in text:
        return text

    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return html.unescape(parser.get_text())


def extract_and_remove_urls(text: str) -> tuple[str, list[str]]:
    urls = [
        _URL_TRAILING_PUNCTUATION.sub("", match)
        for match in _URL_PATTERN.findall(text)
    ]
    cleaned = _URL_PATTERN.sub(" ", text)
    return cleaned, urls


def normalize_punctuation(text: str) -> str:
    return _REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def validate_length(text: str) -> None:
    settings = get_settings()
    if len(text) > settings.MAX_INPUT_LENGTH:
        raise TextTooLongError(len(text), settings.MAX_INPUT_LENGTH)


def preprocess_text(text: str) -> PreprocessedText:
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
