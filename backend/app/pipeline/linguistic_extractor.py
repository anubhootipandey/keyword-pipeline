import re
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Span, Token

from app.core.config import get_settings
from app.pipeline.exceptions import EmptyTextError, LinguisticModelNotAvailableError


DEFAULT_MIN_PHRASE_LENGTH = 1
DEFAULT_MAX_PHRASE_LENGTH = 5
DEFAULT_MAX_CANDIDATES = 50

_BOUNDARY_STRIP_POS = {"DET", "PRON", "ADP", "CCONJ", "SCONJ"}

_POSSESSIVE_TAG = "POS"

_MULTISPACE_PATTERN = re.compile(r"\s+")

_nlp: Language | None = None


def _get_nlp() -> Language:
   
    global _nlp
    if _nlp is not None:
        return _nlp

    model_name = get_settings().SPACY_MODEL_NAME
    try:
        _nlp = spacy.load(model_name, exclude=["ner"])
    except OSError as exc:
        raise LinguisticModelNotAvailableError(model_name) from exc

    return _nlp


@dataclass(frozen=True)
class LinguisticCandidate:

    phrase: str
    normalized_phrase: str
    linguistic_signal: int
    normalized_score: float
    ngram_size: int
    root_pos: str
    source: str = "linguistic"


@dataclass(frozen=True)
class LinguisticExtractionResult:
    candidates: list[LinguisticCandidate] = field(default_factory=list)
    model_name: str = ""
    noun_chunk_count: int = 0


def _trim_chunk_boundaries(chunk: Span) -> list[Token]:
    tokens = list(chunk)

    def is_boundary_noise(tok: Token) -> bool:
        return tok.is_punct or tok.is_stop or tok.pos_ in _BOUNDARY_STRIP_POS

    start = 0
    while start < len(tokens) and is_boundary_noise(tokens[start]):
        start += 1

    end = len(tokens)
    while end > start and is_boundary_noise(tokens[end - 1]):
        end -= 1

    return tokens[start:end]


def _surface_form(tokens: list[Token]) -> str:
    text = "".join(tok.text_with_ws for tok in tokens)
    return _MULTISPACE_PATTERN.sub(" ", text).strip()


def _normalized_form(tokens: list[Token]) -> str:
    parts = [
        tok.lemma_.lower()
        for tok in tokens
        if not tok.is_punct and tok.tag_ != _POSSESSIVE_TAG
    ]
    return " ".join(parts)


def _minmax_normalize(values: list[int]) -> list[float]:
    if not values:
        return []

    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0] * len(values)

    return [(v - lo) / (hi - lo) for v in values]


def extract_linguistic_candidates(
    cleaned_text: str,
    *,
    min_phrase_length: int = DEFAULT_MIN_PHRASE_LENGTH,
    max_phrase_length: int = DEFAULT_MAX_PHRASE_LENGTH,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> LinguisticExtractionResult:
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"extract_linguistic_candidates expects a string, got {type(cleaned_text).__name__}"
        )
    if not cleaned_text.strip():
        raise EmptyTextError("Linguistic extraction requires non-empty text.")
    if min_phrase_length > max_phrase_length:
        raise ValueError(
            f"min_phrase_length ({min_phrase_length}) cannot be greater than "
            f"max_phrase_length ({max_phrase_length})."
        )

    nlp = _get_nlp()
    doc = nlp(cleaned_text)

    noun_chunks = list(doc.noun_chunks)

    accepted: dict[str, dict] = {}

    for chunk in noun_chunks:
        kept_tokens = _trim_chunk_boundaries(chunk)
        if not kept_tokens:
            continue

        word_count = sum(1 for t in kept_tokens if not t.is_punct)
        if not (min_phrase_length <= word_count <= max_phrase_length):
            continue

        normalized = _normalized_form(kept_tokens)
        if not normalized:
            continue

        surface = _surface_form(kept_tokens)

        if normalized not in accepted:
            accepted[normalized] = {
                "phrase": surface,
                "normalized_phrase": normalized,
                "count": 1,
                "ngram_size": word_count,
                "root_pos": chunk.root.pos_,
            }
        else:
            accepted[normalized]["count"] += 1

    if not accepted:
        return LinguisticExtractionResult(
            candidates=[],
            model_name=get_settings().SPACY_MODEL_NAME,
            noun_chunk_count=len(noun_chunks),
        )

    entries = list(accepted.values())
    counts = [entry["count"] for entry in entries]
    normalized_scores = _minmax_normalize(counts)

    candidates = [
        LinguisticCandidate(
            phrase=entry["phrase"],
            normalized_phrase=entry["normalized_phrase"],
            linguistic_signal=entry["count"],
            normalized_score=normalized_scores[i],
            ngram_size=entry["ngram_size"],
            root_pos=entry["root_pos"],
        )
        for i, entry in enumerate(entries)
    ]
    
    candidates.sort(key=lambda c: (-c.linguistic_signal, c.normalized_phrase))
    candidates = candidates[:max_candidates]

    return LinguisticExtractionResult(
        candidates=candidates,
        model_name=get_settings().SPACY_MODEL_NAME,
        noun_chunk_count=len(noun_chunks),
    )
