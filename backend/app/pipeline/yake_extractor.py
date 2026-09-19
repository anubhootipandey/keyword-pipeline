
from dataclasses import dataclass, field

import yake

from app.pipeline.exceptions import EmptyTextError


DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_NGRAM_SIZE = 3
DEFAULT_MIN_NGRAM_SIZE = 1
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_DEDUP_LIM = 0.9


@dataclass(frozen=True)
class YakeCandidate:

    phrase: str
    raw_score: float
    normalized_score: float
    ngram_size: int
    source: str = "yake"


@dataclass(frozen=True)
class YakeExtractionResult:

    candidates: list[YakeCandidate] = field(default_factory=list)
    language: str = DEFAULT_LANGUAGE
    max_ngram_size: int = DEFAULT_MAX_NGRAM_SIZE


def _normalize_yake_scores(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []

    lo, hi = min(raw_scores), max(raw_scores)
    if hi == lo:
        return [1.0] * len(raw_scores)

    return [(hi - score) / (hi - lo) for score in raw_scores]


def extract_yake_candidates(
    cleaned_text: str,
    *,
    language: str = DEFAULT_LANGUAGE,
    max_ngram_size: int = DEFAULT_MAX_NGRAM_SIZE,
    min_ngram_size: int = DEFAULT_MIN_NGRAM_SIZE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    dedup_lim: float = DEFAULT_DEDUP_LIM,
) -> YakeExtractionResult:
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"extract_yake_candidates expects a string, got {type(cleaned_text).__name__}"
        )
    if not cleaned_text.strip():
        raise EmptyTextError("YAKE extraction requires non-empty text.")
    if min_ngram_size > max_ngram_size:
        raise ValueError(
            f"min_ngram_size ({min_ngram_size}) cannot be greater than "
            f"max_ngram_size ({max_ngram_size})."
        )

    extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedup_lim=dedup_lim,
        top=max_candidates * 3,
    )
    raw_results: list[tuple[str, float]] = extractor.extract_keywords(cleaned_text)

    filtered = [
        (phrase, score)
        for phrase, score in raw_results
        if min_ngram_size <= len(phrase.split()) <= max_ngram_size
    ]

    if not filtered:
        return YakeExtractionResult(
            candidates=[],
            language=language,
            max_ngram_size=max_ngram_size,
        )
    filtered.sort(key=lambda pair: (pair[1], pair[0]))
    filtered = filtered[:max_candidates]

    raw_scores = [float(score) for _, score in filtered]
    normalized_scores = _normalize_yake_scores(raw_scores)

    candidates = [
        YakeCandidate(
            phrase=phrase,
            raw_score=raw_scores[i],
            normalized_score=normalized_scores[i],
            ngram_size=len(phrase.split()),
        )
        for i, (phrase, _) in enumerate(filtered)
    ]

    return YakeExtractionResult(
        candidates=candidates,
        language=language,
        max_ngram_size=max_ngram_size,
    )
