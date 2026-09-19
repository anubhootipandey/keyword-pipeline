import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.pipeline.exceptions import EmptyTextError

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_MAX_CANDIDATES = 50


@dataclass(frozen=True)
class TfidfCandidate:

    phrase: str
    raw_score: float
    normalized_score: float
    ngram_size: int
    document_frequency: int
    source: str = "tfidf"


@dataclass(frozen=True)
class TfidfExtractionResult:

    candidates: list[TfidfCandidate] = field(default_factory=list)
    sentence_count: int = 0
    used_single_document_fallback: bool = False
    vocabulary_size: int = 0


def _split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def _is_low_quality_phrase(phrase: str) -> bool:
    words = phrase.split()
    if not words:
        return True
    if all(word in ENGLISH_STOP_WORDS for word in words):
        return True
    if words[0] in ENGLISH_STOP_WORDS or words[-1] in ENGLISH_STOP_WORDS:
        return True
    return False


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0] * len(scores)

    return [(score - lo) / (hi - lo) for score in scores]


def extract_tfidf_candidates(
    cleaned_text: str,
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_df: float = 1.0,
    min_df: int = 1,
) -> TfidfExtractionResult:
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"extract_tfidf_candidates expects a string, got {type(cleaned_text).__name__}"
        )
    if not cleaned_text.strip():
        raise EmptyTextError("TF-IDF extraction requires non-empty text.")

    sentences = _split_into_sentences(cleaned_text)

    if len(sentences) < 2:
        corpus = [cleaned_text]
        used_fallback = True
        effective_max_df: float = 1.0
        effective_min_df: int = 1
    else:
        corpus = sentences
        used_fallback = False
        effective_max_df = max_df
        effective_min_df = min_df

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_df=effective_max_df,
        min_df=effective_min_df,
        sublinear_tf=True, 
        lowercase=True,
        norm="l2",  
    )

    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            return TfidfExtractionResult(
                candidates=[],
                sentence_count=len(corpus),
                used_single_document_fallback=used_fallback,
                vocabulary_size=0,
            )
        raise

    feature_names = vectorizer.get_feature_names_out()
    document_frequencies = np.asarray((matrix > 0).sum(axis=0)).ravel()
    raw_scores = np.asarray(matrix.sum(axis=0)).ravel()

    keep_indices = [
        i for i, name in enumerate(feature_names) if not _is_low_quality_phrase(name)
    ]

    if not keep_indices:
        return TfidfExtractionResult(
            candidates=[],
            sentence_count=len(corpus),
            used_single_document_fallback=used_fallback,
            vocabulary_size=len(feature_names),
        )

    filtered_raw_scores = [float(raw_scores[i]) for i in keep_indices]
    normalized_scores = _minmax_normalize(filtered_raw_scores)

    candidates = [
        TfidfCandidate(
            phrase=feature_names[feat_idx],
            raw_score=filtered_raw_scores[pos],
            normalized_score=normalized_scores[pos],
            ngram_size=len(feature_names[feat_idx].split()),
            document_frequency=int(document_frequencies[feat_idx]),
        )
        for pos, feat_idx in enumerate(keep_indices)
    ]

    candidates.sort(key=lambda c: (-c.raw_score, c.phrase))
    candidates = candidates[:max_candidates]

    return TfidfExtractionResult(
        candidates=candidates,
        sentence_count=len(corpus),
        used_single_document_fallback=used_fallback,
        vocabulary_size=len(feature_names),
    )
