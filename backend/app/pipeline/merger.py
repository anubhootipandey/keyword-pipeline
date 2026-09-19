from dataclasses import dataclass, field
from statistics import mean

from app.pipeline.linguistic_extractor import LinguisticExtractionResult
from app.pipeline.normalizer import normalize_for_matching
from app.pipeline.tfidf_extractor import TfidfExtractionResult
from app.pipeline.yake_extractor import YakeExtractionResult

_SOURCE_ORDER = ("tfidf", "yake", "linguistic")

_SURFACE_PRIORITY = ("linguistic", "yake", "tfidf")

DEFAULT_MAX_CANDIDATES = 50


@dataclass(frozen=True)
class MergedCandidate:
    phrase: str
    normalized_phrase: str
    sources: tuple[str, ...]
    source_signals: dict[str, float] = field(default_factory=dict)
    source_count: int = 0
    source_consensus: float = 0.0
    ngram_size: int = 0


@dataclass(frozen=True)
class MergeResult:
    candidates: list[MergedCandidate] = field(default_factory=list)
    total_sources: int = 0


@dataclass
class _Bucket:

    signals: dict[str, float] = field(default_factory=dict)
    surfaces: dict[str, str] = field(default_factory=dict)
    linguistic_lemma: str | None = None

    def add(self, source: str, surface: str, normalized_score: float) -> None:
        if source in self.signals:
            self.signals[source] = max(self.signals[source], normalized_score)
            self.surfaces[source] = min(self.surfaces[source], surface)
        else:
            self.signals[source] = normalized_score
            self.surfaces[source] = surface


def _select_surface(bucket: _Bucket) -> str:
    for source in _SURFACE_PRIORITY:
        if source in bucket.surfaces:
            return bucket.surfaces[source]
    raise AssertionError("bucket has no surfaces — this should be unreachable")  


def merge_candidates(
    tfidf_result: TfidfExtractionResult | None = None,
    yake_result: YakeExtractionResult | None = None,
    linguistic_result: LinguisticExtractionResult | None = None,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> MergeResult:
    results = {
        "tfidf": tfidf_result,
        "yake": yake_result,
        "linguistic": linguistic_result,
    }
    provided = {name: result for name, result in results.items() if result is not None}

    if not provided:
        raise ValueError(
            "merge_candidates requires at least one extraction result "
            "(tfidf_result, yake_result, or linguistic_result) to be provided."
        )

    total_sources = len(provided)
    buckets: dict[str, _Bucket] = {}

    def get_bucket(key: str) -> _Bucket:
        if key not in buckets:
            buckets[key] = _Bucket()
        return buckets[key]

    if tfidf_result is not None:
        for candidate in tfidf_result.candidates:
            key = normalize_for_matching(candidate.phrase)
            if not key:
                continue
            get_bucket(key).add("tfidf", candidate.phrase, candidate.normalized_score)

    if yake_result is not None:
        for candidate in yake_result.candidates:
            key = normalize_for_matching(candidate.phrase)
            if not key:
                continue
            get_bucket(key).add("yake", candidate.phrase, candidate.normalized_score)

    if linguistic_result is not None:
        for candidate in linguistic_result.candidates:
            key = normalize_for_matching(candidate.phrase)
            if not key:
                continue
            bucket = get_bucket(key)
            bucket.add("linguistic", candidate.phrase, candidate.normalized_score)
            if bucket.linguistic_lemma is None:
                bucket.linguistic_lemma = candidate.normalized_phrase
            else:
                bucket.linguistic_lemma = min(bucket.linguistic_lemma, candidate.normalized_phrase)

    merged: list[MergedCandidate] = []
    for key, bucket in buckets.items():
        sources = tuple(s for s in _SOURCE_ORDER if s in bucket.signals)
        phrase = _select_surface(bucket)
        normalized_phrase = bucket.linguistic_lemma if bucket.linguistic_lemma is not None else key

        merged.append(
            MergedCandidate(
                phrase=phrase,
                normalized_phrase=normalized_phrase,
                sources=sources,
                source_signals=dict(bucket.signals),
                source_count=len(sources),
                source_consensus=len(sources) / total_sources,
                ngram_size=len(phrase.split()),
            )
        )

    merged.sort(
        key=lambda c: (
            -c.source_count,
            -mean(c.source_signals.values()),
            c.normalized_phrase,
        )
    )
    merged = merged[:max_candidates]

    return MergeResult(candidates=merged, total_sources=total_sources)