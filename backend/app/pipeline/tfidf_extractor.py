"""
Stage 2A — TF-IDF candidate extraction.

IMPORTANT METHODOLOGICAL NOTE — read this before changing anything here.

TF-IDF's whole point is comparing term frequency *within* a document
against how rare that term is *across a corpus of many documents*. This
application, at least initially, analyzes a single submitted document at
a time. There is no corpus of other documents to compare against.

If we naively ran TfidfVectorizer on a "corpus" containing exactly one
document (the submitted text), every term that appears at all would have
a document frequency of 1, so scikit-learn's smoothed IDF formula
(`ln((1+n)/(1+df)) + 1`) collapses to the same constant for every term.
The result would just be term frequency wearing a TF-IDF costume — not
wrong, exactly, but not what "TF-IDF" implies, and worth being honest
about rather than passing off as more sophisticated than it is.

To get a real document-frequency signal without reaching for an external
corpus (which would mean either a paid dataset or bundling a large
static wordlist — both against the project's constraints), this module
splits the submitted document into sentences and treats EACH SENTENCE AS
ONE "DOCUMENT" for the vectorizer. This is still entirely self-contained
— the "corpus" is just parts of the same text the user submitted, not
external data — and it gives IDF something real to measure: how
concentrated a term or phrase is within specific parts of this document,
versus spread evenly across all of it.

What this signal IS:
    A measure of term/phrase salience within the submitted text, based
    on how frequently and how selectively it appears across the
    sentences of that text.

What this signal is NOT, and must never be described as:
    - "globally important keywords" (there is no global corpus here)
    - "search popularity" or "SEO volume" (this has nothing to do with
      how often anyone searches for a term — that's what the optional
      trend enrichment stage, added later, is for)
    - corpus-wide rarity in any general-English sense

If the input is too short to produce more than one sentence, there's no
meaningful sentence-level corpus to build, and this module falls back to
treating the whole text as a single document. In that fallback case, IDF
really is a constant for every term (the degenerate case described
above) — this is flagged explicitly in `TfidfExtractionResult.
used_single_document_fallback` rather than silently producing a
misleadingly-labeled score.
"""

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.pipeline.exceptions import EmptyTextError

# --- Compiled patterns / constants ---

# Heuristic sentence splitter: splits after a sentence-ending punctuation
# mark that's followed by whitespace. This is intentionally simple —
# it will get things like "Dr. Smith" wrong — because a linguistically
# correct sentence splitter is spaCy's job (Phase 5), and pulling in a
# second sentence-boundary implementation here just to get this
# extractor's pseudo-corpus is not worth the complexity. Getting this
# occasionally wrong only affects how sentences are grouped for the IDF
# signal, not correctness of the extracted phrases themselves.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

DEFAULT_NGRAM_RANGE = (1, 3)
DEFAULT_MAX_CANDIDATES = 50


@dataclass(frozen=True)
class TfidfCandidate:
    """
    One candidate keyword/keyphrase produced by TF-IDF extraction.

    Attributes:
        phrase: the candidate as scikit-learn produced it — lowercased,
            tokenized on word-character boundaries (so hyphens split
            words apart; see module docstring in the merger stage for
            how this gets reconciled with other extractors later).
        raw_score: the actual TF-IDF value, summed across all pseudo-
            documents (sentences) the phrase appeared in. This is the
            real signal — kept untouched so later stages (merging,
            scoring) can use it directly rather than only ever seeing a
            normalized number.
        normalized_score: raw_score min-max normalized to [0, 1] across
            the candidates returned by this extraction run. Provided for
            convenience when combining with other signals that are on
            different scales — NOT a replacement for raw_score.
        ngram_size: number of words in the phrase (1 = unigram, etc).
        document_frequency: number of pseudo-documents (sentences, or 1
            in the single-document fallback case) the phrase appeared in.
        source: constant tag identifying which extractor produced this
            candidate, for the merging stage (Phase 6) to use.
    """

    phrase: str
    raw_score: float
    normalized_score: float
    ngram_size: int
    document_frequency: int
    source: str = "tfidf"


@dataclass(frozen=True)
class TfidfExtractionResult:
    """
    Full result of a TF-IDF extraction run, including the metadata
    needed to interpret the candidates honestly (see module docstring).
    """

    candidates: list[TfidfCandidate] = field(default_factory=list)
    sentence_count: int = 0
    used_single_document_fallback: bool = False
    vocabulary_size: int = 0


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks. See module docstring for
    why this heuristic is good enough for this specific purpose."""
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def _is_low_quality_phrase(phrase: str) -> bool:
    """
    Filter out candidates that are pure noise for keyword purposes:
    a phrase made entirely of stopwords, or a multi-word phrase that
    STARTS or ENDS with a stopword (e.g. "the react" or "react is").

    Stopwords in the *middle* of a phrase are left alone on purpose —
    "state of the art" is a meaningful phrase despite containing "of"
    and "the". Reusing scikit-learn's built-in ENGLISH_STOP_WORDS list
    here (rather than passing stop_words="english" to the vectorizer
    itself) is deliberate: passing it to the vectorizer removes
    stopword tokens from the stream BEFORE n-grams are built, which
    would silently glue together words that weren't actually adjacent
    in the original text (e.g. "out of the box" could become the
    bigram "out box"). Filtering candidates after generation avoids
    that problem entirely.
    """
    words = phrase.split()
    if not words:
        return True
    if all(word in ENGLISH_STOP_WORDS for word in words):
        return True
    if words[0] in ENGLISH_STOP_WORDS or words[-1] in ENGLISH_STOP_WORDS:
        return True
    return False


def _minmax_normalize(scores: list[float]) -> list[float]:
    """
    Min-max normalize a list of scores to [0, 1].

    If every score is identical (including the edge case of a single
    candidate), they all normalize to 1.0 if that shared value is
    positive, or 0.0 if it's zero — this avoids a division by zero
    while still producing a sensible, unsurprising result.
    """
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
    """
    Extract ranked candidate keywords/keyphrases from already-cleaned
    text using TF-IDF.

    `cleaned_text` is expected to be the output of
    `app.pipeline.preprocessor.preprocess_text(...).cleaned_text` — this
    function does not re-run HTML/URL/whitespace cleaning itself, to
    avoid duplicating that logic (see Phase 2).

    Args:
        cleaned_text: preprocessed input text.
        ngram_range: (min_n, max_n) for candidate phrase length. Default
            covers unigrams through trigrams.
        max_candidates: maximum number of candidates to return, so later
            stages (embeddings, clustering) never have to deal with an
            unbounded candidate list from a long document.
        max_df: maximum document-frequency proportion for a term to be
            considered (ignored — forced to 1.0 — when the input is too
            short to build a multi-sentence pseudo-corpus; see below).
        min_df: minimum document frequency (absolute count) for a term
            to be considered (same fallback behavior as max_df).

    Returns:
        A TfidfExtractionResult. If the text contains no usable
        vocabulary (e.g. only stopwords, numbers, or punctuation), this
        returns an empty candidate list rather than raising — that's a
        legitimate "nothing useful here" outcome, not an error.

    Raises:
        TypeError: if cleaned_text is not a string.
        EmptyTextError: if cleaned_text is empty or whitespace-only.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError(
            f"extract_tfidf_candidates expects a string, got {type(cleaned_text).__name__}"
        )
    if not cleaned_text.strip():
        raise EmptyTextError("TF-IDF extraction requires non-empty text.")

    sentences = _split_into_sentences(cleaned_text)

    if len(sentences) < 2:
        # Degenerate case — see module docstring. Forcing max_df=1.0 and
        # min_df=1 here is not optional: with exactly one pseudo-document,
        # every present term has a document-frequency ratio of 1.0, so
        # any max_df below 1.0 would filter out the ENTIRE vocabulary,
        # not just common terms.
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
        sublinear_tf=True,  # log-scale term frequency (1 + log(tf)) so
        # a term appearing 20 times doesn't linearly dominate a term
        # appearing twice — diminishing returns on raw repetition.
        lowercase=True,
        norm="l2",  # normalize each pseudo-document's vector so longer
        # sentences don't automatically dominate shorter ones.
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

    # Sort by raw score descending; break ties alphabetically so ranking
    # is fully deterministic regardless of any incidental ordering from
    # scikit-learn's internal vocabulary construction.
    candidates.sort(key=lambda c: (-c.raw_score, c.phrase))
    candidates = candidates[:max_candidates]

    return TfidfExtractionResult(
        candidates=candidates,
        sentence_count=len(corpus),
        used_single_document_fallback=used_fallback,
        vocabulary_size=len(feature_names),
    )
