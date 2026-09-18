"""
Stage 2B — YAKE candidate extraction.

WHAT YAKE IS

YAKE (Yet Another Keyword Extractor) is an unsupervised keyword/keyphrase
extraction algorithm. It looks only at the submitted text itself — word
casing, position in the text, frequency, how many different sentences a
term appears in, and how often it appears next to other candidate terms
— and combines those statistical/textual features into a score for each
candidate phrase. It does not use a trained language model, does not
call any API, and does not know anything about the world beyond the
text it's given.

That means YAKE's output must not be described as:
    - search volume / how often people search for a term
    - SEO popularity or Google ranking
    - semantic understanding (YAKE has no notion of meaning or
      synonymy — that's what the embeddings stage, added later, is for)
    - LLM-generated keywords (there is no language model involved)

And its ranking is not "correct" in any objective sense — it's one
statistical signal among several the pipeline will eventually combine
(TF-IDF, YAKE, spaCy linguistic extraction, later phases).

SCORE DIRECTION — IMPORTANT

YAKE's raw score is a "how good is this candidate" score where LOWER IS
BETTER — closer to 0 means more relevant, higher means less relevant.
This is the opposite convention from TF-IDF (where a higher score means
more salient) and from most intuitions about "score". Getting this
backwards would silently invert every ranking downstream, so:

    - `raw_score` on YakeCandidate is YAKE's score, completely
      unmodified — still "lower is better" if you read it directly.
    - `normalized_score` is a separate, clearly-inverted value on
      [0, 1] where HIGHER MEANS BETTER, produced so this extractor's
      output can be compared/combined with TF-IDF's normalized_score
      (also higher-is-better) in the merger stage (Phase 6) without
      every caller having to remember which direction which score
      goes. See `_normalize_yake_scores` for exactly how that
      transformation works. It is a ranking convenience, not an
      "accuracy" or "confidence" value.

WHY THIS STAYS SEPARATE FROM TF-IDF FOR NOW

TF-IDF and YAKE measure different things (corpus-relative document
statistics vs. single-document statistical/positional features) and can
disagree on both which phrases matter and how phrases are tokenized.
Merging them now would mean either silently picking one extractor's
opinion or building the merge logic before Phase 6 (candidate merging)
is supposed to exist. This module's job ends at "here are YAKE's
ranked candidates" — nothing here compares them to TF-IDF's output.

LIMITATIONS

- YAKE's phrase boundaries come from statistical co-occurrence within a
  sliding window, not grammar. It will sometimes produce phrases that
  aren't clean grammatical units (e.g. "components make building" from
  "React components make building apps easier") alongside genuinely
  good ones. That's expected — YAKE isn't a grammar-aware extractor;
  spaCy's noun-phrase extraction (Phase 5) is what covers that angle.
- Like TF-IDF here, YAKE only sees the single document it's given. It
  has no corpus-wide notion of rarity either — its "distinctiveness"
  features are all relative to word positions/co-occurrences within
  this one document.
- Very short input (a handful of words) gives YAKE little to work
  with; scores in that regime are noisier and less meaningful than on
  a full paragraph or article.
"""

from dataclasses import dataclass, field

import yake

from app.pipeline.exceptions import EmptyTextError

# --- Defaults ---

DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_NGRAM_SIZE = 3
DEFAULT_MIN_NGRAM_SIZE = 1
DEFAULT_MAX_CANDIDATES = 50
# How similar (by sequence matching) two candidate phrases can be before
# YAKE treats the weaker one as a near-duplicate and drops it. YAKE's own
# default is 0.9; kept as our default too since it's a reasonable balance
# — low enough to catch obvious near-duplicates ("react hook" / "react
# hooks"), high enough not to eliminate genuinely distinct short phrases.
DEFAULT_DEDUP_LIM = 0.9


@dataclass(frozen=True)
class YakeCandidate:
    """
    One candidate keyword/keyphrase produced by YAKE extraction.

    Attributes:
        phrase: the candidate exactly as YAKE returned it — YAKE
            preserves original casing (unlike scikit-learn's
            TfidfVectorizer, which lowercases everything), so this may
            read as "React Components" rather than "react components".
        raw_score: YAKE's own score, untouched. LOWER IS BETTER — see
            module docstring. Kept as-is so later stages that want
            YAKE's actual signal (not a rescaled version of it) can use
            it directly.
        normalized_score: raw_score transformed to [0, 1] where HIGHER
            IS BETTER, for combining with other higher-is-better signals
            (e.g. TfidfCandidate.normalized_score) in the merger stage.
            See `_normalize_yake_scores`.
        ngram_size: number of words in the phrase.
        source: constant tag identifying which extractor produced this
            candidate, for the merging stage (Phase 6) to use.
    """

    phrase: str
    raw_score: float
    normalized_score: float
    ngram_size: int
    source: str = "yake"


@dataclass(frozen=True)
class YakeExtractionResult:
    """Full result of a YAKE extraction run."""

    candidates: list[YakeCandidate] = field(default_factory=list)
    language: str = DEFAULT_LANGUAGE
    max_ngram_size: int = DEFAULT_MAX_NGRAM_SIZE


def _normalize_yake_scores(raw_scores: list[float]) -> list[float]:
    """
    Convert YAKE's "lower is better" raw scores into a [0, 1] scale
    where HIGHER IS BETTER, via inverted min-max normalization.

    Given the best (lowest) raw score `lo` and worst (highest) raw score
    `hi` in this candidate set:

        normalized = (hi - raw_score) / (hi - lo)

    So the best candidate in the batch normalizes to 1.0, the worst
    normalizes to 0.0, and everything else falls linearly in between.
    This is a per-batch relative scale, not an absolute quality
    threshold — a normalized_score of 0.9 means "among the best of
    these particular candidates", not "90% relevant" in any universal
    sense. That's true of TF-IDF's normalized_score too; the two are
    meant to be comparable to each other on that same relative basis.

    Edge case: if every raw score is identical (including a single
    candidate), everything normalizes to 1.0 — there's no meaningful
    "worse" candidate to be relatively better than, so treating them as
    equally good (rather than arbitrarily all-zero) is the less
    misleading choice.
    """
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
    """
    Extract ranked candidate keywords/keyphrases from already-cleaned
    text using YAKE.

    `cleaned_text` is expected to be the output of
    `app.pipeline.preprocessor.preprocess_text(...).cleaned_text` — this
    function does not re-run HTML/URL/whitespace cleaning itself (Phase
    2 already did that), and it does not modify the text beyond what
    YAKE's own tokenizer does internally.

    Args:
        cleaned_text: preprocessed input text.
        language: YAKE language code for its built-in stopword list.
            "en" (English) is the only language this project documents
            or tests; YAKE supports others, but behavior for them is
            unverified here.
        max_ngram_size: longest phrase (in words) YAKE should consider.
            This maps directly to YAKE's own `n` parameter, which is a
            ceiling, not a range — YAKE always considers 1-word phrases
            up to this size.
        min_ngram_size: shortest phrase (in words) to keep in the
            result. YAKE itself has no minimum-length parameter, so
            this is enforced by filtering YAKE's output after the fact.
        max_candidates: maximum number of candidates to return.
        dedup_lim: passed straight through to YAKE's own `dedup_lim` —
            see DEFAULT_DEDUP_LIM above.

    Returns:
        A YakeExtractionResult. Text with no usable content (e.g. only
        stopwords, only punctuation, only numbers) legitimately produces
        an empty candidate list — YAKE itself returns `[]` for that
        case rather than raising, and this function preserves that
        behavior rather than treating it as an error.

    Raises:
        TypeError: if cleaned_text is not a string.
        EmptyTextError: if cleaned_text is empty or whitespace-only.
        ValueError: if min_ngram_size > max_ngram_size (a genuine
            caller configuration error, not an input-data problem).
    """
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
        # Ask YAKE for more than max_candidates before our own
        # min_ngram_size filtering runs, so filtering out short phrases
        # doesn't leave us with fewer than max_candidates results when
        # enough good longer candidates actually exist.
        top=max_candidates * 3,
    )

    # YAKE's extract_keywords does not raise on empty/whitespace/
    # no-vocabulary input — it returns []. We've already rejected
    # empty/whitespace input above (as an explicit EmptyTextError,
    # consistent with the TF-IDF extractor), so by this point an empty
    # result specifically means "no usable candidates in otherwise
    # valid text" (e.g. stopwords/numbers/punctuation only), which is a
    # legitimate outcome, not an error.
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

    # YAKE already returns results sorted best-first (ascending raw
    # score), but we sort explicitly rather than relying on that,
    # breaking ties alphabetically so output is fully deterministic
    # regardless of any incidental ordering YAKE's internals produce.
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
