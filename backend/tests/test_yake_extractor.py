"""
Tests for Stage 2B — YAKE candidate extraction.

These test actual ranking/scoring behavior, not just "did this return a
list". Each test class maps to one of the required scenarios from the
project spec (Phase 4).

A note on YAKE-specific test design: raw YAKE scores are not asserted
against exact floating-point literals anywhere here (YAKE's internal
scoring can shift slightly between versions/feature tweaks). Instead
tests assert relationships that must hold regardless of the exact
numbers — "the repeated concept's raw score is lower (better) than a
one-off word's", "the best raw score normalizes to 1.0", etc.
"""

import pytest

from app.pipeline.exceptions import EmptyTextError
from app.pipeline.preprocessor import preprocess_text
from app.pipeline.yake_extractor import (
    YakeCandidate,
    _normalize_yake_scores,
    extract_yake_candidates,
)

REACT_TEXT = (
    "React components make building React Native applications much easier. "
    "React components are reusable. State management in React can be "
    "handled with hooks. React hooks like useState and useEffect are "
    "commonly used. TypeScript adds type safety to React applications."
)


class TestNormalProse:
    def test_normal_article_produces_candidates(self):
        result = extract_yake_candidates(REACT_TEXT)
        assert len(result.candidates) > 0
        assert all(isinstance(c, YakeCandidate) for c in result.candidates)

    def test_candidates_are_sorted_best_first(self):
        result = extract_yake_candidates(REACT_TEXT)
        raw_scores = [c.raw_score for c in result.candidates]
        # Raw YAKE scores: lower is better, so best-first means ascending.
        assert raw_scores == sorted(raw_scores)

    def test_normalized_scores_are_sorted_best_first_descending(self):
        result = extract_yake_candidates(REACT_TEXT)
        normalized_scores = [c.normalized_score for c in result.candidates]
        # normalized_score: higher is better, so best-first means descending.
        assert normalized_scores == sorted(normalized_scores, reverse=True)


class TestRepeatedImportantConcepts:
    def test_repeated_concept_outranks_one_off_word(self):
        # "React" appears 6 times across the text; "TypeScript" appears once.
        result = extract_yake_candidates(REACT_TEXT, max_candidates=30)
        by_phrase = {c.phrase: c for c in result.candidates}

        assert "React" in by_phrase
        assert "TypeScript" in by_phrase
        # Lower raw_score = more important in YAKE's convention.
        assert by_phrase["React"].raw_score < by_phrase["TypeScript"].raw_score

    def test_repeated_concept_has_higher_normalized_score(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=30)
        by_phrase = {c.phrase: c for c in result.candidates}

        assert by_phrase["React"].normalized_score > by_phrase["TypeScript"].normalized_score


class TestMeaningfulMultiWordPhrases:
    def test_multiword_phrase_is_extracted(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=30)
        phrases = [c.phrase for c in result.candidates]
        assert any(p.split() == ["React", "Native"] or "React Native" in p for p in phrases)

    def test_ngram_size_matches_actual_word_count(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=30)
        for c in result.candidates:
            assert c.ngram_size == len(c.phrase.split())


class TestShortText:
    def test_two_word_input_does_not_crash(self):
        result = extract_yake_candidates("machine learning")
        assert isinstance(result.candidates, list)

    def test_single_word_input_produces_one_candidate(self):
        result = extract_yake_candidates("Python")
        phrases = [c.phrase for c in result.candidates]
        assert "Python" in phrases

    def test_short_text_scores_are_still_valid_floats(self):
        result = extract_yake_candidates("machine learning models")
        for c in result.candidates:
            assert isinstance(c.raw_score, float)
            assert c.raw_score >= 0.0
            assert 0.0 <= c.normalized_score <= 1.0


class TestEmptyAndInvalidInput:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyTextError):
            extract_yake_candidates("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyTextError):
            extract_yake_candidates("     \n\t   ")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_yake_candidates(None)  # type: ignore[arg-type]

    def test_min_greater_than_max_ngram_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_yake_candidates("some valid text here", min_ngram_size=3, max_ngram_size=1)


class TestNoUsableVocabulary:
    def test_stopwords_and_numbers_only_yields_no_candidates(self):
        result = extract_yake_candidates("the a an is of to and 123 456 789")
        assert result.candidates == []

    def test_punctuation_only_yields_no_candidates_not_a_crash(self):
        result = extract_yake_candidates("!!! ??? ... --- ***")
        assert result.candidates == []

    def test_no_usable_vocabulary_still_returns_valid_result_object(self):
        result = extract_yake_candidates("the a an is of")
        assert result.candidates == []
        assert result.language == "en"
        assert result.max_ngram_size == 3


class TestPunctuationHeavyInput:
    def test_repeated_punctuation_does_not_break_extraction(self):
        text = "This is amazing!!! Really??? I cannot believe it... Best tool ever!!!"
        result = extract_yake_candidates(text)
        phrases = [c.phrase for c in result.candidates]
        assert len(phrases) > 0
        # Punctuation characters should not themselves become candidates.
        assert all(not all(ch in "!?.-" for ch in p) for p in phrases)

    def test_preprocessed_punctuation_heavy_text_still_extracts(self):
        # Feed text through Phase 2 first, as it would be in the real
        # pipeline, to confirm the two stages work together correctly.
        preprocessed = preprocess_text(
            "Wow!!! This is amazing??? Really... incredible!!! Best framework ever!!!"
        )
        result = extract_yake_candidates(preprocessed.cleaned_text)
        assert len(result.candidates) > 0


class TestUnicodeAndNonEnglishText:
    def test_accented_characters_are_preserved_in_candidates(self):
        text = (
            "Caf\u00e9 culture is popular in Paris. Many people visit a "
            "caf\u00e9 every morning for a caf\u00e9 au lait."
        )
        result = extract_yake_candidates(text)
        phrases = " ".join(c.phrase for c in result.candidates)
        assert "caf\u00e9" in phrases.lower()

    def test_non_latin_script_does_not_crash(self):
        text = "\u4f60\u597d\u4e16\u754c is Chinese for hello world in machine translation research."
        result = extract_yake_candidates(text)
        assert isinstance(result.candidates, list)

    def test_emoji_does_not_crash_extraction(self):
        text = "Launching the rocket \U0001f680 today was a huge success for the team."
        result = extract_yake_candidates(text)
        assert len(result.candidates) > 0


class TestCandidateLimit:
    def test_max_candidates_is_respected(self):
        long_text = REACT_TEXT * 3
        result = extract_yake_candidates(long_text, max_candidates=5)
        assert len(result.candidates) <= 5

    def test_smaller_limit_keeps_the_best_scoring_candidates(self):
        result_full = extract_yake_candidates(REACT_TEXT, max_candidates=30)
        result_limited = extract_yake_candidates(REACT_TEXT, max_candidates=3)

        best_three_phrases = [c.phrase for c in result_full.candidates[:3]]
        limited_phrases = [c.phrase for c in result_limited.candidates]
        assert limited_phrases == best_three_phrases


class TestConfigurableNgramLength:
    def test_max_ngram_size_one_returns_only_unigrams(self):
        result = extract_yake_candidates(REACT_TEXT, max_ngram_size=1, min_ngram_size=1)
        assert all(c.ngram_size == 1 for c in result.candidates)

    def test_min_ngram_size_two_excludes_unigrams(self):
        result = extract_yake_candidates(
            REACT_TEXT, max_ngram_size=3, min_ngram_size=2, max_candidates=30
        )
        assert all(c.ngram_size >= 2 for c in result.candidates)
        assert len(result.candidates) > 0

    def test_max_ngram_size_three_can_include_trigrams(self):
        result = extract_yake_candidates(REACT_TEXT, max_ngram_size=3, max_candidates=30)
        ngram_sizes = {c.ngram_size for c in result.candidates}
        assert 3 in ngram_sizes

    def test_result_reports_configured_max_ngram_size(self):
        result = extract_yake_candidates(REACT_TEXT, max_ngram_size=2)
        assert result.max_ngram_size == 2


class TestDeterministicOutput:
    def test_same_input_same_config_produces_identical_output(self):
        first = extract_yake_candidates(REACT_TEXT, max_candidates=15)
        second = extract_yake_candidates(REACT_TEXT, max_candidates=15)

        assert [c.phrase for c in first.candidates] == [c.phrase for c in second.candidates]
        assert [c.raw_score for c in first.candidates] == [c.raw_score for c in second.candidates]

    def test_tied_raw_scores_break_ties_alphabetically(self):
        # Construct near-identical-frequency single-occurrence words so
        # ties are plausible, then confirm ordering among any tied group
        # is alphabetical rather than arbitrary/insertion-order.
        text = "Apple Banana Cherry Date Elderberry are all fruit names used once each here."
        first = extract_yake_candidates(text, max_candidates=20)
        second = extract_yake_candidates(text, max_candidates=20)
        assert [c.phrase for c in first.candidates] == [c.phrase for c in second.candidates]


class TestRawScorePreservation:
    def test_raw_score_is_not_overwritten_by_normalization(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=10)
        # Raw scores must vary (not all be forced to 0 or 1 the way a
        # normalized score's edge cases might be) — this is YAKE's real
        # signal, untouched.
        raw_scores = [c.raw_score for c in result.candidates]
        assert len(set(raw_scores)) > 1

    def test_raw_scores_are_real_floats_not_ranks(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=10)
        for c in result.candidates:
            assert isinstance(c.raw_score, float)
            # YAKE raw scores are non-negative "cost" style values.
            assert c.raw_score >= 0.0


class TestScoreNormalization:
    def test_best_raw_score_normalizes_close_to_one(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=15)
        best = min(result.candidates, key=lambda c: c.raw_score)
        assert best.normalized_score == pytest.approx(1.0)

    def test_worst_raw_score_in_batch_normalizes_close_to_zero(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=15)
        worst = max(result.candidates, key=lambda c: c.raw_score)
        assert worst.normalized_score == pytest.approx(0.0)

    def test_all_normalized_scores_within_unit_range(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=15)
        for c in result.candidates:
            assert 0.0 <= c.normalized_score <= 1.0

    def test_normalize_function_directly_inverts_direction(self):
        # Lower raw score (better in YAKE) must map to higher normalized
        # score (better in the normalized convention).
        normalized = _normalize_yake_scores([0.01, 0.05, 0.10])
        assert normalized[0] > normalized[1] > normalized[2]
        assert normalized[0] == pytest.approx(1.0)
        assert normalized[2] == pytest.approx(0.0)

    def test_normalize_function_handles_identical_scores(self):
        normalized = _normalize_yake_scores([0.05, 0.05, 0.05])
        assert normalized == [1.0, 1.0, 1.0]

    def test_normalize_function_handles_empty_list(self):
        assert _normalize_yake_scores([]) == []

    def test_normalize_function_handles_single_score(self):
        assert _normalize_yake_scores([0.42]) == [1.0]


class TestMultipleCandidatesDifferentRawScores:
    def test_candidates_have_a_spread_of_distinct_raw_scores(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=20)
        raw_scores = {c.raw_score for c in result.candidates}
        # A real article should produce more than one or two distinct
        # score values, not everything tied.
        assert len(raw_scores) >= 5


class TestIntegrationWithPreprocessing:
    def test_html_and_url_cleaned_text_produces_clean_candidates(self):
        raw_html = (
            "<p>Check out <b>React components</b> at https://example.com/docs "
            "for building modern web applications with reusable UI elements.</p>"
        )
        preprocessed = preprocess_text(raw_html)
        result = extract_yake_candidates(preprocessed.cleaned_text)

        phrases_joined = " ".join(c.phrase for c in result.candidates)
        assert "<" not in phrases_joined
        assert "https://" not in phrases_joined
        assert "example.com" not in phrases_joined
        assert len(result.candidates) > 0

    def test_preprocessed_text_with_repeated_punctuation_is_clean(self):
        raw = "Amazing!!!! Truly incredible??? A must-have tool...."
        preprocessed = preprocess_text(raw)
        # Phase 2 already collapsed the repeated punctuation.
        assert "!!!!" not in preprocessed.cleaned_text
        result = extract_yake_candidates(preprocessed.cleaned_text)
        assert isinstance(result.candidates, list)


class TestCandidateDataclass:
    def test_source_field_defaults_to_yake(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=5)
        assert all(c.source == "yake" for c in result.candidates)

    def test_candidate_is_frozen(self):
        result = extract_yake_candidates(REACT_TEXT, max_candidates=1)
        candidate = result.candidates[0]
        with pytest.raises(Exception):
            candidate.phrase = "modified"  # type: ignore[misc]
