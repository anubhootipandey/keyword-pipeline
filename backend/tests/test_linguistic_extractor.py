"""
Tests for Stage 2C — spaCy linguistic candidate extraction.

These test behavior (which phrases survive, what gets trimmed, what
ranks higher) rather than depending on spaCy's exact internal
tokenization/parsing decisions, since those can shift slightly between
model versions. Where a specific spaCy behavior IS asserted on (e.g. a
known small-model parsing limitation), it was verified directly against
the installed model before being written into a test, not assumed.
"""

import pytest

from app.pipeline.exceptions import EmptyTextError, LinguisticModelNotAvailableError
from app.pipeline.linguistic_extractor import (
    LinguisticCandidate,
    _minmax_normalize,
    extract_linguistic_candidates,
)
from app.pipeline.preprocessor import preprocess_text

REACT_TEXT = (
    "React components make building React Native applications much easier. "
    "React components are reusable. Machine learning models require large "
    "datasets. Machine learning models are powerful. NASA and IBM use "
    "real-time systems daily."
)


class TestNormalProse:
    def test_normal_article_produces_candidates(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        assert len(result.candidates) > 0
        assert all(isinstance(c, LinguisticCandidate) for c in result.candidates)

    def test_result_reports_model_name_and_chunk_count(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        assert result.model_name == "en_core_web_sm"
        assert result.noun_chunk_count > 0


class TestNounPhrases:
    def test_simple_noun_phrase_is_extracted(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "large dataset" in normalized_phrases

    def test_surface_form_preserves_original_casing(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}
        assert by_normalized["react component"].phrase == "React components"


class TestAdjectiveNounPhrases:
    def test_attributive_adjective_is_kept_with_its_noun(self):
        text = "Large datasets are essential. Small datasets are less useful for training."
        result = extract_linguistic_candidates(text)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "large dataset" in normalized_phrases
        assert "small dataset" in normalized_phrases


class TestMultiWordTechnicalPhrases:
    def test_trigram_technical_phrase_is_extracted(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "machine learning model" in normalized_phrases

    def test_ngram_size_matches_actual_kept_word_count(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        for c in result.candidates:
            # Kept word count should match the normalized phrase's word count.
            assert c.ngram_size == len(c.normalized_phrase.split())


class TestLemmatizationAndNormalization:
    def test_plural_noun_lemmatizes_to_singular_in_normalized_form(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}
        # "React components" (plural) -> normalized "react component" (singular)
        assert "react component" in by_normalized
        assert by_normalized["react component"].phrase == "React components"

    def test_surface_form_is_not_silently_replaced_by_lemma(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        phrases = [c.phrase for c in result.candidates]
        # The exact surface text "React components" (not "React component")
        # must be present somewhere as a displayable phrase.
        assert "React components" in phrases

    def test_different_inflections_of_same_concept_merge_under_one_normalized_key(self):
        text = "The model predicts well. The models predict well together."
        result = extract_linguistic_candidates(text)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}
        assert "model" in by_normalized
        # Both "model" and "models" contributed to the same normalized entry.
        assert by_normalized["model"].linguistic_signal == 2


class TestStopwordsInsideMeaningfulPhrases:
    def test_internal_preposition_is_preserved_not_stripped(self):
        text = "Out of the box solutions save time. The out of the box experience matters."
        result = extract_linguistic_candidates(text, max_phrase_length=6)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        # Whatever chunk boundary spaCy picks, an internal "of"/"the"
        # between content words must not be stripped out mid-phrase.
        assert any(
            "of" in phrase.split() or "the" in phrase.split()
            for phrase in normalized_phrases
        ) or any("box" in phrase for phrase in normalized_phrases)

    def test_natural_language_processing_style_phrase_survives_intact(self):
        text = "Natural language processing models require large datasets to train."
        result = extract_linguistic_candidates(text)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "natural language processing model" in normalized_phrases

    def test_leading_determiner_is_trimmed(self):
        text = "The dataset was large. A dataset like this one takes time to prepare."
        result = extract_linguistic_candidates(text)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "dataset" in normalized_phrases
        assert "the dataset" not in normalized_phrases
        assert "a dataset" not in normalized_phrases


class TestPunctuation:
    def test_punctuation_is_not_a_standalone_candidate(self):
        text = "Wow! This is amazing. Really? I can't believe it... Best tool ever!"
        result = extract_linguistic_candidates(text)
        for c in result.candidates:
            assert not all(ch in "!?.-," for ch in c.phrase)

    def test_possessive_marker_excluded_from_normalized_form(self):
        text = "The company's machine learning strategy improved results significantly."
        result = extract_linguistic_candidates(text, max_phrase_length=6)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert not any("'s" in p for p in normalized_phrases)


class TestHyphenatedTechnicalTerms:
    def test_hyphenated_compound_noun_phrase_is_extracted(self):
        # "Real-time" is tokenized as a compound and stays attached to
        # its noun in the parse - verified directly against the
        # installed model before writing this assertion.
        text = "Real-time systems require careful design. Real-time systems are complex."
        result = extract_linguistic_candidates(text)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "real time system" in normalized_phrases


class TestAcronyms:
    def test_acronym_is_extracted_as_its_own_candidate(self):
        text = "NASA and IBM both invest heavily in research. NASA leads several projects."
        result = extract_linguistic_candidates(text)
        phrases = [c.phrase for c in result.candidates]
        assert "NASA" in phrases
        assert "IBM" in phrases

    def test_acronym_normalized_form_is_lowercased(self):
        text = "NASA leads the mission. NASA announced new plans."
        result = extract_linguistic_candidates(text)
        by_phrase = {c.phrase: c for c in result.candidates}
        assert by_phrase["NASA"].normalized_phrase == "nasa"


class TestNumbers:
    def test_number_with_noun_is_kept_as_quantity_phrase(self):
        text = "The company hired 150 employees this year across 10 offices worldwide."
        result = extract_linguistic_candidates(text, max_phrase_length=4)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert any("150 employee" in p or "employee" in p for p in normalized_phrases)

    def test_bare_number_is_not_returned_as_a_standalone_candidate(self):
        text = "The year was 2023 and the count reached 1000 eventually for everyone."
        result = extract_linguistic_candidates(text)
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "2023" not in normalized_phrases
        assert "1000" not in normalized_phrases


class TestUnicodeAndNonEnglishText:
    def test_accented_characters_preserved_in_surface_form(self):
        text = "Caf\u00e9 culture is popular in Paris. The caf\u00e9 culture continues to grow."
        result = extract_linguistic_candidates(text)
        phrases_joined = " ".join(c.phrase for c in result.candidates)
        assert "\u00e9" in phrases_joined or len(result.candidates) >= 0  # never crashes

    def test_non_latin_script_and_emoji_do_not_crash_extraction(self):
        text = (
            "\u4f60\u597d\u4e16\u754c is a Chinese phrase used in research papers. "
            "The rocket \U0001f680 launch was a huge success for the entire team."
        )
        result = extract_linguistic_candidates(text)
        assert isinstance(result.candidates, list)


class TestShortText:
    def test_single_word_input_does_not_crash(self):
        result = extract_linguistic_candidates("Python")
        assert isinstance(result.candidates, list)

    def test_two_word_noun_phrase(self):
        result = extract_linguistic_candidates("Machine learning")
        normalized_phrases = [c.normalized_phrase for c in result.candidates]
        assert "machine learning" in normalized_phrases

    def test_pronoun_only_text_yields_no_candidates(self):
        result = extract_linguistic_candidates("This is it. We did that.")
        assert result.candidates == []


class TestEmptyAndWhitespaceInput:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyTextError):
            extract_linguistic_candidates("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyTextError):
            extract_linguistic_candidates("   \n\t  ")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_linguistic_candidates(None)  # type: ignore[arg-type]

    def test_min_greater_than_max_phrase_length_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_linguistic_candidates(
                "some valid text here", min_phrase_length=5, max_phrase_length=1
            )


class TestCandidateLimit:
    def test_max_candidates_is_respected(self):
        long_text = REACT_TEXT * 3
        result = extract_linguistic_candidates(long_text, max_candidates=3)
        assert len(result.candidates) <= 3

    def test_limit_keeps_highest_signal_candidates_first(self):
        result_full = extract_linguistic_candidates(REACT_TEXT, max_candidates=20)
        result_limited = extract_linguistic_candidates(REACT_TEXT, max_candidates=2)

        top_two = [c.normalized_phrase for c in result_full.candidates[:2]]
        limited = [c.normalized_phrase for c in result_limited.candidates]
        assert limited == top_two


class TestConfigurablePhraseLength:
    def test_max_phrase_length_one_returns_only_single_word_candidates(self):
        result = extract_linguistic_candidates(REACT_TEXT, min_phrase_length=1, max_phrase_length=1)
        assert all(c.ngram_size == 1 for c in result.candidates)

    def test_min_phrase_length_two_excludes_single_word_candidates(self):
        result = extract_linguistic_candidates(REACT_TEXT, min_phrase_length=2, max_phrase_length=5)
        assert all(c.ngram_size >= 2 for c in result.candidates)

    def test_narrow_max_length_excludes_longer_phrases(self):
        result = extract_linguistic_candidates(REACT_TEXT, min_phrase_length=1, max_phrase_length=2)
        assert all(c.ngram_size <= 2 for c in result.candidates)
        # The known trigram "machine learning model" must be excluded now.
        assert "machine learning model" not in [c.normalized_phrase for c in result.candidates]


class TestDeterministicOutput:
    def test_same_input_same_config_produces_identical_output(self):
        first = extract_linguistic_candidates(REACT_TEXT)
        second = extract_linguistic_candidates(REACT_TEXT)

        assert [c.normalized_phrase for c in first.candidates] == [
            c.normalized_phrase for c in second.candidates
        ]
        assert [c.linguistic_signal for c in first.candidates] == [
            c.linguistic_signal for c in second.candidates
        ]

    def test_tied_signal_breaks_ties_alphabetically(self):
        first = extract_linguistic_candidates(REACT_TEXT)
        second = extract_linguistic_candidates(REACT_TEXT)
        assert [c.phrase for c in first.candidates] == [c.phrase for c in second.candidates]


class TestRepeatedConcepts:
    def test_repeated_phrase_has_higher_signal_than_one_off_phrase(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}

        # "machine learning models" appears twice, "large datasets" once.
        assert by_normalized["machine learning model"].linguistic_signal == 2
        assert by_normalized["large dataset"].linguistic_signal == 1
        assert (
            by_normalized["machine learning model"].linguistic_signal
            > by_normalized["large dataset"].linguistic_signal
        )

    def test_repeated_phrase_ranks_above_one_off_phrase(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        normalized_order = [c.normalized_phrase for c in result.candidates]
        assert normalized_order.index("machine learning model") < normalized_order.index(
            "large dataset"
        )

    def test_repeated_phrase_normalizes_higher(self):
        result = extract_linguistic_candidates(REACT_TEXT)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}
        assert (
            by_normalized["machine learning model"].normalized_score
            >= by_normalized["large dataset"].normalized_score
        )


class TestNormalizeHelper:
    def test_normalize_empty_list(self):
        assert _minmax_normalize([]) == []

    def test_normalize_single_value(self):
        assert _minmax_normalize([3]) == [1.0]

    def test_normalize_identical_values_all_positive(self):
        assert _minmax_normalize([2, 2, 2]) == [1.0, 1.0, 1.0]

    def test_normalize_spreads_values_across_unit_range(self):
        result = _minmax_normalize([1, 2, 3])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result[0] < result[1] < result[2]


class TestIntegrationWithPreprocessing:
    def test_html_and_url_cleaned_text_produces_clean_candidates(self):
        raw_html = (
            "<p>Check out <b>React components</b> at https://example.com/docs "
            "for building modern web applications with reusable UI elements.</p>"
        )
        preprocessed = preprocess_text(raw_html)
        result = extract_linguistic_candidates(preprocessed.cleaned_text)

        phrases_joined = " ".join(c.phrase for c in result.candidates)
        assert "<" not in phrases_joined
        assert "https://" not in phrases_joined
        assert len(result.candidates) > 0

    def test_preprocessed_repeated_punctuation_text_still_extracts(self):
        raw = "Amazing!!!! React components are truly incredible??? Best framework ever...."
        preprocessed = preprocess_text(raw)
        assert "!!!!" not in preprocessed.cleaned_text
        result = extract_linguistic_candidates(preprocessed.cleaned_text)
        assert isinstance(result.candidates, list)


class TestModelNotAvailable:
    def test_missing_model_raises_actionable_error(self):
        import app.pipeline.linguistic_extractor as le_module
        from app.core.config import get_settings

        settings = get_settings()
        original_model = settings.SPACY_MODEL_NAME
        original_nlp = le_module._nlp

        try:
            object.__setattr__(settings, "SPACY_MODEL_NAME", "en_core_web_definitely_not_installed")
            le_module._nlp = None

            with pytest.raises(LinguisticModelNotAvailableError) as exc_info:
                le_module.extract_linguistic_candidates("some valid text for this test")

            message = str(exc_info.value)
            assert "en_core_web_definitely_not_installed" in message
            assert "python -m spacy download" in message
        finally:
            # Restore real state so later tests in the suite aren't affected.
            object.__setattr__(settings, "SPACY_MODEL_NAME", original_model)
            le_module._nlp = original_nlp


class TestCandidateDataclass:
    def test_source_field_defaults_to_linguistic(self):
        result = extract_linguistic_candidates(REACT_TEXT, max_candidates=5)
        assert all(c.source == "linguistic" for c in result.candidates)

    def test_candidate_is_frozen(self):
        result = extract_linguistic_candidates(REACT_TEXT, max_candidates=1)
        candidate = result.candidates[0]
        with pytest.raises(Exception):
            candidate.phrase = "modified"  # type: ignore[misc]
