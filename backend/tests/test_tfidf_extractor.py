
import pytest

from app.pipeline.exceptions import EmptyTextError
from app.pipeline.preprocessor import preprocess_text
from app.pipeline.tfidf_extractor import (
    TfidfCandidate,
    _is_low_quality_phrase,
    _minmax_normalize,
    extract_tfidf_candidates,
)

ML_TEXT = (
    "Machine learning models require large datasets to train effectively. "
    "Machine learning models are powerful tools for prediction. "
    "Data scientists often use machine learning models in production systems."
)


class TestRepeatedImportantTerms:
    def test_repeated_term_outranks_terms_seen_once(self):
        result = extract_tfidf_candidates(ML_TEXT)
        phrases = [c.phrase for c in result.candidates]

        top_phrases = phrases[:6]
        assert "machine learning models" in top_phrases
        assert "datasets" not in top_phrases

    def test_repeated_term_has_higher_document_frequency(self):
        result = extract_tfidf_candidates(ML_TEXT)
        by_phrase = {c.phrase: c for c in result.candidates}

        assert by_phrase["machine learning models"].document_frequency == 3
        assert by_phrase["datasets"].document_frequency == 1

    def test_sublinear_scaling_dampens_pure_repetition(self):
        text = "spam spam spam spam spam spam spam spam spam spam. Ham appears once here."
        result = extract_tfidf_candidates(text)
        by_phrase = {c.phrase: c for c in result.candidates}

        assert by_phrase["spam"].raw_score < 10 * by_phrase["ham"].raw_score


class TestMultiWordPhrases:
    def test_meaningful_trigram_is_extracted(self):
        result = extract_tfidf_candidates(ML_TEXT)
        phrases = [c.phrase for c in result.candidates]
        assert "machine learning models" in phrases

    def test_meaningful_bigram_is_extracted(self):
        result = extract_tfidf_candidates(ML_TEXT)
        phrases = [c.phrase for c in result.candidates]
        assert "machine learning" in phrases
        assert "learning models" in phrases

    def test_ngram_size_field_matches_word_count(self):
        result = extract_tfidf_candidates(ML_TEXT)
        by_phrase = {c.phrase: c for c in result.candidates}

        assert by_phrase["machine learning models"].ngram_size == 3
        assert by_phrase["machine learning"].ngram_size == 2
        assert by_phrase["machine"].ngram_size == 1


class TestNgramRangeGeneration:
    def test_default_range_produces_unigrams_bigrams_trigrams(self):
        result = extract_tfidf_candidates(ML_TEXT)
        sizes_present = {c.ngram_size for c in result.candidates}
        assert sizes_present == {1, 2, 3}

    def test_unigrams_only_when_range_restricted(self):
        result = extract_tfidf_candidates(ML_TEXT, ngram_range=(1, 1))
        sizes_present = {c.ngram_size for c in result.candidates}
        assert sizes_present == {1}

    def test_bigrams_only_when_range_restricted(self):
        result = extract_tfidf_candidates(ML_TEXT, ngram_range=(2, 2))
        sizes_present = {c.ngram_size for c in result.candidates}
        assert sizes_present == {2}


class TestStopwordAndNoiseHandling:
    def test_no_candidate_is_a_bare_stopword(self):
        text = "The team is going to the market and it is very far from here."
        result = extract_tfidf_candidates(text)
        bare_stopwords = {"the", "is", "to", "and", "it", "from", "here"}
        phrases = {c.phrase for c in result.candidates}
        assert phrases.isdisjoint(bare_stopwords)

    def test_no_candidate_starts_or_ends_with_a_stopword(self):
        text = "The team is going to the market and it is very far from here."
        result = extract_tfidf_candidates(text)
        for c in result.candidates:
            words = c.phrase.split()
            assert words[0] not in {"the", "is", "to", "and", "it", "from"}
            assert words[-1] not in {"the", "is", "to", "and", "it", "from"}

    def test_stopword_only_input_yields_no_candidates(self):
        result = extract_tfidf_candidates("the a an is of it")
        assert result.candidates == []

    def test_is_low_quality_phrase_directly(self):
        assert _is_low_quality_phrase("the") is True
        assert _is_low_quality_phrase("is of") is True
        assert _is_low_quality_phrase("the react") is True
        assert _is_low_quality_phrase("react is") is True
        assert _is_low_quality_phrase("state of the art") is False
        assert _is_low_quality_phrase("react") is False
        assert _is_low_quality_phrase("") is True


class TestShortText:
    def test_two_word_input_does_not_crash(self):
        result = extract_tfidf_candidates("React hooks")
        phrases = {c.phrase for c in result.candidates}
        assert "react" in phrases
        assert "hooks" in phrases
        assert "react hooks" in phrases

    def test_short_text_uses_single_document_fallback(self):
        result = extract_tfidf_candidates("React hooks")
        assert result.used_single_document_fallback is True
        assert result.sentence_count == 1

    def test_short_text_still_produces_valid_scores(self):
        result = extract_tfidf_candidates("React hooks")
        for c in result.candidates:
            assert c.raw_score > 0
            assert 0.0 <= c.normalized_score <= 1.0


class TestEmptyAndInvalidInput:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyTextError):
            extract_tfidf_candidates("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyTextError):
            extract_tfidf_candidates("   \n\t  ")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_tfidf_candidates(None)  


class TestNoUsableVocabulary:
    def test_only_symbols_and_single_digits_yields_no_candidates(self):
        result = extract_tfidf_candidates("1 2 3 ! ? -")
        assert result.candidates == []
        assert result.vocabulary_size == 0

    def test_no_usable_vocabulary_still_returns_valid_metadata(self):
        result = extract_tfidf_candidates("1 2 3 ! ? -")
        assert result.sentence_count >= 1
        assert isinstance(result.used_single_document_fallback, bool)


class TestDeterministicRanking:
    def test_same_input_produces_identical_ranking(self):
        first = extract_tfidf_candidates(ML_TEXT)
        second = extract_tfidf_candidates(ML_TEXT)

        first_seq = [(c.phrase, c.raw_score) for c in first.candidates]
        second_seq = [(c.phrase, c.raw_score) for c in second.candidates]
        assert first_seq == second_seq

    def test_tied_scores_break_ties_alphabetically(self):
        result = extract_tfidf_candidates(ML_TEXT)
        top_score = result.candidates[0].raw_score
        tied = [c.phrase for c in result.candidates if c.raw_score == top_score]
        assert tied == sorted(tied)


class TestCandidateLimit:
    def test_max_candidates_is_respected(self):
        long_text = (
            "Apples bananas cherries dates elderberries figs grapes honeydew. "
            "Ice cream jam kiwi lemons mangoes nectarines oranges papayas. "
            "Quince raspberries strawberries tangerines ugli vanilla watermelon xigua. "
            "Yams zucchini apricots blueberries cranberries dragonfruit eggplant fennel."
        )
        result = extract_tfidf_candidates(long_text, max_candidates=5)
        assert len(result.candidates) == 5

    def test_default_limit_caps_large_vocabulary(self):
        result = extract_tfidf_candidates(ML_TEXT, max_candidates=3)
        assert len(result.candidates) == 3
        full_result = extract_tfidf_candidates(ML_TEXT)
        assert [c.phrase for c in result.candidates] == [
            c.phrase for c in full_result.candidates[:3]
        ]


class TestRawScorePreservation:
    def test_raw_score_is_not_overwritten_by_normalization(self):
        result = extract_tfidf_candidates(ML_TEXT)
        raw_scores = [c.raw_score for c in result.candidates]
        normalized_scores = [c.normalized_score for c in result.candidates]

        assert raw_scores != normalized_scores
        assert all(score > 0 for score in raw_scores)

    def test_raw_scores_are_real_floats_not_ranks(self):
        result = extract_tfidf_candidates(ML_TEXT)
        for c in result.candidates:
            assert isinstance(c.raw_score, float)
            assert c.raw_score > 0.0


class TestScoreNormalization:
    def test_top_candidate_normalizes_close_to_one(self):
        result = extract_tfidf_candidates(ML_TEXT)
        assert result.candidates[0].normalized_score == pytest.approx(1.0)

    def test_lowest_candidate_normalizes_close_to_zero(self):
        result = extract_tfidf_candidates(ML_TEXT)
        assert result.candidates[-1].normalized_score == pytest.approx(0.0, abs=1e-6)

    def test_normalized_scores_stay_within_unit_range(self):
        result = extract_tfidf_candidates(ML_TEXT)
        for c in result.candidates:
            assert 0.0 <= c.normalized_score <= 1.0

    def test_minmax_normalize_function_directly(self):
        assert _minmax_normalize([]) == []
        assert _minmax_normalize([5.0]) == [1.0]
        assert _minmax_normalize([0.0]) == [0.0]
        assert _minmax_normalize([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]
        assert _minmax_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


class TestIntegrationWithPreprocessing:
    def test_html_and_url_cleaned_text_produces_clean_candidates(self):
        raw = (
            "<p>Check out <b>React components</b> at https://example.com/docs "
            "and learn about React components today!!!</p>"
        )
        preprocessed = preprocess_text(raw)
        result = extract_tfidf_candidates(preprocessed.cleaned_text)

        phrases = [c.phrase for c in result.candidates]
        joined = " ".join(phrases)

        assert "example.com" not in joined
        assert "https" not in joined
        assert "<" not in joined and ">" not in joined
        assert "react components" in phrases

    def test_preprocessed_text_with_repeated_punctuation_is_clean(self):
        raw = "Amazing!!! Truly amazing??? This is amazing... Amazing stuff everywhere."
        preprocessed = preprocess_text(raw)
        result = extract_tfidf_candidates(preprocessed.cleaned_text)

        for c in result.candidates:
            assert "!" not in c.phrase
            assert "?" not in c.phrase
            assert "..." not in c.phrase


class TestCandidateDataclass:
    def test_source_field_defaults_to_tfidf(self):
        result = extract_tfidf_candidates(ML_TEXT)
        assert all(c.source == "tfidf" for c in result.candidates)

    def test_candidate_is_frozen(self):
        candidate = TfidfCandidate(
            phrase="test", raw_score=1.0, normalized_score=1.0, ngram_size=1, document_frequency=1
        )
        with pytest.raises(Exception):
            candidate.phrase = "changed"  
