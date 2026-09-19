import pytest

from app.pipeline.linguistic_extractor import (
    LinguisticCandidate,
    LinguisticExtractionResult,
    extract_linguistic_candidates,
)
from app.pipeline.merger import MergedCandidate, merge_candidates
from app.pipeline.tfidf_extractor import (
    TfidfCandidate,
    TfidfExtractionResult,
    extract_tfidf_candidates,
)
from app.pipeline.yake_extractor import (
    YakeCandidate,
    YakeExtractionResult,
    extract_yake_candidates,
)


def _tfidf(*candidates: TfidfCandidate) -> TfidfExtractionResult:
    return TfidfExtractionResult(candidates=list(candidates))


def _yake(*candidates: YakeCandidate) -> YakeExtractionResult:
    return YakeExtractionResult(candidates=list(candidates))


def _linguistic(*candidates: LinguisticCandidate) -> LinguisticExtractionResult:
    return LinguisticExtractionResult(candidates=list(candidates))


class TestThreeSourcesSamePhrase:
    def test_exact_same_phrase_from_all_three_sources_merges_into_one(self):
        tfidf = _tfidf(
            TfidfCandidate(
                phrase="react components",
                raw_score=1.2,
                normalized_score=0.8,
                ngram_size=2,
                document_frequency=2,
            )
        )
        yake = _yake(
            YakeCandidate(
                phrase="React components",
                raw_score=0.01,
                normalized_score=0.9,
                ngram_size=2,
            )
        )
        linguistic = _linguistic(
            LinguisticCandidate(
                phrase="React components",
                normalized_phrase="react component",
                linguistic_signal=2,
                normalized_score=1.0,
                ngram_size=2,
                root_pos="NOUN",
            )
        )

        result = merge_candidates(tfidf, yake, linguistic)

        assert len(result.candidates) == 1
        merged = result.candidates[0]
        assert merged.sources == ("tfidf", "yake", "linguistic")
        assert merged.source_count == 3
        assert merged.source_consensus == pytest.approx(1.0)

    def test_all_three_source_signals_preserved_individually(self):
        tfidf = _tfidf(
            TfidfCandidate("react components", 1.2, 0.8, 2, 2)
        )
        yake = _yake(YakeCandidate("React components", 0.01, 0.9, 2))
        linguistic = _linguistic(
            LinguisticCandidate("React components", "react component", 2, 1.0, 2, "NOUN")
        )

        result = merge_candidates(tfidf, yake, linguistic)
        signals = result.candidates[0].source_signals

        assert signals == {"tfidf": 0.8, "yake": 0.9, "linguistic": 1.0}


class TestSamePhraseDifferentCasing:
    def test_casing_variants_across_sources_merge(self):
        tfidf = _tfidf(TfidfCandidate("nasa", 0.5, 0.5, 1, 1))
        yake = _yake(YakeCandidate("NASA", 0.02, 0.7, 1))

        result = merge_candidates(tfidf, yake)

        assert len(result.candidates) == 1
        assert result.candidates[0].sources == ("tfidf", "yake")

    def test_yake_surface_preferred_over_tfidf_lowercase(self):
        tfidf = _tfidf(TfidfCandidate("nasa", 0.5, 0.5, 1, 1))
        yake = _yake(YakeCandidate("NASA", 0.02, 0.7, 1))

        result = merge_candidates(tfidf, yake)
        assert result.candidates[0].phrase == "NASA"


class TestLemmaNormalization:
    def test_linguistic_lemma_becomes_the_normalized_phrase(self):
        tfidf = _tfidf(TfidfCandidate("react components", 1.0, 0.8, 2, 1))
        linguistic = _linguistic(
            LinguisticCandidate("React components", "react component", 1, 1.0, 2, "NOUN")
        )

        result = merge_candidates(tfidf, linguistic_result=linguistic)
        merged = result.candidates[0]

        assert merged.normalized_phrase == "react component"
        assert merged.phrase in ("React components", "react components")

    def test_without_linguistic_source_normalized_phrase_falls_back_to_merge_key(self):
        tfidf = _tfidf(TfidfCandidate("react components", 1.0, 0.8, 2, 1))
        yake = _yake(YakeCandidate("React Components", 0.01, 0.9, 2))

        result = merge_candidates(tfidf, yake)
        merged = result.candidates[0]

        assert merged.normalized_phrase == "react components"


class TestSingleSourceCandidate:
    def test_candidate_found_by_only_tfidf(self):
        tfidf = _tfidf(TfidfCandidate("datasets", 0.5, 0.6, 1, 1))
        result = merge_candidates(tfidf)

        assert len(result.candidates) == 1
        merged = result.candidates[0]
        assert merged.sources == ("tfidf",)
        assert merged.source_count == 1
        assert merged.source_consensus == pytest.approx(1.0)  
        assert merged.source_signals == {"tfidf": 0.6}

    def test_candidate_found_by_only_yake(self):
        yake = _yake(YakeCandidate("Real-time systems", 0.02, 0.8, 3))
        result = merge_candidates(yake_result=yake)

        merged = result.candidates[0]
        assert merged.sources == ("yake",)
        assert merged.phrase == "Real-time systems"

    def test_candidate_found_by_only_linguistic(self):
        linguistic = _linguistic(
            LinguisticCandidate("machine learning", "machine learning", 1, 1.0, 2, "NOUN")
        )
        result = merge_candidates(linguistic_result=linguistic)

        merged = result.candidates[0]
        assert merged.sources == ("linguistic",)
        assert merged.source_consensus == pytest.approx(1.0)


class TestTwoSourceCandidate:
    def test_consensus_reflects_fraction_of_provided_sources_not_hardcoded_three(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        yake = _yake(YakeCandidate("React", 0.01, 0.85, 1))

        result = merge_candidates(tfidf, yake)
        merged = result.candidates[0]

        assert merged.source_count == 2
        assert merged.source_consensus == pytest.approx(1.0)  

    def test_two_of_three_provided_sources_agreeing_gives_two_thirds_consensus(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        yake = _yake(YakeCandidate("React", 0.01, 0.85, 1))
        linguistic = _linguistic()  

        result = merge_candidates(tfidf, yake, linguistic)
        merged = result.candidates[0]

        assert merged.source_count == 2
        assert merged.source_consensus == pytest.approx(2 / 3)


class TestAllThreeSources:
    def test_three_of_three_gives_full_consensus(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        yake = _yake(YakeCandidate("React", 0.01, 0.85, 1))
        linguistic = _linguistic(
            LinguisticCandidate("React", "react", 1, 1.0, 1, "PROPN")
        )

        result = merge_candidates(tfidf, yake, linguistic)
        merged = result.candidates[0]

        assert merged.source_count == 3
        assert merged.source_consensus == pytest.approx(1.0)


class TestNoAveragingIncompatibleScores:
    def test_source_signals_are_kept_separate_not_averaged(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.1, 1, 1))
        yake = _yake(YakeCandidate("React", 0.01, 0.95, 1))

        result = merge_candidates(tfidf, yake)
        merged = result.candidates[0]

        assert merged.source_signals["tfidf"] == pytest.approx(0.1)
        assert merged.source_signals["yake"] == pytest.approx(0.95)
        assert not hasattr(merged, "combined_score")
        assert not hasattr(merged, "average_score")

    def test_missing_source_is_absent_not_zero(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.7, 1, 1))
        result = merge_candidates(tfidf, yake_result=_yake(), linguistic_result=_linguistic())
        merged = result.candidates[0]

        assert "yake" not in merged.source_signals
        assert "linguistic" not in merged.source_signals
        assert merged.source_signals == {"tfidf": 0.7}


class TestUnrelatedPhrasesStaySeparate:
    def test_different_words_never_merge(self):
        tfidf = _tfidf(
            TfidfCandidate("react", 0.9, 0.9, 1, 1),
            TfidfCandidate("vue", 0.5, 0.5, 1, 1),
        )
        result = merge_candidates(tfidf)

        assert len(result.candidates) == 2
        phrases = {c.normalized_phrase for c in result.candidates}
        assert phrases == {"react", "vue"}

    def test_hyphenated_and_spaced_variants_stay_separate(self):
        yake = _yake(YakeCandidate("machine-learning", 0.02, 0.9, 1))
        tfidf = _tfidf(TfidfCandidate("machine learning", 0.8, 0.8, 2, 1))

        result = merge_candidates(tfidf, yake)

        assert len(result.candidates) == 2
        normalized = {c.normalized_phrase for c in result.candidates}
        assert normalized == {"machine learning", "machine-learning"}


class TestSemanticallyRelatedButTextuallyDifferentPhrasesStaySeparate:
    def test_related_concepts_with_different_wording_do_not_merge(self):
        yake = _yake(
            YakeCandidate("frontend development", 0.02, 0.9, 2),
            YakeCandidate("client-side development", 0.03, 0.85, 2),
            YakeCandidate("web UI engineering", 0.04, 0.8, 3),
        )
        result = merge_candidates(yake_result=yake)

        assert len(result.candidates) == 3


class TestEmptyExtractorOutputs:
    def test_one_empty_extractor_among_others_is_handled(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        yake = _yake()  
        result = merge_candidates(tfidf, yake)

        assert len(result.candidates) == 1
        assert result.total_sources == 2

    def test_all_extractors_empty_produces_empty_merge_result(self):
        result = merge_candidates(_tfidf(), _yake(), _linguistic())
        assert result.candidates == []
        assert result.total_sources == 3

    def test_all_none_raises_value_error(self):
        with pytest.raises(ValueError):
            merge_candidates()


class TestDuplicateCandidatesFromSameSource:
    def test_duplicate_within_one_source_does_not_double_count_source(self):
        tfidf = _tfidf(
            TfidfCandidate("react", 0.9, 0.7, 1, 1),
            TfidfCandidate("REACT", 0.9, 0.9, 1, 1),
        )
        result = merge_candidates(tfidf)

        assert len(result.candidates) == 1
        merged = result.candidates[0]
        assert merged.sources == ("tfidf",)
        assert merged.source_count == 1

    def test_duplicate_within_one_source_keeps_higher_score(self):
        tfidf = _tfidf(
            TfidfCandidate("react", 0.9, 0.7, 1, 1),
            TfidfCandidate("REACT", 0.9, 0.9, 1, 1),
        )
        result = merge_candidates(tfidf)
        assert result.candidates[0].source_signals["tfidf"] == pytest.approx(0.9)

    def test_duplicate_within_one_source_surface_choice_is_deterministic(self):
        tfidf = _tfidf(
            TfidfCandidate("react", 0.9, 0.7, 1, 1),
            TfidfCandidate("REACT", 0.9, 0.9, 1, 1),
        )
        first = merge_candidates(tfidf).candidates[0].phrase
        second = merge_candidates(tfidf).candidates[0].phrase
        assert first == second


class TestCandidateLimit:
    def test_max_candidates_is_respected(self):
        tfidf = _tfidf(
            *[
                TfidfCandidate(f"term{i}", 1.0, 1.0 - i * 0.01, 1, 1)
                for i in range(20)
            ]
        )
        result = merge_candidates(tfidf, max_candidates=5)
        assert len(result.candidates) == 5

    def test_limit_keeps_highest_consensus_candidates_first(self):
        tfidf = _tfidf(
            TfidfCandidate("solo", 1.0, 1.0, 1, 1),
            TfidfCandidate("shared", 0.5, 0.5, 1, 1),
        )
        yake = _yake(YakeCandidate("shared", 0.01, 0.5, 1))

        result = merge_candidates(tfidf, yake, max_candidates=1)
        assert result.candidates[0].normalized_phrase == "shared"


class TestDeterministicOrdering:
    def test_merging_same_inputs_twice_produces_identical_result(self):
        tfidf = _tfidf(
            TfidfCandidate("react components", 1.2, 0.8, 2, 2),
            TfidfCandidate("machine learning", 0.9, 0.6, 2, 1),
        )
        yake = _yake(
            YakeCandidate("React components", 0.01, 0.9, 2),
            YakeCandidate("Machine learning", 0.02, 0.7, 2),
        )

        first = merge_candidates(tfidf, yake)
        second = merge_candidates(tfidf, yake)

        assert [c.phrase for c in first.candidates] == [c.phrase for c in second.candidates]
        assert [c.source_signals for c in first.candidates] == [
            c.source_signals for c in second.candidates
        ]

    def test_tied_candidates_break_ties_alphabetically_by_normalized_phrase(self):
        tfidf = _tfidf(
            TfidfCandidate("zebra", 0.5, 0.5, 1, 1),
            TfidfCandidate("apple", 0.5, 0.5, 1, 1),
        )
        result = merge_candidates(tfidf)
        normalized_order = [c.normalized_phrase for c in result.candidates]
        assert normalized_order == ["apple", "zebra"]

    def test_higher_source_count_ranks_above_lower_source_count(self):
        tfidf = _tfidf(
            TfidfCandidate("solo", 1.0, 1.0, 1, 1),
            TfidfCandidate("shared", 0.1, 0.1, 1, 1),
        )
        yake = _yake(YakeCandidate("shared", 0.01, 0.1, 1))

        result = merge_candidates(tfidf, yake)
        normalized_order = [c.normalized_phrase for c in result.candidates]
        assert normalized_order.index("shared") < normalized_order.index("solo")


class TestMergedCandidateDataclass:
    def test_ngram_size_matches_selected_phrase_word_count(self):
        tfidf = _tfidf(TfidfCandidate("machine learning models", 1.0, 1.0, 3, 1))
        result = merge_candidates(tfidf)
        assert result.candidates[0].ngram_size == 3

    def test_candidate_is_frozen(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        merged = merge_candidates(tfidf).candidates[0]
        with pytest.raises(Exception):
            merged.phrase = "modified"  

    def test_instance_is_merged_candidate_type(self):
        tfidf = _tfidf(TfidfCandidate("react", 0.9, 0.9, 1, 1))
        merged = merge_candidates(tfidf).candidates[0]
        assert isinstance(merged, MergedCandidate)


class TestIntegrationWithRealExtractors:
    REACT_TEXT = (
        "React components make building React Native applications much easier. "
        "React components are reusable. Machine learning models require large "
        "datasets. Machine learning models are powerful. NASA and IBM use "
        "real-time systems daily."
    )

    def test_real_extractors_merge_without_error(self):
        tfidf = extract_tfidf_candidates(self.REACT_TEXT, max_candidates=30)
        yake = extract_yake_candidates(self.REACT_TEXT, max_candidates=30)
        linguistic = extract_linguistic_candidates(self.REACT_TEXT, max_candidates=30)

        result = merge_candidates(tfidf, yake, linguistic, max_candidates=20)

        assert result.total_sources == 3
        assert len(result.candidates) > 0

    def test_react_components_reaches_full_consensus_from_real_extractors(self):
        tfidf = extract_tfidf_candidates(self.REACT_TEXT, max_candidates=30)
        yake = extract_yake_candidates(self.REACT_TEXT, max_candidates=30)
        linguistic = extract_linguistic_candidates(self.REACT_TEXT, max_candidates=30)

        result = merge_candidates(tfidf, yake, linguistic, max_candidates=30)
        by_normalized = {c.normalized_phrase: c for c in result.candidates}

        assert "react component" in by_normalized
        assert by_normalized["react component"].source_count == 3
        assert by_normalized["react component"].normalized_phrase == "react component"

    def test_merging_real_extractor_output_twice_is_deterministic(self):
        tfidf = extract_tfidf_candidates(self.REACT_TEXT, max_candidates=30)
        yake = extract_yake_candidates(self.REACT_TEXT, max_candidates=30)
        linguistic = extract_linguistic_candidates(self.REACT_TEXT, max_candidates=30)

        first = merge_candidates(tfidf, yake, linguistic, max_candidates=20)
        second = merge_candidates(tfidf, yake, linguistic, max_candidates=20)

        assert [c.phrase for c in first.candidates] == [c.phrase for c in second.candidates]