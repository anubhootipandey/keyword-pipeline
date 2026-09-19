import pytest

from app.pipeline.normalizer import normalize_for_matching


class TestCaseDifferences:
    def test_uppercase_and_lowercase_match(self):
        assert normalize_for_matching("React Components") == normalize_for_matching(
            "react components"
        )

    def test_mixed_case_matches(self):
        assert normalize_for_matching("rEACT compONENTS") == normalize_for_matching(
            "React Components"
        )

    def test_result_is_lowercase(self):
        assert normalize_for_matching("REACT") == "react"


class TestWhitespace:
    def test_surrounding_whitespace_is_stripped(self):
        assert normalize_for_matching("  React Components  ") == normalize_for_matching(
            "React Components"
        )

    def test_repeated_internal_whitespace_collapses(self):
        assert normalize_for_matching("React    Components") == normalize_for_matching(
            "React Components"
        )

    def test_tabs_and_newlines_normalize_like_spaces(self):
        assert normalize_for_matching("React\t\nComponents") == normalize_for_matching(
            "React Components"
        )


class TestPunctuation:
    def test_leading_and_trailing_punctuation_is_stripped(self):
        assert normalize_for_matching("-caching-") == "caching"
        assert normalize_for_matching("(caching)") == "caching"
        assert normalize_for_matching("caching!") == "caching"

    def test_internal_hyphen_is_preserved(self):
        assert normalize_for_matching("real-time") == "real-time"

    def test_internal_ampersand_is_preserved(self):
        assert normalize_for_matching("R&D projects") == "r&d projects"

    def test_hyphenated_and_spaced_forms_remain_distinct(self):
        assert normalize_for_matching("machine learning") != normalize_for_matching(
            "machine-learning"
        )

    def test_unrelated_but_overlapping_phrases_remain_distinct(self):
        assert normalize_for_matching("machine learning") != normalize_for_matching(
            "machine learning systems"
        )


class TestPossessives:
    def test_singular_possessive_is_stripped(self):
        assert normalize_for_matching("company's strategy") == normalize_for_matching(
            "company strategy"
        )

    def test_plural_possessive_trailing_apostrophe_is_stripped(self):
        assert normalize_for_matching("developers' productivity") == normalize_for_matching(
            "developers productivity"
        )

    def test_curly_apostrophe_possessive_is_stripped(self):
        assert normalize_for_matching("company\u2019s strategy") == normalize_for_matching(
            "company strategy"
        )

    def test_possessive_stripping_does_not_affect_unrelated_trailing_s(self):
        assert normalize_for_matching("machine learning models") == "machine learning models"


class TestHyphenatedPhrases:
    def test_hyphenated_phrase_normalizes_consistently(self):
        assert normalize_for_matching("Real-Time Systems") == normalize_for_matching(
            "real-time systems"
        )

    def test_multi_hyphen_phrase_is_preserved_as_is(self):
        result = normalize_for_matching("state-of-the-art")
        assert result == "state-of-the-art"


class TestUnicode:
    def test_accented_characters_are_preserved(self):
        assert normalize_for_matching("Caf\u00e9 Culture") == normalize_for_matching(
            "caf\u00e9 culture"
        )
        assert "\u00e9" in normalize_for_matching("Caf\u00e9")

    def test_accents_are_not_stripped_to_ascii(self):
        assert normalize_for_matching("caf\u00e9") != "cafe"

    def test_non_latin_script_does_not_crash(self):
        result = normalize_for_matching("\u4f60\u597d\u4e16\u754c")
        assert result == "\u4f60\u597d\u4e16\u754c"

    def test_nfkc_canonical_equivalence(self):
        precomposed = "caf\u00e9"
        decomposed = "cafe\u0301"
        assert precomposed != decomposed 
        assert normalize_for_matching(precomposed) == normalize_for_matching(decomposed)


class TestEmptyAndInvalidInput:
    def test_empty_string_returns_empty_string(self):
        assert normalize_for_matching("") == ""

    def test_whitespace_only_returns_empty_string(self):
        assert normalize_for_matching("   \n\t  ") == ""

    def test_punctuation_only_returns_empty_string(self):
        assert normalize_for_matching("!!!---...") == ""

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_for_matching(None)  

        with pytest.raises(TypeError):
            normalize_for_matching(123)  


class TestDeterminism:
    def test_same_input_always_produces_same_output(self):
        for _ in range(5):
            assert normalize_for_matching("React Components") == "react components"

    def test_semantically_unrelated_phrases_are_not_accidentally_equal(self):
        assert normalize_for_matching("apple") != normalize_for_matching("orange")