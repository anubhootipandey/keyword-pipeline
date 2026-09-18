"""
Tests for Stage 1 — text preprocessing.

These test behavior (what the output actually looks like / what gets
raised), not just "does the function execute without crashing". Each
test class groups tests for one of the required scenarios from the
project spec (section 27, "preprocessing" bullet list).
"""

import pytest

from app.pipeline.exceptions import EmptyTextError, TextTooLongError
from app.pipeline.preprocessor import (
    extract_and_remove_urls,
    normalize_punctuation,
    normalize_whitespace,
    preprocess_text,
    strip_html,
)


class TestNormalProse:
    def test_normal_prose_is_preserved(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = preprocess_text(text)

        assert result.cleaned_text == text
        assert result.raw_text == text
        assert result.word_count == 9
        assert result.urls == []
        assert result.html_was_present is False

    def test_casing_and_word_choice_are_not_altered(self):
        text = "React Components are NOT the same as react-native Modules."
        result = preprocess_text(text)

        # Casing and word choice must survive untouched — normalization
        # for matching purposes belongs to a later stage, not this one.
        assert "React Components" in result.cleaned_text
        assert "react-native" in result.cleaned_text
        assert "NOT" in result.cleaned_text


class TestEmptyAndWhitespaceInput:
    def test_empty_string_raises(self):
        with pytest.raises(EmptyTextError):
            preprocess_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyTextError):
            preprocess_text("   \n\t  \n   ")

    def test_non_string_input_raises_type_error(self):
        with pytest.raises(TypeError):
            preprocess_text(None)  # type: ignore[arg-type]

    def test_text_reduced_to_nothing_after_cleaning_raises(self):
        # Only a URL and HTML tags, no actual visible text content.
        with pytest.raises(EmptyTextError):
            preprocess_text("<div></div> https://example.com/only-a-url")

    def test_too_long_input_raises(self):
        from app.core.config import get_settings

        max_len = get_settings().MAX_INPUT_LENGTH
        too_long_text = "word " * (max_len // 4)  # comfortably over the limit
        assert len(too_long_text) > max_len

        with pytest.raises(TextTooLongError) as exc_info:
            preprocess_text(too_long_text)

        assert exc_info.value.max_length == max_len


class TestExcessiveWhitespace:
    def test_multiple_newlines_and_spaces_collapse(self):
        text = "Hello   \n\n\n   World  \t\t  Again"
        result = preprocess_text(text)

        assert result.cleaned_text == "Hello World Again"

    def test_normalize_whitespace_function_directly(self):
        assert normalize_whitespace("  a   b\n\nc\t\td  ") == "a b c d"

    def test_leading_and_trailing_whitespace_is_stripped(self):
        result = preprocess_text("   surrounded by spaces   ")
        assert result.cleaned_text == "surrounded by spaces"


class TestHTMLHandling:
    def test_tags_are_removed_text_is_kept(self):
        result = preprocess_text("<p>Hello <b>World</b></p>")
        assert result.cleaned_text == "Hello World"
        assert result.html_was_present is True

    def test_script_and_style_content_is_discarded(self):
        html_text = "<script>alert('x')</script><style>.a{color:red}</style><p>Real content</p>"
        result = preprocess_text(html_text)

        assert result.cleaned_text == "Real content"
        assert "alert" not in result.cleaned_text
        assert "color" not in result.cleaned_text

    def test_html_entities_are_decoded(self):
        result = preprocess_text("<p>Tom &amp; Jerry</p>")
        assert "Tom & Jerry" == result.cleaned_text

    def test_plain_text_without_markup_is_unaffected(self):
        text = "Just plain text, nothing HTML about it."
        result = preprocess_text(text)
        assert result.cleaned_text == text
        assert result.html_was_present is False

    def test_strip_html_function_directly(self):
        assert strip_html("no tags here") == "no tags here"
        assert strip_html("<i>italic</i> text") == "italic text"


class TestURLHandling:
    def test_urls_are_removed_from_cleaned_text(self):
        text = "Check out https://example.com/page?q=1 for more info."
        result = preprocess_text(text)

        assert "https://example.com" not in result.cleaned_text
        assert "example.com" not in result.cleaned_text
        assert "Check out" in result.cleaned_text
        assert "for more info" in result.cleaned_text

    def test_urls_are_captured_in_metadata(self):
        text = "See https://example.com/docs and www.another-site.org for details."
        result = preprocess_text(text)

        assert len(result.urls) == 2
        assert any("example.com" in u for u in result.urls)
        assert any("another-site.org" in u for u in result.urls)

    def test_url_removal_does_not_glue_adjacent_words_together(self):
        cleaned, _ = extract_and_remove_urls("seehttps://example.com forinfo")
        # A space replaces the URL, so words on either side stay separated
        # rather than merging into "seeforinfo".
        assert cleaned.split() == ["see", "forinfo"]

    def test_trailing_sentence_punctuation_excluded_from_url(self):
        _, urls = extract_and_remove_urls("Visit https://example.com/page.")
        assert urls == ["https://example.com/page"]

    def test_text_with_no_urls_is_unaffected(self):
        text = "There are no links in this sentence at all."
        cleaned, urls = extract_and_remove_urls(text)
        assert cleaned == text
        assert urls == []


class TestRepeatedPunctuation:
    def test_repeated_exclamation_marks_collapse(self):
        result = preprocess_text("This is amazing!!!")
        assert result.cleaned_text == "This is amazing!"

    def test_repeated_question_marks_collapse(self):
        assert normalize_punctuation("Really???") == "Really?"

    def test_repeated_periods_collapse(self):
        assert normalize_punctuation("Wait......") == "Wait."

    def test_single_punctuation_is_untouched(self):
        # Regression guard: must not collapse legitimate single
        # occurrences like hyphens in compound words or apostrophes.
        text = "state-of-the-art doesn't break."
        assert normalize_punctuation(text) == text

    def test_mixed_repeated_punctuation_in_full_pipeline(self):
        result = preprocess_text("Wow!!! Is this real??? I can't believe it...")
        assert "!!!" not in result.cleaned_text
        assert "???" not in result.cleaned_text
        assert "can't" in result.cleaned_text  # single apostrophe preserved


class TestUnicodeAndNonAscii:
    def test_accented_characters_preserved(self):
        text = "Caf\u00e9 r\u00e9sum\u00e9 na\u00efve"
        result = preprocess_text(text)
        assert result.cleaned_text == text

    def test_non_latin_scripts_preserved(self):
        text = "\u4f60\u597d\u4e16\u754c means hello world in Chinese"
        result = preprocess_text(text)
        assert "\u4f60\u597d\u4e16\u754c" in result.cleaned_text

    def test_emoji_preserved(self):
        text = "Launching the rocket \U0001f680 today"
        result = preprocess_text(text)
        assert "\U0001f680" in result.cleaned_text

    def test_non_breaking_space_is_normalized_like_regular_whitespace(self):
        text = "Word1\u00a0\u00a0Word2"
        result = preprocess_text(text)
        assert result.cleaned_text == "Word1 Word2"


class TestCodeLikeContent:
    def test_comparison_operators_do_not_crash_html_stripping(self):
        text = "if (x < 10 && y > 5) { doSomething(); }"
        result = preprocess_text(text)

        assert "doSomething" in result.cleaned_text
        assert "&&" in result.cleaned_text

    def test_angle_brackets_without_real_tags_pass_through(self):
        text = "a < b and c > d without tags at all"
        result = preprocess_text(text)
        assert result.cleaned_text == text

    def test_code_snippet_with_braces_and_semicolons_survives(self):
        text = "const value = getValue(); if (value !== null) { console.log(value); }"
        result = preprocess_text(text)

        assert "getValue" in result.cleaned_text
        assert "console.log" in result.cleaned_text
        assert "!==" in result.cleaned_text


class TestMultiWordPhrasesStayIntact:
    def test_named_multi_word_phrases_remain_contiguous(self):
        text = (
            "React components make building React Native applications "
            "much easier with reusable UI components."
        )
        result = preprocess_text(text)

        assert "React components" in result.cleaned_text
        assert "React Native applications" in result.cleaned_text
        assert "reusable UI components" in result.cleaned_text

    def test_hyphenated_compound_terms_stay_together(self):
        text = "This is a state-of-the-art, machine-learning-based system."
        result = preprocess_text(text)

        assert "state-of-the-art" in result.cleaned_text
        assert "machine-learning-based" in result.cleaned_text


class TestDeterminism:
    def test_same_input_produces_identical_output(self):
        text = "Some <b>HTML</b> with a https://example.com link and repeats!!!"
        first = preprocess_text(text)
        second = preprocess_text(text)

        assert first.cleaned_text == second.cleaned_text
        assert first.urls == second.urls
        assert first.word_count == second.word_count
