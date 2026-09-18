"""
Pipeline-specific exceptions.

Each error here represents one specific, expected failure mode. The
point of having distinct exception classes instead of raising a bare
`ValueError` everywhere is that calling code (and eventually the API
layer in Phase 11) can catch and handle each case differently — e.g.
"empty input" and "input too long" should probably produce different
HTTP status codes and messages, not the same generic 500.

None of these are caught and silently discarded anywhere in the
pipeline. They are meant to propagate up to whatever layer knows how to
turn them into a user-facing response.
"""


class EmptyTextError(ValueError):
    """Raised when input text is empty, whitespace-only, or reduces to
    nothing usable after cleaning (e.g. text that was only a URL or only
    HTML markup with no visible text)."""


class TextTooLongError(ValueError):
    """Raised when input text exceeds the configured maximum length."""

    def __init__(self, length: int, max_length: int):
        self.length = length
        self.max_length = max_length
        super().__init__(
            f"Input text is {length} characters long, which exceeds the "
            f"configured maximum of {max_length} characters."
        )


class LinguisticModelNotAvailableError(RuntimeError):
    """
    Raised when the required spaCy model is not installed.

    This is deliberately NOT a bare OSError bubbling up from spaCy —
    that would leave the caller looking at spaCy's internal error
    message and having to guess what to do about it. This exception
    carries an actionable fix (the exact install command) so the
    failure is clear the first time someone sees it, whether that's a
    developer setting up the project for the first time or the API
    layer turning this into a user-facing error later.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            f"The spaCy model '{model_name}' is not installed. Install it with:\n"
            f"    python -m spacy download {model_name}\n"
            f"(run from the backend/ directory, with the virtual environment active)."
        )
