class EmptyTextError(ValueError):
    """Raised when input text is empty, whitespace-only, or reduces to
    nothing usable after cleaning (e.g. text that was only a URL or only
    HTML markup with no visible text)."""


class TextTooLongError(ValueError):

    def __init__(self, length: int, max_length: int):
        self.length = length
        self.max_length = max_length
        super().__init__(
            f"Input text is {length} characters long, which exceeds the "
            f"configured maximum of {max_length} characters."
        )


class LinguisticModelNotAvailableError(RuntimeError):

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            f"The spaCy model '{model_name}' is not installed. Install it with:\n"
            f"    python -m spacy download {model_name}\n"
            f"(run from the backend/ directory, with the virtual environment active)."
        )
