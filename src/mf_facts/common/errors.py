"""Domain exceptions for the MF-Facts-Bot pipeline.

Query-path callers must never propagate these to a client; they are converted to a
safe response or a 503. Build-path callers let them fail the build loudly.
"""


class MFactsError(Exception):
    """Base class for every domain error in this project."""


class ConfigError(MFactsError):
    """Configuration file is missing, unreadable, or contains an unknown key."""


class FetchError(MFactsError):
    """A source could not be retrieved from the network."""


class ParseError(MFactsError):
    """A retrieved document could not be parsed into structure."""


class ChunkSizeError(MFactsError):
    """A chunk exceeds the configured token cap.

    Raised instead of silently truncating, because truncation of a fee or
    exit-load table produces a plausible-looking but wrong number.
    """


class EmbeddingModelMismatch(MFactsError):
    """The configured embedding model differs from the one that built the collection."""


class NoDocumentsForScheme(MFactsError):
    """A scheme ended the build with zero successfully parsed documents."""


class CollectionUnavailable(MFactsError):
    """The vector store could not be reached."""


class LLMError(MFactsError):
    """The language model provider failed or returned an unusable response."""


class UnknownDocumentClass(MFactsError):
    """A SourceSpec referenced a doc_class outside the legal set."""
