"""Frozen dataclasses crossing every module boundary.

Field names here are contracts. They match PRD.md section 8.1 and
architecture.md sections 5.1, 5.3, 5.4, 5.6 and 8.2. Renaming a field is a
breaking change to the manifest schema and the contract tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DocClass = Literal[
    "overview",
    "factsheet",
    "faq",
    "fees",
    "kim",
    "sid",
    "riskometer",
    "statement_guide",
]

DOC_CLASSES: tuple[str, ...] = (
    "overview",
    "factsheet",
    "faq",
    "fees",
    "kim",
    "sid",
    "riskometer",
    "statement_guide",
)

Publisher = Literal["amc", "sebi", "amfi", "aggregator"]

LastUpdatedSource = Literal["document", "spec", "retrieved_at"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One ingestible (scheme_key, doc_class) region.

    Multiple specs may share one ``url`` with different ``locator`` values: the
    page is fetched once via the content-addressed cache and then parsed once
    per spec, so one URL can yield several document classes.
    """

    source_id: str
    scheme_key: str
    scheme_name: str
    doc_class: str
    url: str
    publisher: str
    locator: str
    exclude_selectors: tuple[str, ...] = ()
    exclude_headings: tuple[str, ...] = ()
    effective_date: str | None = None
    last_updated: str | None = None
    enabled: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.doc_class not in DOC_CLASSES:
            from .errors import UnknownDocumentClass

            raise UnknownDocumentClass(
                f"{self.source_id}: doc_class {self.doc_class!r} not in {DOC_CLASSES}"
            )
        if self.publisher not in ("amc", "sebi", "amfi", "aggregator"):
            from .errors import ConfigError

            raise ConfigError(
                f"{self.source_id}: publisher {self.publisher!r} is not a legal value"
            )


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Bytes as retrieved, plus provenance of the retrieval."""

    source_id: str
    url: str
    resolved_url: str
    content: bytes
    content_type: str
    status: int
    retrieved_at: str
    from_cache: bool
    error: str = ""


@dataclass(frozen=True, slots=True)
class Table:
    """A table preserved as ordered rows of cells. Never re-flowed into prose."""

    rows: tuple[tuple[str, ...], ...]
    caption: str = ""

    @property
    def header(self) -> tuple[str, ...]:
        return self.rows[0] if self.rows else ()

    def to_markdown(self, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        padded = [r + ("",) * (width - len(r)) for r in rows]
        out = ["| " + " | ".join(cells) for cells in padded]
        out.insert(1, "|" + "|".join([" --- "] * width) + "|")
        if self.caption:
            out.insert(0, f"{self.caption}")
        return "\n".join(out)


@dataclass(frozen=True, slots=True)
class ParsedDoc:
    """Clean text plus the structure the heading-aware chunker depends on."""

    source_id: str
    text: str
    headings: tuple[tuple[int, str], ...] = ()
    tables: tuple[Table, ...] = ()
    page_count: int = 1


@dataclass(frozen=True, slots=True)
class NormalizedDoc:
    """Parsed text after cleanup and date canonicalization.

    Numbers are never rewritten. A normalizing bug on a numeric value is a
    correctness bug, so this stage only touches whitespace, unicode and dates.
    """

    source_id: str
    scheme_key: str
    scheme_name: str
    doc_class: str
    publisher: str
    source_url: str
    text: str
    headings: tuple[tuple[int, str], ...] = ()
    tables: tuple[Table, ...] = ()
    effective_date: str | None = None
    last_updated: str = ""
    last_updated_source: LastUpdatedSource = "retrieved_at"
    content_hash: str = ""
    duplicate_of: str | None = None
    page_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tables"] = [t.to_markdown() for t in self.tables]
        data["headings"] = [list(h) for h in self.headings]
        return data


@dataclass(frozen=True, slots=True)
class Chunk:
    """The retrieval unit. Carries all eleven PRD section 8.1 metadata fields."""

    text: str
    token_count: int
    scheme_key: str
    scheme_name: str
    doc_class: str
    publisher: str
    source_url: str
    section: str
    effective_date: str | None
    last_updated: str
    content_hash: str
    chunk_index: int

    CHUNK_ID_SEPARATOR = ":"

    @property
    def chunk_id(self) -> str:
        return self.CHUNK_ID_SEPARATOR.join(
            (
                self.scheme_key,
                self.doc_class,
                self.content_hash[:12],
                str(self.chunk_index),
            )
        )

    def to_chroma_metadata(self) -> dict[str, str | int]:
        """Chroma accepts only scalar metadata values; absent dates become ''.

        Kept as a method so a single place owns the None-to-empty-string rule
        that otherwise raises at write time.
        """
        return {
            "scheme_key": self.scheme_key,
            "scheme_name": self.scheme_name,
            "doc_class": self.doc_class,
            "publisher": self.publisher,
            "source_url": self.source_url,
            "section": self.section,
            "effective_date": self.effective_date or "",
            "last_updated": self.last_updated or "",
            "content_hash": self.content_hash,
            "chunk_index": int(self.chunk_index),
            "token_count": int(self.token_count),
        }


@dataclass(frozen=True, slots=True)
class FetchRun:
    """Per-spec build outcome, used to render sources.csv and sources.md."""

    source_id: str
    scheme_key: str
    scheme_name: str
    doc_class: str
    url: str
    publisher: str
    effective_date: str
    last_updated: str
    retrieved_at: str
    content_hash: str
    status: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Counts emitted by the offline pipeline for observability and P2 evidence."""

    strategy: str
    docs_fetched: int = 0
    docs_failed: int = 0
    docs_deduped: int = 0
    chunks: int = 0
    tokens_total: int = 0
    tokens_min: int = 0
    tokens_mean: float = 0.0
    tokens_max: int = 0
    build_duration_s: float = 0.0
    chunks_per_scheme: dict[str, int] = field(default_factory=dict)
    chunks_per_doc_class: dict[str, int] = field(default_factory=dict)
    per_strategy_tokens: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """The reproducibility contract for the corpus (PRD section 8.2)."""

    corpus_version: str
    embedding_model: str
    embedding_dim: int
    chunking_strategy: str
    chunking_params: dict[str, Any]
    collection_name: str
    doc_count: int
    chunk_count: int
    chunks_per_scheme: dict[str, int]
    chunks_per_doc_class: dict[str, int]
    sources: list[dict[str, Any]]
    built_at: str
    corpus_hash: str
    tokens_max: int
    build_duration_s: float
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Online path (architecture.md sections 5.11-5.14, 7.4).
# ---------------------------------------------------------------------------

QueryClass = Literal[
    "pii",
    "factual_scheme",
    "how_to",
    "opinionated",
    "performance",
    "portfolio_personal",
    "out_of_scope",
    "grounding_fail",
]

# PRD.md section 6 FR3. ``grounding_fail`` is added by architecture section 5.13
# for the post-retrieval refusal and is not produced by the classifier.
QUERY_CLASSES: tuple[str, ...] = (
    "pii",
    "factual_scheme",
    "how_to",
    "opinionated",
    "performance",
    "portfolio_personal",
    "out_of_scope",
    "grounding_fail",
)

# The only labels the residual LLM stage may return (architecture section 5.12).
RESIDUAL_LABELS: tuple[str, ...] = ("factual_scheme", "how_to", "opinionated")

REFUSAL_CLASSES: tuple[str, ...] = (
    "pii",
    "opinionated",
    "performance",
    "portfolio_personal",
    "out_of_scope",
    "grounding_fail",
)


@dataclass(frozen=True, slots=True)
class PiiResult:
    """Output of node [11]. ``sanitized_query`` is "" on a hit, never masked text."""

    is_pii: bool
    categories: tuple[str, ...]
    sanitized_query: str

    def to_log_fields(self) -> dict[str, Any]:
        """The only two PII fields that may ever be persisted (architecture 5.11)."""
        return {"pii_detected": self.is_pii, "pii_categories": list(self.categories)}


@dataclass(frozen=True, slots=True)
class Classification:
    """Output of node [12]: exactly one class, the rule that decided it."""

    query_class: str
    rule_id: str
    confidence: float
    stage: str

    def __post_init__(self) -> None:
        if self.query_class not in QUERY_CLASSES:
            from .errors import ConfigError

            raise ConfigError(
                f"Classification: query_class {self.query_class!r} not in {QUERY_CLASSES}"
            )
        if self.stage not in ("rule", "residual"):
            from .errors import ConfigError

            raise ConfigError(
                f"Classification: stage {self.stage!r} must be 'rule' or 'residual'"
            )
        if not 0.0 <= self.confidence <= 1.0:
            from .errors import ConfigError

            raise ConfigError(
                f"Classification: confidence {self.confidence!r} outside [0.0, 1.0]"
            )


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    """Output of node [14]. Hints are a soft boost, never a hard filter."""

    text: str
    scheme_keys: tuple[str, ...]
    doc_class_hints: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [h for h in self.doc_class_hints if h not in DOC_CLASSES]
        if unknown:
            from .errors import ConfigError

            raise ConfigError(f"RewrittenQuery: unknown doc_class_hints {unknown}")


@dataclass(frozen=True, slots=True)
class RefusalResponse:
    """Output of node [13]. Templated; ``text`` is never LLM-generated."""

    query_class: str
    text: str
    link_label: str = ""
    link_url: str = ""
    rule_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AskResponse:
    """Output of the online pipeline [10]+[11]+[12]+[13]+[14].

    ``route`` is ``refusal`` for a composed refusal, ``answer`` once nodes
    [15]-[19] exist, and ``not_implemented`` for the in-scope classes that P3
    deliberately leaves unbuilt.
    """

    query_class: str
    route: str
    text: str
    link_label: str = ""
    link_url: str = ""
    rule_id: str = ""
    confidence: float = 0.0
    pii_detected: bool = False
    pii_categories: tuple[str, ...] = ()
    scheme_keys: tuple[str, ...] = ()
    doc_class_hints: tuple[str, ...] = ()
    # P4 answer-path fields. All defaulted so every P3 construction site and
    # refusal path keeps working unchanged.
    citation_url: str = ""
    last_updated: str = ""
    grounding_passed: bool = False
    grounding_score: float = 0.0
    validator_checks: tuple[str, ...] = ()
    answer_model: str = ""
    #: Reranked chunk ids, best first. Debugging and eval only - architecture
    #: 5.19 keeps ``route`` in the same category. It is what lets the eval report
    #: separate "the answer was wrong" from "the fact was never retrieved".
    retrieved_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# P4 - retrieval, grounding, validation contracts (architecture.md 5.15-5.18)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrieved chunk, plus the ranking bookkeeping the later nodes need.

    ``metadata`` is the chunk's own metadata, unmodified. That is deliberate: the
    citation and the freshness stamp are read from here (rule GR5), so a passage
    that lost a metadata field upstream would silently produce an answer with no
    source rather than an error.
    """

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float = 0.0
    vector_rank: int = 0
    lexical_rank: int = 0
    fused_score: float = 0.0
    boost_reasons: tuple[str, ...] = ()

    @property
    def scheme_key(self) -> str:
        return str(self.metadata.get("scheme_key", ""))

    @property
    def doc_class(self) -> str:
        return str(self.metadata.get("doc_class", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))

    @property
    def last_updated(self) -> str:
        return str(self.metadata.get("last_updated", ""))

    @property
    def has_digits(self) -> bool:
        return any(ch.isdigit() for ch in self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Output of node [17] (architecture.md 5.16). ``passed`` is the gate."""

    passed: bool
    best_passage: Passage | None
    score: float
    reason: str
    fact_anchor_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "fact_anchor_present": self.fact_anchor_present,
            "best_passage": self.best_passage.to_dict() if self.best_passage else None,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Output of node [19] (architecture.md 5.18).

    ``safe_response`` is what the caller must show. It is never the raw model
    output on a failure (rule GR10), so a caller cannot accidentally prefer the
    original text: the original is not even carried here.
    """

    passed: bool
    text: str
    safe_response: str
    fired: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "fired": list(self.fired),
            "reasons": list(self.reasons),
        }
