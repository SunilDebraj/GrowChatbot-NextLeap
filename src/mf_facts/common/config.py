"""Strict configuration loading (architecture.md section 8).

Every threshold referenced anywhere in the project lives in ``config/config.yaml``
and nowhere in code. Parsing is strict: an unknown key raises rather than being
ignored, so a typo cannot silently disable a safety threshold.

Amendment to architecture.md section 8: ``safety.performance_exclude_patterns``
was added during P1 after inspecting the live corpus. The scheme pages carry a
"Returns and rankings" / "Return calculator" / "Compare similar funds" region
holding real return figures. Ingesting it would put return numbers one retrieval
away from the generator, which contradicts driver D5 and rule GR6. The exclusion
is therefore a config key rather than a hardcoded parser behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

LEGAL_CHUNKING_STRATEGIES: tuple[str, ...] = (
    "heading_aware",
    "fixed_window",
    "atomic_fact",
)


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    name: str
    collection: str
    raw_cache_dir: str
    embedding_cache_dir: str


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_id: str
    embedding_dim: int
    batch_size: int
    normalize: bool


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    strategy: str
    max_chunk_tokens: int
    target_tokens: int
    overlap_ratio: float
    protect_tables: bool
    repeat_table_headers: bool


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int
    top_n: int
    rrf_k: int
    doc_class_boost: float
    section_match_boost: float
    numeric_anchor_boost: float


@dataclass(frozen=True, slots=True)
class GroundingConfig:
    min_relevance: float
    require_fact_anchor: bool


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    max_sentences: int
    #: OQ2. An OpenAI-compatible ``/chat/completions`` endpoint. A URL, not a
    #: provider name: the wire shape is the contract, and the vendor behind it is
    #: a deployment detail that can change without a code change.
    base_url: str = ""
    #: Name of the environment variable holding the key. The value itself is
    #: never in config - implementation.md 2.3 puts secrets in the environment.
    api_key_env: str = ""
    timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    pii_scan_enabled: bool
    log_raw_queries: bool
    pii_categories: tuple[str, ...]
    performance_exclude_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UIConfig:
    examples: tuple[str, ...]
    disclaimer_file: str


@dataclass(frozen=True, slots=True)
class APIConfig:
    """P5-T4 rate and size guards.

    Lives in config like every other threshold (implementation.md 3.4) so a
    reviewer can see the limits without reading the request handler.
    """

    max_question_length: int
    max_body_bytes: int
    rate_limit_requests: int
    rate_limit_window_s: int
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Short conversation memory, used only to carry a scheme into a follow-up.

    The window is the number of previous questions the client may send. Nothing
    is stored server-side: the history arrives with each request and is dropped
    when the request ends.
    """

    window_turns: int = 10


@dataclass(frozen=True, slots=True)
class AppConfig:
    corpus: CorpusConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    grounding: GroundingConfig
    generation: GenerationConfig
    safety: SafetyConfig
    ui: UIConfig
    api: APIConfig
    root: Path
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: expected a mapping, got {type(value).__name__}")
    return value


def _strict(section: dict[str, Any], allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"{name}: unknown config key(s) {unknown}; allowed {list(allowed)}")


def _get(section: dict[str, Any], key: str, default: Any, *, cast: type | None = None) -> Any:
    if key not in section or section[key] is None:
        return default
    value = section[key]
    if cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key}: expected {cast.__name__}, got {value!r}") from exc


def load_config(path: str | Path) -> AppConfig:
    """Load and validate ``config.yaml``.

    Raises ConfigError on a missing file, a non-mapping root, an unknown key at
    any level, or an illegal chunking strategy.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc

    root = _require_mapping(raw, "<root>")

    corpus_raw = _require_mapping(root.get("corpus"), "corpus")
    _strict(
        corpus_raw,
        ("name", "collection", "raw_cache_dir", "embedding_cache_dir"),
        "corpus",
    )
    corpus = CorpusConfig(
        name=_get(corpus_raw, "name", "mf_facts"),
        collection=_get(corpus_raw, "collection", "mf_facts_v1"),
        raw_cache_dir=_get(corpus_raw, "raw_cache_dir", "raw_cache"),
        embedding_cache_dir=_get(corpus_raw, "embedding_cache_dir", "cache/embeddings"),
    )

    embedding_raw = _require_mapping(root.get("embedding"), "embedding")
    _strict(embedding_raw, ("model_id", "embedding_dim", "batch_size", "normalize"), "embedding")
    embedding = EmbeddingConfig(
        model_id=_get(
            embedding_raw,
            "model_id",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_dim=_get(embedding_raw, "embedding_dim", 384, cast=int),
        batch_size=_get(embedding_raw, "batch_size", 32, cast=int),
        normalize=_get(embedding_raw, "normalize", True, cast=bool),
    )

    chunking_raw = _require_mapping(root.get("chunking"), "chunking")
    _strict(
        chunking_raw,
        (
            "strategy",
            "max_chunk_tokens",
            "target_tokens",
            "overlap_ratio",
            "protect_tables",
            "repeat_table_headers",
        ),
        "chunking",
    )
    strategy = _get(chunking_raw, "strategy", "heading_aware")
    if strategy not in LEGAL_CHUNKING_STRATEGIES:
        raise ConfigError(
            f"chunking.strategy: {strategy!r} is not legal; expected one of {LEGAL_CHUNKING_STRATEGIES}"
        )
    max_chunk_tokens = _get(chunking_raw, "max_chunk_tokens", 220, cast=int)
    if max_chunk_tokens > 256:
        raise ConfigError(
            f"chunking.max_chunk_tokens: {max_chunk_tokens} exceeds the all-MiniLM-L6-v2 "
            "input limit of 256 tokens; chunks would be silently truncated by the model"
        )
    chunking = ChunkingConfig(
        strategy=strategy,
        max_chunk_tokens=max_chunk_tokens,
        target_tokens=_get(chunking_raw, "target_tokens", 160, cast=int),
        overlap_ratio=float(_get(chunking_raw, "overlap_ratio", 0.15)),
        protect_tables=_get(chunking_raw, "protect_tables", True, cast=bool),
        repeat_table_headers=_get(chunking_raw, "repeat_table_headers", True, cast=bool),
    )
    if chunking.target_tokens > chunking.max_chunk_tokens:
        raise ConfigError(
            "chunking.target_tokens must not exceed chunking.max_chunk_tokens"
        )

    retrieval_raw = _require_mapping(root.get("retrieval"), "retrieval")
    _strict(
        retrieval_raw,
        (
            "top_k",
            "top_n",
            "rrf_k",
            "doc_class_boost",
            "section_match_boost",
            "numeric_anchor_boost",
        ),
        "retrieval",
    )
    retrieval = RetrievalConfig(
        top_k=_get(retrieval_raw, "top_k", 12, cast=int),
        top_n=_get(retrieval_raw, "top_n", 4, cast=int),
        rrf_k=_get(retrieval_raw, "rrf_k", 60, cast=int),
        doc_class_boost=float(_get(retrieval_raw, "doc_class_boost", 0.15)),
        section_match_boost=float(_get(retrieval_raw, "section_match_boost", 0.10)),
        numeric_anchor_boost=float(_get(retrieval_raw, "numeric_anchor_boost", 0.10)),
    )

    grounding_raw = _require_mapping(root.get("grounding"), "grounding")
    _strict(grounding_raw, ("min_relevance", "require_fact_anchor"), "grounding")
    grounding = GroundingConfig(
        min_relevance=float(_get(grounding_raw, "min_relevance", 0.15)),
        require_fact_anchor=_get(grounding_raw, "require_fact_anchor", True, cast=bool),
    )

    generation_raw = _require_mapping(root.get("generation"), "generation")
    _strict(
        generation_raw,
        (
            "provider",
            "model",
            "temperature",
            "max_tokens",
            "max_sentences",
            "base_url",
            "api_key_env",
            "timeout_s",
        ),
        "generation",
    )
    generation = GenerationConfig(
        provider=str(_get(generation_raw, "provider", "")),
        model=str(_get(generation_raw, "model", "")),
        temperature=float(_get(generation_raw, "temperature", 0.0)),
        max_tokens=_get(generation_raw, "max_tokens", 180, cast=int),
        max_sentences=_get(generation_raw, "max_sentences", 3, cast=int),
        base_url=str(_get(generation_raw, "base_url", "")),
        api_key_env=str(_get(generation_raw, "api_key_env", "")),
        timeout_s=float(_get(generation_raw, "timeout_s", 30.0)),
    )

    safety_raw = _require_mapping(root.get("safety"), "safety")
    _strict(
        safety_raw,
        (
            "pii_scan_enabled",
            "log_raw_queries",
            "pii_categories",
            "performance_exclude_patterns",
        ),
        "safety",
    )
    safety = SafetyConfig(        pii_scan_enabled=_get(safety_raw, "pii_scan_enabled", True, cast=bool),
        log_raw_queries=_get(safety_raw, "log_raw_queries", False, cast=bool),
        pii_categories=tuple(
            _get(safety_raw, "pii_categories", ("pan", "aadhaar", "account", "otp", "email", "phone"))
        ),
        performance_exclude_patterns=tuple(
            _get(
                safety_raw,
                "performance_exclude_patterns",
                (
                    "return calculator",
                    "returns and rankings",
                    "compare similar funds",
                    "fund returns",
                    "category average",
                    "rank (",
                ),
            )
        ),
    )

    ui_raw = _require_mapping(root.get("ui"), "ui")
    _strict(ui_raw, ("examples", "disclaimer_file"), "ui")
    ui = UIConfig(
        examples=tuple(_get(ui_raw, "examples", ())),
        disclaimer_file=_get(ui_raw, "disclaimer_file", "assets/disclaimer.txt"),
    )

    api_raw = _require_mapping(root.get("api"), "api")
    _strict(
        api_raw,
        (
            "max_question_length",
            "max_body_bytes",
            "rate_limit_requests",
            "rate_limit_window_s",
            "host",
            "port",
        ),
        "api",
    )
    api = APIConfig(
        max_question_length=_get(api_raw, "max_question_length", 400, cast=int),
        max_body_bytes=_get(api_raw, "max_body_bytes", 8192, cast=int),
        rate_limit_requests=_get(api_raw, "rate_limit_requests", 20, cast=int),
        rate_limit_window_s=_get(api_raw, "rate_limit_window_s", 60, cast=int),
        host=_get(api_raw, "host", "127.0.0.1"),
        port=_get(api_raw, "port", 8000, cast=int),
    )
    if api.max_question_length < 1 or api.max_body_bytes < 1:
        raise ConfigError("api: max_question_length and max_body_bytes must be positive")
    if api.rate_limit_requests < 1 or api.rate_limit_window_s < 1:
        raise ConfigError("api: rate_limit_requests and rate_limit_window_s must be positive")

    memory_raw = _require_mapping(root.get("memory"), "memory")
    _strict(memory_raw, ("window_turns",), "memory")
    memory = MemoryConfig(window_turns=_get(memory_raw, "window_turns", 10, cast=int))
    if memory.window_turns < 0:
        raise ConfigError("memory: window_turns must be zero (off) or positive")

    # A half-configured provider is caught here rather than on the first request.
    # Otherwise the service boots, answers every question with a refusal, and the
    # operator has no idea whether the provider is down or the config is short.
    if generation.provider.strip() in ("http", "chat_completions"):
        missing = [
            name
            for name, value in (
                ("base_url", generation.base_url),
                ("api_key_env", generation.api_key_env),
            )
            if not value.strip()
        ]
        if missing:
            raise ConfigError(
                f"generation.provider is {generation.provider!r} but "
                f"{', '.join(missing)} is empty. Both are required: the endpoint is a "
                "URL, and the key stays in the environment (implementation.md 2.3)."
            )

    return AppConfig(
        corpus=corpus,
        embedding=embedding,
        chunking=chunking,
        retrieval=retrieval,
        grounding=grounding,
        generation=generation,
        safety=safety,
        ui=ui,
        api=api,
        root=config_path.resolve().parent.parent,
        memory=memory,
        raw=root,
    )
