"""Structured logging with a deny-by-default field allowlist.

Two independent protections, both required by architecture.md section 14:

1. ``log_event`` drops any field not on :data:`ALLOWED_EVENT_FIELDS`, so
   ``log_event("x", query="...")`` silently logs nothing sensitive. This is
   deny-by-default: a new field must be added to the allowlist deliberately.
2. The formatter runs :func:`mf_facts.common.redaction.redact` over the final
   rendered record, so even a raw ``logger.warning`` with an interpolated PAN
   cannot write the value to disk.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .redaction import redact

ALLOWED_EVENT_FIELDS: frozenset[str] = frozenset(
    {
        "query_hash",
        "route",
        "class",
        "rule_id",
        "retrieved_ids",
        "validator_flags",
        "latency_ms",
        "pii_detected",
        "pii_categories",
        "grounding_score",
        "no_answer",
        "scheme_key",
        "doc_class",
        "chunk_count",
        "token_count",
        "strategy",
        "status",
        "source_id",
        "count",
    }
)

#: Keys that are always dropped even if someone widens the allowlist later.
FORBIDDEN_EVENT_FIELDS: frozenset[str] = frozenset(
    {"query", "question", "text", "raw", "passage", "answer", "prompt", "content"}
)


class RedactingFormatter(logging.Formatter):
    """Renders a record then masks any PII span in the rendered output."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))[0]


def _filter_fields(fields: dict[str, Any]) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for key, value in fields.items():
        if key in FORBIDDEN_EVENT_FIELDS:
            continue
        if key not in ALLOWED_EVENT_FIELDS:
            continue
        kept[key] = value
    return kept


def setup_logging(level: str | int = "INFO") -> None:
    """Install a single stdout handler with the redacting formatter.

    Idempotent: repeated calls replace the handler rather than stacking them,
    so tests that reconfigure logging do not duplicate output.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured event, keeping only allowlisted fields."""
    kept = _filter_fields(fields)
    if kept:
        logger.info("%s %s", event, json.dumps(kept, sort_keys=True, default=str))
    else:
        logger.info("%s", event)
