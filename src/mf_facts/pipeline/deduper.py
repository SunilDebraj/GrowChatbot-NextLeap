"""[5] Deduper: remove duplicate content across sources (architecture.md 5.5).

Two passes. Exact matching on the SHA-256 of normalized text catches a page
fetched through two specs. Section-signature matching catches the same fact
restated in a second document, where a high token overlap means one copy can be
shadowed without losing information.

The shadowed copy is recorded as ``duplicate_of`` rather than discarded, so
sources.csv still shows every source that was consulted.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..common.models import NormalizedDoc

_TOKEN = re.compile(r"[a-z0-9%]+")

#: Jaccard overlap above which two documents in the same section signature are
#: treated as the same content.
NEAR_DUPLICATE_THRESHOLD = 0.95


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def _content_key(doc: NormalizedDoc) -> tuple[str, str, str]:
    """Identity of a piece of content, scoped to its scheme and doc class.

    Scoping matters: five HDFC scheme pages share boilerplate, so an unscoped
    content hash would collapse four schemes' documents into the first one and
    silently leave the corpus answering for a single scheme.
    """
    return (doc.scheme_key, doc.doc_class, doc.content_hash)


def _section_signature(doc: NormalizedDoc) -> tuple[str, str]:
    return (doc.scheme_key, doc.doc_class)


def dedupe(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    """Mark duplicates, keeping the first occurrence of each piece of content.

    Order of the input decides the winner, and the registry is already ordered by
    publisher precedence, so the higher-precedence copy is the one retained.
    """
    kept: list[NormalizedDoc] = []
    seen_exact: set[tuple[str, str, str]] = set()
    signatures: list[tuple[tuple[str, str], str, set[str]]] = []

    for doc in docs:
        if not doc.text:
            continue

        content_key = _content_key(doc)
        if content_key in seen_exact:
            kept.append(replace(doc, duplicate_of=_source_id_for(kept, content_key)))
            continue

        signature = _section_signature(doc)
        tokens = _tokens(doc.text)
        winner: str | None = None
        for existing_signature, existing_source_id, existing_tokens in signatures:
            if existing_signature != signature:
                continue
            if jaccard(tokens, existing_tokens) >= NEAR_DUPLICATE_THRESHOLD:
                winner = existing_source_id
                break

        if winner is not None:
            kept.append(replace(doc, duplicate_of=winner))
            continue

        seen_exact.add(content_key)
        signatures.append((signature, doc.source_id, tokens))
        kept.append(doc)

    return kept


def _source_id_for(kept: list[NormalizedDoc], content_key: tuple[str, str, str]) -> str:
    for doc in kept:
        if _content_key(doc) == content_key:
            return doc.source_id
    return "unknown"


def live_docs(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    """The documents that should be chunked: everything not marked duplicate."""
    return [doc for doc in docs if doc.duplicate_of is None]
