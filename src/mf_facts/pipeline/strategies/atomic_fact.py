"""[6c] Per-fact atomic strategy (architecture.md 7.2 candidate C).

One chunk per labelled fact, phrased so the embedded text is close to the
question that will retrieve it:

    "Expense ratio: 1.03%. Expense ratio 1.03%. fees HDFC Large Cap Fund"

Highest precision for numeric lookups, which is the metric the P2 gate weights
first. Its weakness is coverage: it needs a label extractor, and partial
extraction silently drops facts, which is worse than a coarse chunk. It is
therefore restricted to the ``overview`` and ``fees`` doc classes, and its
coverage gap is recorded in the build report rather than hidden.
"""

from __future__ import annotations

import re

from ...common.models import Chunk, NormalizedDoc
from ..chunker import BaseStrategy

#: Doc classes whose content is predominantly labelled facts.
SUPPORTED_DOC_CLASSES = frozenset({"overview", "fees"})

#: "Label: value" where the value carries a figure, amount or a term.
_LABELLED = re.compile(
    r"(?m)^(?P<label>[A-Za-z][A-Za-z0-9 .()/&'-]{2,44}?)\s*:\s*(?P<value>[^\n]{1,160})$"
)

#: A value worth its own chunk: it contains a digit or an explicit negation.
_INTERESTING = re.compile(r"[0-9₹]|nil|no |not applicable", re.IGNORECASE)

#: Labels that must never become a chunk even when numeric.
_BLOCKED = re.compile(
    r"(?i)\b(return|cagr|xirr|rank|performance|nav\b|aum\b|fund size|"
    r"holdings?|1y|3y|5y|1 year|3 year|5 year|since inception)\b"
)


class AtomicFactStrategy(BaseStrategy):
    name = "atomic_fact"

    def split(self, doc: NormalizedDoc) -> list[Chunk]:
        if doc.doc_class not in SUPPORTED_DOC_CLASSES:
            return []

        chunks: list[Chunk] = []
        index = 0
        seen: set[str] = set()

        for match in _LABELLED.finditer(doc.text):
            label = match.group("label").strip()
            value = match.group("value").strip()

            if not _INTERESTING.search(value):
                continue
            if _BLOCKED.search(label):
                continue

            key = f"{label.lower()}={value.lower()}"
            if key in seen:
                continue
            seen.add(key)

            section = label
            text = (
                f"{label}: {value}. "
                f"{label} {value}. "
                f"{section}. {doc.scheme_name}."
            )
            chunk = self.make_chunk(doc, text, section, index)
            if chunk:
                chunks.append(chunk)
                index += 1

        return chunks

    def coverage_report(self, doc: NormalizedDoc) -> dict[str, int]:
        """How many labelled facts this strategy found, for the build report."""
        if doc.doc_class not in SUPPORTED_DOC_CLASSES:
            return {"candidates": 0, "kept": 0}
        candidates = len(_LABELLED.findall(doc.text))
        return {"candidates": candidates, "kept": len(self.split(doc))}
