"""Sentence handling shared by the generator and the validator.

These two nodes must agree on where a sentence ends, and they get there from
opposite directions: the generator tries to stay inside the limit, the validator
enforces it. If each owned its own split, a change to one would silently stop
enforcing the other's assumption - a compliance bug with no failing test.

The split is a lookbehind on terminal punctuation rather than ``split(". ")``
because the corpus is full of decimals ("Expense ratio: 1.03%."), and ``". "``
matching inside a number is how a well-formed answer gets truncated mid-fact.
"""

from __future__ import annotations

import re

#: A boundary is terminal punctuation followed by whitespace. The lookbehind
#: keeps the punctuation attached to the sentence it terminates.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Non-empty sentences in ``text``."""
    return [part for part in SENTENCE_SPLIT.split(text or "") if part.strip()]


def truncate_sentences(text: str, limit: int) -> str:
    """The first ``limit`` sentences, terminated if the cut left an open clause.

    A truncation that leaves "the exit load is" hanging is worse than refusing,
    so the fragment is closed. Returns ``text`` unchanged when it already fits.
    """
    parts = sentences(text)
    if len(parts) <= limit:
        return text
    kept = " ".join(parts[:limit]).strip()
    if not kept.endswith((".", "!", "?")):
        kept = kept.rstrip(",;: ") + "."
    return kept
