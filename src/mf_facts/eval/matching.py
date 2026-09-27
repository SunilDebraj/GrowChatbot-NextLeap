"""Deterministic answer matching for the eval harness (P2-T4).

architecture.md 7.6 and implementation.md P2-T4 both forbid an LLM judge: the
scoring has to be reproducible and it has to be possible to argue with a number
in a review. Everything here is string and arithmetic.

The distinction that matters is between *the fact is somewhere in the retrieved
chunks* (a retrieval property) and *the fact survives intact inside one
sentence* (a chunking property). A strategy that slices a fee slab across a chunk
boundary can still pass the first test and fail the second, and the second is the
one that predicts a wrong answer at query time.
"""

from __future__ import annotations

import re

# A period between two digits is part of a number, not a sentence end. Without
# this, "Rs 1.25 lakh" splits into "Rs 1" and "25 lakh" and every decimal answer
# silently fails to match.
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop edge punctuation."""
    return _WS.sub(" ", (text or "").strip().lower()).strip(" .,;:")


def sentences(text: str) -> list[str]:
    """Split into sentences without breaking decimals or version numbers."""
    if not text:
        return []
    guarded = _DECIMAL.sub("\x00", text)
    parts = _SENTENCE_BREAK.split(guarded)
    return [part.replace("\x00", ".").strip() for part in parts if part.strip()]


def numbers(text: str) -> set[str]:
    """Every numeric literal in the text, normalised.

    "1,07,295.79" -> {"107295.79"}; "0.78%" -> {"0.78"}. Thousands separators go
    so that an AUM written with Indian digit grouping matches the same value
    written without it.
    """
    found = set()
    for raw in _NUMBER.findall(text or ""):
        cleaned = raw.replace(",", "")
        try:
            found.add(f"{float(cleaned):.6f}".rstrip("0").rstrip("."))
        except ValueError:
            continue
    return {value for value in found if value}


def fact_in_sentence(expected_fact: str, sentence: str, numeric: bool) -> bool:
    """Is `expected_fact` stated by `sentence`?

    Numeric items compare the extracted number exactly, per P2-T4. Prose items
    use normalized containment, which is the "key value" match the spec asks
    for rather than a fuzzy similarity that would let a near-miss pass.
    """
    if not sentence or not expected_fact:
        return False
    if numeric:
        return bool(numbers(expected_fact) & numbers(sentence))
    return normalize(expected_fact) in normalize(sentence)


def fact_in_text(expected_fact: str, text: str, numeric: bool) -> bool:
    """Is the fact present anywhere in the text, boundaries ignored."""
    if not text or not expected_fact:
        return False
    if numeric:
        return bool(numbers(expected_fact) & numbers(text))
    return normalize(expected_fact) in normalize(text)


def extract_sentence(expected_fact: str, text: str, numeric: bool) -> str | None:
    """The first sentence stating the fact, or None.

    This is the extractive stand-in for generation. With OQ2 still open there is
    no model to generate an answer during the P2 gate, and the question this needs
    to answer is narrower than "can a model answer it": it is "did the chunker
    leave this fact whole and findable". See harness.evaluate for the caveat this
    puts on grounded_accuracy.
    """
    for sentence in sentences(text):
        if fact_in_sentence(expected_fact, sentence, numeric):
            return sentence
    return None
