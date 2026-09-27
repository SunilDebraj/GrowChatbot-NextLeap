"""Log-time PII masking (architecture.md section 14).

This is a backstop, not the primary control. The primary control is that the
query pipeline discards a PII-bearing query before anything else sees it, and
that ``common.logging.log_event`` refuses to log raw query text at all. This
module exists so that a mistake in a log call, a third-party warning, or an
exception traceback cannot leak a PAN into a log file.
"""

from __future__ import annotations

from .pii_patterns import find_pii

MASK = "[REDACTED]"


def redact(text: str) -> tuple[str, list[str]]:
    """Mask every PII span in ``text``.

    Returns the cleaned text and the deduplicated categories that were masked,
    so the caller can record *that* a redaction happened without recording the
    value.
    """
    if not text:
        return text, []

    hits = find_pii(text)
    if not hits:
        return text, []

    ordered = sorted(hits, key=lambda h: h.start)
    out: list[str] = []
    cursor = 0
    for hit in ordered:
        out.append(text[cursor : hit.start])
        out.append(MASK)
        cursor = hit.end
    out.append(text[cursor:])

    categories = list(dict.fromkeys(hit.category for hit in ordered))
    return "".join(out), categories


def contains_pii(text: str) -> bool:
    return bool(find_pii(text))
