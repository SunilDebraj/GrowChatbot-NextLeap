"""[4] Normalizer: clean text, canonicalize dates, resolve last_updated (architecture.md 5.4).

This stage touches whitespace, unicode and dates only. It never rewrites a
number: no rounding, no unit conversion, no percent stripping. A normalizing bug
on a numeric value is a correctness bug, and the highest-severity failure in this
product is a wrong fee or exit-load figure.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone

from ..common.models import NormalizedDoc, ParsedDoc, SourceSpec

# Unicode and typographic variants that carry the same meaning as ASCII.
_UNICODE_MAP = {
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u20b9": "Rs ",
    "\uff05": "%", "\u00b7": "-", "\u2022": "-",
}

_TRANSLATE = str.maketrans(_UNICODE_MAP)
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

# Date patterns, most specific first. All are resolved to ISO-8601.
# The ``kind`` names the group layout, so a numeric dd/mm/yyyy pattern is never
# read as if its middle group were a month name.
_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b", re.I), "day_month_year"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "month_day_year"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "year_month_day"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "numeric_day_month_year"),
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Contexts that indicate the document's own currency date.
_LAST_UPDATED_CONTEXT = re.compile(
    r"(?i)(last\s*updated|as\s*on|updated\s*on|updated|valid\s*upto|"
    r"as\s*of)\s*[:\-]?\s*([^\n]{4,40})"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace without touching numeric content."""
    if not text:
        return ""
    translated = text.translate(_TRANSLATE)
    lines = [_WS.sub(" ", line).strip() for line in translated.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if line or (collapsed and collapsed[-1]):
            collapsed.append(line)
    return _MULTI_NL.sub("\n\n", "\n".join(collapsed)).strip()


def parse_date(text: str) -> str | None:
    """Extract an ISO-8601 date from ``text``, or None."""
    if not text:
        return None
    for pattern, kind in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            if kind == "year_month_day":
                return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
            if kind == "numeric_day_month_year":
                return date(int(match[3]), int(match[2]), int(match[1])).isoformat()
            if kind == "day_month_year":
                return date(int(match[3]), _MONTHS[match[2][:3].lower()], int(match[1])).isoformat()
            return date(int(match[3]), _MONTHS[match[1][:3].lower()], int(match[2])).isoformat()
        except (ValueError, KeyError):
            continue
    return None


def extract_last_updated(text: str) -> str | None:
    """Find the document's own last-updated date from an explicit context line."""
    for match in _LAST_UPDATED_CONTEXT.finditer(text or ""):
        found = parse_date(match.group(2))
        if found:
            return found
    return None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(
    parsed: ParsedDoc,
    spec: SourceSpec,
    retrieved_at: str,
) -> NormalizedDoc:
    """Turn a ParsedDoc into a NormalizedDoc.

    last_updated precedence (architecture.md 5.4): the document's own value
    first, then the registry's declared value, then the retrieval time. Which
    one was used is recorded in ``last_updated_source`` so a reviewer can tell a
    genuinely fresh document from one that merely has a recent crawl time.
    """
    text = clean_text(parsed.text)

    document_date = extract_last_updated(text)
    if document_date:
        last_updated, source = document_date, "document"
    elif spec.last_updated:
        last_updated, source = spec.last_updated, "spec"
    else:
        last_updated = (retrieved_at or _now_iso())[:10]
        source = "retrieved_at"

    effective = spec.effective_date or extract_last_updated(text) or None

    return NormalizedDoc(
        source_id=parsed.source_id,
        scheme_key=spec.scheme_key,
        scheme_name=spec.scheme_name,
        doc_class=spec.doc_class,
        publisher=spec.publisher,
        source_url=spec.url,
        text=text,
        headings=parsed.headings,
        tables=parsed.tables,
        effective_date=effective,
        last_updated=last_updated,
        last_updated_source=source,
        content_hash=content_hash(text),
        page_count=parsed.page_count,
    )
