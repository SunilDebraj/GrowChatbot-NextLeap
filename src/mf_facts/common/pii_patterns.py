"""Named PII detection patterns (architecture.md section 5.11).

These are the single source of truth for what counts as PII in this project.
They are used by two independent callers:

* ``online.pii.PiiScanner`` at query time, which *discards* the query on a hit.
* ``common.redaction.redact`` at log time, which masks the value in place.

Keeping the patterns here and the policies separate is deliberate: a change to
detection must not quietly change what the logger does, or vice versa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PAN = "pan"
AADHAAR = "aadhaar"
ACCOUNT = "account"
OTP = "otp"
EMAIL = "email"
PHONE = "phone"

PII_CATEGORIES: tuple[str, ...] = (PAN, AADHAAR, ACCOUNT, OTP, EMAIL, PHONE)


@dataclass(frozen=True, slots=True)
class PIIPattern:
    category: str
    name: str
    regex: re.Pattern[str]


def _p(category: str, name: str, pattern: str) -> PIIPattern:
    return PIIPattern(category=category, name=name, regex=re.compile(pattern))


# Order matters for overlap resolution: the more specific pattern must come
# first so a 12-digit Aadhaar is not also reported as a 10-digit account number.
PII_PATTERNS: tuple[PIIPattern, ...] = (
    _p(PAN, "pan:strict", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    _p(PAN, "pan:masked", r"\b[X]{4,5}[0-9]{4}[X]{1,2}\b"),
    _p(
        AADHAAR,
        "aadhaar:spaced",
        r"(?<![0-9A-Za-z])[2-9][0-9]{3}\s[0-9]{4}\s[0-9]{4}(?![0-9A-Za-z])",
    ),
    _p(AADHAAR, "aadhaar:plain", r"(?<![0-9A-Za-z])[2-9][0-9]{11}[A-Za-z]?(?![0-9A-Za-z])"),
    _p(EMAIL, "email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    _p(
        OTP,
        "otp:contextual",
        r"(?i)\b(?:otp|one[\s-]?time[\s-]?pass(?:word)?|verification[\s-]?code|"
        r"auth(?:entication)?[\s-]?code|passcode|pin)\b\s*(?:is|:|-)?\s*[0-9]{4,8}\b",
    ),
    _p(PHONE, "phone:intl", r"(?<![0-9])\+?91[\s-]?[6-9][0-9]{4}[\s-]?[0-9]{5}(?![0-9])"),
    _p(
        PHONE,
        "phone:local",
        r"(?<![0-9])[6-9][0-9]{4}[\s-]?[0-9]{5}(?![0-9])",
    ),
    _p(ACCOUNT, "account:ten", r"(?<![0-9])[0-9]{10}(?![0-9])"),
    _p(ACCOUNT, "account:dashes", r"(?<![0-9])[0-9]{4}-[0-9]{4}-[0-9]{4}(?![0-9])"),
    _p(ACCOUNT, "account:long", r"(?<![0-9])[0-9]{12,}(?![0-9])"),
)

# Patterns evaluated only when an OTP keyword is present elsewhere in the
# query; a bare 4-6 digit number is far too common in fund questions
# (minimum SIP amounts, lock-in years) to treat as an OTP on its own.
_OTP_KEYWORDS = re.compile(
    r"(?i)\b(?:otp|one[\s-]?time[\s-]?pass(?:word)?|verification[\s-]?code|"
    r"auth(?:entication)?[\s-]?code|passcode|pin)\b"
)


@dataclass(frozen=True, slots=True)
class PIIHit:
    category: str
    pattern: str
    start: int
    end: int
    text: str


def _overlaps(a: PIIHit, b: PIIHit) -> bool:
    return a.start < b.end and b.start < a.end


def find_pii(text: str) -> list[PIIHit]:
    """Return non-overlapping PII hits, most specific pattern winning.

    Patterns are evaluated in declaration order, so an Aadhaar reported by
    ``aadhaar:plain`` suppresses the ``account:ten`` and ``phone:local``
    findings covering the same characters.
    """
    if not text:
        return []

    has_otp_keyword = bool(_OTP_KEYWORDS.search(text))
    hits: list[PIIHit] = []

    for pattern in PII_PATTERNS:
        if pattern.category == OTP and not has_otp_keyword:
            continue
        for match in pattern.regex.finditer(text):
            hit = PIIHit(
                category=pattern.category,
                pattern=pattern.name,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
            )
            if any(_overlaps(hit, existing) for existing in hits):
                continue
            hits.append(hit)

    return sorted(hits, key=lambda h: (h.start, h.end))


def categories_in(text: str) -> list[str]:
    """Deduplicated PII categories present in ``text``, in canonical order."""
    found = {hit.category for hit in find_pii(text)}
    return [category for category in PII_CATEGORIES if category in found]
