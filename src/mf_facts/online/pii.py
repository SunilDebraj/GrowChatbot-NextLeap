"""Node [11] - PII scanner (architecture.md section 5.11, PRD FR6, driver D6).

First node in the query pipeline. Detection logic lives in
``common.pii_patterns`` so that this node and the log-time redactor cannot drift
apart; this module owns the *policy* around a hit, and that policy is the whole
point of the node:

* the raw query is discarded, not masked - ``sanitized_query`` is ``""``;
* the discarded value is never returned to the caller, never handed to the
  rewriter, retriever, generator or any LLM, and never persisted;
* the only things that survive a hit are the boolean and the category names.

"Discarded, not masked" is not a stylistic choice. A masked query - one where
only the identifier is blanked and the trailing question is left intact - is
still a forwarded query that can still be logged, cached or retried, and it is
still answerable, which is exactly the situation where a half-redacted query
quietly becomes answerable with the identifier attached. Refusing the whole
query is the only behaviour that makes the guarantee testable.
"""

from __future__ import annotations

from ..common.config import AppConfig
from ..common.models import PiiResult
from ..common.pii_patterns import PII_CATEGORIES, categories_in


class PiiScanner:
    """Detects PII in a raw query and, on a hit, destroys the query."""

    def __init__(self, config: AppConfig) -> None:
        self._enabled = config.safety.pii_scan_enabled
        self._categories = tuple(config.safety.pii_categories)

    def scan(self, query: str) -> PiiResult:
        """Return the scan result. On a PII hit the query is not represented
        anywhere in the returned object, by construction."""
        if not self._enabled:
            # Scanning disabled is a misconfiguration in a financial product, not
            # a supported mode: D6 is non-negotiable. Fail closed to "detect
            # everything" rather than to "detect nothing".
            return PiiResult(is_pii=True, categories=PII_CATEGORIES, sanitized_query="")

        text = query or ""
        if not text.strip():
            return PiiResult(is_pii=False, categories=(), sanitized_query=text)

        categories = categories_in(text)
        if not categories:
            return PiiResult(is_pii=False, categories=(), sanitized_query=text)

        # A hit is only reported for categories the config opted into, so a
        # narrowed pii_categories list narrows what the notice says too.
        reported = tuple(c for c in PII_CATEGORIES if c in categories and c in self._categories)
        if not reported:
            return PiiResult(is_pii=False, categories=(), sanitized_query=text)

        return PiiResult(is_pii=True, categories=reported, sanitized_query="")
