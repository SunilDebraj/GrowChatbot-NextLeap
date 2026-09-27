"""Node [17] - grounding check (architecture.md 5.16 and 7.7, PRD FR2 step 6).

The anti-hallucination gate, and the last point before generation where a wrong
answer is still preventable. Two conditions, **both** required:

1. **Relevance floor.** Max lexical overlap between the query's content terms and
   the best passage is at least ``grounding.min_relevance``, *or* the passage's
   doc_class matches a hint and a query term appears in it.
2. **Fact-anchor presence.** For a numeric ask, at least one candidate carries
   digits. For a lock-in ask, at least one mentions lock-in.

On failure the caller composes a ``grounding_fail`` refusal. There is deliberately
no third outcome: no "answer from general knowledge", no "answer with a caveat",
no partial answer from the model's priors. In a financial facts product the
failure mode of a permissive grounding check is a confident invented expense
ratio, and the cost of a conservative one is a user reading a source link
instead. The asymmetry is the whole design.

The check is a deterministic proxy rather than an NLI model, which makes it
testable with no latency and no key. A proxy that is too strict produces an
unnecessary refusal, which is the safe direction; a proxy that is too loose is the
dangerous direction, so the thresholds are set from the measured corpus rather
than tuned for pass rate.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..common.config import AppConfig
from ..common.models import GroundingResult, Passage
from .reranker import content_terms

# Anchor vocabularies. A "numeric question" is inferred rather than asked for,
# so that a query the classifier routed to the answer path but which asks for a
# figure still gets the digit requirement applied.
_NUMERIC_QUERY = re.compile(
    r"\d|how\s+much|how\s+many|rate|ratio|percentage|fee|charge|load|aum|size|"
    r"tax|limit|minimum|maximum|cost|price|amount|value|weight|age|tenure",
    re.IGNORECASE,
)
_LOCKIN_QUERY = re.compile(r"lock[\s-]?in|locked|3\s*year|80c|elss", re.IGNORECASE)
_LOCKIN_PASSAGE = re.compile(r"lock[\s-]?in|locked|3y|3\s*year|80c", re.IGNORECASE)

# Scheme-name fragments, so "HDFC Large Cap" in a query matches a passage that
# says "HDFC Large Cap Fund" and a metadata key that says hdfc_large_cap.
_SCHEME_FRAGMENT = re.compile(r"[a-z]+")


class Grounding:
    def __init__(self, config: AppConfig) -> None:
        self.min_relevance = float(config.grounding.min_relevance)
        self.require_fact_anchor = bool(config.grounding.require_fact_anchor)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def is_numeric_question(query: str) -> bool:
        return bool(_NUMERIC_QUERY.search(query or ""))

    @staticmethod
    def is_lockin_question(query: str) -> bool:
        return bool(_LOCKIN_QUERY.search(query or ""))

    @staticmethod
    def relevance(query_terms: set[str], passage: Passage) -> float:
        """|query ∩ passage| / |query|, over content terms (architecture.md 7.7).

        The scheme's own name is excluded from the denominator. Those terms are
        present in *every* chunk of the scheme - the name, "fund", "direct",
        "growth" - so counting them means any in-scope question scores high
        against any passage of that scheme, and the gate cannot tell "the corpus
        contains this fact" from "the corpus contains this scheme". Concretely,
        "What is the ticker symbol of HDFC Large Cap Fund?" scored 0.67 against
        the overview chunk and was answered, when the corpus contains no ticker
        symbol anywhere. Subtracting them leaves only the terms that carry the
        fact being asked about, which is what the threshold was calibrated for.
        """
        if not query_terms:
            return 0.0
        name_terms = set(content_terms(str(passage.metadata.get("scheme_name", ""))))
        informative = {term for term in query_terms if term not in name_terms}
        if not informative:
            # A query that is nothing but the scheme name carries no fact to
            # ground, so it cannot be relevant; refusing is the safe direction.
            return 0.0
        passage_terms = set(content_terms(f"{passage.section} {passage.text}"))
        return len(informative & passage_terms) / len(informative)

    # -- the gate -----------------------------------------------------------

    def check(
        self,
        query: str,
        passages: Sequence[Passage],
        doc_class_hints: Sequence[str] = (),
    ) -> GroundingResult:
        if not passages:
            return GroundingResult(
                passed=False,
                best_passage=None,
                score=0.0,
                reason="no_candidates",
                fact_anchor_present=False,
            )

        query_terms = set(content_terms(query))
        hints = {hint for hint in (doc_class_hints or ()) if hint}

        best: Passage | None = None
        best_score = -1.0
        for passage in passages:
            score = self.relevance(query_terms, passage)
            if score > best_score:
                best, best_score = passage, score
        assert best is not None  # non-empty passages

        # Condition 1: relevance floor, or a hint match backed by a real term.
        relevance_ok = best_score >= self.min_relevance
        hint_ok = bool(
            hints
            and best.doc_class in hints
            and bool(query_terms & set(content_terms(f"{best.section} {best.text}")))
        )
        if not (relevance_ok or hint_ok):
            return GroundingResult(
                passed=False,
                best_passage=best,
                score=round(max(best_score, 0.0), 6),
                reason=(
                    f"relevance {best_score:.3f} < {self.min_relevance:.2f} and no "
                    f"doc_class hint match (hints={sorted(hints) or 'none'}, "
                    f"best doc_class={best.doc_class!r})"
                ),
                fact_anchor_present=True,
            )

        # Condition 2: the fact anchor must exist in the candidate set.
        if self.require_fact_anchor:
            anchor, anchor_reason = self._fact_anchor(query, passages)
            if not anchor:
                return GroundingResult(
                    passed=False,
                    best_passage=best,
                    score=round(max(best_score, 0.0), 6),
                    reason=anchor_reason,
                    fact_anchor_present=False,
                )
        else:
            anchor_reason = "fact_anchor_not_required"

        return GroundingResult(
            passed=True,
            best_passage=best,
            score=round(max(best_score, 0.0), 6),
            reason=f"relevance {best_score:.3f}, {anchor_reason}",
            fact_anchor_present=True,
        )

    def _fact_anchor(self, query: str, passages: Sequence[Passage]) -> tuple[bool, str]:
        """Does the candidate set structurally contain the kind of evidence asked for?"""
        if self.is_lockin_question(query):
            found = any(_LOCKIN_PASSAGE.search(p.text) for p in passages)
            return found, (
                "lock-in anchor present" if found else "no lock-in anchor in candidates"
            )
        if self.is_numeric_question(query):
            found = any(p.has_digits for p in passages)
            return found, (
                "digit-bearing anchor present" if found else "no digit-bearing passage"
            )
        return True, "no numeric or lock-in anchor required"
