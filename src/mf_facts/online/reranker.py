"""Node [16] - reranker (architecture.md 5.15, PRD FR2 steps 4-5).

Reciprocal Rank Fusion over two orderings of the *same* candidate set:

    score(d) = sum_r 1 / (k + rank_r(d)),   k = 60

The second ordering is lexical (BM25 over the candidates' own text). It is not a
substitute for the vector store - it is a second opinion on the identical
candidates, and that is what makes the fusion useful. A cross-encoder was
deliberately not used (decision D9): at five schemes it would add a dependency,
latency, and a tuning surface to solve a problem RRF plus a keyword ordering
already answers, and it would make the ranking impossible to read off by hand.

After fusion come the three fact-anchoring boosts from architecture.md 5.15. They
are small, additive, and each records *why* it fired, because a ranking that
cannot explain itself is a ranking nobody can debug when it picks the wrong
passage.

The citation is decided here, structurally: ``passages[0]`` after fusion. Not the
union of candidate URLs, which is the most common way "exactly one link" breaks
(implementation.md P4 pitfalls).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Sequence

from ..common.config import AppConfig
from ..common.models import Passage

_WORD = re.compile(r"[a-z0-9]+")

# Terms carrying no retrieval signal. Kept short on purpose: an aggressive stop
# list is the usual reason a lexical second opinion stops disagreeing with the
# vector ranking at all.
_STOPWORDS = frozenset(
    """
    a an and are as at be been by for from has have how in is it its of on or
    that the their there this to was were what when where which who whom whose
    with do does did can could should would will shall my your our their his her
    """.split()
)


def content_terms(text: str) -> list[str]:
    """Lowercased, stop-worded alphanumeric tokens (architecture.md 7.7)."""
    return [word for word in _WORD.findall((text or "").lower()) if word not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed, small candidate set.

    Scoped to the passages handed in rather than the whole collection on purpose:
    the fusion is a second opinion on the same candidates, and IDF computed over
    20 chunks is already a weak signal. Using corpus-wide statistics would be more
    "correct" and would quietly reintroduce a corpus-sized dependency into a
    component whose value is that it is inspectable.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [content_terms(doc) for doc in documents]
        self.lengths = [len(doc) for doc in self.documents]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.documents else 0.0
        self.frequencies = [Counter(doc) for doc in self.documents]
        self.n_docs = len(self.documents)

    def scores(self, query: str) -> list[float]:
        terms = content_terms(query)
        if not terms or not self.n_docs:
            return [0.0] * self.n_docs
        result = []
        for index in range(self.n_docs):
            frequency = self.frequencies[index]
            length = self.lengths[index] or 1
            score = 0.0
            for term in terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                containing = sum(1 for doc in self.documents if term in doc)
                # +1 inside the log keeps the IDF positive for a term that occurs
                # in every candidate, where the true IDF is 0 or negative.
                idf = math.log(1.0 + (self.n_docs - containing + 0.5) / (containing + 0.5))
                score += idf * (count * (self.k1 + 1.0)) / (
                    count + self.k1 * (1.0 - self.b + self.b * length / (self.avg_length or 1.0))
                )
            result.append(score)
        return result


class Reranker:
    def __init__(self, config: AppConfig) -> None:
        retrieval = config.retrieval
        self.rrf_k = int(retrieval.rrf_k)
        self.top_n = int(retrieval.top_n)
        self.doc_class_boost = float(retrieval.doc_class_boost)
        self.section_match_boost = float(retrieval.section_match_boost)
        self.numeric_anchor_boost = float(retrieval.numeric_anchor_boost)

    def rerank(
        self,
        passages: Sequence[Passage],
        query: str,
        doc_class_hints: Sequence[str] = (),
        numeric_question: bool = False,
        top_n: int | None = None,
    ) -> list[Passage]:
        """Fuse, boost, and return the top ``n`` passages.

        Input order is the vector order, which is the only ordering Chroma
        guarantees, so ``vector_rank`` is taken from the caller's ordering rather
        than re-derived.
        """
        passages = list(passages)
        if not passages:
            return []

        lexical_scores = BM25([passage.text for passage in passages]).scores(query)
        lexical_order = sorted(
            range(len(passages)),
            key=lambda index: (-lexical_scores[index], index),
        )
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order)}

        hints = {hint for hint in (doc_class_hints or ()) if hint}
        query_terms = set(content_terms(query))

        scored: list[Passage] = []
        for index, passage in enumerate(passages):
            reasons: list[str] = []
            fused = 1.0 / (self.rrf_k + passage.vector_rank + 1) + 1.0 / (
                self.rrf_k + lexical_rank[index] + 1
            )

            if hints and passage.doc_class in hints:
                fused += self.doc_class_boost
                reasons.append(f"doc_class:{passage.doc_class}")

            section_terms = set(content_terms(passage.section))
            if query_terms & section_terms:
                fused += self.section_match_boost
                reasons.append("section_keyword")

            if numeric_question and passage.has_digits:
                fused += self.numeric_anchor_boost
                reasons.append("numeric_anchor")

            scored.append(
                replace(
                    passage,
                    lexical_rank=lexical_rank[index],
                    fused_score=round(fused, 6),
                    boost_reasons=tuple(reasons),
                )
            )

        scored.sort(
            key=lambda passage: (
                -passage.fused_score,
                passage.vector_rank,
                passage.id,
            )
        )
        return scored[: int(top_n if top_n is not None else self.top_n)]
