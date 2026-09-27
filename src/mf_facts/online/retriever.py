"""Node [15] - retriever (architecture.md 5.15, PRD FR2 step 3).

One job: get candidate passages for the schemes the user actually named, and get
nothing else. The scheme filter is hard (decision D3) because a passage from an
unrequested scheme can only ever produce a confident answer about the wrong fund,
and "we also found this in a similar fund" is not a safe failure mode here.

``top_k`` over-fetches on purpose. Retrieval is not the gate - grounding is - so
this stage is allowed to be generous, and the cost of a wide candidate set is
paid downstream by node [17] rather than by a wrong answer.
"""

from __future__ import annotations

from typing import Sequence

from ..common.config import AppConfig
from ..common.models import Passage
from ..pipeline.embedder import Embedder
from ..pipeline.store import ChromaStore


class Retriever:
    """Vector search over the collection, hard-filtered to resolved schemes."""

    def __init__(
        self,
        config: AppConfig,
        store: ChromaStore,
        embedder: Embedder,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.top_k = int(top_k if top_k is not None else config.retrieval.top_k)
        self.top_n = int(top_n if top_n is not None else config.retrieval.top_n)

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed([query])[0]

    def retrieve(
        self,
        query_vec: Sequence[float],
        scheme_keys: Sequence[str],
        top_k: int | None = None,
    ) -> list[Passage]:
        """Return candidates for ``scheme_keys``, best first.

        An empty ``scheme_keys`` returns nothing rather than searching the whole
        collection. The rewriter only produces an empty tuple for a query that
        named no scheme, and the classifier sends those to ``out_of_scope`` before
        retrieval; returning everything here would be an unguarded path to
        answering about a fund the user never mentioned.
        """
        keys = [key for key in (scheme_keys or ()) if key]
        if not keys:
            return []

        limit = int(top_k if top_k is not None else self.top_k)
        result = self.store.query(
            query_vec,
            n_results=limit,
            where={"scheme_key": {"$in": keys}},
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        passages: list[Passage] = []
        for rank, chunk_id in enumerate(ids):
            distance = float(distances[rank]) if rank < len(distances) else 1.0
            passages.append(
                Passage(
                    id=str(chunk_id),
                    text=documents[rank] if rank < len(documents) else "",
                    metadata=dict(metadatas[rank] or {}) if rank < len(metadatas) else {},
                    # 1 - distance, per architecture.md 5.15. Chroma's default is L2
                    # on normalised vectors, which is monotonic with cosine, so the
                    # ordering is what matters and the absolute value is a score.
                    similarity=round(1.0 - distance, 6),
                    vector_rank=rank,
                )
            )
        return passages

    def candidate_source_urls(self, passages: Sequence[Passage]) -> set[str]:
        """The set validator check 3 measures the attached URL against.

        Built from the candidates rather than the final four, so an over-fetched
        URL is still recognised as legitimate and not stripped as an invention.
        """
        return {passage.source_url for passage in passages if passage.source_url}
