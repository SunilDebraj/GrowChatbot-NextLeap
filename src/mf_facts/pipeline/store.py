"""[8] ChromaDB store (architecture.md 5.8).

Every Chroma access in the project lives in this module (rule GR1). The
collection carries its embedding model in its own metadata, and a mismatch
against config raises rather than writing, because mixing vector spaces in one
collection produces retrieval that looks plausible and is wrong.

Ids are deterministic (``scheme_key:doc_class:hash:index``), which is what makes
re-ingestion idempotent: the same corpus produces the same ids and upserts onto
itself instead of accumulating duplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..common.errors import CollectionUnavailable, EmbeddingModelMismatch
from ..common.logging import log_event
from ..common.models import Chunk


class ChromaStore:
    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection = None

    # -- lifecycle --------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:  # pragma: no cover
                raise CollectionUnavailable(
                    "chromadb is required. Install dependencies with: pip install -e ."
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.path), settings=Settings(anonymized_telemetry=False)
                )
            except Exception as exc:
                raise CollectionUnavailable(
                    f"could not open the persistent client at {self.path}: {exc}"
                ) from exc
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "embedding_model": self.embedding_model,
                        "embedding_dim": int(self.embedding_dim),
                    },
                )
            except Exception as exc:
                raise CollectionUnavailable(
                    f"could not open collection {self.collection_name!r}: {exc}"
                ) from exc
            self._verify_model()
        return self._collection

    def _verify_model(self) -> None:
        try:
            stored = self._collection.metadata or {}
        except Exception:
            return
        recorded = stored.get("embedding_model")
        if recorded and recorded != self.embedding_model:
            raise EmbeddingModelMismatch(
                f"collection {self.collection_name!r} was built with "
                f"{recorded!r} but config specifies {self.embedding_model!r}. "
                f"Delete {self.path} and rebuild, or restore the previous model."
            )

    # -- writes -----------------------------------------------------------

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector length mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            duplicates = _duplicates(ids)
            raise ValueError(
                "deterministic chunk ids collided, which would make re-ingestion "
                f"non-idempotent: {duplicates[:5]}"
            )

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=[list(map(float, v)) for v in vectors],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.to_chroma_metadata() for chunk in chunks],
            )
        except Exception as exc:
            raise CollectionUnavailable(f"upsert failed: {exc}") from exc

        log_event(
            _logger(),
            "store.upsert",
            count=len(ids),
            chunk_count=len(ids),
        )
        return len(ids)

    def delete_all(self) -> None:
        try:
            self.collection.delete(where={"scheme_key": {"$ne": ""}})
        except Exception:
            pass

    def reconcile(self, keep_ids: Sequence[str]) -> int:
        """Drop stored chunks that are not in the current corpus.

        Ids are content-addressed, so a locator or parser fix produces new ids and
        leaves the previous ones behind. Upsert alone would then serve stale facts
        that no longer exist in any source document, which for a facts-only bot is
        worse than a missing answer. Returns the number removed.
        """
        keep = set(keep_ids)
        try:
            existing = self.collection.get(include=[])["ids"]
        except Exception as exc:
            raise CollectionUnavailable(f"could not list stored ids: {exc}") from exc

        stale = [chunk_id for chunk_id in existing if chunk_id not in keep]
        if not stale:
            return 0
        try:
            self.collection.delete(ids=stale)
        except Exception as exc:
            raise CollectionUnavailable(f"could not delete stale chunks: {exc}") from exc
        return len(stale)

    # -- reads ------------------------------------------------------------

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            raise CollectionUnavailable(f"count failed: {exc}") from exc

    def get_all(self, limit: int | None = None) -> dict[str, Any]:
        try:
            return self.collection.get(limit=limit, include=["documents", "metadatas"])
        except Exception as exc:
            raise CollectionUnavailable(f"get failed: {exc}") from exc

    def get_all_with_vectors(self) -> dict[str, Any]:
        """Everything in the collection, vectors included.

        Separate from get_all on purpose: the vectors are ~1.5 KB per chunk as
        float64 text and no caller except the corpus dump wants them, so loading
        them into a rebuild or a manifest is pure cost. IDs are returned because
        an embedding is meaningless without the id that ties it to a chunk.
        """
        try:
            return self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as exc:
            raise CollectionUnavailable(f"get with vectors failed: {exc}") from exc

    def where(self, where: dict[str, Any], limit: int = 1) -> dict[str, Any]:
        try:            return self.collection.get(where=where, limit=limit, include=["metadatas"])
        except Exception as exc:
            raise CollectionUnavailable(f"filtered get failed: {exc}") from exc

    def query(
        self,
        vector: Sequence[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Nearest-neighbour search, optionally hard-filtered by metadata.

        Lives here so rule GR1 stays true: no other module talks to Chroma. The
        caller passes the filter, because "which scheme_keys are in scope" is a
        query-pipeline decision, not a storage one.

        Distances are L2 by default, so callers wanting cosine similarity use
        ``1 - distance``. The embedder normalises vectors, which makes that
        equivalent to cosine - see architecture.md 7.5.
        """
        try:
            return self.collection.query(
                query_embeddings=[list(map(float, vector))],
                n_results=int(n_results),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise CollectionUnavailable(f"query failed: {exc}") from exc


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for value in values:
        if value in seen:
            repeated.append(value)
        seen.add(value)
    return repeated


def _logger():
    import logging

    return logging.getLogger("mf_facts.store")
