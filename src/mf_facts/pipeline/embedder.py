"""[7] Embedder: chunks -> vectors (architecture.md 5.7).

``all-MiniLM-L6-v2`` is 384-dimensional and mean-pooled with L2 normalization.
Both the model id and the dimension are asserted, because a silent dimension
change would mix incompatible vectors in one collection.

Embeddings are cached on ``sha256(model_id + text)``, so a re-run after a
chunking change only re-embeds the chunks whose text actually moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..common.config import EmbeddingConfig
from ..common.errors import ConfigError
from ..common.models import Chunk


class Embedder:
    def __init__(self, config: EmbeddingConfig, cache_dir: str | Path | None = None) -> None:
        self.config = config
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None

    @property
    def model(self):
        """Load the sentence-transformers model on first use, not at import."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ConfigError(
                    "sentence-transformers is required to embed chunks. "
                    "Install dependencies with: pip install -e ."
                ) from exc
            self._model = SentenceTransformer(self.config.model_id)
        return self._model

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.config.model_id}\x00{text}".encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / key[:2] / f"{key}.json"

    def _load_cached(self, texts: Sequence[str]) -> dict[int, list[float]]:
        if not self._cache_dir:
            return {}
        found: dict[int, list[float]] = {}
        for position, text in enumerate(texts):
            path = self._cache_path(self._cache_key(text))
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            vector = payload.get("vector")
            if isinstance(vector, list) and len(vector) == self.config.embedding_dim:
                found[position] = vector
        return found

    def _store_cached(self, texts: Sequence[str], vectors: list[list[float]]) -> None:
        if not self._cache_dir:
            return
        for text, vector in zip(texts, vectors):
            path = self._cache_path(self._cache_key(text))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"model": self.config.model_id, "vector": vector}),
                encoding="utf-8",
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, returning one 384-dim vector per input, in order."""
        if not texts:
            return []

        cached = self._load_cached(texts)
        missing_positions = [i for i in range(len(texts)) if i not in cached]
        if missing_positions:
            missing_texts = [texts[i] for i in missing_positions]
            computed = self.model.encode(
                missing_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            computed_list = [list(map(float, row)) for row in computed]
            for position, vector in zip(missing_positions, computed_list):
                if len(vector) != self.config.embedding_dim:
                    raise ConfigError(
                        f"{self.config.model_id} returned {len(vector)} dimensions but "
                        f"embedding.embedding_dim is {self.config.embedding_dim}"
                    )
                cached[position] = vector
            self._store_cached(missing_texts, computed_list)

        return [cached[i] for i in range(len(texts))]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        return self.embed([chunk.text for chunk in chunks])
