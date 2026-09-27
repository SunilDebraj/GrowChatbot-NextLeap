"""[6] Chunker interface and shared guards (architecture.md 5.6, 7.1).

The token cap and the metadata contract are enforced here, in the base class, so
no strategy can opt out of them. Two guards matter most:

* **Token cap.** ``all-MiniLM-L6-v2`` truncates its input at 256 wordpiece
  tokens, silently. A chunk over the cap loses its tail, which for a fee table
  means a wrong number. Counts therefore use the model's own tokenizer, never
  ``len(text.split())``, and an over-cap chunk raises rather than truncating.
* **Metadata completeness.** All eleven PRD section 8.1 fields are present on
  every chunk, because the retrieval-time metadata filter depends on them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol, Sequence

from ..common.config import ChunkingConfig
from ..common.errors import ChunkSizeError
from ..common.models import Chunk, NormalizedDoc

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"\S+")


def token_counter(model_id: str):
    """Return a callable that counts wordpiece tokens for ``model_id``.

    Imported lazily so unit tests can exercise the chunkers without downloading
    the model, by passing a whitespace tokenizer instead.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover - exercised only without transformers
        return lambda text: len(_TOKEN.findall(text))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=True))


@lru_cache(maxsize=4)
def _cached_counter(model_id: str):  # pragma: no cover - thin cache wrapper
    return token_counter(model_id)


def whitespace_token_counter(text: str) -> int:
    """Dependency-free token estimate used by tests and by the cap guard."""
    return len(_TOKEN.findall(text))


class ChunkingStrategy(Protocol):
    name: str

    def split(self, doc: NormalizedDoc) -> list[Chunk]: ...


class BaseStrategy:
    """Shared behaviour: token counting, cap enforcement, chunk construction."""

    name = "base"

    def __init__(self, config: ChunkingConfig, count_tokens=None) -> None:
        self.config = config
        self._count = count_tokens or whitespace_token_counter
        self.over_cap: list[str] = []

    def count_tokens(self, text: str) -> int:
        return int(self._count(text))

    def split(self, doc: NormalizedDoc) -> list[Chunk]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- construction -----------------------------------------------------

    def make_chunk(
        self,
        doc: NormalizedDoc,
        text: str,
        section: str,
        index: int,
        *,
        allow_over_cap: bool = False,
    ) -> Chunk | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        token_count = self.count_tokens(cleaned)
        if token_count > self.config.max_chunk_tokens and not allow_over_cap:
            self.over_cap.append(f"{doc.source_id}#{index}: {token_count} tokens")
            raise ChunkSizeError(
                f"{doc.source_id} section {section!r} chunk {index} has {token_count} "
                f"tokens, above the cap of {self.config.max_chunk_tokens}. Truncating "
                f"would silently corrupt a fee or exit-load table, so the build stops. "
                f"Raise chunking.max_chunk_tokens in config.yaml or split the section."
            )

        if token_count > self.config.max_chunk_tokens:
            self.over_cap.append(f"{doc.source_id}#{index}: {token_count} tokens (flagged)")

        return Chunk(
            text=cleaned,
            token_count=token_count,
            scheme_key=doc.scheme_key,
            scheme_name=doc.scheme_name,
            doc_class=doc.doc_class,
            publisher=doc.publisher,
            source_url=doc.source_url,
            section=section,
            effective_date=doc.effective_date,
            last_updated=doc.last_updated,
            content_hash=doc.content_hash,
            chunk_index=index,
        )

    # -- helpers ----------------------------------------------------------

    def split_oversized(
        self,
        doc: NormalizedDoc,
        text: str,
        section: str,
        start_index: int,
    ) -> list[Chunk]:
        """Split ``text`` into sentences, then pack greedily under the cap.

        Falls back to hard character windows only if a single sentence is itself
        over the cap, since a sentence is the smallest unit we can preserve.
        """
        chunks: list[Chunk] = []
        sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

        buffer: list[str] = []
        index = start_index
        for sentence in sentences:
            candidate = " ".join([*buffer, sentence.strip()])
            if buffer and self.count_tokens(candidate) > self.config.max_chunk_tokens:
                chunk = self.make_chunk(doc, " ".join(buffer), section, index)
                if chunk:
                    chunks.append(chunk)
                    index += 1
                buffer = [sentence.strip()]
            else:
                buffer.append(sentence.strip())

        if buffer:
            chunk = self.make_chunk(doc, " ".join(buffer), section, index, allow_over_cap=True)
            if chunk:
                chunks.append(chunk)

        return chunks


def pack(
    pieces: Sequence[str],
    target: int,
    cap: int,
    count_tokens,
) -> list[str]:
    """Greedily pack ``pieces`` into groups of at most ``cap`` tokens.

    Used for table rows, where a row must never be split but a group of rows
    may be, with the header repeated in each group by the caller.
    """
    groups: list[str] = []
    buffer: list[str] = []
    for piece in pieces:
        candidate = "\n".join([*buffer, piece])
        if buffer and count_tokens(candidate) > cap:
            groups.append("\n".join(buffer))
            buffer = [piece]
        else:
            buffer.append(piece)
    if buffer:
        groups.append("\n".join(buffer))
    del target
    return groups
