"""[6b] Fixed-token-window strategy (architecture.md 7.2 candidate B).

The baseline that candidates A and C must beat. It makes no structural
assumptions, which is its virtue and its limitation: a plain window can cut a
fee slab or an exit-load table in half, and a half-table is how this product ends
up quoting a wrong number. It is retained as the control in the P2 comparison,
not as a production candidate.
"""

from __future__ import annotations

from ...common.models import Chunk, NormalizedDoc
from ..chunker import BaseStrategy

#: A row is a pipe-delimited or whitespace-column line. Rows are never split.
_ROW = (" | ", "\t")


class FixedWindowStrategy(BaseStrategy):
    name = "fixed_window"

    def split(self, doc: NormalizedDoc) -> list[Chunk]:
        window = self.config.target_tokens
        stride = max(window - int(window * self.config.overlap_ratio), 1)

        units = self._units(doc.text)
        chunks: list[Chunk] = []
        index = 0
        cursor = 0
        buffer: list[str] = []
        buffered_tokens = 0

        while cursor < len(units):
            unit = units[cursor]
            unit_tokens = self.count_tokens(unit)

            if buffered_tokens and buffered_tokens + unit_tokens > window:
                chunk = self.make_chunk(doc, "\n".join(buffer), "full", index)
                if chunk:
                    chunks.append(chunk)
                    index += 1
                buffer = []
                buffered_tokens = 0

            buffer.append(unit)
            buffered_tokens += unit_tokens
            cursor += stride if self.config.overlap_ratio and len(units) > window else 1

        if buffer:
            chunk = self.make_chunk(doc, "\n".join(buffer), "full", index)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _units(self, text: str) -> list[str]:
        """Split into indivisible units: table rows, else sentences, else words."""
        lines = [line for line in text.splitlines() if line.strip()]
        if lines and all(any(marker in line for marker in _ROW) for line in lines):
            return lines

        sentences: list[str] = []
        for line in lines:
            sentences.extend(part for part in line.split(". ") if part.strip())
        if sentences:
            return [s if s.endswith(".") else s + "." for s in sentences]

        return text.split()
