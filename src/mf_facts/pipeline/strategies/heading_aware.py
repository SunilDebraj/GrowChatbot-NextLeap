"""[6a] Heading-aware strategy (architecture.md 7.2 candidate A).

The recommended starting point. It keeps "Exit load" and "Expense ratio" tables
whole, which is what the highest-severity questions in this product need, and it
gives every chunk its section heading as context so the embedded text is
self-describing.

Risk: it depends on heading extraction quality, which is weaker for PDF
factsheets. The P2 evaluation gate decides.
"""

from __future__ import annotations

import re

from ...common.models import Chunk, NormalizedDoc, Table
from ..chunker import BaseStrategy, pack

_HEADING_LINE = re.compile(r"^([A-Z][A-Z0-9 \-&/(),.'%]{3,70}):\s*(.*)$")


class HeadingAwareStrategy(BaseStrategy):
    name = "heading_aware"

    def split(self, doc: NormalizedDoc) -> list[Chunk]:
        sections = self._sections(doc)
        chunks: list[Chunk] = []
        index = 0

        for heading, body in sections:
            tables = self._tables_for(body, doc)
            for table in tables:
                for group in self._table_groups(table):
                    chunk = self.make_chunk(doc, group, heading, index)
                    if chunk:
                        chunks.append(chunk)
                        index += 1

            prose = self._strip_tables(body, tables)
            for piece in self._prose_groups(doc, prose, heading, index):
                chunk = self.make_chunk(doc, piece, heading, index)
                if chunk:
                    chunks.append(chunk)
                    index += 1

            if not tables and not prose.strip():
                chunk = self.make_chunk(doc, heading, heading, index)
                if chunk:
                    chunks.append(chunk)
                    index += 1

        return chunks

    # -- section construction --------------------------------------------

    def _sections(self, doc: NormalizedDoc) -> list[tuple[str, str]]:
        """Prefer the parser's heading tree; fall back to H1/H2/H3-style lines."""
        if doc.headings:
            text = doc.text
            pieces: list[tuple[str, str]] = []
            markers: list[tuple[int, int, str]] = []
            cursor = 0
            for level, heading in doc.headings:
                line_match = self._heading_at_line_start(heading, text, cursor)
                if line_match is not None:
                    start, end = line_match
                    markers.append((start, level, heading))
                    cursor = end
            if markers:
                for position, (start, _level, heading) in enumerate(markers):
                    end = markers[position + 1][0] if position + 1 < len(markers) else len(text)
                    pieces.append((heading, text[start:end].strip()))
                return pieces

        pieces = []
        current_heading = doc.scheme_name
        buffer: list[str] = []
        for line in doc.text.splitlines():
            match = _HEADING_LINE.match(line.strip())
            if match and len(line.strip()) < 72:
                if buffer:
                    pieces.append((current_heading, "\n".join(buffer).strip()))
                    buffer = []
                current_heading = match.group(1).strip()
                remainder = match.group(2).strip()
                if remainder:
                    buffer.append(remainder)
            else:
                buffer.append(line)
        if buffer:
            pieces.append((current_heading, "\n".join(buffer).strip()))
        return pieces or [(doc.scheme_name, doc.text)]

    @staticmethod
    def _heading_at_line_start(heading: str, text: str, cursor: int):
        """Return (start, end) only where a heading begins a line, else None.

        The heading tree covers the whole page while the text is one located
        region, so a generic heading such as "Tax" would otherwise match as a
        substring inside "stamp duty and tax" and cut that line in two, emitting
        a stray "tax" chunk that answers nothing. Requiring a line start keeps
        the match a real heading; a leading "H3: " tag prefix is allowed.
        """
        pattern = re.compile(
            r"^[ \t]*(?:H[1-6]:[ \t]*)?" + re.escape(heading),
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(text, cursor)
        if match is None:
            return None
        return match.start(), match.end()

    def _tables_for(self, body: str, doc: NormalizedDoc) -> list[Table]:
        wanted = {table.to_markdown() for table in doc.tables}
        if not wanted:
            return []
        found: list[Table] = []
        for table in doc.tables:
            rendered = table.to_markdown()
            if rendered and rendered in body:
                found.append(table)
        return found

    def _strip_tables(self, body: str, tables: list[Table]) -> str:
        remainder = body
        for table in tables:
            rendered = table.to_markdown()
            if rendered:
                remainder = remainder.replace(rendered, " ")
        return remainder.strip()

    def _table_groups(self, table: Table) -> list[str]:
        """Render a table as markdown, never splitting a row.

        A table longer than the cap is split on row boundaries and the header row
        is repeated in each part, so every chunk remains readable on its own. The
        header's own token cost is reserved up front; packing the rows to the full
        cap and then prepending a header would put the chunk back over the limit.
        """
        rendered = table.to_markdown()
        if self.count_tokens(rendered) <= self.config.max_chunk_tokens:
            return [rendered] if rendered.strip() else []

        header, *rows = table.rows
        header_text = " | ".join(header)
        repeat = self.config.repeat_table_headers
        overhead = self.count_tokens(header_text) + 2 if repeat else 0
        row_cap = max(self.config.max_chunk_tokens - overhead, 10)

        groups = pack(
            [" | ".join(row) for row in rows],
            target=self.config.target_tokens,
            cap=row_cap,
            count_tokens=self.count_tokens,
        )
        if repeat:
            return [f"{header_text}\n{group}" for group in groups]
        return list(groups)

    def _prose_groups(
        self, doc: NormalizedDoc, prose: str, section: str, start_index: int
    ) -> list[str]:
        """Chunk a section's prose, falling back to character windows.

        Sentence packing is preferred because it preserves meaning; a character
        window is used only when the text has no sentence boundaries at all.
        """
        if not prose.strip():
            return []
        if self.count_tokens(prose) <= self.config.max_chunk_tokens:
            return [prose]

        sentence_chunks = self.split_oversized(doc, prose, section, start_index)
        if sentence_chunks:
            return [chunk.text for chunk in sentence_chunks]
        return _char_windows(prose, self.config)


def _char_windows(text: str, config) -> list[str]:
    """Last-resort windowing for prose with no sentence boundaries.

    A character window is an approximation of the token budget; it is only
    reached when sentence packing found no boundaries, and the cap guard in
    ``make_chunk`` remains authoritative either way.
    """
    window = config.target_tokens * 4
    step = max(window - int(window * config.overlap_ratio), 1)
    return [
        text[start : start + window]
        for start in range(0, len(text), step)
        if text[start : start + window].strip()
    ]
