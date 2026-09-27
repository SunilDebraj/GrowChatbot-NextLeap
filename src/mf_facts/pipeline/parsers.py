"""[3] Parsers: bytes -> ParsedDoc (architecture.md 5.3).

Two implementations behind one entry point. The HTML parser is the one that
carries the corpus, and it implements the locator grammar documented in
``config/sources.yaml`` so that no selector or label is hardcoded here (GR12).

PDF support exists for the factsheet/KIM/SID doc classes. Those documents are
not currently reachable from any source we can fetch (see the ``unavailable:``
block in sources.yaml), so the PDF path is exercised by tests and fixtures only.
"""

from __future__ import annotations

import io
import re
from typing import Callable, Iterable

from ..common.errors import ParseError
from ..common.models import ParsedDoc, SourceSpec, Table

_WS = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

#: Elements stripped unconditionally: chrome, not content.
_DROP_TAGS = ("script", "style", "svg", "noscript", "nav", "footer", "form")

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def normalize_ws(text: str) -> str:
    return _MULTI_NEWLINE.sub("\n\n", _WS.sub(" ", text)).strip()


def _table_rows(table) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for row in table.find_all("tr"):
        cells = [
            normalize_ws(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(tuple(cells))
    return rows


def _table_markdown(table, rows) -> str:
    markdown = Table(rows=tuple(rows), caption="").to_markdown()
    heading = table.find_previous(HEADING_TAGS)
    if heading is not None:
        caption = _clean_text(heading)
        if caption:
            markdown = f"{caption}\n{markdown}"
    return markdown


def _clean_text(element) -> str:
    return normalize_ws(element.get_text("\n", strip=True))


def _tables_in_region(tables: tuple[Table, ...], text: str) -> tuple[Table, ...]:
    """Keep only the tables that the locator actually extracted.

    Tables are collected from the whole page, so a spec for the fees section
    would otherwise carry the holdings table with it. That mislabels portfolio
    allocation percentages as fee data and hands the chunker rows that belong to
    a different document class.
    """
    return tuple(
        table
        for table in tables
        if normalize_ws(table.to_markdown()).strip() in normalize_ws(text)
    )


def _heading_level(tag: str) -> int:
    return int(tag[1]) if tag[1:].isdigit() else 6


class HtmlParser:
    """Extracts a spec's region from an HTML page."""

    def __init__(self, exclude_patterns: Iterable[str] = ()) -> None:
        self.exclude_patterns = tuple(p.lower() for p in exclude_patterns)

    def parse(self, raw_bytes: bytes, spec: SourceSpec) -> ParsedDoc:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise ParseError("beautifulsoup4 is required to parse HTML sources") from exc

        try:
            soup = BeautifulSoup(raw_bytes, "html.parser")
        except Exception as exc:
            raise ParseError(f"{spec.source_id}: {type(exc).__name__}: {exc}") from exc

        self._strip(soup)
        self._drop_performance_blocks(soup)

        headings = self._collect_headings(soup)
        tables = self._collect_tables(soup)

        parts = self._apply_locator(soup, spec.locator)
        text = normalize_ws("\n\n".join(part for part in parts if part))

        if not text:
            raise ParseError(
                f"{spec.source_id}: locator {spec.locator!r} matched no text "
                f"(page layout may have changed; check config/sources.yaml)"
            )

        return ParsedDoc(
            source_id=spec.source_id,
            text=text,
            headings=headings,
            tables=_tables_in_region(tables, text),
        )

    def _strip(self, soup) -> None:
        for tag in _DROP_TAGS:
            for element in soup.find_all(tag):
                element.decompose()
        for selector in ("[aria-hidden='true']", "[role='presentation']"):
            for element in soup.select(selector):
                element.decompose()

    def _matches_exclusion(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern in lowered for pattern in self.exclude_patterns)

    def _drop_performance_blocks(self, soup) -> None:
        """Remove sections holding return figures before anything is extracted.

        Driver D5 / rule GR6: the scheme pages carry "Return calculator",
        "Returns and rankings" and "Compare similar funds" regions with real
        return percentages. They must never reach the vector store, because that
        would place performance numbers one retrieval away from the generator.
        This runs ahead of locator evaluation so every locator inherits it.

        Removal walks maximal document-order units, not raw find_all_next()
        output. Yielding descendants and siblings together can capture the
        container shared with the next section and delete it, which silently
        removed the fees section that followed "Returns and rankings". Restricting
        to maximal units removes the block without touching its neighbours.
        """
        for heading in list(soup.find_all(HEADING_TAGS)):
            if heading.parent is None:
                continue
            if not self._matches_exclusion(_clean_text(heading)):
                continue
            level = _heading_level(heading.name)
            for element in [heading, *self._section_units(heading, level)]:
                if element.parent is not None:
                    element.decompose()

        for element in list(soup.find_all(HEADING_TAGS)):
            if self._matches_exclusion(_clean_text(element)):
                element.decompose()

    def _next_heading_at_or_above(self, heading, level: int):
        for element in heading.find_all_next():
            if element.name in HEADING_TAGS and _heading_level(element.name) <= level:
                return element
        return None

    @staticmethod
    def _contains(node, target) -> bool:
        if target is None:
            return False
        return any(descendant is target for descendant in node.descendants)

    def _collect_headings(self, soup) -> tuple[tuple[int, str], ...]:
        found: list[tuple[int, str]] = []
        for tag in soup.find_all(HEADING_TAGS):
            text = _clean_text(tag)
            if text:
                found.append((_heading_level(tag.name), text))
        return tuple(found)

    def _collect_tables(self, soup) -> tuple[Table, ...]:
        tables: list[Table] = []
        for element in soup.find_all("table"):
            rows: list[tuple[str, ...]] = []
            for row in element.find_all("tr"):
                cells = [normalize_ws(c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"])]
                if any(cells):
                    rows.append(tuple(cells))
            if rows:
                caption = ""
                heading = element.find_previous(HEADING_TAGS)
                if heading is not None:
                    caption = _clean_text(heading)
                tables.append(Table(rows=tuple(rows), caption=caption))
        return tuple(tables)

    def _apply_locator(self, soup, locator: str) -> list[str]:
        parts: list[str] = []
        for raw_part in locator.split(";"):
            part = raw_part.strip()
            if not part:
                continue
            kind, _, argument = part.partition(":")
            kind = kind.strip().lower()
            handler: Callable[[], str] | None = {
                "full": lambda: self._region_text(soup),
                "css": lambda: self._by_css(soup, argument),
                "heading": lambda: self._by_heading(soup, argument),
                "labels": lambda: self._by_labels(soup, argument),
                "pill": lambda: self._by_pill(soup, argument),
            }.get(kind)

            if handler is None:
                raise ParseError(
                    f"{kind!r} is not a legal locator kind; "
                    "expected full, css, heading, labels or pill"
                )
            parts.append(handler())
        return parts

    def _region_text(self, element) -> str:
        """Extract a region's text with its tables rendered as markdown.

        get_text() flattens a table into one line per cell, which destroys the
        row structure the chunker needs in order to avoid splitting a fee slab
        mid-row. Replacing each table with its markdown rendering keeps the rows
        intact and makes the extracted text match what the chunker looks for.
        """
        self._render_tables(element)
        return _clean_text(element)

    def _render_tables(self, element) -> None:
        from bs4 import NavigableString

        # find_all() does not include the element itself, and a located section
        # can hand back the table as its own unit.
        targets = list(element.find_all("table"))
        if getattr(element, "name", None) == "table":
            targets.insert(0, element)

        for table in targets:
            rows = _table_rows(table)
            if not rows:
                table.decompose()
                continue
            markdown = _table_markdown(table, rows)
            table.replace_with(NavigableString("\n" + markdown + "\n"))

    def _by_css(self, soup, selector: str) -> str:
        element = soup.select_one(selector.strip())
        return self._region_text(element) if element is not None else ""

    def _by_heading(self, soup, text: str) -> str:
        """Capture a heading and the content that belongs to it.

        Content is collected as document-order units that are not nested inside
        another collected unit. Reading a container's text and then each nested
        element's text duplicates the same sentence several times, and Groww
        wraps sections in nested containers, so the heading's own next_siblings
        are often empty even though content follows it.
        """
        target = normalize_ws(text).lower()
        for heading in soup.find_all(HEADING_TAGS):
            if _clean_text(heading).lower() != target:
                continue
            level = _heading_level(heading.name)
            buffer: list[str] = [f"{heading.name.upper()}: {_clean_text(heading)}"]
            title = _clean_text(heading).lower()
            for unit in self._section_units(heading, level):
                if getattr(unit, "name", None) == "table":
                    # Rendering replaces the table in the tree, so its text has to
                    # be taken from the markdown rather than read back afterwards.
                    rows = _table_rows(unit)
                    value = _table_markdown(unit, rows) if rows else ""
                else:
                    self._render_tables(unit)
                    value = _clean_text(unit)
                if not value or self._adds_nothing(value, title):
                    continue
                buffer.append(value)
            return normalize_ws("\n".join(buffer))
        return ""

    @staticmethod
    def _adds_nothing(value: str, title: str) -> bool:
        """True when a unit is already covered by its own section heading.

        Groww repeats a bare label ("tax") as a separate element inside the
        section it names. Left in place it becomes a one-word chunk in the store
        that matches no question and displaces better passages.
        """
        lowered = value.lower()
        return len(lowered) <= 40 and lowered in title

    def _section_units(self, heading, level: int) -> list:
        """Maximal elements between a heading and the next heading of its level.

        Stops before any ancestor of the boundary heading, so the section that
        follows is never absorbed into this one.
        """
        stop = self._next_heading_at_or_above(heading, level)
        stop_ancestors = {id(node) for node in stop.parents} if stop is not None else set()

        units: list = []
        collected: set[int] = set()
        for element in heading.next_elements:
            if element is stop or id(element) in stop_ancestors:
                break
            if getattr(element, "name", None) is None:
                continue
            if any(id(node) in collected for node in element.parents):
                continue
            collected.add(id(element))
            units.append(element)
        return units

    def _by_labels(self, soup, argument: str) -> str:
        buffer: list[str] = []
        for label in (part.strip() for part in argument.split("|")):
            if not label:
                continue
            wanted = normalize_ws(label).lower()
            for element in soup.find_all(string=True):
                if normalize_ws(str(element)).lower() != wanted:
                    continue
                parent = element.parent
                if parent is None:
                    continue
                sibling = parent.find_next_sibling()
                if sibling is None:
                    continue
                value = _clean_text(sibling)
                if value:
                    buffer.append(f"{label.strip()}: {value}")
                break
        return "\n".join(buffer)

    def _by_pill(self, soup, value_text: str) -> str:
        """Capture a badge-style value such as the riskometer rating.

        The visible text sits on an inner <span> while the badge styling lives on
        an ancestor, so the class hint is matched against the whole ancestor
        chain rather than the direct parent only.
        """
        wanted = normalize_ws(value_text).lower()
        for element in soup.find_all(string=True):
            if normalize_ws(str(element)).lower() != wanted:
                continue
            if not self._has_badge_ancestor(element):
                continue
            return f"Riskometer: {normalize_ws(str(element))}"
        return ""

    @staticmethod
    def _has_badge_ancestor(element, depth: int = 3) -> bool:
        node = element.parent
        for _ in range(depth):
            if node is None:
                return False
            classes = " ".join(node.get("class") or []).lower()
            if "pill" in classes or "badge" in classes or "chip" in classes:
                return True
            node = node.parent
        return False


class PdfParser:
    """Extracts text and best-effort tables from a PDF (factsheet/KIM/SID).

    pypdf has no table model. Lines containing two or more multi-space column
    gaps are reconstructed as pipe-delimited rows, which is what the
    heading-aware chunker needs to protect a fee table from being split. Rows
    that come out ragged are flagged by the caller rather than emitted, because
    a garbled fee slab is worse than a missing one.
    """

    def __init__(self, exclude_patterns: Iterable[str] = ()) -> None:
        self.exclude_patterns = tuple(p.lower() for p in exclude_patterns)

    def parse(self, raw_bytes: bytes, spec: SourceSpec) -> ParsedDoc:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ParseError("pypdf is required to parse PDF sources") from exc

        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ParseError(f"{spec.source_id}: unreadable PDF: {exc}") from exc

        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(normalize_ws(page.extract_text() or ""))
            except Exception:
                pages.append("")

        text = self._drop_excluded_lines("\n\n".join(p for p in pages if p))
        if not text:
            raise ParseError(f"{spec.source_id}: PDF produced no extractable text")

        return ParsedDoc(
            source_id=spec.source_id,
            text=normalize_ws(text),
            headings=self._infer_headings(text),
            tables=self._infer_tables(text),
            page_count=len(reader.pages),
        )

    def _drop_excluded_lines(self, text: str) -> str:
        if not self.exclude_patterns:
            return text
        kept = []
        skipping = False
        for line in text.splitlines():
            lowered = line.lower()
            if any(pattern in lowered for pattern in self.exclude_patterns):
                skipping = True
                continue
            if skipping and not line.strip():
                skipping = False
                continue
            if not skipping:
                kept.append(line)
        return "\n".join(kept)

    def _infer_headings(self, text: str) -> tuple[tuple[int, str], ...]:
        found: list[tuple[int, str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 80:
                continue
            if stripped.isupper() and len(stripped.split()) <= 8:
                found.append((2, stripped))
        return tuple(found)

    def _infer_tables(self, text: str) -> tuple[Table, ...]:
        tables: list[Table] = []
        current: list[tuple[str, ...]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    tables.append(Table(rows=tuple(current)))
                    current = []
                continue
            cells = [cell.strip() for cell in re.split(r"\s{3,}|\|", stripped) if cell.strip()]
            if len(cells) >= 2:
                current.append(tuple(cells))
            elif current:
                tables.append(Table(rows=tuple(current)))
                current = []
        if current:
            tables.append(Table(rows=tuple(current)))

        # A table with ragged rows is dropped rather than emitted: a half-parsed
        # fee schedule can produce a wrong number, which is the one failure this
        # project cannot tolerate.
        return tuple(
            table
            for table in tables
            if len(table.rows) >= 2
            and len({len(row) for row in table.rows}) == 1
        )


def build_parser(content_type: str, exclude_patterns: Iterable[str] = ()) -> HtmlParser | PdfParser:
    if "pdf" in content_type.lower() or content_type.lower().endswith(".pdf"):
        return PdfParser(exclude_patterns)
    return HtmlParser(exclude_patterns)
