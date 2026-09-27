"""[1] Source registry: config/sources.yaml -> List[SourceSpec] (architecture.md 5.1).

Also owns writing ``artifacts/sources.csv`` and ``artifacts/sources.md``, which is
the source-transparency deliverable in PRD.md FR8. Both are regenerated on every
build from the actual run and are never hand-edited.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..common.errors import ConfigError
from ..common.models import FetchRun, SourceSpec

#: Columns of sources.csv, in order. PRD.md FR8 / architecture.md 5.1.
SOURCE_CSV_COLUMNS: tuple[str, ...] = (
    "source_id",
    "scheme_key",
    "scheme_name",
    "doc_class",
    "url",
    "publisher",
    "effective_date",
    "last_updated",
    "retrieved_at",
    "content_hash",
    "status",
    "detail",
)

#: Publisher precedence for de-duplication (PRD.md 5.3, OQ3). Lower wins.
PUBLISHER_PRECEDENCE: dict[str, int] = {
    "amc": 0,
    "sebi": 1,
    "amfi": 2,
    "aggregator": 3,
}


def load_sources(path: str | Path) -> list[SourceSpec]:
    """Parse sources.yaml into SourceSpec records.

    Rejects duplicate ``source_id`` values, and marks a spec disabled when a
    lower-precedence publisher already covers the same (scheme_key, doc_class).
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise ConfigError(f"sources file not found: {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source_path}: invalid YAML: {exc}") from exc

    defaults = raw.get("defaults") or {}
    default_publisher = defaults.get("publisher", "aggregator")

    specs: list[SourceSpec] = []
    seen_ids: set[str] = set()

    for scheme in raw.get("schemes") or []:
        try:
            scheme_key = scheme["scheme_key"]
            scheme_name = scheme["scheme_name"]
            url = scheme["url"]
        except KeyError as exc:
            raise ConfigError(f"scheme entry missing required key {exc}") from exc

        for entry in scheme.get("specs") or []:
            source_id = entry.get("source_id")
            if not source_id:
                raise ConfigError(f"{scheme_key}: a spec is missing source_id")
            if source_id in seen_ids:
                raise ConfigError(f"duplicate source_id: {source_id}")
            seen_ids.add(source_id)

            publisher = entry.get("publisher", default_publisher)
            specs.append(
                SourceSpec(
                    source_id=source_id,
                    scheme_key=scheme_key,
                    scheme_name=scheme_name,
                    doc_class=entry["doc_class"],
                    url=entry.get("url", url),
                    publisher=publisher,
                    locator=entry.get("locator", "full"),
                    exclude_selectors=tuple(entry.get("exclude_selectors") or ()),
                    exclude_headings=tuple(entry.get("exclude_headings") or ()),
                    effective_date=entry.get(
                        "effective_date", defaults.get("effective_date")
                    ),
                    last_updated=entry.get("last_updated", defaults.get("last_updated")),
                    enabled=bool(entry.get("enabled", True)),
                    note=str(entry.get("note", "")),
                )
            )

    return _apply_precedence(specs)


def _apply_precedence(specs: list[SourceSpec]) -> list[SourceSpec]:
    """Disable a spec whose (scheme, doc_class) is already covered by a
    higher-precedence publisher, keeping the winner enabled (PRD.md 5.3)."""
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for spec in specs:
        if not spec.enabled:
            continue
        key = (spec.scheme_key, spec.doc_class)
        rank = PUBLISHER_PRECEDENCE.get(spec.publisher, 99)
        current = best.get(key)
        if current is None or rank < current[0]:
            best[key] = (rank, spec.source_id)

    resolved: list[SourceSpec] = []
    for spec in specs:
        winner = best.get((spec.scheme_key, spec.doc_class))
        if spec.enabled and winner is not None and winner[1] != spec.source_id:
            resolved.append(
                SourceSpec(
                    **{
                        **{f: getattr(spec, f) for f in spec.__slots__},
                        "enabled": False,
                        "note": (
                            f"{spec.note} superseded by {winner[1]} "
                            f"(publisher precedence)".strip()
                        ),
                    }
                )
            )
        else:
            resolved.append(spec)
    return resolved


def load_unavailable(path: str | Path) -> list[dict[str, str]]:
    """Return the recorded corpus gaps declared under ``unavailable:``."""
    source_path = Path(path)
    if not source_path.is_file():
        return []
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    return [dict(item) for item in (raw.get("unavailable") or [])]


def group_by_url(specs: Iterable[SourceSpec]) -> dict[str, list[SourceSpec]]:
    """Group enabled specs by URL.

    The fetcher downloads each URL once; the parser then runs once per spec with
    that spec's locator, which is how one page yields several doc classes.
    """
    grouped: dict[str, list[SourceSpec]] = {}
    for spec in specs:
        if spec.enabled:
            grouped.setdefault(spec.url, []).append(spec)
    return grouped


def _csv_row(run: FetchRun) -> dict[str, Any]:
    return {column: getattr(run, column) for column in SOURCE_CSV_COLUMNS}


def write_sources_csv(runs: Iterable[FetchRun], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_CSV_COLUMNS))
        writer.writeheader()
        for run in runs:
            writer.writerow(_csv_row(run))
    return out


def write_sources_md(
    runs: Iterable[FetchRun],
    unavailable: Iterable[dict[str, str]],
    path: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_runs = list(runs)
    by_scheme: dict[str, list[FetchRun]] = {}
    for run in all_runs:
        by_scheme.setdefault(run.scheme_key, []).append(run)

    lines: list[str] = [
        "# Source list",
        "",
        "Regenerated by `python -m mf_facts.pipeline.build`. Do not hand-edit.",
        "",
        f"Specs: {len(all_runs)}"
        f" | OK: {sum(1 for r in all_runs if r.status == 'ok')}"
        f" | failed: {sum(1 for r in all_runs if r.status != 'ok')}",
        "",
    ]

    for scheme_key in sorted(by_scheme):
        first = by_scheme[scheme_key][0]
        lines.append(f"## {first.scheme_name}")
        lines.append("")
        lines.append("| doc_class | status | source | last updated |")
        lines.append("| --- | --- | --- | --- |")
        for run in sorted(by_scheme[scheme_key], key=lambda r: r.doc_class):
            lines.append(
                f"| {run.doc_class} | {run.status} | "
                f"[{run.source_id}]({run.url}) | {run.last_updated or 'n/a'} |"
            )
        lines.append("")

    gaps = list(unavailable)
    if gaps:
        lines.append("## Document classes with no reachable source")
        lines.append("")
        lines.append("| doc_class | reason |")
        lines.append("| --- | --- |")
        for gap in gaps:
            lines.append(f"| {gap.get('doc_class', '?')} | {gap.get('reason', '')} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
