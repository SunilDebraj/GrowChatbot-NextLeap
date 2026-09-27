"""[9] Manifest writer (architecture.md 5.9).

The manifest is the reproducibility contract for the demo: a reviewer must be able
to tell which corpus, which embedding model and which chunking strategy produced a
given answer. The build report is also the evidence the P2 chunking gate needs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..common.models import BuildReport, CorpusManifest, FetchRun, SourceSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_corpus_hash(runs: Iterable[FetchRun], chunks: int) -> str:
    """A stable fingerprint of the corpus, used to detect a real change."""
    parts = sorted(f"{run.source_id}:{run.content_hash}" for run in runs)
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
    digest.update(str(chunks).encode("utf-8"))
    return digest.hexdigest()


def build_manifest(
    *,
    corpus_version: str,
    embedding_model: str,
    embedding_dim: int,
    chunking_strategy: str,
    chunking_params: dict[str, Any],
    collection_name: str,
    runs: list[FetchRun],
    report: BuildReport,
) -> CorpusManifest:
    return CorpusManifest(
        corpus_version=corpus_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunking_strategy=chunking_strategy,
        chunking_params=chunking_params,
        collection_name=collection_name,
        doc_count=sum(1 for run in runs if run.status == "ok"),
        chunk_count=report.chunks,
        chunks_per_scheme=dict(sorted(report.chunks_per_scheme.items())),
        chunks_per_doc_class=dict(sorted(report.chunks_per_doc_class.items())),
        sources=[_source_entry(run) for run in runs],
        built_at=_now(),
        corpus_hash=compute_corpus_hash(runs, report.chunks),
        tokens_max=report.tokens_max,
        build_duration_s=round(report.build_duration_s, 2),
    )


def _source_entry(run: FetchRun) -> dict[str, Any]:
    return {
        "source_id": run.source_id,
        "scheme_key": run.scheme_key,
        "doc_class": run.doc_class,
        "url": run.url,
        "publisher": run.publisher,
        "effective_date": run.effective_date,
        "last_updated": run.last_updated,
        "retrieved_at": run.retrieved_at,
        "content_hash": run.content_hash,
        "status": run.status,
    }


def write_manifest(manifest: CorpusManifest, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def write_build_report(
    report: BuildReport,
    path: str | Path,
    *,
    strategy_params: dict[str, Any],
    coverage: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": report.strategy,
        "strategy_params": strategy_params,
        "docs_fetched": report.docs_fetched,
        "docs_failed": report.docs_failed,
        "docs_deduped": report.docs_deduped,
        "chunks": report.chunks,
        "tokens_total": report.tokens_total,
        "tokens_min": report.tokens_min,
        "tokens_mean": round(report.tokens_mean, 1),
        "tokens_max": report.tokens_max,
        "build_duration_s": round(report.build_duration_s, 2),
        "chunks_per_scheme": report.chunks_per_scheme,
        "chunks_per_doc_class": report.chunks_per_doc_class,
        "per_strategy_tokens": report.per_strategy_tokens,
        "coverage": coverage or {},
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def summarize(manifest: CorpusManifest, report: BuildReport) -> str:
    lines = [
        f"corpus_version   {manifest.corpus_version}",
        f"collection       {manifest.collection_name}",
        f"embedding        {manifest.embedding_model} ({manifest.embedding_dim}d)",
        f"chunking         {manifest.chunking_strategy}",
        f"docs ok/failed   {report.docs_fetched}/{report.docs_failed}",
        f"chunks           {report.chunks}",
        f"tokens           min={report.tokens_min} mean={report.tokens_mean:.1f} max={report.tokens_max}",
        f"chunks/scheme    {report.chunks_per_scheme}",
        f"chunks/class     {report.chunks_per_doc_class}",
        f"build_time       {report.build_duration_s:.1f}s",
    ]
    return "\n".join(lines)


def strategy_params(spec: Any) -> dict[str, Any]:
    return {
        "max_chunk_tokens": spec.max_chunk_tokens,
        "target_tokens": spec.target_tokens,
        "overlap_ratio": spec.overlap_ratio,
        "protect_tables": spec.protect_tables,
        "repeat_table_headers": spec.repeat_table_headers,
    }


def enabled_summary(specs: list[SourceSpec]) -> str:
    enabled = [spec for spec in specs if spec.enabled]
    disabled = [spec for spec in specs if not spec.enabled]
    parts = [f"{len(enabled)} enabled"]
    if disabled:
        parts.append(f"{len(disabled)} disabled")
    schemes = sorted({spec.scheme_key for spec in enabled})
    return f"{', '.join(parts)} across {len(schemes)} schemes"
