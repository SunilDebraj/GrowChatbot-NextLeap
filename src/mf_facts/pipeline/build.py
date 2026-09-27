"""[14] Build CLI: runs stages S1-S4 end to end (architecture.md 6.1).

    python -m mf_facts.pipeline.build --strategy heading_aware --refresh

Stage order is load -> chunk -> embed -> store -> manifest, and each stage is
separately observable through ``--stage`` so the demo can show them individually
(PRD.md G7). Failure propagation is deliberate: a fetch failure marks that spec
and continues, but a scheme with zero documents, an over-cap chunk, or an
embedding-model mismatch fails the build. A partially built corpus must never be
published.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..common.config import AppConfig, load_config
from ..common.errors import ConfigError, MFactsError, NoDocumentsForScheme
from ..common.logging import log_event, setup_logging
from ..common.models import Chunk, FetchRun, NormalizedDoc
from .chunker import whitespace_token_counter
from .deduper import dedupe, live_docs
from .embedder import Embedder
from .fetcher import Fetcher
from .manifest import (
    build_manifest,
    strategy_params,
    summarize,
    write_build_report,
    write_manifest,
)
from .normalizer import normalize
from .parsers import HtmlParser, PdfParser
from .sources import (
    group_by_url,
    load_sources,
    load_unavailable,
    write_sources_csv,
    write_sources_md,
)
from .store import ChromaStore
from .strategies import build_strategy

STAGES = ("load", "chunk", "embed", "store", "manifest")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parser_for(spec, content_type: str, exclude_patterns):
    if "pdf" in content_type.lower() or content_type.lower().endswith(".pdf"):
        return PdfParser(exclude_patterns)
    return HtmlParser(exclude_patterns)


# --------------------------------------------------------------------------
# S1 LOAD
# --------------------------------------------------------------------------


def stage_load(
    config: AppConfig, specs, fetcher: Fetcher, verbose: bool
) -> tuple[list[NormalizedDoc], list[FetchRun]]:
    exclude_patterns = config.safety.performance_exclude_patterns
    grouped = group_by_url(specs)

    docs: list[NormalizedDoc] = []
    runs: list[FetchRun] = []

    for url in sorted(grouped):
        url_specs = grouped[url]
        document = fetcher.fetch(url)
        fetcher.record_in_cache_index(url, document)

        for spec in url_specs:
            if document.error or not document.content:
                runs.append(
                    FetchRun(
                        source_id=spec.source_id,
                        scheme_key=spec.scheme_key,
                        scheme_name=spec.scheme_name,
                        doc_class=spec.doc_class,
                        url=url,
                        publisher=spec.publisher,
                        effective_date=spec.effective_date or "",
                        last_updated=spec.last_updated or "",
                        retrieved_at=document.retrieved_at,
                        content_hash="",
                        status="fetch_failed",
                        detail=document.error or f"HTTP {document.status}",
                    )
                )
                if verbose:
                    print(f"  [s1] FAIL {spec.source_id}: {document.error or document.status}")
                continue

            try:
                parser = _parser_for(spec, document.content_type, exclude_patterns)
                parsed = parser.parse(document.content, spec)
            except MFactsError as exc:
                runs.append(
                    FetchRun(
                        source_id=spec.source_id,
                        scheme_key=spec.scheme_key,
                        scheme_name=spec.scheme_name,
                        doc_class=spec.doc_class,
                        url=url,
                        publisher=spec.publisher,
                        effective_date=spec.effective_date or "",
                        last_updated=spec.last_updated or "",
                        retrieved_at=document.retrieved_at,
                        content_hash="",
                        status="parse_failed",
                        detail=str(exc)[:200],
                    )
                )
                if verbose:
                    print(f"  [s1] PARSE FAIL {spec.source_id}: {exc}")
                continue

            normalized = normalize(parsed, spec, document.retrieved_at)
            docs.append(normalized)
            runs.append(
                FetchRun(
                    source_id=spec.source_id,
                    scheme_key=spec.scheme_key,
                    scheme_name=spec.scheme_name,
                    doc_class=spec.doc_class,
                    url=url,
                    publisher=spec.publisher,
                    effective_date=normalized.effective_date or "",
                    last_updated=normalized.last_updated,
                    retrieved_at=document.retrieved_at,
                    content_hash=normalized.content_hash,
                    status="ok",
                )
            )
            if verbose:
                print(
                    f"  [s1] ok   {spec.source_id:<18} {spec.doc_class:<12} "
                    f"{len(normalized.text):>6} chars  last_updated={normalized.last_updated}"
                    f" ({normalized.last_updated_source})"
                )

    ok_schemes = {run.scheme_key for run in runs if run.status == "ok"}
    declared = {spec.scheme_key for spec in specs if spec.enabled}
    missing = declared - ok_schemes
    if missing:
        raise NoDocumentsForScheme(
            f"no successfully parsed document for scheme(s) {sorted(missing)}. "
            f"The corpus would silently omit a scheme, so the build stops. "
            f"Check config/sources.yaml locators and network reachability."
        )

    deduped = dedupe(docs)
    for doc, original in zip(deduped, docs):
        if doc.duplicate_of:
            if verbose:
                print(f"  [s1] dedup {doc.source_id} -> {doc.duplicate_of}")
    return deduped, runs


# --------------------------------------------------------------------------
# S2 CHUNK
# --------------------------------------------------------------------------


def stage_chunk(
    config: AppConfig,
    docs: list[NormalizedDoc],
    strategy_name: str | None,
    count_tokens,
    verbose: bool,
):
    live = live_docs(docs)
    strategy = build_strategy(config.chunking, count_tokens=count_tokens, name=strategy_name)

    chunks: list[Chunk] = []
    for doc in live:
        try:
            produced = strategy.split(doc)
        except MFactsError:
            raise
        if verbose:
            print(f"  [s2] {doc.source_id:<18} {len(produced):>4} chunks")
        chunks.extend(produced)

    if not chunks:
        raise ConfigError(
            f"strategy {strategy.name!r} produced zero chunks. Atomic-fact extraction "
            f"is restricted to the overview and fees doc classes; if it was selected "
            f"for a corpus of other classes it will legitimately yield nothing."
        )

    per_scheme: dict[str, int] = {}
    per_class: dict[str, int] = {}
    for chunk in chunks:
        per_scheme[chunk.scheme_key] = per_scheme.get(chunk.scheme_key, 0) + 1
        per_class[chunk.doc_class] = per_class.get(chunk.doc_class, 0) + 1

    return chunks, per_scheme, per_class, strategy


# --------------------------------------------------------------------------
# S3 EMBED + S4 STORE
# --------------------------------------------------------------------------


def stage_embed_store(
    config: AppConfig, chunks: list[Chunk], strategy_name: str, verbose: bool
):
    embedder = Embedder(config.embedding, cache_dir=config.root / config.corpus.embedding_cache_dir)

    if verbose:
        print(f"  [s3] embedding {len(chunks)} chunks with {config.embedding.model_id}")
    vectors = embedder.embed_chunks(chunks)
    if verbose:
        print(f"  [s3] {len(vectors)} vectors of dim {len(vectors[0]) if vectors else 0}")

    store = ChromaStore(
        path=config.root / "chroma",
        collection_name=config.corpus.collection,
        embedding_model=config.embedding.model_id,
        embedding_dim=config.embedding.embedding_dim,
    )
    written = store.upsert(chunks, vectors)
    removed = store.reconcile([chunk.chunk_id for chunk in chunks])
    if verbose:
        print(f"  [s4] upserted {written} chunks into {config.corpus.collection}")
        if removed:
            print(f"  [s4] removed {removed} stale chunks not in this corpus")

    return store, written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mf_facts.pipeline.build",
        description="Build the MF-Facts-Bot corpus: load, chunk, embed, store.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sources", default="config/sources.yaml")
    parser.add_argument(
        "--strategy",
        default=None,
        help="override chunking.strategy for this run (P2 comparison)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        action="append",
        help="run only these stages; repeatable. Default: all.",
    )
    parser.add_argument("--refresh", action="store_true", help="bypass the raw cache")
    parser.add_argument("--limit", type=int, default=None, help="cap sources processed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    setup_logging("WARNING" if args.quiet else "INFO")
    verbose = not args.quiet
    started = time.perf_counter()

    try:
        config = load_config(args.config)
        specs = load_sources(args.sources)
        if args.limit:
            specs = specs[: args.limit]
    except MFactsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    artifacts = config.root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    selected_stages = set(args.stage) if args.stage else set(STAGES)

    if verbose:
        print(f"[build] strategy={args.strategy or config.chunking.strategy}")
        print(f"[build] sources={len(specs)} stages={sorted(selected_stages)}")

    runs: list[FetchRun] = []
    chunks: list[Chunk] = []
    per_scheme: dict[str, int] = {}
    per_class: dict[str, int] = {}
    store = None
    deduped: list[NormalizedDoc] = []

    # The chunker must count tokens with the model's own tokenizer, otherwise the
    # 256-token cap is enforced in the wrong unit (rule GR2).
    count_tokens = whitespace_token_counter
    if "chunk" in selected_stages:
        try:
            from .chunker import token_counter

            if verbose:
                print("[build] loading the tokenizer for the configured model")
            count_tokens = token_counter(config.embedding.model_id)
        except Exception as exc:  # model download may be unavailable offline
            print(
                f"warning: falling back to a whitespace token estimate ({exc}). "
                f"tokens_max in the manifest is then approximate.",
                file=sys.stderr,
            )

    with Fetcher(
        cache_dir=config.root / config.corpus.raw_cache_dir, refresh=args.refresh
    ) as fetcher:
        if "load" in selected_stages or "chunk" in selected_stages:
            if verbose:
                print("[build] S1 loading")
            deduped, runs = stage_load(config, specs, fetcher, verbose)
            write_sources_csv(runs, artifacts / "sources.csv")
            write_sources_md(
                runs, load_unavailable(args.sources), artifacts / "sources.md"
            )
            if verbose:
                print(
                    f"[build] S1 done: {sum(1 for r in runs if r.status == 'ok')} ok, "
                    f"{sum(1 for r in runs if r.status != 'ok')} failed"
                )

        if "chunk" in selected_stages:
            if verbose:
                print("[build] S2 chunking")
            chunks, per_scheme, per_class, strategy = stage_chunk(
                config, deduped, args.strategy, count_tokens, verbose
            )
            if verbose:
                print(f"[build] S2 done: {len(chunks)} chunks, {strategy.name}")

        if "embed" in selected_stages or "store" in selected_stages:
            if not chunks:
                print("no chunks to embed; run the chunk stage first", file=sys.stderr)
                return 3
            if verbose:
                print("[build] S3 embedding")
            store, _ = stage_embed_store(
                config, chunks, args.strategy or config.chunking.strategy, verbose
            )

    tokens = [chunk.token_count for chunk in chunks]
    report_kwargs = {
        "strategy": args.strategy or config.chunking.strategy,
        "docs_fetched": sum(1 for run in runs if run.status == "ok"),
        "docs_failed": sum(1 for run in runs if run.status != "ok"),
        "docs_deduped": sum(1 for doc in deduped if doc.duplicate_of),
        "chunks": len(chunks),
        "tokens_total": sum(tokens),
        "tokens_min": min(tokens) if tokens else 0,
        "tokens_mean": (sum(tokens) / len(tokens)) if tokens else 0.0,
        "tokens_max": max(tokens) if tokens else 0,
        "build_duration_s": time.perf_counter() - started,
        "chunks_per_scheme": per_scheme,
        "chunks_per_doc_class": per_class,
    }

    if "manifest" in selected_stages:
        from ..common.models import BuildReport

        report = BuildReport(**report_kwargs)
        manifest = build_manifest(
            corpus_version=_now(),
            embedding_model=config.embedding.model_id,
            embedding_dim=config.embedding.embedding_dim,
            chunking_strategy=report.strategy,
            chunking_params=strategy_params(config.chunking),
            collection_name=config.corpus.collection,
            runs=runs,
            report=report,
        )
        write_manifest(manifest, artifacts / "corpus_manifest.json")
        write_build_report(
            report,
            artifacts / "build_report.json",
            strategy_params=strategy_params(config.chunking),
            coverage={
                "schemes": per_scheme,
                "doc_classes": per_class,
                "stored_chunks": store.count() if store is not None else 0,
            },
        )
        if verbose:
            print("\n[build] manifest written")
            print(summarize(manifest, report))

    log_event(
        __import__("logging").getLogger("mf_facts.build"),
        "build.complete",
        strategy=report_kwargs["strategy"],
        chunk_count=report_kwargs["chunks"],
        status="ok",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
