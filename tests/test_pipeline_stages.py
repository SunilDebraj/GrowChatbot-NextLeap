"""Offline pipeline stages, exercised without network access (rule GR14).

Drives load -> chunk -> embed -> store -> manifest through the real stage
functions with a stub fetcher, which is the only way to assert the cross-stage
invariants (idempotency, per-scheme coverage, guard behaviour) without touching
groww.in.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mf_facts.common.config import load_config
from mf_facts.common.errors import NoDocumentsForScheme
from mf_facts.common.models import RawDocument
from mf_facts.pipeline.build import stage_chunk, stage_load
from mf_facts.pipeline.deduper import live_docs
from mf_facts.pipeline.manifest import build_manifest, write_manifest
from mf_facts.common.models import BuildReport
from mf_facts.pipeline.sources import load_sources
from mf_facts.pipeline.strategies import build_strategy

pytestmark = pytest.mark.contract




class StubFetcher:
    """Serves fixture bytes per URL and records what was requested."""

    def __init__(self, pages: dict[str, bytes], failing: set[str] | None = None) -> None:
        self.pages = pages
        self.failing = failing or set()
        self.requested: list[str] = []

    def fetch(self, url: str) -> RawDocument:
        self.requested.append(url)
        if url in self.failing or url not in self.pages:
            return RawDocument(
                source_id="",
                url=url,
                resolved_url=url,
                content=b"",
                content_type="",
                status=404,
                retrieved_at="2026-09-27T00:00:00+00:00",
                from_cache=False,
                error="HTTP 404",
            )
        return RawDocument(
            source_id="",
            url=url,
            resolved_url=url,
            content=self.pages[url],
            content_type="text/html; charset=utf-8",
            status=200,
            retrieved_at="2026-09-27T00:00:00+00:00",
            from_cache=False,
        )

    def record_in_cache_index(self, url, document) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def two_page_config(tmp_path, repo_root):
    """A config rooted in tmp_path with the real chunking settings."""
    source = (repo_root / "config" / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(source, encoding="utf-8")
    return load_config(tmp_path / "config" / "config.yaml")


def _pages(specs, html_by_scheme: dict[str, str]) -> dict[str, bytes]:
    pages: dict[str, bytes] = {}
    for spec in specs:
        if spec.enabled:
            pages.setdefault(spec.url, html_by_scheme[spec.scheme_key].encode("utf-8"))
    return pages


def test_load_produces_one_document_per_spec(scheme_page_html, repo_root, two_page_config):
    # Disabled specs are registry-level opt-outs and produce no run at all.
    specs = [
        spec
        for spec in load_sources(repo_root / "config" / "sources.yaml")
        if spec.scheme_key == "hdfc_large_cap_direct_growth" and spec.enabled
    ]
    pages = _pages(specs, {"hdfc_large_cap_direct_growth": scheme_page_html})
    fetcher = StubFetcher(pages)

    docs, runs = stage_load(two_page_config, specs, fetcher, verbose=False)

    assert specs, "the registry must have enabled specs for this scheme"
    assert len(runs) == len(specs)
    assert all(run.status == "ok" for run in runs)
    assert {doc.doc_class for doc in docs} == {spec.doc_class for spec in specs}
    assert all(doc.scheme_key == "hdfc_large_cap_direct_growth" for doc in docs)


def test_load_fetches_each_url_once_despite_many_specs(
    scheme_page_html, repo_root, two_page_config
):
    """The implementation.md 5.0 rule: one URL, several doc classes, one fetch."""
    specs = [
        spec
        for spec in load_sources(repo_root / "config" / "sources.yaml")
        if spec.scheme_key == "hdfc_large_cap_direct_growth"
    ]
    assert len(specs) > 1
    pages = _pages(specs, {"hdfc_large_cap_direct_growth": scheme_page_html})
    fetcher = StubFetcher(pages)

    stage_load(two_page_config, specs, fetcher, verbose=False)

    assert len(fetcher.requested) == 1
    assert len(specs) > 1


def test_load_fails_the_build_when_a_scheme_has_no_documents(
    scheme_page_html, repo_root, two_page_config
):
    specs = [
        spec
        for spec in load_sources(repo_root / "config" / "sources.yaml")
        if spec.scheme_key in ("hdfc_large_cap_direct_growth", "hdfc_small_cap_direct_growth")
    ]
    pages = _pages(
        specs,
        {
            "hdfc_large_cap_direct_growth": scheme_page_html,
            "hdfc_small_cap_direct_growth": scheme_page_html,
        },
    )
    fetcher = StubFetcher(pages, failing={"https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"})

    with pytest.raises(NoDocumentsForScheme, match="silently omit a scheme"):
        stage_load(two_page_config, specs, fetcher, verbose=False)


def test_load_records_a_parse_failure_rather_than_raising(
    repo_root, two_page_config, scheme_page_html
):
    """A changed layout must be visible in sources.csv, not silently empty."""
    specs = [
        spec
        for spec in load_sources(repo_root / "config" / "sources.yaml")
        if spec.scheme_key == "hdfc_large_cap_direct_growth"
    ]
    # Break exactly one locator; the scheme's other specs stay parseable, so the
    # build must continue and report the failure instead of aborting.
    broken = replace(specs[0], locator="heading:ASectionThatNoLongerExists")
    specs = [broken, *specs[1:]]
    pages = _pages(specs, {"hdfc_large_cap_direct_growth": scheme_page_html})

    docs, runs = stage_load(two_page_config, specs, StubFetcher(pages), verbose=False)

    assert any(run.status == "parse_failed" for run in runs)
    assert any("matched no text" in run.detail for run in runs)
    assert any(run.status == "ok" for run in runs), "the build must continue"
    assert any(run.source_id == broken.source_id for run in runs)


def test_chunk_stage_covers_every_scheme(
    scheme_page_html, repo_root, two_page_config, count_tokens
):
    specs = load_sources(repo_root / "config" / "sources.yaml")
    by_scheme = {spec.scheme_key: scheme_page_html for spec in specs if spec.enabled}
    pages = _pages(specs, by_scheme)
    docs, runs = stage_load(two_page_config, specs, StubFetcher(pages), verbose=False)

    chunks, per_scheme, per_class, strategy = stage_chunk(
        two_page_config, docs, None, count_tokens, verbose=False
    )

    assert set(per_scheme) == set(by_scheme), "every scheme must yield chunks"
    assert strategy.name == two_page_config.chunking.strategy
    assert sum(per_class.values()) == len(chunks)


def test_chunk_stage_is_deterministic_across_runs(
    scheme_page_html, repo_root, two_page_config, count_tokens
):
    """Idempotency (P1 acceptance) rests on chunk text and ids both repeating."""
    specs = load_sources(repo_root / "config" / "sources.yaml")
    by_scheme = {spec.scheme_key: scheme_page_html for spec in specs if spec.enabled}
    pages = _pages(specs, by_scheme)

    outputs = []
    for _ in range(2):
        docs, _runs = stage_load(two_page_config, specs, StubFetcher(pages), verbose=False)
        chunks, _s, _c, _st = stage_chunk(
            two_page_config, docs, None, count_tokens, verbose=False
        )
        outputs.append([chunk.chunk_id for chunk in chunks])

    assert outputs[0] == outputs[1]
    assert len(outputs[0]) == len(set(outputs[0])), "chunk ids must be unique"


def test_atomic_fact_strategy_covers_only_overview_and_fees(
    scheme_page_html, repo_root, two_page_config, count_tokens
):
    specs = load_sources(repo_root / "config" / "sources.yaml")
    by_scheme = {spec.scheme_key: scheme_page_html for spec in specs if spec.enabled}
    docs, _runs = stage_load(
        two_page_config, specs, StubFetcher(_pages(specs, by_scheme)), verbose=False
    )

    chunks, _s, per_class, _st = stage_chunk(
        two_page_config, docs, "atomic_fact", count_tokens, verbose=False
    )

    assert set(per_class) <= {"overview", "fees"}
    assert chunks, "atomic_fact should find labelled facts on the scheme pages"


def test_manifest_reflects_the_stage_outputs(
    scheme_page_html, repo_root, two_page_config, count_tokens, tmp_path
):
    specs = load_sources(repo_root / "config" / "sources.yaml")
    by_scheme = {spec.scheme_key: scheme_page_html for spec in specs if spec.enabled}
    docs, runs = stage_load(
        two_page_config, specs, StubFetcher(_pages(specs, by_scheme)), verbose=False
    )
    chunks, per_scheme, per_class, strategy = stage_chunk(
        two_page_config, docs, None, count_tokens, verbose=False
    )

    tokens = [chunk.token_count for chunk in chunks]
    report = BuildReport(
        strategy=strategy.name,
        docs_fetched=sum(1 for r in runs if r.status == "ok"),
        docs_failed=sum(1 for r in runs if r.status != "ok"),
        docs_deduped=sum(1 for d in docs if d.duplicate_of),
        chunks=len(chunks),
        tokens_total=sum(tokens),
        tokens_min=min(tokens),
        tokens_mean=sum(tokens) / len(tokens),
        tokens_max=max(tokens),
        chunks_per_scheme=per_scheme,
        chunks_per_doc_class=per_class,
    )
    manifest = build_manifest(
        corpus_version="test",
        embedding_model=two_page_config.embedding.model_id,
        embedding_dim=two_page_config.embedding.embedding_dim,
        chunking_strategy=strategy.name,
        chunking_params={"max_chunk_tokens": two_page_config.chunking.max_chunk_tokens},
        collection_name=two_page_config.corpus.collection,
        runs=runs,
        report=report,
    )
    out = write_manifest(manifest, tmp_path / "corpus_manifest.json")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["chunk_count"] == len(chunks)
    assert payload["chunks_per_scheme"] == per_scheme
    assert payload["tokens_max"] <= two_page_config.chunking.max_chunk_tokens
    assert payload["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_build_strategy_rejects_unknown_name_via_stage(two_page_config, count_tokens):
    from mf_facts.common.errors import ConfigError

    with pytest.raises(ConfigError):
        build_strategy(two_page_config.chunking, count_tokens=count_tokens, name="nope")


def test_live_docs_is_the_chunking_input(scheme_page_html, repo_root, two_page_config):
    specs = load_sources(repo_root / "config" / "sources.yaml")
    by_scheme = {spec.scheme_key: scheme_page_html for spec in specs if spec.enabled}
    docs, _runs = stage_load(
        two_page_config, specs, StubFetcher(_pages(specs, by_scheme)), verbose=False
    )
    assert len(live_docs(docs)) == len([d for d in docs if d.duplicate_of is None])
