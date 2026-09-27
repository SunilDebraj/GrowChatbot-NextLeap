"""Store and manifest contracts (P1-T12, P1-T13)."""

from __future__ import annotations

import json

import pytest

from mf_facts.common.errors import CollectionUnavailable, EmbeddingModelMismatch
from mf_facts.common.models import BuildReport, FetchRun
from mf_facts.pipeline.manifest import build_manifest, write_manifest
from mf_facts.pipeline.store import ChromaStore

pytestmark = pytest.mark.contract



pytest.importorskip("chromadb", reason="chromadb is required for the store contract")


def _chunk(**overrides):
    from mf_facts.common.models import Chunk

    base = dict(
        text="Expense ratio: 1.03%",
        token_count=6,
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class="fees",
        publisher="aggregator",
        source_url="https://example.invalid/lc",
        section="Expense ratio",
        effective_date=None,
        last_updated="2026-09-25",
        content_hash="0123456789abcdef0123",
        chunk_index=0,
    )
    base.update(overrides)
    return Chunk(**base)


def _store(tmp_path, name="mf_facts_v1", model="all-MiniLM-L6-v2", dim=384):
    return ChromaStore(
        path=tmp_path / "chroma",
        collection_name=name,
        embedding_model=model,
        embedding_dim=dim,
    )


def _vector(seed: float = 0.1, dim: int = 384):
    return [seed] * dim


def test_upsert_and_count(tmp_path):
    store = _store(tmp_path)
    chunks = [_chunk(), _chunk(chunk_index=1, text="Exit load 1%")]
    written = store.upsert(chunks, [_vector(0.1), _vector(0.2)])
    assert written == 2
    assert store.count() == 2


def test_reingest_is_idempotent(tmp_path):
    """Deterministic ids make a repeated build upsert onto itself (architecture.md 5.8)."""
    store = _store(tmp_path)
    chunks = [_chunk(), _chunk(chunk_index=1, text="Exit load 1%")]
    store.upsert(chunks, [_vector(0.1), _vector(0.2)])
    store.upsert(chunks, [_vector(0.1), _vector(0.2)])
    assert store.count() == 2


def test_none_metadata_is_rejected_at_the_helper_not_at_write_time():
    metadata = _chunk().to_chroma_metadata()
    assert metadata["effective_date"] == ""
    assert all(value is not None for value in metadata.values())


def test_colliding_ids_are_rejected(tmp_path):
    """A collision would silently drop a chunk on upsert."""
    store = _store(tmp_path)
    chunks = [_chunk(chunk_index=0), _chunk(chunk_index=0)]
    with pytest.raises(ValueError, match="idempotent"):
        store.upsert(chunks, [_vector(0.1), _vector(0.2)])


def test_length_mismatch_is_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert([_chunk()], [])


def test_empty_upsert_is_a_noop(tmp_path):
    store = _store(tmp_path)
    assert store.upsert([], []) == 0


def test_model_mismatch_raises(tmp_path):
    """Mixing vector spaces in one collection looks plausible and is wrong."""
    store = _store(tmp_path)
    store.upsert([_chunk()], [_vector()])
    other = _store(tmp_path, model="some-other-model")
    with pytest.raises(EmbeddingModelMismatch, match="was built with"):
        _ = other.collection


def test_metadata_filter_returns_only_the_requested_scheme(tmp_path):
    """D3: the scheme filter is the primary retrieval precision lever."""
    store = _store(tmp_path)
    store.upsert(
        [
            _chunk(),
            _chunk(
                scheme_key="hdfc_elss_tax_saver_direct_growth",
                scheme_name="HDFC ELSS",
                source_url="https://example.invalid/elss",
                content_hash="abcdef0123456789abcd",
            ),
        ],
        [_vector(0.1), _vector(0.2)],
    )
    found = store.where(
        {"scheme_key": {"$in": ["hdfc_elss_tax_saver_direct_growth"]}}
    )
    keys = {meta["scheme_key"] for meta in found["metadatas"]}
    assert keys == {"hdfc_elss_tax_saver_direct_growth"}


# -- manifest ---------------------------------------------------------------


def _run(status="ok"):
    return FetchRun(
        source_id="lc_overview",
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class="overview",
        url="https://example.invalid/lc",
        publisher="aggregator",
        effective_date="",
        last_updated="2026-09-25",
        retrieved_at="2026-09-27T00:00:00+00:00",
        content_hash="deadbeef",
        status=status,
    )


def test_manifest_matches_the_prd_schema(tmp_path):
    report = BuildReport(
        strategy="heading_aware",
        docs_fetched=1,
        chunks=3,
        tokens_total=30,
        tokens_min=8,
        tokens_mean=10.0,
        tokens_max=14,
        chunks_per_scheme={"hdfc_large_cap_direct_growth": 3},
        chunks_per_doc_class={"overview": 3},
    )
    manifest = build_manifest(
        corpus_version="v1",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        chunking_strategy="heading_aware",
        chunking_params={"max_chunk_tokens": 220},
        collection_name="mf_facts_v1",
        runs=[_run()],
        report=report,
    )
    out = write_manifest(manifest, tmp_path / "corpus_manifest.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "corpus_version",
        "embedding_model",
        "embedding_dim",
        "chunking_strategy",
        "chunking_params",
        "collection_name",
        "doc_count",
        "chunk_count",
        "chunks_per_scheme",
        "chunks_per_doc_class",
        "sources",
        "built_at",
        "corpus_hash",
    ):
        assert key in payload, f"manifest is missing {key}"
    assert payload["embedding_dim"] == 384
    assert payload["chunk_count"] == 3


def test_corpus_hash_changes_with_content(tmp_path):
    report_a = BuildReport(strategy="heading_aware", chunks=1)
    report_b = BuildReport(strategy="heading_aware", chunks=2)
    a = build_manifest(
        corpus_version="v1",
        embedding_model="m",
        embedding_dim=384,
        chunking_strategy="heading_aware",
        chunking_params={},
        collection_name="c",
        runs=[_run()],
        report=report_a,
    )
    b = build_manifest(
        corpus_version="v1",
        embedding_model="m",
        embedding_dim=384,
        chunking_strategy="heading_aware",
        chunking_params={},
        collection_name="c",
        runs=[_run()],
        report=report_b,
    )
    assert a.corpus_hash != b.corpus_hash


def test_manifest_records_the_strategy_that_built_it():
    """A stored collection must be attributable to a chunking strategy."""
    report = BuildReport(strategy="atomic_fact", chunks=2)
    manifest = build_manifest(
        corpus_version="v1",
        embedding_model="m",
        embedding_dim=384,
        chunking_strategy="atomic_fact",
        chunking_params={"max_chunk_tokens": 220},
        collection_name="c",
        runs=[_run()],
        report=report,
    )
    assert manifest.chunking_strategy == "atomic_fact"
    assert manifest.chunking_params["max_chunk_tokens"] == 220


def test_reconcile_removes_chunks_absent_from_the_current_corpus(tmp_path):
    """A parser fix changes content-addressed ids; the old ones must not survive.

    Stale chunks stay retrievable and answer with facts no source document
    contains any more, which is the failure mode this guards.
    """
    store = _store(tmp_path)
    keep = _chunk(chunk_index=0, text="Expense ratio: 1.03%")
    stale = _chunk(chunk_index=1, text="Exit load of 1% if redeemed within 1 year.")
    store.upsert([keep, stale], [[0.01] * 384, [0.02] * 384])
    assert store.count() == 2

    removed = store.reconcile([keep.chunk_id])

    assert removed == 1
    assert store.count() == 1
    remaining = store.get_all()
    assert remaining["ids"] == [keep.chunk_id]


def test_reconcile_is_a_no_op_when_the_store_already_matches(tmp_path):
    store = _store(tmp_path)
    chunk = _chunk()
    store.upsert([chunk], [[0.01] * 384])

    assert store.reconcile([chunk.chunk_id]) == 0
    assert store.count() == 1
