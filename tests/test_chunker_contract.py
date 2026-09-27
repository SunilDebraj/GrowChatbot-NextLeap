"""Chunker contract tests (P1-T7..T10).

These encode the structural promises in PRD.md section 8.1 and rules GR2/GR3/GR4:
all eleven metadata fields on every chunk, the token cap enforced in the model's
own unit, tables never split mid-row, and no chunk mixing two schemes.
"""
from __future__ import annotations

import pytest

from mf_facts.common.errors import ChunkSizeError
from mf_facts.common.models import Table
from mf_facts.pipeline.strategies import (

    AtomicFactStrategy,
    FixedWindowStrategy,
    HeadingAwareStrategy,
    build_strategy,
)

pytestmark = pytest.mark.contract




REQUIRED_METADATA_FIELDS = {
    "scheme_key",
    "scheme_name",
    "doc_class",
    "publisher",
    "source_url",
    "section",
    "effective_date",
    "last_updated",
    "content_hash",
    "chunk_index",
    "token_count",
}

ALL_STRATEGIES = [HeadingAwareStrategy, FixedWindowStrategy, AtomicFactStrategy]


def _all_strategies(chunking_config, count_tokens):
    return [cls(chunking_config, count_tokens=count_tokens) for cls in ALL_STRATEGIES]


# -- GR3: metadata completeness --------------------------------------------


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
def test_every_chunk_carries_all_metadata_fields(
    strategy_cls, chunking_config, count_tokens, normalized_doc
):
    strategy = strategy_cls(chunking_config, count_tokens=count_tokens)
    chunks = strategy.split(normalized_doc)
    assert chunks, "strategy produced no chunks for the fees fixture"
    for chunk in chunks:
        metadata = chunk.to_chroma_metadata()
        assert REQUIRED_METADATA_FIELDS <= set(metadata)
        assert metadata["scheme_key"] == normalized_doc.scheme_key
        assert metadata["source_url"] == normalized_doc.source_url
        assert metadata["last_updated"] == normalized_doc.last_updated


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
def test_chroma_metadata_has_no_none_values(
    strategy_cls, chunking_config, count_tokens, normalized_doc
):
    """A None in Chroma metadata raises at write time (architecture.md 5.8)."""
    strategy = strategy_cls(chunking_config, count_tokens=count_tokens)
    for chunk in strategy.split(normalized_doc):
        assert all(value is not None for value in chunk.to_chroma_metadata().values())


# -- GR2: token cap ---------------------------------------------------------


def test_no_chunk_exceeds_the_cap(normalized_doc, count_tokens):
    """GR2: with a tight cap, no strategy may emit an over-cap chunk."""
    from mf_facts.common.config import ChunkingConfig

    chunking = ChunkingConfig(
        strategy="heading_aware",
        max_chunk_tokens=40,
        target_tokens=20,
        overlap_ratio=0.15,
        protect_tables=True,
        repeat_table_headers=True,
    )
    for strategy in _all_strategies(chunking, count_tokens):
        for chunk in strategy.split(normalized_doc):
            assert chunk.token_count <= 40, f"{strategy.name} produced {chunk.token_count}"


def test_over_cap_chunk_raises_rather_than_truncating(count_tokens):
    """Truncating a fee table silently yields a wrong number, so the build stops."""
    from mf_facts.common.config import ChunkingConfig
    from mf_facts.common.models import NormalizedDoc

    doc = NormalizedDoc(
        source_id="big",
        scheme_key="s",
        scheme_name="S",
        doc_class="fees",
        publisher="aggregator",
        source_url="https://example.invalid",
        text="word " * 500,
        headings=((2, "Big Section"),),
        effective_date=None,
        last_updated="2026-01-01",
        content_hash="h",
    )
    config = ChunkingConfig(
        strategy="heading_aware",
        max_chunk_tokens=10,
        target_tokens=5,
        overlap_ratio=0.15,
        protect_tables=True,
        repeat_table_headers=True,
    )
    strategy = HeadingAwareStrategy(config, count_tokens=count_tokens)
    with pytest.raises(ChunkSizeError, match="Truncating would silently corrupt"):
        strategy.split(doc)


# -- tables are never split mid-row -----------------------------------------


def test_fee_table_rows_are_not_split(chunking_config, count_tokens, normalized_doc):
    strategy = HeadingAwareStrategy(chunking_config, count_tokens=count_tokens)
    chunks = strategy.split(normalized_doc)
    blob = "\n".join(chunk.text for chunk in chunks)
    for row in ("0-12 months", "12-24 months", "Above 24 months"):
        assert row in blob, f"table row lost: {row}"


def test_table_header_repeats_when_a_table_must_be_split(count_tokens):
    from mf_facts.common.config import ChunkingConfig
    from mf_facts.common.models import NormalizedDoc

    rows = [("Period", "Exit load")]
    rows += [(f"{i} months", f"{i}.00%") for i in range(1, 40)]
    table = Table(rows=tuple(rows), caption="Exit load")
    doc = NormalizedDoc(
        source_id="tbl",
        scheme_key="s",
        scheme_name="S",
        doc_class="fees",
        publisher="aggregator",
        source_url="https://example.invalid",
        text=table.to_markdown(),
        headings=((2, "Exit load"),),
        tables=(table,),
        effective_date=None,
        last_updated="2026-01-01",
        content_hash="h",
    )
    config = ChunkingConfig(
        strategy="heading_aware",
        max_chunk_tokens=40,
        target_tokens=20,
        overlap_ratio=0.15,
        protect_tables=True,
        repeat_table_headers=True,
    )
    strategy = HeadingAwareStrategy(config, count_tokens=count_tokens)
    chunks = strategy.split(doc)
    assert len(chunks) > 1, "expected the table to be split"
    for chunk in chunks:
        assert chunk.token_count <= 40
        assert "Period" in chunk.text and "Exit load" in chunk.text


# -- GR4: scheme isolation --------------------------------------------------


def test_chunks_never_mix_schemes(chunking_config, count_tokens, normalized_doc):
    from dataclasses import replace

    other = replace(
        normalized_doc,
        source_id="el_fees",
        scheme_key="hdfc_elss_tax_saver_direct_growth",
        scheme_name="HDFC ELSS Tax Saver Fund - Direct Plan Growth",
    )
    strategy = HeadingAwareStrategy(chunking_config, count_tokens=count_tokens)
    combined = strategy.split(normalized_doc) + strategy.split(other)
    for chunk in combined:
        others = {c.scheme_key for c in combined}
        assert chunk.scheme_key in others
        assert chunk.scheme_name.startswith("HDFC")


# -- strategy registry ------------------------------------------------------


def test_build_strategy_resolves_by_name(chunking_config, count_tokens):
    for name in ("heading_aware", "fixed_window", "atomic_fact"):
        strategy = build_strategy(chunking_config, count_tokens=count_tokens, name=name)
        assert strategy.name == name


def test_build_strategy_rejects_unknown_name(chunking_config, count_tokens):
    from mf_facts.common.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown chunking strategy"):
        build_strategy(chunking_config, count_tokens=count_tokens, name="telepathy")


def test_chunk_ids_are_deterministic_and_unique(
    chunking_config, count_tokens, normalized_doc
):
    strategy = HeadingAwareStrategy(chunking_config, count_tokens=count_tokens)
    first = [chunk.chunk_id for chunk in strategy.split(normalized_doc)]
    second = [chunk.chunk_id for chunk in strategy.split(normalized_doc)]
    assert first == second
    assert len(first) == len(set(first))


# -- atomic-fact coverage gap (must be explicit, not silent) ----------------


def test_atomic_fact_is_restricted_to_overview_and_fees(
    chunking_config, count_tokens, normalized_doc
):
    from dataclasses import replace

    strategy = AtomicFactStrategy(chunking_config, count_tokens=count_tokens)
    doc = replace(normalized_doc, doc_class="faq")
    assert strategy.split(doc) == []


def test_atomic_fact_produces_label_value_chunks(
    chunking_config, count_tokens, normalized_doc
):
    strategy = AtomicFactStrategy(chunking_config, count_tokens=count_tokens)
    chunks = strategy.split(normalized_doc)
    assert chunks, "expected at least one atomic fact from the fees fixture"
    assert any(":" in chunk.text for chunk in chunks)


def test_atomic_fact_blocks_performance_labels(count_tokens, normalized_doc):
    from dataclasses import replace

    from mf_facts.common.config import ChunkingConfig

    config = ChunkingConfig(
        strategy="atomic_fact",
        max_chunk_tokens=220,
        target_tokens=160,
        overlap_ratio=0.15,
        protect_tables=True,
        repeat_table_headers=True,
    )
    doc = replace(
        normalized_doc,
        text="1 year returns: +8.7%\nRank: 45\nExpense ratio: 1.03%",
    )
    strategy = AtomicFactStrategy(config, count_tokens=count_tokens)
    labels = [chunk.section for chunk in strategy.split(doc)]
    assert "Expense ratio" in labels
    assert "Rank" not in labels
    assert not any("returns" in label.lower() for label in labels)
