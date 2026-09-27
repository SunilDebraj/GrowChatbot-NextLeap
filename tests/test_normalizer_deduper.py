"""Normalizer and deduper (P1-T5, P1-T6)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mf_facts.pipeline.deduper import dedupe, jaccard, live_docs
from mf_facts.pipeline.normalizer import (
    clean_text,
    content_hash,
    extract_last_updated,
    parse_date,
)

pytestmark = pytest.mark.contract

# -- dates ------------------------------------------------------------------


def test_parse_date_forms():
    assert parse_date("as on 25 Sep 2026") == "2026-09-25"
    assert parse_date("Last updated: 1 Jan 2025") == "2025-01-01"
    assert parse_date("as of September 3, 2024") == "2024-09-03"
    assert parse_date("25/09/2026") == "2026-09-25"
    assert parse_date("no date here") is None


def test_extract_last_updated_requires_a_context_word():
    """A bare date is not a freshness claim; the context word is required."""
    assert extract_last_updated("Last updated: 25 Sep 2026") == "2026-09-25"
    assert extract_last_updated("NAV: 25 Sep 2026") is None


# -- text hygiene -----------------------------------------------------------


def test_rupee_and_unicode_variants_are_normalized():
    assert "\u20b9" not in clean_text("NAV \u20b91,189.08")
    assert "Rs 1,189.08" in clean_text("NAV \u20b91,189.08")


def test_numbers_are_never_altered():
    """A normalizing bug on a numeric value is a correctness bug (5.4)."""
    text = "Expense ratio 1.03% and Rs 39,933.37 Cr"
    cleaned = clean_text(text)
    assert "1.03%" in cleaned
    assert "39,933.37" in cleaned


def test_whitespace_is_collapsed():
    assert clean_text("a  \t\t b\n\n\n\nc   ") == "a b\n\nc"


# -- last_updated precedence (architecture.md 5.4) --------------------------


def test_document_date_beats_spec_and_retrieval(normalized_doc):
    from mf_facts.common.models import ParsedDoc
    from mf_facts.common.models import SourceSpec
    from mf_facts.pipeline.normalizer import normalize

    spec = SourceSpec(
        source_id="x",
        scheme_key=normalized_doc.scheme_key,
        scheme_name=normalized_doc.scheme_name,
        doc_class="fees",
        url="https://example.invalid",
        publisher="aggregator",
        locator="full",
        last_updated="2020-01-01",
    )
    parsed = ParsedDoc(
        source_id="x", text="Fees as on 12 Jun 2026 apply.", headings=(), tables=()
    )
    result = normalize(parsed, spec, "2026-09-27T10:00:00+00:00")
    assert result.last_updated == "2026-06-12"
    assert result.last_updated_source == "document"


def test_spec_date_used_when_document_has_none(normalized_doc):
    from mf_facts.common.models import ParsedDoc
    from mf_facts.common.models import SourceSpec
    from mf_facts.pipeline.normalizer import normalize

    spec = SourceSpec(
        source_id="x",
        scheme_key=normalized_doc.scheme_key,
        scheme_name=normalized_doc.scheme_name,
        doc_class="fees",
        url="https://example.invalid",
        publisher="aggregator",
        locator="full",
        last_updated="2020-01-01",
    )
    parsed = ParsedDoc(source_id="x", text="No context date here.", headings=(), tables=())
    result = normalize(parsed, spec, "2026-09-27T10:00:00+00:00")
    assert result.last_updated == "2020-01-01"
    assert result.last_updated_source == "spec"


def test_retrieval_time_is_the_last_resort_and_is_recorded(normalized_doc):
    from mf_facts.common.models import ParsedDoc
    from mf_facts.common.models import SourceSpec
    from mf_facts.pipeline.normalizer import normalize

    spec = SourceSpec(
        source_id="x",
        scheme_key=normalized_doc.scheme_key,
        scheme_name=normalized_doc.scheme_name,
        doc_class="fees",
        url="https://example.invalid",
        publisher="aggregator",
        locator="full",
    )
    parsed = ParsedDoc(source_id="x", text="Nothing dated.", headings=(), tables=())
    result = normalize(parsed, spec, "2026-09-27T10:00:00+00:00")
    assert result.last_updated == "2026-09-27"
    assert result.last_updated_source == "retrieved_at"


# -- deduper ----------------------------------------------------------------


def test_jaccard_bounds():
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), {"a"}) == 0.0


def test_identical_documents_are_deduped(normalized_doc):
    duplicate = replace(normalized_doc, source_id="lc_fees_copy")
    result = dedupe([normalized_doc, duplicate])
    assert len(result) == 2
    assert result[0].duplicate_of is None
    assert result[1].duplicate_of == "lc_fees"


def test_near_duplicate_is_shadowed(normalized_doc):
    near = replace(
        normalized_doc,
        source_id="lc_fees_near",
        text=normalized_doc.text + " ",
    )
    near = replace(near, content_hash=content_hash(near.text))
    result = dedupe([normalized_doc, near])
    assert result[1].duplicate_of == "lc_fees"


def test_different_schemes_are_never_deduped(normalized_doc):
    other = replace(
        normalized_doc,
        source_id="el_fees",
        scheme_key="hdfc_elss_tax_saver_direct_growth",
    )
    result = dedupe([normalized_doc, other])
    assert all(doc.duplicate_of is None for doc in result)


def test_live_docs_excludes_shadowed(normalized_doc):
    duplicate = replace(normalized_doc, source_id="copy")
    result = dedupe([normalized_doc, duplicate])
    assert [doc.source_id for doc in live_docs(result)] == ["lc_fees"]


def test_empty_documents_are_dropped(normalized_doc):
    empty = replace(normalized_doc, source_id="empty", text="")
    result = dedupe([empty])
    assert result == []
