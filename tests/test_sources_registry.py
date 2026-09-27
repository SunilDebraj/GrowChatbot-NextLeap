"""Source registry (P1-T1): one spec per (scheme, doc_class), shared URLs allowed."""

from __future__ import annotations

import csv

import pytest

from mf_facts.common.errors import ConfigError
from mf_facts.pipeline.sources import (
    SOURCE_CSV_COLUMNS,
    group_by_url,
    load_sources,
    load_unavailable,
    write_sources_csv,
    write_sources_md,
)

pytestmark = pytest.mark.contract
from mf_facts.common.models import FetchRun


@pytest.fixture
def specs(repo_root):
    return load_sources(repo_root / "config" / "sources.yaml")


def test_five_schemes_are_registered(specs):
    assert len({spec.scheme_key for spec in specs}) == 5


def test_spec_ids_are_unique(specs):
    ids = [spec.source_id for spec in specs]
    assert len(ids) == len(set(ids))


def test_no_duplicate_scheme_doc_class_pair(specs):
    """One spec per (scheme_key, doc_class): the rule in implementation.md 5.0."""
    pairs = [(spec.scheme_key, spec.doc_class) for spec in specs]
    assert len(pairs) == len(set(pairs))


def test_one_url_serves_several_doc_classes(specs):
    """The groww scheme pages are the structural case GR12 and section 5.0 call out."""
    grouped = group_by_url(specs)
    multi = {url: items for url, items in grouped.items() if len(items) > 1}
    assert multi, "expected at least one URL to yield several doc classes"
    for items in multi.values():
        assert len({item.scheme_key for item in items}) == 1


def test_every_scheme_has_an_overview_and_fees(specs):
    for scheme_key in {spec.scheme_key for spec in specs}:
        classes = {spec.doc_class for spec in specs if spec.scheme_key == scheme_key}
        assert "overview" in classes
        assert "fees" in classes


def test_publisher_precedence_disables_lower_priority_copies(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
schemes:
  - scheme_key: s1
    scheme_name: Scheme One
    url: https://example.invalid/a
    specs:
      - {source_id: amc_one, doc_class: overview, publisher: amc, locator: "full"}
      - {source_id: agg_one, doc_class: overview, publisher: aggregator, locator: "full"}
""",
        encoding="utf-8",
    )
    loaded = load_sources(path)
    by_id = {spec.source_id: spec for spec in loaded}
    assert by_id["amc_one"].enabled is True
    assert by_id["agg_one"].enabled is False
    assert "superseded by amc_one" in by_id["agg_one"].note


def test_duplicate_source_id_is_rejected(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
schemes:
  - scheme_key: s1
    scheme_name: Scheme One
    url: https://example.invalid/a
    specs:
      - {source_id: dup, doc_class: overview, locator: "full"}
      - {source_id: dup, doc_class: fees, locator: "full"}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate source_id"):
        load_sources(path)


def test_unknown_doc_class_is_rejected(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
schemes:
  - scheme_key: s1
    scheme_name: Scheme One
    url: https://example.invalid/a
    specs:
      - {source_id: x, doc_class: horoscope, locator: "full"}
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="doc_class"):
        load_sources(path)


def test_unavailable_classes_are_recorded(repo_root):
    gaps = load_unavailable(repo_root / "config" / "sources.yaml")
    classes = {gap["doc_class"] for gap in gaps}
    assert {"factsheet", "kim", "sid", "statement_guide"} <= classes
    assert all(gap.get("reason") for gap in gaps)


def test_sources_csv_has_the_contract_columns(tmp_path):
    run = FetchRun(
        source_id="lc_overview",
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class="overview",
        url="https://example.invalid/lc",
        publisher="aggregator",
        effective_date="",
        last_updated="2026-09-25",
        retrieved_at="2026-09-27T00:00:00+00:00",
        content_hash="abc123",
        status="ok",
    )
    out = write_sources_csv([run], tmp_path / "sources.csv")
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(SOURCE_CSV_COLUMNS)
    assert rows[0]["source_id"] == "lc_overview"


def test_sources_md_reports_gaps(tmp_path):
    run = FetchRun(
        source_id="lc_overview",
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class="overview",
        url="https://example.invalid/lc",
        publisher="aggregator",
        effective_date="",
        last_updated="2026-09-25",
        retrieved_at="2026-09-27T00:00:00+00:00",
        content_hash="abc",
        status="ok",
    )
    out = write_sources_md(
        [run], [{"doc_class": "factsheet", "reason": "403"}], tmp_path / "sources.md"
    )
    body = out.read_text(encoding="utf-8")
    assert "HDFC Large Cap Fund" in body
    assert "factsheet" in body
    assert "403" in body
