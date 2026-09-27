"""Shared pytest fixtures.

Tests never touch the network (rule GR14). Corpus behaviour is exercised against
local HTML fixtures that mimic the structure of a scheme page, and the embedding
model is replaced by a deterministic fake so the suite runs in seconds.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from mf_facts.common.config import ChunkingConfig, load_config
from mf_facts.common.models import NormalizedDoc

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def app_config():
    return load_config(REPO_ROOT / "config" / "config.yaml")


@pytest.fixture(scope="session")
def alias_map():
    """The query rewriter's alias table. Shared by the P3 routing tests."""
    from mf_facts.online.rewriter import AliasMap

    return AliasMap.load(REPO_ROOT / "config" / "aliases.yaml")


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        strategy="heading_aware",
        max_chunk_tokens=220,
        target_tokens=60,
        overlap_ratio=0.15,
        protect_tables=True,
        repeat_table_headers=True,
    )


@pytest.fixture
def count_tokens():
    """Dependency-free token estimate, so chunker tests need no model download."""
    import re

    pattern = re.compile(r"\S+")

    def _count(text: str) -> int:
        return len(pattern.findall(text))

    return _count


# --------------------------------------------------------------------------
# Fake embedder: deterministic, dimension-correct, no model download.
# --------------------------------------------------------------------------


class FakeEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])
        return vectors

    def embed_chunks(self, chunks):
        return self.embed([chunk.text for chunk in chunks])


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --------------------------------------------------------------------------
# Fake LLM: P3+ only, defined here so the fixture set is complete.
# --------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, responses=None) -> None:
        self.responses = list(responses or ["NO_ANSWER"])
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


# --------------------------------------------------------------------------
# HTML fixture mimicking a scheme page: hashed classes, label/value pairs,
# a header pill cluster, a fees table, and a returns region that must be dropped.
# --------------------------------------------------------------------------

SCHEME_PAGE_HTML = """
<html><body>
<nav><a href="/x">nav junk</a></nav>
<header>
  <h1>HDFC Large Cap Fund Direct Growth</h1>
  <span class="pill">Equity</span>
  <span class="pill">Large Cap</span>
  <span class="pill">Very High Risk</span>
</header>
<div class="flex flex-column hashed__abc">
  <div class="valign-wrapper contentTertiary">Expense ratio</div>
  <div class="bodyXLargeHeavy contentPrimary">1.03%</div>
</div>
<div><div class="lbl">Min. for SIP</div><div class="val">Rs 100</div></div>
<div><div class="lbl">Fund benchmark</div><div class="val">NIFTY 100 Total Return Index</div></div>
<div><div class="lbl">Fund size (AUM)</div><div class="val">Rs 39,933.37 Cr</div></div>

<h3>Minimum investments</h3>
<p>Min. for 1st investment Rs 100.</p>
<p>Min. for 2nd investment Rs 100.</p>
<p>Min. for SIP Rs 100.</p>

<h3>Exit load, stamp duty and tax</h3>
<table>
  <tr><th>Period</th><th>Exit load</th></tr>
  <tr><td>0-12 months</td><td>1.00%</td></tr>
  <tr><td>12-24 months</td><td>0.75%</td></tr>
  <tr><td>Above 24 months</td><td>Nil</td></tr>
</table>
<p>Stamp duty on investment: 0.005% (from July 1st, 2020).</p>
<p>If you redeem within one year, returns are taxed at 20%.</p>

<h3>Returns and rankings</h3>
<p>Fund returns +8.7% +10.3% +12.0%</p>
<p>Category average ( Equity Large Cap ) +10.3%</p>
<table><tr><th>Name</th><th>3Y</th></tr><tr><td>Rank</td><td>45</td></tr></table>

<h3>Return calculator</h3>
<p>1 year -2.62 %, 3 years +2.06 %</p>

<footer>footer junk</footer>
</body></html>
"""


@pytest.fixture
def scheme_page_html() -> str:
    return SCHEME_PAGE_HTML


@pytest.fixture
def normalized_doc(chunking_config) -> NormalizedDoc:
    from mf_facts.common.models import Table

    table = Table(
        rows=(
            ("Period", "Exit load"),
            ("0-12 months", "1.00%"),
            ("12-24 months", "0.75%"),
            ("Above 24 months", "Nil"),
        ),
        caption="Exit load, stamp duty and tax",
    )
    text = (
        "Exit load, stamp duty and tax:\n"
        "Exit load of 1% if redeemed within 1 year.\n"
        "Stamp duty on investment: 0.005% (from July 1st, 2020).\n"
        f"{table.to_markdown()}\n"
        "Minimum investments:\n"
        "Min. for SIP Rs 100.\n"
    )
    return NormalizedDoc(
        source_id="lc_fees",
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class="fees",
        publisher="aggregator",
        source_url="https://example.invalid/large-cap",
        text=text,
        headings=((3, "Exit load, stamp duty and tax"), (3, "Minimum investments")),
        tables=(table,),
        effective_date=None,
        last_updated="2026-09-25",
        last_updated_source="document",
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
