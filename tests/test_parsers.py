"""Parser behaviour: locator grammar and the performance exclusion (P1-T3/T4)."""

from __future__ import annotations

import pytest

from mf_facts.common.errors import ParseError
from mf_facts.common.models import SourceSpec
from mf_facts.pipeline.parsers import HtmlParser, normalize_ws

pytestmark = pytest.mark.contract



EXCLUSIONS = (
    "return calculator",
    "returns and rankings",
    "compare similar funds",
    "category average",
    "rank (",
)


def _spec(locator: str, doc_class: str = "overview") -> SourceSpec:
    return SourceSpec(
        source_id="t",
        scheme_key="hdfc_large_cap_direct_growth",
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        doc_class=doc_class,
        url="https://example.invalid/lc",
        publisher="aggregator",
        locator=locator,
    )


@pytest.fixture
def parser() -> HtmlParser:
    return HtmlParser(EXCLUSIONS)


def _parse(parser, html, locator, doc_class="overview"):
    return parser.parse(html.encode("utf-8"), _spec(locator, doc_class))


# -- performance exclusion (D5 / GR6) ---------------------------------------


def test_returns_section_is_dropped(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "full")
    lowered = parsed.text.lower()
    assert "returns and rankings" not in lowered
    assert "return calculator" not in lowered
    assert "category average" not in lowered


def test_return_percentages_do_not_survive(parser, scheme_page_html):
    """A leaked '+8.7%' would be a performance figure one retrieval from the LLM."""
    parsed = _parse(parser, scheme_page_html, "full")
    for leaked in ("+8.7%", "+10.3%", "-2.62"):
        assert leaked not in parsed.text


def test_named_exclusion_heading_and_its_table_are_removed(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "full")
    assert "Rank" not in parsed.text


def test_factual_sections_survive_exclusion(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "full")
    lowered = parsed.text.lower()
    assert "expense ratio" in lowered
    assert "exit load" in lowered
    assert "min. for sip" in lowered


# -- locator grammar ---------------------------------------------------------


def test_labels_locator_extracts_pairs(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "labels:Expense ratio|Min. for SIP|Fund benchmark")
    assert "Expense ratio: 1.03%" in parsed.text
    assert "Min. for SIP: Rs 100" in parsed.text
    assert "Fund benchmark: NIFTY 100 Total Return Index" in parsed.text


def test_heading_locator_captures_its_section(parser, scheme_page_html):
    parsed = _parse(
        parser, scheme_page_html, "heading:Exit load, stamp duty and tax", doc_class="fees"
    )
    assert "0-12 months" in parsed.text
    assert "Stamp duty" in parsed.text


def test_heading_locator_stops_at_the_next_heading(parser, scheme_page_html):
    parsed = _parse(
        parser, scheme_page_html, "heading:Minimum investments", doc_class="faq"
    )
    assert "Min. for SIP" in parsed.text
    assert "0-12 months" not in parsed.text


def test_css_locator(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "css:header")
    assert "HDFC Large Cap Fund Direct Growth" in parsed.text


def test_pill_locator_emits_a_riskometer_fact(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "pill:Very High Risk", doc_class="riskometer")
    assert "Riskometer: Very High Risk" in parsed.text


def test_locator_parts_are_concatenated_in_order(parser, scheme_page_html):
    parsed = _parse(
        parser,
        scheme_page_html,
        "labels:Expense ratio;heading:Minimum investments",
    )
    assert "Expense ratio: 1.03%" in parsed.text
    assert "Min. for SIP" in parsed.text


def test_nav_and_footer_are_stripped(parser, scheme_page_html):
    parsed = _parse(parser, scheme_page_html, "full")
    assert "nav junk" not in parsed.text
    assert "footer junk" not in parsed.text


def test_tables_are_captured(parser, scheme_page_html):
    parsed = _parse(
        parser, scheme_page_html, "heading:Exit load, stamp duty and tax", doc_class="fees"
    )
    assert parsed.tables, "expected the fee table to be captured"
    assert "Exit load" in parsed.tables[0].header


def test_locator_matching_nothing_raises_rather_than_returning_empty(
    parser, scheme_page_html
):
    with pytest.raises(ParseError, match="matched no text"):
        _parse(parser, scheme_page_html, "heading:Nonexistent Section Name")


def test_illegal_locator_kind_raises(parser, scheme_page_html):
    with pytest.raises(ParseError, match="not a legal locator kind"):
        _parse(parser, scheme_page_html, "xpath://div")


def test_normalize_ws_collapses_runs():
    assert normalize_ws("a  \t b\n\n\n\nc  ") == "a b\n\nc"

# -- regression: sibling-only section walking (P1 live-build fixes) ---------


NESTED_FEES_PAGE = """
<html><body>
  <h3>Return calculator</h3>
  <div class="rc"><span>1Y returns</span><b>24.5%</b></div>
  <div class="wrap">
    <h3>Exit load, stamp duty and tax</h3>
    <div class="fees">
      <p>Exit load of 1% if redeemed within 1 year.</p>
      <table>
        <tr><th>Particulars</th><th>Rate</th></tr>
        <tr><td>Stamp duty</td><td>0.005%</td></tr>
      </table>
    </div>
  </div>
  <h3>Fund management</h3>
  <p>Manager bio.</p>
</body></html>
"""


def test_exclusion_does_not_delete_the_next_section(parser):
    """The excluded block must not take the shared container with it.

    find_all_next() yields descendants before siblings, so collecting removals
    from it captured the wrapper of both the return calculator and the fees
    section; decomposing that wrapper silently deleted the fees.
    """
    parsed = _parse(parser, NESTED_FEES_PAGE, "full")
    assert "1Y returns" not in parsed.text
    assert "24.5%" not in parsed.text
    assert "Exit load of 1% if redeemed within 1 year." in parsed.text
    assert "Manager bio." in parsed.text


def test_heading_section_is_not_duplicated(parser):
    """A wrapped section contributes its text once, not once per nesting level."""
    parsed = _parse(parser, NESTED_FEES_PAGE, "heading:Exit load, stamp duty and tax", "fees")
    assert parsed.text.count("Exit load of 1% if redeemed within 1 year.") == 1
    assert "Manager bio." not in parsed.text


def test_table_inside_a_located_section_is_kept_as_markdown(parser):
    parsed = _parse(parser, NESTED_FEES_PAGE, "heading:Exit load, stamp duty and tax", "fees")
    assert "Particulars | Rate" in parsed.text
    assert "Stamp duty | 0.005%" in parsed.text


PILL_PAGE = """
<html><body>
  <header>
    <div class="pills_container__x">
      <a href="#"><div class="pill12Pill absolute-center"><span class="bodySmallHeavy">Very High Risk</span></div></a>
    </div>
  </header>
</body></html>
"""


def test_pill_matches_badge_styling_on_an_ancestor(parser):
    """The value sits on an inner span; the badge class is on its grandparent."""
    parsed = _parse(parser, PILL_PAGE, "pill:Very High Risk", "riskometer")
    assert "Riskometer: Very High Risk" in parsed.text


def test_label_already_named_by_the_section_heading_is_dropped(parser):
    """A bare "tax" element inside "Exit load, stamp duty and tax" adds nothing.

    Left in the text it becomes a one-word chunk that matches no question.
    """
    page = """
    <html><body>
      <h3>Exit load, stamp duty and tax</h3>
      <div>tax</div>
      <div>Exit load of 1% if redeemed within 1 year.</div>
    </body></html>
    """
    parsed = _parse(parser, page, "heading:Exit load, stamp duty and tax", "fees")
    body = parsed.text.split("H3: Exit load, stamp duty and tax", 1)[1]
    assert "tax" not in body
    assert "Exit load of 1% if redeemed within 1 year." in body


def test_word_appearing_in_the_title_is_kept_when_the_unit_carries_more(parser):
    page = """
    <html><body>
      <h3>Exit load, stamp duty and tax</h3>
      <div>tax is charged at 20% under 12.5% slabs above Rs 1.25 lakh</div>
    </body></html>
    """
    parsed = _parse(parser, page, "heading:Exit load, stamp duty and tax", "fees")
    assert "tax is charged at 20%" in parsed.text
