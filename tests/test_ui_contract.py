"""P5-T8: the page contract (PRD FR7, line by line).

``implementation.md`` P5-T8 asks for four assertions - the disclaimer, the
facts-only note, exactly 3 examples, the no-PII reminder. The P5 definition of
done also requires the FR7 checklist to be satisfied line by line, so the rest of
FR7 is asserted here too rather than left to a manual read.

Two of these tests are guards on the guards:

* the disclaimer asset is compared against PRD 10.5 itself, so "rendered on every
  page load" cannot quietly become "rendered, but edited"
* the client script is checked for ``innerHTML``, because the answer is
  model-adjacent text and the only way this page could execute something is if it
  were treated as markup
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Coroutine

import httpx
import pytest

from api.app import FactsApp
from mf_facts.common.config import load_config
from mf_facts.online.ask import FOOTER

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.contract

FACTS_ONLY = "Facts-only. No investment advice."


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT / "config" / "config.yaml")


@pytest.fixture(scope="module")
def page() -> str:
    """The served HTML, fetched once: the page is a pure function of config."""
    app = FactsApp(config=load_config(REPO_ROOT / "config" / "config.yaml"), root=REPO_ROOT)

    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/")

    response = _run(go())
    assert response.status_code == 200
    return response.text


@pytest.fixture(scope="module")
def script() -> str:
    return (REPO_ROOT / "ui" / "app.js").read_text(encoding="utf-8")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_js_comments(source: str) -> str:
    """Remove /* */ and // comments so a scan reads code, not prose.

    The client's own comments name the things it avoids ("never innerHTML"), so a
    naive substring scan over the file would flag the explanation as the offence.
    """
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", " ", without_block)


def page_without_footer(page: str) -> str:
    """The page minus the disclaimer footer.

    The disclaimer legitimately says the assistant does not "recommend, compare,
    or rate any fund", so a forbidden-word scan over the whole page would flag the
    very sentence that promises the restriction.
    """
    return re.sub(r"<footer.*?</footer>", " ", page, flags=re.S)


# --------------------------------------------------------------------------
# PRD 10.5 disclaimer (P5-T6)
# --------------------------------------------------------------------------


def test_the_disclaimer_asset_is_prd_10_5_verbatim():
    """The strongest form of 'verbatim': read the PRD and compare, so an edit to
    either side fails rather than passing by eye."""
    prd = (REPO_ROOT / "PRD.md").read_text(encoding="utf-8")
    section = re.search(r"### 10\.5[^\n]*\n(.*?)\n---", prd, re.S)
    assert section, "PRD 10.5 disclaimer section not found"

    # Strip markdown presentation only - the blockquote marker and bold markers.
    expected = re.sub(r"^>\s*", "", section.group(1).strip()).replace("**", "")
    asset = (REPO_ROOT / "assets" / "disclaimer.txt").read_text(encoding="utf-8")

    assert norm(asset) == norm(expected)


def test_the_page_contains_the_disclaimer(page):
    asset = (REPO_ROOT / "assets" / "disclaimer.txt").read_text(encoding="utf-8")
    assert norm(asset) in norm(page)


def test_the_disclaimer_is_in_the_footer_not_behind_a_link(page):
    """FR7: 'in the footer or an About panel'. A link to it would fail the
    requirement to show it."""
    footer = re.search(r"<footer[^>]*>(.*?)</footer>", page, re.S)
    assert footer, "no <footer> in the page"
    assert "SEBI-registered" in footer.group(1)


def test_the_disclaimer_is_rendered_on_every_page_load(config):
    """P5-T6. Two separate loads, because a client-side fetch would show up on
    the second load only if the script ran - and the test must not depend on a
    browser."""
    app = FactsApp(config=config, root=REPO_ROOT)
    asset = norm((REPO_ROOT / "assets" / "disclaimer.txt").read_text(encoding="utf-8"))

    async def load() -> str:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return norm((await client.get("/")).text)

    assert asset in _run(load())
    assert asset in _run(load())


def test_the_disclaimer_is_not_fetched_by_the_client(page, script):
    """It has to be in the bytes the server sent, or a blocked script would take
    it away with it."""
    code = strip_js_comments(script).lower()
    assert "disclaimer" not in code
    assert "innerhtml" not in code


# --------------------------------------------------------------------------
# the facts-only note and welcome line (FR7)
# --------------------------------------------------------------------------


def test_the_facts_only_note_is_present(page):
    assert FACTS_ONLY in page


def test_the_facts_only_note_is_the_same_string_the_pipeline_appends():
    """One source of truth: the UI note and the refusal footer must not drift
    into two different sentences."""
    assert FOOTER == FACTS_ONLY


def test_the_welcome_line_names_the_assistant_and_its_scope(page):
    assert "HDFC Mutual Fund Facts Assistant" in page
    assert "5 HDFC schemes" in page
    assert "official public sources" in page


# --------------------------------------------------------------------------
# exactly 3 clickable examples (FR7, P5-T2)
# --------------------------------------------------------------------------


def test_there_are_exactly_three_examples(page):
    assert page.count('class="example"') == 3


def test_the_examples_are_the_config_examples(page, config):
    for question in config.ui.examples:
        assert question in page, f"missing example: {question}"


def test_each_example_is_a_button_that_fills_the_input(page, script):
    """'Clickable to auto-fill the input' - not a link, and not an immediate send."""
    assert 'type="button"' in page
    assert "data-question=" in page
    assert ".example" in script
    assert "input.value" in script


def test_an_example_does_not_send_immediately(page, script):
    """The demo is narrated before the answer appears."""
    click = script[script.index('".example"'):]
    assert "ask(" not in click.split("});")[0]


# --------------------------------------------------------------------------
# the no-PII reminder (FR7)
# --------------------------------------------------------------------------


def test_the_input_carries_the_no_pii_reminder(page):
    assert "PAN" in page and "Aadhaar" in page
    assert "placeholder=" in page


def test_the_no_pii_reminder_lists_every_forbidden_field(page):
    for field in ("PAN", "Aadhaar", "account number", "OTP", "email", "phone"):
        assert field in page, f"reminder omits {field}"


def test_the_reminder_matches_the_disclaimer_safety_list(page):
    """PRD 10.5 and the input reminder must not name different sets of fields;
    a user who reads one and follows the other should not be misled."""
    disclaimer = norm((REPO_ROOT / "assets" / "disclaimer.txt").read_text(encoding="utf-8"))
    for field in ("PAN", "Aadhaar", "account numbers", "OTPs", "email addresses", "phone numbers"):
        assert field in disclaimer


# --------------------------------------------------------------------------
# the message thread (FR7, P5-T7)
# --------------------------------------------------------------------------


def test_the_thread_region_exists(page):
    assert 'id="thread"' in page
    assert "aria-live" in page, "an answer appearing should be announced"


def test_the_client_renders_the_citation_as_a_real_link(script):
    """P5-T7: a real link, from the structured field, not parsed out of the text."""
    assert "payload.citation" in script
    assert "a.href = url" in script or ".href = " in script
    assert "rel =" in script and "noopener" in script


def test_the_client_renders_the_last_updated_stamp(script):
    assert "Last updated from sources:" in script


def test_the_client_renders_the_refusal_footer_on_refusals(script):
    """P5-T7: the refusal footer on every refusal."""
    assert "refusal-footer" in script
    assert FACTS_ONLY in script


def test_the_client_marks_a_refusal_distinctly_from_an_answer(script):
    assert "refusal" in script


def test_the_client_never_assigns_innerhtml(script):
    """The answer is model-adjacent text. Treating it as markup is the only way
    this page could execute something, so it is asserted rather than reviewed."""
    code = strip_js_comments(script)
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in code, f"{forbidden} in the client script"


def test_the_client_puts_text_in_with_textcontent(script):
    assert "textContent" in strip_js_comments(script)


# --------------------------------------------------------------------------
# FR7: what v1 must not have
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        'type="password"',
        "login",
        "sign in",
        "signin",
        "signup",
        "sign up",
        "dashboard",
        "settings",
        "account settings",
        "scheme comparison",
    ],
)
def test_v1_has_no_accounts_settings_or_dashboards(page, forbidden):
    """FR7: 'No accounts, no settings, no dashboards, no scheme comparison
    table in v1.' Checked because a prototype grows by accident.

    Scanned outside the footer, whose disclaimer legitimately contains the words
    'recommend, compare, or rate'.
    """
    assert forbidden not in page_without_footer(page).lower()


def test_there_is_no_comparison_table(page):
    assert "<table" not in page.lower()


def test_there_is_no_conversation_history(page, script):
    """Known Limit #6, and a P5 pitfall: no history, no scheme carry-over."""
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "history" not in script.lower()


def test_the_page_loads_no_third_party_resource(page):
    """Every src/href is local. A remote asset would be a privacy leak on a page
    whose whole promise is that nothing leaves except the question."""
    for url in re.findall(r'(?:src|href)="([^"]+)"', page):
        assert url.startswith("/") or url.startswith("#"), url


def test_the_page_loads_its_script_as_a_file_not_inline(script, page):
    """Pairs with the CSP the app sends: script-src 'self' with no 'unsafe-inline'
    means an inline script would not run at all."""
    assert "<script>" not in page
    assert '<script src="/static/app.js">' in page
