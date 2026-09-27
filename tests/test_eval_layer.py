"""The eval layer itself: matching, the P2 harness, the P4 answer eval.

The eval sets decide whether P2 and P4 claims are true, so an error in the
measuring instrument is more dangerous than an error in the thing measured: a
matcher that is too lenient reports a corpus as better than it is, and a report
that silently skips a failing item hides the failure it exists to surface.

These tests therefore check the instrument, not the system:

* every eval item's expected fact is actually present in the corpus
* the matcher agrees with a hand-built case in both directions
* the harness and the answer eval refuse to report a partial run as a pass
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mf_facts.common.config import load_config
from mf_facts.eval import matching
from mf_facts.eval.answer_eval import evaluate_answers, format_row
from mf_facts.eval.harness import load_eval_set

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.contract

EVAL_SETS = ("factual", "table_facts", "howto", "pii", "refusal", "performance", "out_of_scope")


def corpus_text() -> str:
    """Every indexed chunk, concatenated."""
    from mf_facts.pipeline.store import ChromaStore

    config = load_config(REPO_ROOT / "config" / "config.yaml")
    store = ChromaStore(
        path=REPO_ROOT / "chroma",
        collection_name=config.corpus.collection,
        embedding_model=config.embedding.model_id,
        embedding_dim=config.embedding.embedding_dim,
    )
    stored = store.get_all()
    return "\n".join(stored.get("documents") or [])


@pytest.fixture(scope="module")
def corpus() -> str:
    return corpus_text()


# --------------------------------------------------------------------------
# The matcher
# --------------------------------------------------------------------------


def test_numeric_match_tolerates_formatting_only():
    assert matching.fact_in_text("1.03%", "Expense ratio: 1.03 %", True) is True
    assert matching.fact_in_text("0.58%", "Expense ratio: 1.03%", True) is False


def test_a_non_numeric_fact_is_matched_literally():
    assert matching.fact_in_text("Very High Risk", "Riskometer: Very High Risk", False) is True
    assert matching.fact_in_text("Very Low Risk", "Riskometer: Very High Risk", False) is False


def test_extract_sentence_returns_the_supporting_sentence():
    text = "Expense ratio: 1.03%. Exit load of 1% if redeemed within 1 year."
    sentence = matching.extract_sentence("1%", text, True)

    assert sentence is not None
    assert "1%" in sentence


def test_extract_sentence_returns_none_when_absent():
    assert matching.extract_sentence("0.58%", "Expense ratio: 1.03%.", True) is None


# --------------------------------------------------------------------------
# The eval sets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", EVAL_SETS)
def test_every_eval_set_loads_and_is_non_empty(name):
    items = load_eval_set(name, REPO_ROOT)

    assert items, f"{name}.jsonl is empty"
    for item in items:
        assert item.get("question") or item.get("query"), item


@pytest.mark.parametrize("name", ("factual", "table_facts", "howto"))
def test_answerable_items_declare_what_they_expect(name):
    for item in load_eval_set(name, REPO_ROOT):
        assert item.get("scheme_key"), item
        assert item.get("expected_fact"), item
        assert item.get("expected_url", "").startswith("https://"), item
        assert isinstance(item.get("numeric"), bool), item


@pytest.mark.parametrize("name", ("factual", "table_facts", "howto"))
def test_every_expected_fact_is_really_in_the_corpus(name, corpus):
    """The whole eval rests on this. An item whose fact is not in the corpus
    measures nothing except how the system handles a question it cannot answer."""
    missing = []
    for item in load_eval_set(name, REPO_ROOT):
        if not matching.fact_in_text(
            item["expected_fact"], corpus, bool(item.get("numeric"))
        ):
            missing.append(item["id"])
    assert not missing, f"{name}: expected facts absent from the corpus: {missing}"


def test_howto_items_declare_the_class_they_are_measuring():
    for item in load_eval_set("howto", REPO_ROOT):
        assert item.get("expected_class") == "how_to", item["id"]


def test_item_ids_are_unique():
    for name in ("factual", "table_facts", "howto"):
        ids = [item["id"] for item in load_eval_set(name, REPO_ROOT)]
        assert len(ids) == len(set(ids)), f"{name} has duplicate ids"


def test_the_howto_set_covers_every_scheme():
    items = load_eval_set("howto", REPO_ROOT)
    schemes = {item["scheme_key"] for item in load_eval_set("factual", REPO_ROOT)}

    assert {item["scheme_key"] for item in items} == schemes


# --------------------------------------------------------------------------
# The P4 answer eval
# --------------------------------------------------------------------------


def test_answer_eval_reports_the_output_contract_on_answered_items():
    """Exactly one link, at most three sentences, a stamp, no unsupported numbers.

    This is the part of the P4 acceptance that is verifiable without a provider,
    so it is asserted on the real pipeline rather than on a hand-built context.
    """
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"),
        "table_facts",
        root=REPO_ROOT,
        limit=10,
    )

    answered = [o for o in metrics.outcomes if o.answered]
    assert answered, "no item was answered, so this proves nothing"
    assert metrics.one_valid_link == 1.0
    assert metrics.within_sentence_limit == 1.0
    assert metrics.has_last_updated == 1.0
    assert metrics.unsupported_numbers == 0
    for outcome in answered:
        # contract_failure, not failure: a missing fact is a quality miss, and
        # conflating the two would make a weak fake look like a broken validator.
        assert outcome.contract_failure == "", (
            f"{outcome.item_id}: {outcome.contract_failure}"
        )


def test_answer_eval_without_fake_resolves_the_provider_from_config(monkeypatch):
    """The non-``--fake`` run must reach the configured provider.

    ``AnswerPipeline.build`` reads an explicit ``llm=None`` as "no LLM", so an
    eval that forwarded its own ``None`` default scored every item as a 0 ms
    fail-closed refusal against a live provider. ``build_llm`` is patched to a
    fake so this asserts the wiring without a network call (GR14).
    """
    from mf_facts.online import ask
    from mf_facts.online.generator import FakeLLM

    calls = []

    def fake_build_llm(config, allow_fake=False):
        calls.append(allow_fake)
        return FakeLLM(max_sentences=int(config.generation.max_sentences))

    monkeypatch.setattr(ask, "build_llm", fake_build_llm)
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"),
        "table_facts",
        root=REPO_ROOT,
        llm=None,
        allow_fake=False,
        limit=3,
    )

    assert calls == [False], "the provider was not resolved from config"
    assert any(o.answered for o in metrics.outcomes)


def test_answer_eval_separates_retrieval_misses_from_generation_misses():
    """`grounded` and `fact_in_top_n` must be able to disagree.

    If they always agreed, one of them would be measuring nothing, and the report
    could not tell a retrieval bug from a generation bug.
    """
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"),
        "table_facts",
        root=REPO_ROOT,
    )

    assert metrics.fact_in_top_n_accuracy > metrics.grounded_accuracy, (
        "with the extractive fake the top-1 chunk is often not the one holding the "
        "fact; if this ever inverts, the two metrics have stopped being distinct"
    )


def test_answer_eval_row_is_json_serializable():
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"), "howto", root=REPO_ROOT, limit=3
    )

    row = metrics.as_row()
    assert json.loads(json.dumps(row)) == row
    # The row is what a human reads; both numbers must be in it, because reading
    # only `grounded` is exactly the mistake this metric exists to prevent.
    assert "grounded" in format_row(metrics)
    assert "fact_in_top4" in format_row(metrics)


def test_answer_eval_latency_is_measured():
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"), "factual", root=REPO_ROOT, limit=5
    )

    assert metrics.mean_latency_ms > 0
    assert metrics.p95_latency_ms >= 0


def test_a_refusal_is_not_scored_against_the_answer_contract():
    """A refusal carries one educational link and a short body on purpose.

    Scoring it against "exactly one link, at most three sentences" would report
    every correctly-refused item as a contract failure and make the metric
    unusable as an acceptance gate.
    """
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"), "factual", root=REPO_ROOT
    )
    refused = [o for o in metrics.outcomes if not o.answered]

    assert refused, "no refusals, so this proves nothing"
    for outcome in refused:
        assert outcome.contract_failure == "", f"{outcome.item_id}: {outcome.contract_failure}"


def test_contract_metrics_use_answered_items_as_the_denominator():
    metrics = evaluate_answers(
        load_config(REPO_ROOT / "config" / "config.yaml"), "factual", root=REPO_ROOT
    )

    assert metrics.n_answered == sum(o.answered for o in metrics.outcomes)
    assert metrics.n_answered < metrics.n_items, "expected some refusals in this set"
    # A clean contract gives 1.0 over answered items even though some items were
    # refused, which is only true if the denominator excludes the refusals.
    assert metrics.one_valid_link == 1.0
    assert metrics.within_sentence_limit == 1.0
    assert metrics.has_last_updated == 1.0
