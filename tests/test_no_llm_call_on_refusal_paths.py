"""GR8 / D5 / AD6: no model is involved in producing any refusal.

There is no generator in P3, so the "model is not called" claim is easy to make
and easy to break later. These tests state it as a property of the refusal paths
themselves, using a spy that fails loudly rather than a mock that silently
records nothing: if a future P4 wires a generator in and something routes to it
before the refusal branch, this fails at that point rather than in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mf_facts.common.models import REFUSAL_CLASSES
from mf_facts.online.ask import AnswerPipeline
from mf_facts.online.classifier import RESIDUAL_PROMPT
from mf_facts.online.prompts import SYSTEM_PROMPT

pytestmark = [pytest.mark.contract, pytest.mark.adversarial]

REPO_ROOT = Path(__file__).resolve().parent.parent

EVERY_REFUSAL_QUERY = [
    "my PAN ABCDE1234F, what is the expense ratio?",
    "Should I invest in HDFC Large Cap Fund?",
    "Is HDFC Small Cap a good option?",
    "Which is the best fund among these five?",
    "Recommend a fund for my son",
    "How much should I invest in HDFC Large Cap?",
    "Is my portfolio balanced?",
    "What is my risk profile?",
    "What are the returns of HDFC Large Cap Fund?",
    "What is the 3 year return of HDFC Small Cap Fund?",
    "What is the NAV history of HDFC Flexi Cap?",
    "Which of these is the best performing?",
    "What is the weather in Mumbai?",
    "Explain the share market crash of 2008",
]


class ForbiddenLLM:
    """Any call at all is a failure, not a recorded call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError(
            f"an LLM was called on a refusal path with {sorted(kwargs)}"
        )


@pytest.mark.parametrize("query", EVERY_REFUSAL_QUERY, ids=lambda q: q[:38])
def test_no_llm_call_on_refusal_paths(query):
    llm = ForbiddenLLM()
    pipeline = AnswerPipeline.build(root=REPO_ROOT, llm=llm)

    response = pipeline.ask(query)

    assert response.route == "refusal", response.query_class
    assert response.query_class in REFUSAL_CLASSES
    assert llm.calls == []


def test_performance_refusal_cannot_contain_a_return_number():
    """D5: performance is terminal. A percentage or a CAGR-shaped number in the
    response would mean a return figure leaked out of a templated string."""
    pipeline = AnswerPipeline.build(
        root=REPO_ROOT,
        llm=ForbiddenLLM(),
    )
    for query in (
        "What are the returns of HDFC Large Cap Fund?",
        "What is the 3 year return of HDFC Small Cap Fund?",
        "What is the XIRR of HDFC Balanced Advantage Fund?",
        "What are the returns since inception?",
    ):
        response = pipeline.ask(query)
        assert response.query_class == "performance"
        text = response.text
        assert "%" not in text
        for token in ("cagr", "xirr", "irr"):
            assert token not in text.lower().replace("cagr or", "")


def test_in_scope_path_calls_the_classifier_then_the_generator():
    """P3 proved the in-scope branch called the model once, to route only.

    P4 adds the generator, so the count is now two and the second call carries
    passages. What must not change is the shape of each call and, above all, the
    refusal paths above them: this file's subject is that refusing costs nothing,
    and that is unaffected by generation existing.
    """
    calls: list[dict] = []

    class RecordingLLM:
        def generate(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("json_only"):
                return '{"label": "factual_scheme", "confidence": 0.9}'
            return "The expense ratio is 1.03%."

    pipeline = AnswerPipeline.build(root=REPO_ROOT, llm=RecordingLLM())
    response = pipeline.ask("What is the expense ratio of HDFC Large Cap Fund?")

    assert response.route == "answer"

    routing, generation = calls
    # The routing call is unchanged from P3: prompt, not messages, temperature 0.
    assert routing["prompt"] == RESIDUAL_PROMPT
    assert routing["temperature"] == 0.0
    for banned in ("passages", "context", "system", "max_sentences", "citation"):
        assert banned not in routing

    # The generation call is the new one, and it is the only call that sees
    # retrieved text.
    assert generation["system_prompt"] == SYSTEM_PROMPT
    assert "PASSAGE" in generation["user_prompt"]
    assert generation["temperature"] == 0.0
    assert generation["max_tokens"] == 180


def test_refusal_text_is_templated_not_generated():
    """Two different questions of the same class must produce byte-identical
    refusal text apart from the link, which is what "templated" means."""
    pipeline = AnswerPipeline.build(
        root=REPO_ROOT, llm=ForbiddenLLM()
    )
    first = pipeline.ask("Should I invest in HDFC Large Cap Fund?")
    second = pipeline.ask("Is HDFC Small Cap a good option?")

    assert first.query_class == second.query_class == "opinionated"
    strip = lambda r: "\n".join(
        line for line in r.text.splitlines() if not line.startswith("Educational link:")
    )
    assert strip(first) == strip(second)


def test_no_pii_survives_into_a_refusal_of_another_class():
    """A PII-bearing query is a PII refusal, never an opinionated or performance
    refusal that happens to echo part of it."""
    pipeline = AnswerPipeline.build(
        root=REPO_ROOT,
        llm=ForbiddenLLM(),
    )
    response = pipeline.ask("my PAN ABCDE1234F, should I invest in HDFC Large Cap?")

    assert response.query_class == "pii"
    assert "ABCDE1234F" not in response.text
    assert "should I invest" not in response.text.lower()
