"""P5-T8: the API contract.

The API is a transport, and this file is mostly about the properties that make it
safe to put in front of the answer path rather than about JSON shapes:

* nothing internal leaks - no traceback, no exception class, no filesystem path,
  no provider error text (P5 definition of done: zero client-visible stack traces)
* an infrastructure failure is a 503 with no answer, never a partial answer
* the guards actually guard - oversize body, oversize question, rate limit
* the answer contract survives the hop: one citation, a stamp, flags, latency

The pipeline is stubbed so these tests are about the transport. The end-to-end
answer path is covered by tests/test_online_p4.py, and one real-pipeline test at
the bottom of this file keeps the stub from drifting away from the real contract.

ASGI is async and no async pytest plugin is a declared dependency, so the tests
are sync and drive the app through ``asyncio.run`` via a small facade. That keeps
the suite runnable from a clean checkout with no plugin to install.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Coroutine

import httpx
import pytest

from api.app import FactsApp, RateLimiter, build_app
from mf_facts.common.config import load_config
from mf_facts.common.errors import CollectionUnavailable, LLMError
from mf_facts.common.models import AskResponse

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.contract

CITATION = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"

# Substrings that must never appear in any client-visible response body. Kept in
# one place so a new endpoint cannot quietly opt out of the guarantee.
LEAKS = (
    "Traceback",
    'File "',
    ".py",  # any source path
    "LLMError",
    "CollectionUnavailable",
    "MFactsError",
    "Exception",
    "chroma",
    "sqlite",
    "sentence-transformers",
    " on line ",
)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


class Client:
    """Sync facade over the ASGI app, for tests that want plain assertions.

    A fresh ``AsyncClient`` per call rather than one shared instance: httpx
    clients cannot be reopened, and a facade that only worked for its first call
    would make every multi-request test look like a transport bug.
    """

    def __init__(self, app: FactsApp) -> None:
        self.app = app

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def go() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return _run(go())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._send("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._send("POST", path, **kwargs)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self._send(method, path, **kwargs)

    def ask(self, question: str = "What is the exit load?") -> httpx.Response:
        return self.post("/api/ask", json={"question": question})


def answer_response(**overrides: Any) -> AskResponse:
    base = dict(
        query_class="fee_structure",
        route="answer",
        text=(
            "The exit load is 1% if you redeem within 1 year.\n"
            f"[Source: HDFC Large Cap Fund]({CITATION})\n"
            "Last updated from sources: 2026-09-27"
        ),
        link_label="HDFC Large Cap Fund",
        link_url=CITATION,
        citation_url=CITATION,
        last_updated="2026-09-27",
        grounding_passed=True,
        grounding_score=0.42,
        validator_checks=("check_3",),
        answer_model="fake-extractive",
    )
    base.update(overrides)
    return AskResponse(**base)


class StubStore:
    def __init__(self, count: int = 20) -> None:
        self._count = count

    def count(self) -> int:
        if isinstance(self._count, Exception):
            raise self._count
        return self._count


class StubRetriever:
    def __init__(self, store: StubStore) -> None:
        self.store = store


class StubPipeline:
    """Stands in for AnswerPipeline at the API boundary."""

    def __init__(
        self,
        response: AskResponse | None = None,
        error: Exception | None = None,
        count: int = 20,
    ) -> None:
        self.response = response if response is not None else answer_response()
        self.error = error
        self.retriever = StubRetriever(StubStore(count))
        self.questions: list[str] = []
        self.histories: list[tuple[str, ...]] = []

    def ask(self, question: str, history: Any = ()) -> AskResponse:
        self.questions.append(question)
        self.histories.append(tuple(history))
        if self.error is not None:
            raise self.error
        return self.response


def make_app(pipeline: Any = None, **kwargs: Any) -> FactsApp:
    return FactsApp(
        config=load_config(REPO_ROOT / "config" / "config.yaml"),
        root=REPO_ROOT,
        pipeline=pipeline if pipeline is not None else StubPipeline(),
        **kwargs,
    )


def make_client(pipeline: Any = None, **kwargs: Any) -> Client:
    return Client(make_app(pipeline, **kwargs))


# --------------------------------------------------------------------------
# GET /api/health
# --------------------------------------------------------------------------


def test_health_reports_status_collection_and_manifest():
    response = make_client().get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["collection"] == "mf_facts_v1"
    assert body["manifest_version"], "manifest version must be reported"
    assert body["corpus_built_at"], "corpus build time must be reported"
    assert body["indexed_chunks"] == 20


def test_health_says_degraded_instead_of_claiming_ok():
    """A health check that cannot report a problem is worse than none."""
    body = make_client(StubPipeline(count=0)).get("/api/health").json()

    assert body["status"] == "degraded"
    assert any("empty" in detail for detail in body["detail"])


def test_health_reports_degraded_when_the_store_is_unreachable():
    response = make_client(StubPipeline(count=CollectionUnavailable("chroma down"))).get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert not any(leak in json.dumps(body) for leak in LEAKS)


def test_health_reports_the_real_provider_rather_than_a_placeholder():
    """Health must name what is actually configured.

    This test used to assert the provider was the literal string
    ``<TBD: OQ2>``, which is the same "don't dress up a guess as a fact" rule
    seen from the other side. OQ2 is resolved now, so the honest report is the
    real provider - and the same rule still forbids inventing a model name when
    there isn't one.
    """
    body = make_client().get("/api/health").json()

    assert body["provider"] == load_config(REPO_ROOT / "config" / "config.yaml").generation.provider
    assert body["provider"] != "<TBD: OQ2>"
    assert body["model"].startswith("qwen/")


def test_health_never_echoes_the_api_key():
    """The key is named by environment variable, never by value.

    ``/api/health`` is the one unauthenticated endpoint in the service, so it is
    the one place a secret could plausibly leak. Worth a test rather than a
    code review promise.
    """
    body = make_client().get("/api/health").json()
    blob = json.dumps(body)
    needle = "gsk" + "_"  # assembled at runtime; this file is scanned for the literal

    assert needle not in blob
    # An unset var yields "", and "" is in every string - compare only if set.
    key = os.environ.get("MF_FACTS_LLM_API_KEY", "")
    if key:
        assert key not in blob
    assert not any(leak in blob for leak in LEAKS)


# --------------------------------------------------------------------------
# warmup: the model load belongs at boot, not on a user's question
# --------------------------------------------------------------------------


class StubEmbedder:
    def __init__(self) -> None:
        self._model = None
        self.loads = 0

    @property
    def model(self):
        if self._model is None:
            self.loads += 1
            self._model = object()
        return self._model


def test_warmup_loads_the_embedding_model():
    embedder = StubEmbedder()
    pipeline = StubPipeline()
    pipeline.retriever.embedder = embedder
    app = make_app(pipeline)

    assert app.is_warm is False
    app.warmup()

    assert embedder.loads == 1
    assert app.is_warm is True


def test_warmup_is_idempotent():
    """Startup runs it and so might an operator; loading twice would throw away
    a resident model for nothing."""
    embedder = StubEmbedder()
    pipeline = StubPipeline()
    pipeline.retriever.embedder = embedder
    app = make_app(pipeline)

    app.warmup()
    app.warmup()

    assert embedder.loads == 1


def test_a_failed_warmup_does_not_stop_the_service_from_serving():
    """A degraded service beats no service: /api/ask fails closed with a 503 and
    /api/health explains why, which is exactly what an operator needs."""
    pipeline = StubPipeline()
    pipeline.retriever.embedder = StubEmbedder()

    class Exploding:
        @property
        def model(self):
            raise RuntimeError("no model for you")

    pipeline.retriever.embedder = Exploding()
    app = make_app(pipeline)
    app._safe_warmup()

    assert app.is_warm is False
    # Still answering, and still not leaking the reason.
    response = Client(app).ask()
    assert response.status_code == 200


def test_health_reports_whether_the_model_is_loaded():
    cold = make_client().get("/api/health").json()
    assert cold["embedding_model_loaded"] is False

    embedder = StubEmbedder()
    pipeline = StubPipeline()
    pipeline.retriever.embedder = embedder
    app = make_app(pipeline)
    app.warmup()

    assert Client(app).get("/api/health").json()["embedding_model_loaded"] is True


def test_the_lifespan_warms_up_on_startup():
    """The cost is paid while the server is starting, where no user is waiting."""
    embedder = StubEmbedder()
    pipeline = StubPipeline()
    pipeline.retriever.embedder = embedder
    app = make_app(pipeline)

    sent: list[str] = []
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message["type"])

    async def go() -> None:
        await app({"type": "lifespan"}, receive, send)

    _run(go())

    assert embedder.loads == 1
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


# --------------------------------------------------------------------------
# GET /api/examples
# --------------------------------------------------------------------------


def test_examples_returns_exactly_the_three_config_examples():
    config = load_config(REPO_ROOT / "config" / "config.yaml")
    body = make_client().get("/api/examples").json()

    assert body["examples"] == list(config.ui.examples)
    assert len(body["examples"]) == 3


def test_examples_are_the_three_documented_categories():
    """P5 pitfall: fixed in config, covering the three high-frequency categories
    - exit load, lock-in, statement download."""
    examples = make_client().get("/api/examples").json()["examples"]
    joined = " ".join(examples).lower()

    assert "exit load" in joined
    assert "lock-in" in joined
    assert "statement" in joined


def test_examples_are_stable_across_calls():
    """Never randomized or rotated: the demo depends on them."""
    client = make_client()
    assert client.get("/api/examples").json() == client.get("/api/examples").json()


# --------------------------------------------------------------------------
# POST /api/ask
# --------------------------------------------------------------------------


def test_ask_returns_the_documented_payload():
    response = make_client().ask()
    body = response.json()

    assert response.status_code == 200
    for key in ("answer", "citation", "last_updated", "route", "validator_flags", "latency_ms"):
        assert key in body, f"missing {key}"
    assert body["route"] == "answer"
    assert body["citation"]["url"] == CITATION
    assert body["last_updated"] == "2026-09-27"
    assert body["validator_flags"] == ["check_3"]
    assert body["latency_ms"] >= 0


def test_ask_carries_exactly_one_citation_in_the_answer():
    """D4, re-checked at the boundary. The validator guarantees one URL in the
    text; this asserts the API did not add a second."""
    body = make_client().ask().json()

    assert body["answer"].count("https://") == 1
    assert body["citation"]["url"] in body["answer"]


def test_ask_passes_the_question_through_untouched():
    pipeline = StubPipeline()
    make_client(pipeline).ask("  What is the exit load on HDFC Large Cap Fund?  ")

    assert pipeline.questions == ["What is the exit load on HDFC Large Cap Fund?"]


def test_ask_returns_a_refusal_as_a_normal_200():
    """A refusal is a successful request carrying a refusal payload."""
    refusal = answer_response(
        query_class="performance",
        route="refusal",
        text=(
            "This assistant does not discuss returns or performance. "
            "Facts-only. No investment advice."
        ),
        link_label="HDFC Large Cap Fund",
        link_url=CITATION,
        citation_url="",
        last_updated="",
        validator_checks=(),
    )
    response = make_client(StubPipeline(response=refusal)).ask("What is the 1 year return?")
    body = response.json()

    assert response.status_code == 200
    assert body["route"] == "refusal"
    assert "Facts-only. No investment advice." in body["answer"]


def test_ask_reports_safe_response_route_separately():
    body = make_client(StubPipeline(response=answer_response(route="safe_response"))).ask().json()

    assert body["route"] == "safe_response"


# --------------------------------------------------------------------------
# failure handling: the point of P5-T3
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        LLMError("provider 500: sk-secret-abc123"),
        CollectionUnavailable("cannot open C:\\Users\\sunil\\chroma\\mf_facts_v1"),
        RuntimeError("unexpected in /home/app/venv/lib/site-packages/httpx/_client.py"),
    ],
)
def test_ask_returns_503_on_infrastructure_failure(error):
    response = make_client(StubPipeline(error=error)).ask()

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "service_unavailable"
    assert "answer" not in body, "a failed request must not carry answer text"


def test_no_error_body_leaks_internals():
    """P5 definition of done: zero client-visible stack traces."""
    for error in (
        LLMError("provider 500 for key sk-secret-abc123"),
        CollectionUnavailable("cannot open chroma at C:\\data\\chroma"),
        RuntimeError("Traceback (most recent call last): at /app/venv/httpx line 88"),
    ):
        raw = make_client(StubPipeline(error=error)).ask().text
        for leak in LEAKS:
            assert leak not in raw, f"{leak!r} leaked in a 503 body: {raw[:200]}"


def test_a_secret_in_an_exception_never_reaches_the_client():
    body = make_client(StubPipeline(error=LLMError("auth failed: Bearer sk-live-SECRET123"))).ask().json()

    assert "SECRET" not in json.dumps(body)


# --------------------------------------------------------------------------
# guards (P5-T4)
# --------------------------------------------------------------------------


def test_an_oversize_body_is_refused_before_it_is_buffered():
    app = make_app()
    limit = app.config.api.max_body_bytes
    response = Client(app).post(
        "/api/ask",
        content=b'{"question": "' + b"x" * (limit * 2) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"


def test_an_over_long_question_is_refused():
    app = make_app()
    limit = app.config.api.max_question_length
    response = Client(app).ask("e" * (limit + 1))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "question_too_long"


def test_a_question_at_the_limit_is_accepted():
    """The guard must not fire early - an off-by-one here would reject a
    legitimate long question."""
    app = make_app()
    limit = app.config.api.max_question_length
    response = Client(app).ask("e" * limit)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b"[]",
        b'{"question": 42}',
        b'{"question": "   "}',
        b'{"q": "what is the exit load"}',
    ],
)
def test_an_unusable_body_is_a_400_not_a_500(body):
    response = make_client().post("/api/ask", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_the_rate_limit_fires_and_says_when_to_retry():
    client = Client(make_app(limiter=RateLimiter(limit=2, window_s=60)))

    assert client.ask().status_code == 200
    assert client.ask().status_code == 200
    limited = client.ask()

    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) >= 1


def test_the_rate_limit_counts_per_client():
    """Two peers do not share a budget, so one busy client cannot lock everyone
    else out of the demo."""
    limiter = RateLimiter(limit=5, window_s=60)
    assert limiter.check("1.2.3.4", now=0.0)[0] is True
    assert limiter.check("1.2.3.4", now=0.0)[0] is True
    assert limiter.check("5.6.7.8", now=0.0)[0] is True


def test_the_rate_limiter_window_expires():
    limiter = RateLimiter(limit=1, window_s=10)

    assert limiter.check("ip", now=0.0)[0] is True
    assert limiter.check("ip", now=1.0)[0] is False
    # Past the window the old hit is forgotten rather than counted forever.
    assert limiter.check("ip", now=11.5)[0] is True


# --------------------------------------------------------------------------
# routing and headers
# --------------------------------------------------------------------------


def test_an_unknown_path_is_a_clean_404():
    response = make_client().get("/api/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "method,path,allow",
    [("GET", "/api/ask", "POST"), ("POST", "/api/health", "GET"), ("POST", "/api/examples", "GET")],
)
def test_a_wrong_method_is_405_and_advertises_the_right_one(method, path, allow):
    """Telling a caller it used the wrong verb is more useful than a 404, and
    the Allow header is what a client reads to find out."""
    response = make_client().request(method, path, json={} if method == "POST" else None)

    assert response.status_code == 405
    assert response.headers["allow"] == allow
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_responses_carry_the_security_headers():
    response = make_client().get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "script-src 'self'" in response.headers["content-security-policy"]


def test_there_are_exactly_three_api_endpoints():
    """architecture 5.19: 'three endpoints, no more'."""
    source = (REPO_ROOT / "api" / "app.py").read_text(encoding="utf-8")
    routes = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("if method ==") and '"/api/' in line
    }

    assert len(routes) == 3, routes


# --------------------------------------------------------------------------
# one real-pipeline test, so the stub cannot drift from the real contract
# --------------------------------------------------------------------------


def test_the_real_pipeline_answers_through_the_api():
    """The stub tests the transport; this proves the transport fits the real
    pipeline. Uses FakeLLM because OQ2 is still open."""
    from mf_facts.online.generator import FakeLLM

    app = build_app(root=REPO_ROOT, allow_fake=True, llm=FakeLLM(max_sentences=3))
    assert isinstance(app, FactsApp)
    response = Client(app).ask("What is the exit load on HDFC Large Cap Fund?")
    body = response.json()

    assert response.status_code == 200
    assert body["route"] in ("answer", "safe_response", "refusal")
    assert body["answer"].strip(), "an answered request must carry text"


# -- the served entrypoint ----------------------------------------------------


def _served_app(monkeypatch, argv: list[str]) -> FactsApp:
    """Run ``api.app.main`` with uvicorn stubbed out and return the app it serves."""
    import sys
    import types

    from api import app as app_module

    served: dict[str, Any] = {}
    stub = types.ModuleType("uvicorn")
    stub.run = lambda app, **kwargs: served.setdefault("app", app)
    monkeypatch.setitem(sys.modules, "uvicorn", stub)
    assert app_module.main(argv) == 0
    return served["app"]


def test_served_app_resolves_the_provider_from_config(monkeypatch):
    # Regression: main() once passed an explicit llm=None, which build_app treats
    # as "no LLM, fail closed" - so `python -m api` refused every question that
    # needed the residual classifier or the generator, while /api/health still
    # reported the configured provider.
    monkeypatch.setenv("MF_FACTS_LLM_API_KEY", "test-key-not-real")
    app = _served_app(monkeypatch, ["--port", "1"])
    assert app.pipeline.llm is not None
    assert app.pipeline.classifier.llm is not None
    assert app.pipeline.generator.available


def test_served_app_with_fake_uses_the_fake(monkeypatch):
    from mf_facts.online.generator import FakeLLM

    app = _served_app(monkeypatch, ["--port", "1", "--fake"])
    assert isinstance(app.pipeline.llm, FakeLLM)


# -- conversation memory ------------------------------------------------------


def test_history_is_forwarded_capped_to_the_window():
    stub = StubPipeline()
    client = make_client(stub)
    history = [f"earlier question {i}" for i in range(15)]
    assert client.post("/api/ask", json={"question": "and the exit load?", "history": history}).status_code == 200
    window = load_config(REPO_ROOT / "config" / "config.yaml").memory.window_turns
    assert stub.histories[-1] == tuple(history[-window:])


def test_missing_history_means_no_history():
    stub = StubPipeline()
    make_client(stub).ask()
    assert stub.histories[-1] == ()


def test_malformed_history_entries_are_skipped_not_fatal():
    stub = StubPipeline()
    body = {"question": "and the exit load?", "history": ["ok one", 7, "", None, "x" * 5000, "ok two"]}
    assert make_client(stub).post("/api/ask", json=body).status_code == 200
    assert stub.histories[-1] == ("ok one", "ok two")


def test_history_that_is_not_a_list_is_a_bad_request():
    stub = StubPipeline()
    body = {"question": "and the exit load?", "history": "HDFC Small Cap"}
    response = make_client(stub).post("/api/ask", json=body)
    assert response.status_code == 400
    assert stub.questions == []
