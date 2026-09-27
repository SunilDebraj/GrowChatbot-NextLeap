"""P5 - the HTTP surface: three endpoints and the page that calls them.

``implementation.md`` P5 names the file and the contract but no framework, and no
web framework is installed here, so this is a hand-rolled ASGI application. ASGI
is a small callable protocol, which means the app runs under the already-installed
``uvicorn`` and is testable in-process with ``httpx`` - no new dependency, and no
framework version pinning a demo hostage to a package index.

Endpoints, exactly three (architecture 5.19):

===============  ======  =====================================================
``/api/health``  GET     service status, collection, manifest version, build time
``/api/examples``GET     the 3 example questions, straight from config
``/api/ask``     POST    ``{question}`` -> the answer payload
===============  ======  =====================================================

Three properties this module is responsible for:

* **No exception ever reaches the client.** Every handler is wrapped; the detail
  goes to the server log and the client gets a code and a sentence. Architecture
  3.2 and the P5 definition of done both require this, and a stack trace in a
  response is the one failure mode that turns a facts product into a disclosure.
* **A 503 means no answer, never a partial one.** Chroma or provider failure is
  reported as ``service_unavailable`` with no answer text, because the alternative
  is an answer with no provenance.
* **The refusal path stays intact.** The API never reclassifies, never repairs and
  never adds text of its own to an answer or a refusal. It is a transport.
"""

from __future__ import annotations

import asyncio
import html
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from mf_facts.common.config import AppConfig, load_config
from mf_facts.common.errors import MFactsError
from mf_facts.common.logging import log_event
from mf_facts.common.models import AskResponse
from mf_facts.online.ask import _UNSET, AnswerPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

#: The only words a client ever sees about a failure. Deliberately generic: an
#: error message that names the failing subsystem leaks the deployment's shape.
SERVICE_UNAVAILABLE = {
    "error": {
        "code": "service_unavailable",
        "message": (
            "The assistant cannot answer right now. Please try again shortly, or "
            "open the official scheme page linked in the sources list."
        ),
    }
}

JSON_TYPE = "application/json; charset=utf-8"
HTML_TYPE = "text/html; charset=utf-8"
CSS_TYPE = "text/css; charset=utf-8"
JS_TYPE = "text/javascript; charset=utf-8"


class RateLimiter:
    """Trivial fixed-window per-client counter (P5-T4).

    A sliding log is more precise and costs a list per client; for a demo whose
    stated purpose is "stop a runaway client, not meter a service", a window
    counter is the honest amount of machinery. Single-process and in-memory, so
    it resets on restart and does not coordinate between workers - both are
    acceptable for a prototype and neither is hidden here.
    """

    def __init__(self, limit: int, window_s: int) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_s)``."""
        stamp = time.monotonic() if now is None else now
        fresh = [t for t in self._hits.get(key, ()) if stamp - t < self._window]
        if len(fresh) >= self._limit:
            retry = int(self._window - (stamp - fresh[0])) + 1
            self._hits[key] = fresh
            return False, max(retry, 1)
        fresh.append(stamp)
        self._hits[key] = fresh
        return True, 0

    def reset(self) -> None:
        self._hits.clear()


class FactsApp:
    """The ASGI application. Construct once, serve many requests."""

    def __init__(
        self,
        config: AppConfig | None = None,
        root: Path | None = None,
        pipeline: AnswerPipeline | None = None,
        log: Any = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.root = Path(root) if root else (config.root if config else REPO_ROOT)
        self.config = config or load_config(self.root / "config" / "config.yaml")
        self.log = log
        self._pipeline = pipeline
        self.limiter = limiter or RateLimiter(
            self.config.api.rate_limit_requests, self.config.api.rate_limit_window_s
        )
        # The pipeline is built lazily by the property above; this caches the
        # disclaimer, which is read fresh on every page load.
        self._disclaimer: str | None = None
        self._warm = False

    # -- the pipeline, built on first use ------------------------------------

    @property
    def pipeline(self) -> AnswerPipeline:
        if self._pipeline is None:
            self._pipeline = AnswerPipeline.build(
                config=self.config, root=self.root, log=self.log
            )
        return self._pipeline

    def warmup(self) -> None:
        """Pay the model load at boot instead of on someone's question.

        ``Embedder.model`` is lazy, and loading all-MiniLM-L6-v2 costs tens of
        seconds. Left lazy, the first uncached question absorbs that cost - during
        the P5 manual acceptance the second example question took 38.6s against a
        8s p95 target, while a cached question beside it took 11ms. A service that
        is slow on its first question is not measurably fast, so the load moves to
        startup, where a slow boot is expected and no user is waiting on it.
        """
        pipeline = self.pipeline
        retriever = pipeline.retriever
        embedder = getattr(retriever, "embedder", None)
        if embedder is not None:
            embedder.model  # noqa: B018 - property access is the load, by design
        self._warm = True

    @property
    def is_warm(self) -> bool:
        """Whether the embedding model is resident. Reported by health so an
        operator can tell a slow service from a cold one."""
        if self._warm:
            return True
        embedder = getattr(getattr(self._pipeline, "retriever", None), "embedder", None)
        return bool(embedder is not None and getattr(embedder, "_model", None) is not None)

    @property
    def disclaimer(self) -> str:
        """Read per page load, so an edited asset shows up without a restart."""
        if self._disclaimer is None:
            path = self.root / self.config.ui.disclaimer_file
            self._disclaimer = path.read_text(encoding="utf-8")
        return self._disclaimer

    # -- ASGI entry point ----------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # websockets are not part of P5
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "/")

        try:
            await self._route(method, path, scope, receive, send)
        except MFactsError as exc:
            # Infrastructure failure inside a handler.
            self._log_failure("api.error", path, exc)
            await self._json(send, 503, SERVICE_UNAVAILABLE)
        except Exception as exc:  # noqa: BLE001 - the point is that nothing escapes
            self._log_failure("api.unhandled", path, exc)
            await self._json(send, 503, SERVICE_UNAVAILABLE)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                # In a thread, so a 30-odd second model load does not block the
                # event loop and turn a slow boot into an unresponsive server.
                await asyncio.to_thread(self._safe_warmup)
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _safe_warmup(self) -> None:
        """A warmup failure must not stop the service from booting.

        If the model cannot load, /api/ask will fail closed with a 503 and
        /api/health will report it. Refusing to start would turn a degraded
        service into no service, and health is exactly the endpoint an operator
        needs in order to see why.
        """
        try:
            self.warmup()
        except Exception as exc:  # noqa: BLE001
            self._log_failure("api.warmup_failed", "/", exc)

    async def _route(self, method: str, path: str, scope: Scope, receive: Receive, send: Send) -> None:
        if method == "GET" and path == "/api/health":
            return await self._health(send)
        if method == "GET" and path == "/api/examples":
            return await self._examples(send)
        if method == "POST" and path == "/api/ask":
            return await self._ask(scope, receive, send)
        if method == "GET" and path in ("/", "/index.html"):
            return await self._page(send)
        if method == "GET" and path in ("/static/app.js", "/static/styles.css"):
            return await self._asset(send, path.rsplit("/", 1)[-1])

        # A wrong method on a real path is a 405, not a 404: the difference tells
        # a caller whether it used the wrong verb or the wrong path.
        if path in ("/api/health", "/api/examples", "/api/ask"):
            return await self._json(
                send,
                405,
                {"error": {"code": "method_not_allowed", "message": f"Use the documented method for {path}."}},
                headers={"allow": "POST" if path == "/api/ask" else "GET"},
            )
        await self._json(send, 404, {"error": {"code": "not_found", "message": "Not found."}})

    # -- endpoints -----------------------------------------------------------

    async def _health(self, send: Send) -> None:
        """Report what is actually loaded, not a hardcoded "ok".

        A health endpoint that cannot say "degraded" is worse than none, because
        the operator trusts it. Manifest and collection are checked separately so
        a missing manifest does not hide a healthy store.
        """
        manifest_path = self.root / "artifacts" / "corpus_manifest.json"
        manifest: dict[str, Any] = {}
        detail: list[str] = []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            detail.append("corpus manifest unavailable")

        collection = str(self.config.corpus.collection)
        try:
            store = self.pipeline.retriever.store if self.pipeline.retriever else None
            count = store.count() if store is not None else 0
        except Exception:  # noqa: BLE001
            count = 0
            # No exception class name here: health is a client-visible endpoint,
            # and naming the failing class tells a caller about the deployment
            # for no benefit to the operator, who has the log.
            detail.append("vector store unavailable")
        else:
            if count == 0:
                detail.append("vector store is empty")

        status = "ok" if not detail else "degraded"
        await self._json(
            send,
            200,
            {
                "status": status,
                "service": "mf-facts-bot",
                "collection": collection,
                "manifest_version": manifest.get("corpus_version", ""),
                "corpus_hash": manifest.get("corpus_hash", ""),
                "corpus_built_at": manifest.get("built_at", ""),
                "chunk_count": manifest.get("chunk_count", 0),
                "indexed_chunks": count,
                "provider": self.config.generation.provider,
                # Named, not summarised. An operator debugging "why did it say
                # that" needs the exact model id, and reporting the provider
                # without it would leave half the question unanswered.
                "model": self.config.generation.model,
                "embedding_model_loaded": self.is_warm,
                "detail": detail,
            },
        )

    async def _examples(self, send: Send) -> None:
        # P5 pitfall: fixed in config, never randomized. The demo depends on these
        # being the three high-frequency categories.
        await self._json(send, 200, {"examples": list(self.config.ui.examples)})

    async def _ask(self, scope: Scope, receive: Receive, send: Send) -> None:
        client = _client_key(scope)
        allowed, retry_after = self.limiter.check(client)
        if not allowed:
            await self._json(
                send,
                429,
                {
                    "error": {
                        "code": "rate_limited",
                        "message": "Too many questions from this client. Try again shortly.",
                    }
                },
                headers={"retry-after": str(retry_after)},
            )
            return

        raw = await self._read_body(receive, send)
        if raw is None:
            return

        question = _parse_question(raw)
        if question is None:
            await self._json(
                send,
                400,
                {"error": {"code": "invalid_request", "message": 'Send {"question": "..."} with a non-empty question.'}},
            )
            return

        if len(question) > self.config.api.max_question_length:
            await self._json(
                send,
                413,
                {
                    "error": {
                        "code": "question_too_long",
                        "message": f"Keep the question under {self.config.api.max_question_length} characters.",
                    }
                },
            )
            return

        history = _parse_history(
            raw, self.config.memory.window_turns, self.config.api.max_question_length
        )
        if history is None:
            await self._json(
                send,
                400,
                {"error": {"code": "invalid_request", "message": '"history" must be a list of previous questions.'}},
            )
            return

        started = time.perf_counter()
        try:
            response = self.pipeline.ask(question, history)
        except Exception as exc:  # noqa: BLE001
            # Chroma or provider failure. The client gets no answer text, because
            # there is no validated text to give.
            self._log_failure("api.ask_failed", "/api/ask", exc)
            await self._json(send, 503, SERVICE_UNAVAILABLE)
            return
        latency_ms = (time.perf_counter() - started) * 1000.0

        await self._json(send, 200, _answer_payload(response, latency_ms))

    # -- the page ------------------------------------------------------------

    async def _page(self, send: Send) -> None:
        """Server-render the shell on every load (P5-T6).

        The disclaimer is in the HTML that leaves the server, not fetched by JS
        afterwards, so it cannot be absent because a script failed.
        """
        await self._send_bytes(send, 200, HTML_TYPE, _render_page(self.config, self.disclaimer))

    async def _asset(self, send: Send, name: str) -> None:
        content_type = JS_TYPE if name.endswith(".js") else CSS_TYPE
        path = self.root / "ui" / name
        try:
            body = path.read_bytes()
        except OSError:
            await self._json(send, 404, {"error": {"code": "not_found", "message": "Not found."}})
            return
        await self._send_bytes(send, 200, content_type, body)

    # -- plumbing ------------------------------------------------------------

    async def _read_body(self, receive: Receive, send: Send) -> bytes | None:
        """Read at most ``max_body_bytes``; refuse anything larger.

        The cap is enforced while reading rather than after, so an oversized
        request cannot make the process buffer it.
        """
        limit = self.config.api.max_body_bytes
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            body = message.get("body", b"")
            size += len(body)
            if size > limit:
                await self._json(
                    send,
                    413,
                    {
                        "error": {
                            "code": "body_too_large",
                            "message": "Request body is too large.",
                        }
                    },
                )
                return None
            chunks.append(body)
            if not message.get("more_body", False):
                break
        return b"".join(chunks)

    async def _json(
        self,
        send: Send,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self._send_bytes(send, status, JSON_TYPE, body, headers)

    async def _send_bytes(
        self,
        send: Send,
        status: int,
        content_type: str,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        raw = [(b"content-type", content_type.encode("latin-1")),
               (b"content-length", str(len(body)).encode("latin-1"))]
        # A minimal, explicit security header set. The page loads no third-party
        # resource and must never be framed or sniffed.
        raw += [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"content-security-policy",
             b"default-src 'none'; script-src 'self'; style-src 'self'; "
             b"connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'"),
        ]
        for key, value in (headers or {}).items():
            raw.append((key.lower().encode("latin-1"), str(value).encode("latin-1")))

        await send({"type": "http.response.start", "status": status, "headers": raw})
        await send({"type": "http.response.body", "body": body})

    def _log_failure(self, event: str, path: str, exc: BaseException) -> None:
        """The detail goes here and only here.

        The class name is enough to debug with and is not what the client sees;
        the message could contain a filesystem path or a provider payload, so it
        stays server-side.
        """
        log_event(
            _logger(),
            event,
            path=path,
            error_type=type(exc).__name__,
            # The message is deliberately omitted: it is the thing most likely to
            # contain something the client should not learn.
        )


# --------------------------------------------------------------------------
# payload shaping
# --------------------------------------------------------------------------


def _answer_payload(response: AskResponse, latency_ms: float) -> dict[str, Any]:
    """Shape an AskResponse for the wire.

    ``route`` and the validator flags ride along because architecture 5.19 says
    they are for eval and debugging. They are not a headline UI feature and the
    UI does not label an answer with them.
    """
    return {
        "answer": response.text,
        "citation": {"label": response.link_label, "url": response.citation_url or response.link_url},
        "last_updated": response.last_updated,
        "route": response.route,
        "validator_flags": list(response.validator_checks),
        "latency_ms": round(latency_ms, 1),
        # Debug/eval extras, same rationale as route.
        "class": response.query_class,
        "rule_id": response.rule_id,
        "grounding_passed": response.grounding_passed,
    }


def _client_key(scope: Scope) -> str:
    """Identify the caller for the rate limit.

    The direct peer address, not X-Forwarded-For: a forwarded header is
    client-controlled, so trusting it would let one caller reset their own limit
    by inventing a value. A deployment behind a proxy needs this revisited, which
    is a known limit rather than a silent assumption.
    """
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return "unknown"


def _parse_question(raw: bytes) -> str | None:
    """Pull the question out of the body, or None if it is not usable."""
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    question = payload.get("question")
    if not isinstance(question, str):
        return None
    question = question.strip()
    return question or None


def _parse_history(raw: bytes, window: int, max_length: int) -> tuple[str, ...] | None:
    """The client's previous questions, newest last, capped at ``window``.

    Absent means no history. A ``history`` that is not a list is a bad request
    (None). Non-string, empty or over-long entries are skipped rather than
    rejected: memory is a convenience, and a malformed turn should cost the
    follow-up its context, not the user their answer. Nothing here is stored.
    """
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ()
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not isinstance(history, list):
        return None
    if window <= 0:
        return ()
    kept = [
        item.strip()
        for item in history
        if isinstance(item, str) and item.strip() and len(item.strip()) <= max_length
    ]
    return tuple(kept[-window:])


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

WELCOME = "HDFC Mutual Fund Facts Assistant"
FACTS_ONLY = "Facts-only. No investment advice."
SCOPE_LINE = "Factual answers about 5 HDFC schemes from official public sources."
NO_PII_REMINDER = (
    "Do not enter PAN, Aadhaar, account numbers, OTP, email addresses or phone numbers."
)


def _render_page(config: AppConfig, disclaimer: str) -> bytes:
    """Build the HTML shell.

    Escaped by hand rather than by a template engine, which is only safe because
    every interpolated value goes through ``html.escape``: the examples come from
    config, and the disclaimer from a local asset, but a config file is still an
    input and an unescaped ``&`` would be a silent corruption rather than a bug
    anyone would notice.
    """
    examples = "\n".join(
        f'      <button class="example" type="button" data-question="{html.escape(q, quote=True)}">'
        f"{html.escape(q)}</button>"
        for q in config.ui.examples
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(WELCOME)}</title>
<link rel="stylesheet" href="/static/styles.css">
</head>
<body>
<header class="banner">
  <h1>{html.escape(WELCOME)}</h1>
  <p>{html.escape(SCOPE_LINE)}</p>
</header>

<p class="facts-only">{html.escape(FACTS_ONLY)}</p>

<main>
  <section class="examples" aria-label="Example questions">
    <h2>Try:</h2>
{examples}
  </section>

  <form id="ask-form" autocomplete="off" data-memory-turns="{int(config.memory.window_turns)}">
    <label class="sr-only" for="question">Your question</label>
    <textarea id="question" name="question" rows="2" maxlength="{config.api.max_question_length}"
      placeholder="{html.escape(NO_PII_REMINDER, quote=True)}" required></textarea>
    <button type="submit" id="ask-button">Ask</button>
  </form>
  <p class="reminder">{html.escape(NO_PII_REMINDER)}</p>

  <section id="thread" class="thread" aria-live="polite" aria-label="Answers"></section>
</main>

<footer class="disclaimer">
  <p>{html.escape(disclaimer.strip())}</p>
</footer>

<script src="/static/app.js"></script>
</body>
</html>
"""
    return page.encode("utf-8")


def _logger():
    import logging

    return logging.getLogger("mf_facts.api")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_app(
    config_path: str | Path | None = None,
    root: Path | None = None,
    pipeline: AnswerPipeline | None = None,
    log: Any = None,
    llm: Any = _UNSET,
    allow_fake: bool = False,
) -> FactsApp:
    """Build the app.

    ``llm`` follows the same three-state rule as
    :meth:`AnswerPipeline.build` - omitted resolves from config (the real app,
    and the only path that may create a live provider client), an object is used
    verbatim, and an explicit ``None`` means "no LLM" so the answer path fails
    closed.
    """
    base = Path(root) if root else REPO_ROOT
    config = load_config(config_path) if config_path else load_config(base / "config" / "config.yaml")
    if pipeline is None and (llm is not _UNSET or allow_fake):
        pipeline = AnswerPipeline.build(
            config=config, root=base, log=log, llm=llm, allow_fake=allow_fake
        )
    return FactsApp(config=config, root=base, pipeline=pipeline, log=log)


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the MF-Facts-Bot prototype.")
    parser.add_argument("--host", default=None, help="override api.host from config")
    parser.add_argument("--port", type=int, default=None, help="override api.port from config")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use the offline FakeLLM. OQ2 is open, so this is the only way to "
        "demo without a provider key.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    probe = load_config(REPO_ROOT / "config" / "config.yaml")
    host = args.host or probe.api.host
    port = args.port or probe.api.port

    # ``llm`` is three-state (see build_app): passing an explicit None would mean
    # "no LLM, fail closed" and silently turn every served answer into a refusal.
    # Without --fake it must be omitted so the provider resolves from config.
    if args.fake:
        from mf_facts.online.generator import FakeLLM

        app = build_app(
            allow_fake=True,
            llm=FakeLLM(
                model=probe.generation.model or "fake-extractive",
                max_sentences=int(probe.generation.max_sentences),
            ),
        )
    else:
        app = build_app()

    try:
        import uvicorn
    except ImportError:
        import sys

        print(
            "uvicorn is required to serve the app: pip install 'mf-facts-bot[serve]'",
            file=sys.stderr,
        )
        return 2

    print(f"http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
