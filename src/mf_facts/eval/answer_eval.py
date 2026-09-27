"""P4 end-to-end evaluation: does the *answer path* hold up (P4-T9 acceptance).

The P2 harness in :mod:`mf_facts.eval.harness` measures retrieval: whether the
expected fact is in the passages at all. That is a necessary condition and not a
sufficient one, because it never runs a model. This module measures the other
half, through the real :class:`~mf_facts.online.ask.AnswerPipeline`:

* ``grounded_accuracy`` - the answer the user actually receives contains the
  expected fact. This is the number implementation.md P4 acceptance asks for, and
  it is the one that would catch a generator that writes fluently and wrongly.
* ``one_valid_link`` - exactly one URL, and it is the one from the cited chunk.
* ``within_sentence_limit`` / ``has_last_updated`` - the 5.18 output contract.
* ``unsupported_numbers`` - counted from the validator's own ``numbers_grounded``
  check, so this measures the shipped gate rather than a reimplementation of it.
* ``class_mismatch`` - how often the routing stage disagreed with the item's
  declared class. Reported rather than corrected, because with OQ2 open there is
  no real residual classifier to evaluate.

READ THIS BEFORE TRUSTING A NUMBER
----------------------------------
Every run needs an :class:`LLMClient`, and the only one available without an API
key is ``FakeLLM``, which is *extractive*: it copies the top passage. So
``grounded_accuracy`` here measures retrieval, routing, grounding, citation and
validation - the deterministic machinery - and **not** generation quality. A pass
is a statement about the plumbing. It is not evidence that a real model would
answer correctly, and it must not be reported as if it were.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..common.config import AppConfig, load_config
from ..common.errors import LLMError, MFactsError
from ..common.text import sentences
from ..online.ask import AnswerPipeline
from ..online.generator import FakeLLM, LLMClient
from . import matching
from .harness import REPO_ROOT, load_eval_set

_URL = re.compile(r"https?://[^\s)\]>\"']+")

#: How many times one item is retried after a provider rate-limit (HTTP 429).
_RATE_LIMIT_RETRIES = 6


@dataclass(slots=True)
class ItemOutcome:
    item_id: str
    question: str
    expected_fact: str
    route: str
    query_class: str
    answered: bool
    grounded: bool
    fact_in_top_n: bool
    citation_ok: bool
    within_sentence_limit: bool
    has_last_updated: bool
    unsupported_numbers: bool
    class_mismatch: bool
    text: str
    citation_url: str
    validator_checks: tuple[str, ...] = ()
    #: Hard output-contract breaches only (link / sentence count / stamp), and
    #: only for answered items. A quality miss is not one of these.
    contract_failure: str = ""
    failure: str = ""


@dataclass(slots=True)
class AnswerMetrics:
    eval_set: str
    n_items: int
    #: Items that reached the answer route. The output-contract metrics divide by
    #: this, not by n_items.
    n_answered: int
    grounded_accuracy: float
    fact_in_top_n_accuracy: float
    answered_rate: float
    one_valid_link: float
    within_sentence_limit: float
    has_last_updated: float
    unsupported_numbers: int
    class_mismatch: int
    mean_latency_ms: float
    p95_latency_ms: float
    provider: str
    model: str
    outcomes: list[ItemOutcome] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "eval_set": self.eval_set,
            "n_items": self.n_items,
            "n_answered": self.n_answered,
            "grounded_accuracy": round(self.grounded_accuracy, 4),
            "fact_in_top_n_accuracy": round(self.fact_in_top_n_accuracy, 4),
            "answered_rate": round(self.answered_rate, 4),
            "one_valid_link": round(self.one_valid_link, 4),
            "within_sentence_limit": round(self.within_sentence_limit, 4),
            "has_last_updated": round(self.has_last_updated, 4),
            "unsupported_numbers": self.unsupported_numbers,
            "class_mismatch": self.class_mismatch,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "provider": self.provider,
            "model": self.model,
        }


def _prose(text: str) -> str:
    """The answer without its citation and stamp lines."""
    return "\n".join(
        line
        for line in text.splitlines()
        if "](http" not in line and not line.startswith("Last updated from sources:")
    )


def _grade(
    response: Any, item: dict[str, Any], passages: dict[str, str]
) -> ItemOutcome:
    numeric = bool(item.get("numeric"))
    expected = item["expected_fact"]
    text = response.text
    urls = _URL.findall(text)
    answered = response.route == "answer"

    # One link, and it is the one the pipeline attached from chunk metadata.
    citation_ok = bool(urls) and len(urls) == 1 and urls[0] == response.citation_url

    grounded = answered and matching.fact_in_text(expected, _prose(text), numeric)

    # Whether the fact was in the passages the generator was actually shown. This
    # is a different question from `grounded` and the gap between them is the
    # whole diagnosis: a miss here is a retrieval or grounding problem, a miss
    # there with a hit here is a generation problem. With FakeLLM - which copies
    # passages[0] rather than reading all four - the gap mostly measures the
    # reranker, not the model.
    seen = "\n".join(passages.get(pid, "") for pid in response.retrieved_ids)
    fact_in_top_n = bool(seen) and matching.fact_in_text(expected, seen, numeric)

    # Hard output-contract breaches, judged only on items that were answered. A
    # refusal legitimately carries one educational link and a short body, so
    # scoring refusals against the answer contract would report every safe
    # refusal as a contract failure.
    contract: list[str] = []
    if answered:
        if not citation_ok:
            contract.append(f"citation invalid ({len(urls)} url(s))")
        if len(sentences(_prose(text))) > 3:
            contract.append("over 3 sentences")
        if "Last updated from sources:" not in text:
            contract.append("no last-updated stamp")

    # A quality miss is not a contract breach, and keeping them apart is what
    # stops a weak extractive fake from being reported as a broken validator.
    quality: list[str] = []
    if not answered:
        quality.append(f"route={response.route}")
    elif not grounded:
        quality.append(
            "expected fact absent from the answer"
            + ("" if fact_in_top_n else " and from every retrieved passage")
        )

    return ItemOutcome(
        item_id=item.get("id", item["question"]),
        question=item["question"],
        expected_fact=expected,
        route=response.route,
        query_class=response.query_class,
        answered=answered,
        grounded=grounded,
        fact_in_top_n=fact_in_top_n,
        citation_ok=citation_ok,
        within_sentence_limit=len(sentences(_prose(text))) <= 3,
        has_last_updated="Last updated from sources:" in text,
        # The shipped gate, not a second implementation of it.
        unsupported_numbers="numbers_grounded" in response.validator_checks,
        class_mismatch=bool(item.get("expected_class")) and response.query_class
        != item["expected_class"],
        text=text,
        citation_url=response.citation_url,
        validator_checks=tuple(response.validator_checks),
        contract_failure="; ".join(contract),
        failure="; ".join(quality + contract),
    )


def evaluate_answers(
    config: AppConfig,
    eval_set: str,
    root: Path = REPO_ROOT,
    llm: LLMClient | None = None,
    allow_fake: bool = True,
    limit: int | None = None,
    pace_s: float = 0.0,
) -> AnswerMetrics:
    """Run every item of ``eval_set`` through the real answer pipeline.

    ``pace_s`` is a pause between items, and also the back-off unit when the
    provider answers HTTP 429, so a free-tier token-per-minute cap slows the run
    rather than aborting it. Only the successful attempt is timed. Any other
    ``LLMError``, or a 429 that outlasts the retries, propagates.
    """
    items = load_eval_set(eval_set, root)
    if limit:
        items = items[:limit]

    if llm is None and allow_fake:
        # Deliberately not delegated to build_llm: an unresolved provider must
        # stay fail-closed in the app path, so the fake is requested here by name
        # rather than becoming the default for a config that forgot to pick one.
        llm = FakeLLM(
            model=config.generation.model or "fake-extractive",
            max_sentences=int(config.generation.max_sentences),
        )

    # ``llm=None`` must mean "resolve the provider from config" here, but to
    # ``AnswerPipeline.build`` an explicit None means "no LLM, fail closed".
    # Forwarding it would score every item as a 0 ms refusal against a live
    # provider, so the argument is only passed when there is a client to pass.
    build_kwargs: dict[str, Any] = {"config": config, "root": root, "allow_fake": allow_fake}
    if llm is not None:
        build_kwargs["llm"] = llm
    pipeline = AnswerPipeline.build(**build_kwargs)

    # id -> text for every indexed chunk, read once. Lets the report say whether a
    # miss was retrieval or generation without asking the pipeline to hand back
    # passage bodies.
    from ..pipeline.store import ChromaStore

    store = ChromaStore(
        path=root / "chroma",
        collection_name=config.corpus.collection,
        embedding_model=config.embedding.model_id,
        embedding_dim=config.embedding.embedding_dim,
    )
    stored = store.get_all()
    passages = {
        chunk_id: text or ""
        for chunk_id, text in zip(
            stored.get("ids") or [], stored.get("documents") or []
        )
    }

    outcomes: list[ItemOutcome] = []
    latencies: list[float] = []

    for index, item in enumerate(items):
        if pace_s and index:
            time.sleep(pace_s)
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            started = time.perf_counter()
            try:
                response = pipeline.ask(item["question"])
            except LLMError as exc:
                if not pace_s or "HTTP 429" not in str(exc) or attempt == _RATE_LIMIT_RETRIES:
                    raise
                time.sleep(pace_s * (attempt + 1))
                continue
            latencies.append((time.perf_counter() - started) * 1000)
            break
        outcomes.append(_grade(response, item, passages))

    n = len(outcomes) or 1
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    # The output contract is a property of answers, so its denominators are
    # answered items. n_answered is reported alongside so a reader can see the
    # denominator rather than having to assume it.
    answered_outcomes = [o for o in outcomes if o.answered]
    a = len(answered_outcomes) or 1

    return AnswerMetrics(
        eval_set=eval_set,
        n_items=len(outcomes),
        n_answered=len(answered_outcomes),
        grounded_accuracy=sum(o.grounded for o in outcomes) / n,
        fact_in_top_n_accuracy=sum(o.fact_in_top_n for o in outcomes) / n,
        answered_rate=len(answered_outcomes) / n,
        one_valid_link=sum(o.citation_ok for o in answered_outcomes) / a,
        within_sentence_limit=sum(o.within_sentence_limit for o in answered_outcomes) / a,
        has_last_updated=sum(o.has_last_updated for o in answered_outcomes) / a,
        unsupported_numbers=sum(o.unsupported_numbers for o in outcomes),
        class_mismatch=sum(o.class_mismatch for o in outcomes),
        mean_latency_ms=sum(latencies) / n,
        p95_latency_ms=p95,
        provider=str(pipeline.generator.provider if pipeline.generator else "none"),
        model=str(pipeline.generator.model if pipeline.generator else "none"),
        outcomes=outcomes,
    )


def format_row(metrics: AnswerMetrics) -> str:
    return (
        f"[{metrics.eval_set:<12}] n={metrics.n_items:<3} "
        f"answered={metrics.answered_rate:6.1%}({metrics.n_answered}) "
        f"grounded={metrics.grounded_accuracy:6.1%} "
        f"fact_in_top4={metrics.fact_in_top_n_accuracy:6.1%} "
        f"one_link={metrics.one_valid_link:6.1%} "
        f"<=3sent={metrics.within_sentence_limit:6.1%} "
        f"stamped={metrics.has_last_updated:6.1%} "
        f"unsupported_numbers={metrics.unsupported_numbers} "
        f"class_mismatch={metrics.class_mismatch} "
        f"p95={metrics.p95_latency_ms:6.0f}ms"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mf_facts.eval.answer_eval",
        description="Score the P4 answer path end to end. Needs a provider or --fake.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--set", action="append", dest="sets", help="eval set; repeatable"
    )
    parser.add_argument("--fake", action="store_true", help="use the offline FakeLLM")
    parser.add_argument("--limit", type=int, help="only the first N items")
    parser.add_argument("--json-out", help="write full metrics, with per-item detail")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument(
        "--pace-s",
        type=float,
        default=0.0,
        help="pause between items, and back-off unit on HTTP 429 (for rate-limited providers)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except MFactsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not args.fake and (config.generation.provider or "").strip() in ("", "<TBD: OQ2>"):
        print(
            "no provider configured (OQ2 is open). Pass --fake to exercise the "
            "deterministic path; the resulting grounded_accuracy measures "
            "plumbing, not generation quality.",
            file=sys.stderr,
        )
        return 2

    llm = FakeLLM() if args.fake else None

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for name in args.sets or ["factual", "table_facts", "howto"]:
        try:
            metrics = evaluate_answers(
                config, name, root=REPO_ROOT, llm=llm, allow_fake=args.fake, limit=args.limit,
                pace_s=args.pace_s,
            )
        except MFactsError as exc:
            print(f"[{name}] {exc}", file=sys.stderr)
            return 1
        print(format_row(metrics))
        rows.append(metrics.as_row())
        details.extend(
            {
                "eval_set": name,
                "item_id": o.item_id,
                "question": o.question,
                "expected_fact": o.expected_fact,
                "route": o.route,
                "query_class": o.query_class,
                "grounded": o.grounded,
                "fact_in_top_n": o.fact_in_top_n,
                "citation_url": o.citation_url,
                "validator_checks": list(o.validator_checks),
                "failure": o.failure,
                "text": o.text,
            }
            for o in metrics.outcomes
        )
        if args.show_failures:
            for o in metrics.outcomes:
                if o.failure:
                    print(f"    FAIL {o.item_id}: {o.failure}")

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"summary": rows, "items": details}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
