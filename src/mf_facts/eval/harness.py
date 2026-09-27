"""[10] Eval harness - the chunking gate (architecture.md 5.10, 7.6).

Scores a candidate chunking strategy end to end so PRD OQ1 can be closed with
numbers rather than adjectives:

    python -m mf_facts.eval.harness --strategy heading_aware
    python -m mf_facts.eval.harness --strategy fixed_window --only table_facts
    python -m mf_facts.eval.harness --all          # every strategy, both sets

Metrics (architecture.md 7.6 step 2):
    hit@k                 expected fact present in the top-k passages
    grounded_accuracy     a single sentence in the top-k states the fact
    table_fact_accuracy   grounded_accuracy over the numeric table set
    mean/max_chunk_tokens chunk size shape
    build_time_s          from the build report

**What grounded_accuracy is and is not, during P2.** implementation.md P2-T4
defines it as "generation restricted to the passage, compared to expected_fact".
OQ2 is still open, so no model exists to generate anything, and inventing a stub
that pretends to generate would make the gate unfalsifiable. It is therefore
computed as *extractive* grounding: does one intact sentence in the top-k state
the fact. That is the chunking question the gate exists to answer, and it is
strictly harder to pass by accident than containment, because a fact split across
a chunk boundary fails it. P4 re-measures the same metric with a real generator
and treats the two numbers as different measurements - the P2 number is an
upper bound on the P4 one, never a substitute.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..common.config import AppConfig, load_config
from ..common.errors import CollectionUnavailable, ConfigError, MFactsError
from ..pipeline.embedder import Embedder
from ..pipeline.store import ChromaStore
from . import matching

DEFAULT_TOP_K = 5
REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIRNAME = "eval"
# Metrics are persisted per strategy so `python -m mf_facts.eval.report` can
# assemble the decision table without re-running three builds. The three
# collections cannot coexist - one collection name - so without a persisted
# artifact the report would only ever see the last strategy built.
METRICS_DIRNAME = "artifacts/eval_metrics"


@dataclass
class ItemResult:
    item_id: str
    scheme_key: str
    category: str
    numeric: bool
    hit_at_1: bool
    hit_at_2: bool
    hit_at_k: bool
    grounded: bool
    expected_fact: str
    supporting_sentence: str | None = None
    top_passage_ids: list[str] = field(default_factory=list)


@dataclass
class Metrics:
    strategy: str
    eval_set: str
    n_items: int
    hit_at_1: float
    hit_at_2: float
    hit_at_5: float
    grounded_accuracy: float
    table_fact_accuracy: float | None
    mean_chunk_tokens: float
    max_chunk_tokens: int
    build_time_s: float | None
    chunk_count: int
    query_time_s: float
    results: list[ItemResult] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        """The scalar view, for the metrics table in the decision doc."""
        return {
            "strategy": self.strategy,
            "eval_set": self.eval_set,
            "n_items": self.n_items,
            "hit@1": self.hit_at_1,
            "hit@2": self.hit_at_2,
            "hit@5": self.hit_at_5,
            "grounded_accuracy": self.grounded_accuracy,
            "table_fact_accuracy": self.table_fact_accuracy,
            "mean_chunk_tokens": self.mean_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "build_time_s": self.build_time_s,
            "chunk_count": self.chunk_count,
            "query_time_s": self.query_time_s,
        }


def load_eval_set(name: str, root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    path = root / EVAL_DIRNAME / f"{name}.jsonl"
    if not path.exists():
        raise ConfigError(f"eval set {name!r} not found at {path}")
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name}:{number} is not valid JSON: {exc}") from exc
    if not items:
        raise ConfigError(f"eval set {name!r} is empty")
    return items


def _retrieve(
    store: ChromaStore,
    embedder: Embedder,
    item: dict[str, Any],
    top_k: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Embed the question, hard-filter to the scheme, return top-k."""
    vector = embedder.embed([item["question"]])[0]
    result = store.query(
        vector,
        n_results=top_k,
        where={"scheme_key": {"$in": [item["scheme_key"]]}},
    )
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    return list(ids), list(docs), [dict(m or {}) for m in metas]


def evaluate(
    config: AppConfig,
    strategy: str,
    eval_set: str,
    top_k: int = DEFAULT_TOP_K,
    root: Path = REPO_ROOT,
) -> Metrics:
    """Run one strategy against one eval set and return the metrics."""
    items = load_eval_set(eval_set, root)
    store = ChromaStore(
        path=root / "chroma",
        collection_name=config.corpus.collection,
        embedding_model=config.embedding.model_id,
        embedding_dim=config.embedding.embedding_dim,
    )
    embedder = Embedder(
        config.embedding, cache_dir=root / config.corpus.embedding_cache_dir
    )

    # Chunk-size shape, read from what is actually stored rather than recomputed.
    stored = store.get_all()
    token_counts = [
        int((meta or {}).get("token_count") or 0) for meta in (stored.get("metadatas") or [])
    ]
    token_counts = [count for count in token_counts if count > 0]

    started = time.perf_counter()
    results: list[ItemResult] = []
    for item in items:
        ids, docs, _metas = _retrieve(store, embedder, item, top_k)
        numeric = bool(item.get("numeric"))
        expected = item["expected_fact"]

        hit_at_k = any(
            matching.fact_in_text(expected, doc, numeric) for doc in docs
        )
        supporting = None
        grounded = False
        for doc in docs:
            supporting = matching.extract_sentence(expected, doc, numeric)
            if supporting is not None:
                grounded = True
                break
        # hit@1 is the first passage alone, so a numeric fact that only appears
        # in passage 3 counts against ranking but not against retrieval.
        hit_at_1 = bool(docs) and matching.fact_in_text(expected, docs[0], numeric)
        hit_at_2 = any(
            matching.fact_in_text(expected, doc, numeric) for doc in docs[:2]
        )

        results.append(
            ItemResult(
                item_id=item.get("id", item["question"]),
                scheme_key=item["scheme_key"],
                category=item.get("category", ""),
                numeric=numeric,
                hit_at_1=hit_at_1,
                hit_at_2=hit_at_2,
                hit_at_k=hit_at_k,
                grounded=grounded,
                expected_fact=expected,
                supporting_sentence=supporting,
                top_passage_ids=ids,
            )
        )
    query_time = time.perf_counter() - started

    def rate(flag: str) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if getattr(r, flag)) / len(results)

    manifest_path = root / "artifacts" / "corpus_manifest.json"
    build_time = None
    strategy_in_manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            build_time = manifest.get("build_duration_s")
            strategy_in_manifest = manifest.get("chunking_strategy")
        except (OSError, json.JSONDecodeError):
            pass

    table_accuracy = None
    if eval_set == "table_facts":
        table_accuracy = rate("grounded")

    return Metrics(
        strategy=strategy_in_manifest or strategy,
        eval_set=eval_set,
        n_items=len(results),
        hit_at_1=rate("hit_at_1"),
        hit_at_2=rate("hit_at_2"),
        hit_at_5=rate("hit_at_k"),
        grounded_accuracy=rate("grounded"),
        table_fact_accuracy=table_accuracy,
        mean_chunk_tokens=round(statistics.fmean(token_counts), 2) if token_counts else 0.0,
        max_chunk_tokens=max(token_counts) if token_counts else 0,
        build_time_s=build_time,
        chunk_count=len(token_counts),
        query_time_s=round(query_time, 3),
        results=results,
    )


def evaluate_all(
    config: AppConfig,
    strategies: Sequence[str],
    eval_sets: Sequence[str],
    top_k: int = DEFAULT_TOP_K,
    root: Path = REPO_ROOT,
) -> list[Metrics]:
    return [
        evaluate(config, strategy, eval_set, top_k=top_k, root=root)
        for strategy in strategies
        for eval_set in eval_sets
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mf_facts.eval.harness",
        description="Score chunking strategies against the factual and table eval sets.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--strategy", action="append", help="strategy to score; repeatable")
    parser.add_argument(
        "--only",
        action="append",
        choices=("factual", "table_facts", "howto"),
        help="restrict to one eval set; repeatable",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="score every strategy in config against every eval set",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--json-out", help="also write the full metrics as JSON here")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help=f"skip writing {METRICS_DIRNAME}/<strategy>.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except MFactsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    strategies = args.strategy or (
        None if args.all else [config.chunking.strategy]
    )
    if args.all and not strategies:
        from ..pipeline.strategies import STRATEGIES

        strategies = list(STRATEGIES)
    if not strategies:
        strategies = [config.chunking.strategy]

    eval_sets = args.only or ["factual", "table_facts"]

    all_metrics: list[Metrics] = []
    for strategy in strategies:
        for eval_set in eval_sets:
            try:
                metrics = evaluate(
                    config, strategy, eval_set, top_k=args.top_k, root=REPO_ROOT
                )
            except CollectionUnavailable as exc:
                print(f"[{strategy}/{eval_set}] collection unavailable: {exc}", file=sys.stderr)
                return 1
            except MFactsError as exc:
                print(f"[{strategy}/{eval_set}] {exc}", file=sys.stderr)
                return 1
            all_metrics.append(metrics)
            if not args.quiet:
                print(format_row(metrics))

    if not args.no_persist:
        metrics_dir = REPO_ROOT / METRICS_DIRNAME
        metrics_dir.mkdir(parents=True, exist_ok=True)
        # Group by strategy: one file per strategy holding a row per eval set.
        # Writing metrics.strategy.json per row would have the second eval set
        # overwrite the first, since both rows carry the same strategy name.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for metrics in all_metrics:
            grouped.setdefault(metrics.strategy, []).append(metrics.as_row())
        for strategy, rows in grouped.items():
            target = metrics_dir / f"{strategy}.json"
            target.write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not args.quiet:
                print(f"  persisted {target.relative_to(REPO_ROOT)} ({len(rows)} eval set(s))")

    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                [m.as_row() for m in all_metrics], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"wrote {target}")
    return 0


def format_row(metrics: Metrics) -> str:
    table = metrics.table_fact_accuracy
    table_text = f"{table:6.1%}" if table is not None else "   -  "
    return (
        f"[{metrics.strategy:<13} {metrics.eval_set:<12}] "
        f"n={metrics.n_items:<3} "
        f"hit@1={metrics.hit_at_1:6.1%} hit@2={metrics.hit_at_2:6.1%} "
        f"hit@5={metrics.hit_at_5:6.1%} "
        f"grounded={metrics.grounded_accuracy:6.1%} "
        f"table={table_text} "
        f"tok(mean/max)={metrics.mean_chunk_tokens:.0f}/{metrics.max_chunk_tokens} "
        f"chunks={metrics.chunk_count} build={metrics.build_time_s}s"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
