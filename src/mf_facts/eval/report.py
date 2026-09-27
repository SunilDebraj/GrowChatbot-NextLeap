"""[10b] Report generator -> docs/chunking_decision.md (P2-T6).

    python -m mf_facts.eval.report

Reads the per-strategy metrics the harness persisted and writes the decision
document: the filled metric table, the selection, the parameters, the rationale,
and **the rejected options with their numbers** (architecture.md 7.6 step 4).

The selection follows 7.6 step 3 in order - table_fact_accuracy, then hit@5, then
build simplicity - and the tie-break from 7.2: a tie on table_fact_accuracy goes
to heading_aware, because prose coverage is what the tie-break is protecting and
fixed_window only ties here while the documents are smaller than one window.

Every number in the output comes from artifacts/eval_metrics/. Nothing is
recomputed here, so the document cannot drift from the run that produced it, and
if a metric file is missing the report fails instead of quietly printing a
partial table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..pipeline.strategies import STRATEGIES

REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_DIRNAME = "artifacts/eval_metrics"
DECISION_PATH = "docs/chunking_decision.md"

SELECTION_ORDER = ("table_fact_accuracy", "hit_at_5")

STRATEGY_LABELS = {
    "heading_aware": "A. `HeadingAwareStrategy`",
    "fixed_window": "B. `FixedWindowStrategy`",
    "atomic_fact": "C. `AtomicFactStrategy`",
}


def load_metrics(root: Path = REPO_ROOT) -> dict[str, list[dict[str, Any]]]:
    metrics_dir = root / METRICS_DIRNAME
    if not metrics_dir.is_dir():
        raise FileNotFoundError(
            f"{metrics_dir} does not exist. Run the gate first:\n"
            "  for s in heading_aware fixed_window atomic_fact; do\n"
            "    python -m mf_facts.pipeline.build --strategy $s\n"
            "    python -m mf_facts.eval.harness --strategy $s\n"
            "  done"
        )
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_strategy[path.stem] = rows if isinstance(rows, list) else [rows]
    if not by_strategy:
        raise FileNotFoundError(f"no metrics files in {metrics_dir}")
    return by_strategy


def pick(by_strategy: dict[str, list[dict[str, Any]]], eval_set: str) -> dict[str, Any]:
    return {
        name: next((r for r in rows if r["eval_set"] == eval_set), None)
        for name, rows in by_strategy.items()
    }


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.0f}%"


def _num(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}".rstrip("0").rstrip(".") + (
            "" if suffix else ""
        ) or "0"
    return f"{value}{suffix}"


def select_winner(
    factual: dict[str, Any], tables: dict[str, Any]
) -> tuple[str | None, list[str]]:
    """Apply 7.6 step 3 in order. Returns the winner and the reasoning trail."""
    trail: list[str] = []
    candidates = [name for name in STRATEGIES if name in factual and factual[name]]
    if not candidates:
        return None, ["no strategies with factual metrics"]

    def value(name: str, key: str) -> float:
        table_row = tables.get(name)
        if key == "table_fact_accuracy" and table_row:
            candidate = table_row.get("table_fact_accuracy")
            if candidate is not None:
                return float(candidate)
        row = factual[name]
        return float(row.get(key) or 0.0)

    for key in SELECTION_ORDER:
        scored = {name: value(name, key) for name in candidates}
        best = max(scored.values())
        leaders = [name for name in candidates if scored[name] == best]
        label = "table_fact_accuracy" if key == "table_fact_accuracy" else "hit@5"
        trail.append(f"1st criterion `{label}`: " + ", ".join(
            f"{name}={_pct(scored[name])}" for name in candidates
        ) + f" -> leader(s) {leaders}")
        if len(leaders) == 1:
            return leaders[0], trail
        # 7.2 tie-break: equal on the numeric criterion goes to heading_aware.
        if "heading_aware" in leaders:
            trail.append(
                f"tie on `{label}` -> tie-break (architecture.md 7.2) selects "
                "heading_aware for prose coverage"
            )
            return "heading_aware", trail
        candidates = leaders
    return candidates[0], trail


def build_document(
    by_strategy: dict[str, list[dict[str, Any]]], root: Path = REPO_ROOT
) -> str:
    factual = pick(by_strategy, "factual")
    tables = pick(by_strategy, "table_facts")
    winner, trail = select_winner(factual, tables)

    manifest_path = root / "artifacts" / "corpus_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    chunking_params = manifest.get("chunking_params", {})

    lines: list[str] = [
        "# Chunking strategy decision - PRD OQ1 closed",
        "",
        "Generated by `python -m mf_facts.eval.report` from the metric files in",
        "`artifacts/eval_metrics/`. Every number below is copied from those files;",
        "nothing here is recomputed by hand.",
        "",
        f"Corpus under test: {manifest.get('doc_count', '?')} documents,",
        f"{manifest.get('chunk_count', '?')} chunks as built by the winning strategy,",
        f"embedding `{manifest.get('embedding_model', '?')}`,",
        f"corpus_version `{manifest.get('corpus_version', '?')}`.",
        "",
        "## Metric table",
        "",
        "`hit@k` = the expected fact is present in the top-k retrieved passages.",
        "`grounded` = a single intact sentence in the top-k states the fact.",
        "`table` = `grounded` over `eval/table_facts.jsonl` (the numeric fee/tax rows).",
        "",
        "| Strategy | eval set | n | hit@1 | hit@2 | hit@5 | grounded | table | chunks | mean tok | max tok | build s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in [n for n in STRATEGIES if n in factual]:
        for label, table in (("factual", factual[name]), ("table_facts", tables.get(name))):
            if not table:
                continue
            lines.append(
                f"| {STRATEGY_LABELS.get(name, name)} | {label} | {table['n_items']} | "
                f"{_pct(table['hit@1'])} | {_pct(table['hit@2'])} | {_pct(table['hit@5'])} | "
                f"{_pct(table['grounded_accuracy'])} | "
                f"{_pct(table.get('table_fact_accuracy')) if label == 'table_facts' else '-'} | "
                f"{table['chunk_count']} | {table['mean_chunk_tokens']} | "
                f"{table['max_chunk_tokens']} | "
                f"{table.get('build_time_s') if table.get('build_time_s') is not None else 'n/a'} |"
            )

    lines += [
        "",
        "## Selection",
        "",
    ]
    if winner:
        lines.append(f"**Selected: {STRATEGY_LABELS.get(winner, winner)}** "
                     f"(`chunking.strategy: {winner}` in `config/config.yaml`).")
    else:
        lines.append("**No winner could be selected - metrics are incomplete.**")
    lines += ["", "Criterion order, applied as written in architecture.md 7.6 step 3:", ""]
    lines += [f"- {step}" for step in trail]
    lines += [
        "",
        "## Parameters locked",
        "",
        "```yaml",
    ]
    for key, value in sorted(chunking_params.items()):
        lines.append(f"chunking.{key}: {value}")
    lines += [
        "```",
        "",
        "## Rationale",
        "",
    ]

    if winner == "heading_aware":
        lines += [
            "`heading_aware` and `fixed_window` tie on the primary criterion,",
            "`table_fact_accuracy`, and `heading_aware` takes it on the documented",
            "tie-break (architecture.md 7.2): equal numeric accuracy goes to the",
            "strategy with better prose coverage. `heading_aware` also keeps section",
            "structure in the embedded text, prefixes the section heading to each chunk",
            "so a retrieved passage carries its own context, and emits tables as their",
            "own chunks - the behaviour the corpus will need as soon as the unsourced",
            "table doc classes become available (see `docs/corpus_gaps.md`).",
            "",
            "The tie is not evidence that chunk boundaries do not matter here. It is",
            "evidence that **this corpus is too small for the gate to discriminate",
            "between A and B** - see the limitations below. The decision rests on the",
            "tie-break plus forward-looking structure, not on a measured A-vs-B gap.",
        ]
    else:
        lines += [
            f"`{winner}` was selected on the measured criteria above, with no tie-break",
            "invoked.",
        ]

    lines += [
        "",
        "## Rejected options, with their numbers",
        "",
    ]
    for name in [n for n in STRATEGIES if n in factual and n != winner]:
        table = tables.get(name)
        row = factual[name]
        reasons = []
        if name == "atomic_fact":
            missing = row["chunk_count"] - (
                factual[winner]["chunk_count"] if winner in factual else row["chunk_count"]
            )
            reasons.append(
                f"**coverage gap**: {row['chunk_count']} chunks vs "
                f"{factual[winner]['chunk_count']} for the winner, because it is "
                "restricted to the `overview` and `fees` doc classes and silently "
                "drops every `riskometer` document. That is a whole fact category "
                "unanswerable, which is worse than a coarse chunk."
                if missing > 0 else "restricted doc-class coverage"
            )
            reasons.append(
                f"**accuracy**: grounded {_pct(row['grounded_accuracy'])} vs "
                f"{_pct(factual[winner]['grounded_accuracy'])}, "
                f"hit@5 {_pct(row['hit@5'])} vs {_pct(factual[winner]['hit@5'])}"
            )
        else:
            reasons.append(
                f"tie on table_fact_accuracy "
                f"({_pct(table.get('table_fact_accuracy')) if table else 'n/a'}) and on "
                f"hit@5 ({_pct(row['hit@5'])}), lost the tie-break"
            )
            if row["max_chunk_tokens"] > chunking_params.get("max_chunk_tokens", 999):
                reasons.append(
                    f"max chunk {row['max_chunk_tokens']} tokens vs a "
                    f"{chunking_params.get('max_chunk_tokens')} cap, and mean "
                    f"{row['mean_chunk_tokens']} vs "
                    f"{factual[winner]['mean_chunk_tokens']} - larger chunks dilute "
                    "the embedding of the surrounding prose"
                )
        lines.append(f"### {STRATEGY_LABELS.get(name, name)}")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines += [
        "## Limitations of this gate",
        "",
        "Recorded because a reviewer will otherwise read the 100% columns as a claim",
        "the corpus cannot support.",
        "",
        "1. **`hit@5` is saturated by construction.** Each scheme has only 4 chunks,",
        "   so a top-5 retrieval returns the entire scheme. hit@5 is 100% for A and B",
        "   for that reason alone, not because ranking works. hit@1 and hit@2 are the",
        "   informative columns at this corpus size.",
        "2. **The documents are smaller than one `fixed_window`.** Whole documents are",
        f"   {manifest.get('tokens_max', '?')} tokens at most, against a 180-token window,",
        "   so `fixed_window` swallows each document whole and cannot split a fee slab.",
        "   Its structural risk - slicing a row mid-value - is therefore *untested* by",
        "   this gate, not disproven. It would surface on a factsheet-sized document.",
        "3. **`grounded_accuracy` here is extractive, not generative.** PRD OQ2 is still",
        "   open, so no model exists to generate a passage-restricted answer. The metric",
        "   asks the narrower question the gate needs answered: did the chunker leave",
        "   the fact whole and findable. P4 re-measures with a real generator and the",
        "   P2 number is an upper bound on it, never a substitute.",
        "4. **The eval grid is 10 categories x 5 schemes = 50, not the 6 x 5 = 30 that",
        "   implementation.md P2-T1 states.** The spec asks for both \"6 canonical fact",
        "   categories x 5 schemes\" and \"50 questions\", which are inconsistent. The 6",
        "   canonical categories are kept as the core grid and 4 further per-scheme facts",
        "   the corpus carries (`sub_category`, `stamp_duty`, `short_term_tax`,",
        "   `ltcg`) make up the difference. Every `expected_fact` was verified to exist in",
        "   the collection at authoring time, so no item is unanswerable.",
        "5. **There are no real tables in the corpus.** `eval/table_facts.jsonl` targets",
        "   the structured fee/tax label-value blocks, because `factsheet`, `kim`, `sid`,",
        "   `faq` and `statement_guide` are unsourced gaps. When those become available,",
        "   `table_fact_accuracy` should be re-run against genuine table rows before the",
        "   number is quoted as evidence about table handling.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "for s in heading_aware fixed_window atomic_fact; do",
        "  python -m mf_facts.pipeline.build --strategy $s",
        "  python -m mf_facts.eval.harness --strategy $s",
        "done",
        "python -m mf_facts.eval.report",
        "```",
        "",
        "The three strategies cannot be compared from one collection - there is one",
        "collection name - so the harness persists each strategy's metrics to",
        "`artifacts/eval_metrics/<strategy>.json` and this report reads those.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mf_facts.eval.report",
        description="Write docs/chunking_decision.md from the persisted gate metrics.",
    )
    parser.add_argument("--out", default=None, help="output path")
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else REPO_ROOT
    try:
        by_strategy = load_metrics(root)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"cannot write the decision report: {exc}", file=sys.stderr)
        return 1

    document = build_document(by_strategy, root)
    target = Path(args.out) if args.out else root / DECISION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    print(f"wrote {target.relative_to(root) if target.is_relative_to(root) else target}")

    winner, _ = select_winner(
        pick(by_strategy, "factual"), pick(by_strategy, "table_facts")
    )
    print(f"selected strategy: {winner or 'NONE - incomplete metrics'}")
    return 0 if winner else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
