# Implementation Guide — MF-Facts-Bot

| Field | Value |
|---|---|
| Document | Phase-wise Implementation Runbook v1.0 |
| Purpose | Drive implementation phase-by-phase with an AI coding agent (Cursor) |
| Authoritative for | **What to build, in what order, and how to verify each phase** |
| Subordinate to | `architecture.md` (component contracts) and `PRD.md` (requirements) |
| Total phases | 7 (P0 → P6) |

---

## 1. How to use this document

This runbook is written to be executed in order. Each phase is a self-contained unit of work with its own file list, task list, tests, and acceptance gate.

**Recommended workflow, repeated per phase:**

1. Open Cursor on the repo root.
2. Paste the phase's **Cursor prompt block** (§ per phase) as the instruction.
3. Let the agent implement the phase's tasks **in task-ID order**.
4. Run the phase's **acceptance commands** yourself and confirm each passes.
5. Fix in place until green. Do not proceed while red.
6. Check off the **Definition of Done**. Only then move to the next phase.

**Precedence rule.** If a conflict arises, this order wins:

```
architecture.md  (component contracts, algorithms, invariants)
    ↓
PRD.md          (requirements, constraints, acceptance criteria)
    ↓
implementation.md (this file — order, tasks, verification only)
```

This document never invents behaviour. Where it names an algorithm, threshold, or interface, it is quoting `architecture.md`; where it names a requirement, it is quoting `PRD.md`. **If a task here contradicts either document, the higher document wins — fix this file, not the architecture.**

### 1.1 Phase map

| Phase | Name | Output | Gate |
|---|---|---|---|
| **P0** | Scaffold & Safety Foundations | Runnable skeleton, config, models, redaction, test harness | `pytest` runs, config loads, no secrets |
| **P1** | Offline Pipeline (Load → Chunk → Embed → Store) | Built corpus + manifest + `sources.csv` | Manifest consistent with Chroma; 5 schemes indexed |
| **P2** | Chunking Strategy Decision | `docs/chunking_decision.md` | **OQ1 resolved and documented** |
| **P3** | Query Pipeline (Safety & Routing) | PII scan, classifier, refusal, rewriter | 100% on refusal + PII eval sets |
| **P4** | Retrieval, Grounding, Generation, Validation | Full RAG answer path + `docs/p4_answer_eval.md` | 8/8 validator checks fire; hit@5 ≥95% |
| **P5** | API + Minimal UI | Working prototype: `api/app.py`, `ui/`, `assets/disclaimer.txt` | Meets `PRD.md` FR7 checklist |
| **P6** | Deliverables, Eval Report, Demo | Submission package | `PRD.md` §13 checklist complete |

> **Delta from `architecture.md` §18:** that table had 6 phases. Here the API and UI are split into P5, and validation is separated from P3/P4 by safety-first ordering. P3 contains *no* generation, so a green P3 means the safety rails are proven before any text is ever produced.

### 1.2 Hard dependencies

```
P0 ──► P1 ──► P2 ──┬──► P3 ──► P4 ──► P5 ──► P6
                    │          (P3 needs the final collection, P2 locks its chunking)
                    └── P2 gates P4: retrieval tuning against the wrong
                        chunking wastes the work and must be redone.
```

P3 also needs a collection to filter against, so **P1 must be green before P3**. P2 (chunking decision) must be green before P4, because retrieval quality is measured against chunks.

---

## 2. Ground rules

Non-negotiable. An implementation that violates one of these is wrong even if it passes the tests.

| # | Rule | Source |
|---|---|---|
| GR1 | No module writes to a Chroma collection except `store.py`. | arch §5.8 |
| GR2 | No chunk may exceed `chunking.max_chunk_tokens` (default 220), measured with the **model's own tokenizer**. Truncation is a hard error, not a silent fix. | arch §7.1, D2 |
| GR3 | Every chunk carries all 11 metadata fields (`PRD.md` §8.1). | arch §5.6 |
| GR4 | Chunks from different schemes are never merged. | arch §5.6 |
| GR5 | Citation URL is read from chunk metadata and attached **after** generation. Any URL emitted by the model is discarded. | arch §10.2, D4 |
| GR6 | The `performance` class is terminal — it never reaches retrieval. | arch §5.12, D5 |
| GR7 | PII: on detection the query value is **discarded**, not masked or forwarded. Nothing downstream of `pii.py` may see it. | arch §5.11, D6 |
| GR8 | Refusals are templated, not LLM-generated. | arch §5.12, AD6 |
| GR9 | No LLM output reaches the user without passing all 8 validator checks. | arch §5.18, D7 |
| GR10 | Any validator failure produces a safe response, never the raw model output. | arch §5.18 |
| GR11 | Logs contain no raw query text by default. Redaction is a formatter-level backstop, not a per-callsite convention. | arch §14 |
| GR12 | Page-structure assumptions live in `config/sources.yaml` locators only — never inline in parser code. | arch §5.1 |
| GR13 | No tool in the generation path. The generator has no network, no filesystem, no tools. | arch §10.3 |
| GR14 | Tests never hit the network. Corpus tests run against fixtures. | arch §16 |
| GR15 | Every phase ends with the acceptance commands actually run and shown passing. A phase with unrun tests is not done. | this doc |

### 2.1 STOP conditions

Halt and escalate to the user rather than improvising:

| Trigger | Why |
|---|---|
| Fetching all 5 scheme pages yields no usable content (auth wall, layout change, region blocked) | The corpus premise fails; locators must be re-derived with the user. |
| Fact sheets / KIM / SID PDFs cannot be located or parsed at all | Ingestion scope shrinks; PRD §5.2 doc classes must be renegotiated. |
| **OQ2 (LLM provider) still undecided when P3 starts** | P4 needs a real generator. Do not hardcode a provider in code — keep it a config key and implement a `FakeLLM` for tests. |
| A guard fires repeatedly (chunk over cap, model mismatch, zero docs for a scheme) | These are designed to stop the build, not be worked around. |
| Any task would require storing user PII | Regulatory and PRD Non-Goal violation. |
| Phase scope would exceed ~1 day of work | Split the phase and confirm the split with the user. |

---

## 3. Global conventions

Apply in every phase. These are the defaults Cursor will otherwise reinvent inconsistently.

### 3.1 Package & naming

- Package root: `src/mf_facts/`, importable as `mf_facts`. Editable install via `pyproject.toml`.
- Module names match `architecture.md` §9 exactly. Do not rename files to suit local preference.
- One responsibility per module. If a module needs the words "and" in its description, split it.
- All cross-module data crosses as a **dataclass** defined in `common/models.py`. No bare dicts between layers.
- Private helpers `_prefixed`. Public functions get docstrings stating input, output, and failure behaviour.

### 3.2 Errors

- Domain failures raise typed exceptions from `common/errors.py`: `ConfigError`, `FetchError`, `ParseError`, `ChunkSizeError`, `EmbeddingModelMismatch`, `NoDocumentsForScheme`, `CollectionUnavailable`, `LLMError`.
- The **query** path (P3–P5) must never propagate a raw exception to the client. It converts to a safe response or a 503 with `service_unavailable` (arch §11).
- The **build** path fails loudly and non-zero. A partially built corpus must never be published.

### 3.3 Logging & redaction

- One `setup_logging(level)` in `common/logging.py`; a formatter installs `common/redaction.py::redact` on every record.
- `log_event(name, **fields)` is the only sanctioned logging call for query events. It enforces the field allowlist from arch §14: `query_hash, route, class, rule_id, retrieved_ids, validator_flags, latency_ms, pii_detected, pii_categories, grounding_score, no_answer`.
- Anything not on the allowlist is dropped, so a careless `log_event(x, query=...)` is a no-op rather than a leak.

### 3.4 Config

- `AppConfig` is a frozen dataclass tree loaded once at startup by `common/config.py::load_config(path)`.
- Every threshold from arch §8 lives in `config/config.yaml` and nowhere in code. Tests override via a fixture, not by monkeypatching module constants.
- Fail fast on unknown keys (strict parse) so a typo'd config key is not silently ignored.

### 3.5 Testing

- `pytest`, with `tests/conftest.py` providing: `tmp_config`, `fixture_corpus_dir` (small HTML/PDF fixtures), `fake_embedder` (deterministic 384-dim vectors), `fake_llm`, `chroma_tmp` (in-memory Chroma), and `pii_eval_cases`.
- Markers: `@pytest.mark.contract` for the structural promises (metadata completeness, idempotency), `@pytest.mark.adversarial` for the safety eval sets.
- Every refusal/PII/format check must be table-driven from a JSON file in `eval/` so the sets are data, not code.

### 3.6 Definition of done, every phase

```
[ ] Acceptance commands run and passing, output shown
[ ] `pytest` green with no new warnings
[ ] No secrets, tokens, or absolute machine paths committed
[ ] Config-driven: no new magic numbers in code
[ ] Docstrings present on public functions
[ ] Phase's tasks ticked in this document
[ ] Cursor prompt block updated to reflect what actually shipped
```

---

## 4. P0 — Scaffold & Safety Foundations

**Goal.** A runnable, empty-but-wired skeleton where the config loads, the package imports, tests run, and the PII regex table + log redaction exist. Nothing is retrieved and nothing is answered yet.

**Deliberately first:** the redaction layer must land before the first real query (arch §18 P0 note). Doing it later risks a PII value in a log file.

**Architecture refs.** §9 (layout), §8 (config), §14 (security), §5.11 (pattern table), §3 (tracing).

### Tasks

| ID | Task | File |
|---|---|---|
| `P0-T1` | Create the `architecture.md` §9 directory tree, minus the modules added in later phases. Add `.gitignore`: `raw_cache/`, `cache/`, `chroma/`, `.env`, `artifacts/` (except the three named deliverables), `__pycache__/`, `*.pyc`, `.pytest_cache/` | repo root |
| `P0-T2` | `pyproject.toml`: package `mf-facts-bot`, src layout, `pytest` + `pytest-cov` dev extras, `python>=3.10`. `requirements.txt` pinning only what P0/P1 need: `sentence-transformers`, `chromadb`, `pydantic`, `pyyaml`, `beautifulsoup4`, `pypdf`, `httpx` | root |
| `P0-T3` | `config/config.yaml` transcribed **verbatim** from `architecture.md` §8, with `generation.provider`/`model` left as `<TBD: OQ2>` and the two `_split_runs` keys documented as comments | `config/` |
| `P0-T4` | `common/errors.py` — the exception classes from §3.2 | `src/mf_facts/common/` |
| `P0-T5` | `common/models.py` — the frozen dataclasses from `architecture.md` §5.1, §5.3, §5.4, §5.6, §8.2. **These are the contracts every later phase codes against**; get the field names exactly right | `common/` |
| `P0-T6` | `common/pii_patterns.py` — the six categories from arch §5.11 as named compiled patterns, each with a `CATEGORY` id used later as the logging `rule_id`. Unit-test the PAN and Aadhaar patterns hardest | `common/` |
| `P0-T7` | `common/redaction.py` — `redact(text) -> (clean_text, categories)` using `pii_patterns`. The redactor is **independent of** the query-time scanner: this one protects logs, that one protects the pipeline | `common/` |
| `P0-T8` | `common/logging.py` — `setup_logging`, `log_event` with the §3.3 field allowlist, formatter redactor. `log_event(query=...)` must be dropped, not logged | `common/` |
| `P0-T9` | `common/config.py` — `load_config(path) -> AppConfig`, strict parsing, defaults from `config.yaml`, `ChunkingConfig.strategy` validated against the three legal values | `common/` |
| `P0-T10` | `tests/conftest.py` — all fixtures from §3.5, with `fake_llm` returning fixed strings and `fake_embedder` returning seeded deterministic vectors | `tests/` |
| `P0-T11` | `tests/test_pii_patterns.py`, `tests/test_redaction.py`, `tests/test_config.py`, `tests/test_logging_no_leak.py` (assert a PAN in a log call never reaches the handler) | `tests/` |
| `P0-T12` | `.env.example` naming the LLM provider env vars only, no values | root |

### Acceptance

```bash
pip install -e ".[dev]"
python -c "from mf_facts.common.config import load_config; print(load_config('config/config.yaml'))"
pytest -q
```
All three must succeed.

### Definition of done

- [ ] P0-T1 … P0-T12 complete
- [ ] `common/models.py` field names match `PRD.md` §8.1 exactly
- [ ] A PAN string passed to `log_event` provably does not appear in captured log output
- [ ] No `TODO`/placeholder left except the two `<TBD: OQ2>` config values

### Pitfalls

- **Do not** build Chroma or load the embedding model in P0. Both are slow; they land in P1.
- **`log_event` allowlist must be a deny-by-default filter**, not an allowlist of what callers pass.
- `pyproject.toml` must be `[tool.setuptools.packages.find] where = ["src"]` or the editable install silently pulls nothing.

### Cursor prompt block

```text
Implement phase P0 of implementation.md in this repo (read implementation.md §4 and
architecture.md §9, §8, §14 first).

Rules: execute tasks P0-T1 through P0-T12 in order. Field names in common/models.py
must match PRD.md §8.1 and architecture.md §5.1/§5.3/§5.4/§5.6/§8.2 exactly — do not
rename or add fields. Config values come from architecture.md §8 verbatim; do not
introduce any new config key. Do not install or import chromadb or
sentence_transformers yet. Do not create files for phases P1+.

Two rules are non-negotiable: (1) log_event must drop any field not on the allowlist,
so a call like log_event(x, query="...") logs nothing sensitive; (2) redaction must be
installed in the logging formatter so it applies to every record regardless of callsite.

When done, run: pytest -q and report the output. Do not claim success without it.
```

---

## 5. P1 — Offline Pipeline (Load → Chunk → Embed → Store)

**Goal.** The four required stages run end-to-end and produce an inspectable corpus. This is the phase that proves RAG stage 1–4 (PRD G7).

**Architecture refs.** §5.1–§5.9, §6.1, §7.1–§7.3, §8.2.

### 5.0 Sub-step: seed the source registry first

**Do this before writing fetchers.** The registry is the spec for everything downstream, and seeding it forces the corpus questions to surface now rather than in week three.

1. Populate `config/sources.yaml` with the 5 scheme URLs from `PRD.md` §5.1, one `SourceSpec` per `(scheme_key, doc_class)` for `overview`.
2. Determine which additional doc classes are reachable. On aggregator scheme pages, `fees`, `faq`, `riskometer` and `statement_guide` are usually **sections of the same page, not separate URLs**.

> **Implementation rule that follows from this:** one `SourceSpec` per `(scheme_key, doc_class)`, and **multiple specs may share one URL with different `locator` values.** A page is fetched once (content-addressed cache dedupes it); the parser is then invoked once per spec with that spec's locator. Do not model this as "one URL = one document" — it is the most common structural mistake in this phase, and it silently collapses `doc_class` metadata that retrieval depends on.

3. Factsheet / KIM / SID are separate PDFs. Locate them, or record in `sources.yaml` that they are unavailable and escalate per §2.1 if none are.
4. For any `(scheme, doc_class)` where both an AMC-hosted and an aggregator-hosted page exist, the AMC-hosted spec wins (`PRD.md` §5.3, OQ3).
5. Resolve `last_updated` per spec from the page itself where visible; otherwise the spec value; the normalizer records which.

### Tasks

| ID | Task | File | Key detail |
|---|---|---|---|
| `P1-T1` | Source registry: load `sources.yaml` → `List[SourceSpec]`, validate uniqueness of `source_id`, apply precedence, mark duplicates | `pipeline/sources.py` | `load_sources(path) -> list[SourceSpec]`; `write_sources_csv/md(specs, runs) -> None` emitting the 11 columns of arch §5.1 |
| `P1-T2` | Fetcher with content-addressed cache | `pipeline/fetcher.py` | `RawDocument` per spec. Cache at `raw_cache/{hash[:2]}/{hash}.{ext}`. `--refresh` to bypass. Non-200 → `status=failed`, continue |
| `P1-T3` | HTML parser | `pipeline/parsers.py` | `HtmlParser.parse(raw, spec)`. Apply `spec.locator` first, else `main`/`article` fallback. Strip `script, style, nav, footer`, cookie banners. Preserve heading hierarchy for strategy A. Run **once per spec** for specs sharing a URL |
| `P1-T4` | PDF parser | `pipeline/parsers.py` | `PdfParser.parse(raw, spec)`. Per-page text; headings from font-size deltas; **tables emitted pipe-delimited** (arch §7.3). Flag ragged rows |
| `P1-T5` | Normalizer | `pipeline/normalizer.py` | Whitespace/unicode/₹ normalization; ISO-8601 date extraction; `last_updated` precedence chain; record `last_updated_source ∈ {document, spec, retrieved_at}`. **Never alter numbers** |
| `P1-T6` | Deduper | `pipeline/deduper.py` | SHA-256 exact + section-signature near-dup (same scheme/doc_class/section, ≥0.95 token Jaccard). Keep higher-precedence copy, record `duplicate_of` |
| `P1-T7` | Chunker interface + guard | `pipeline/chunker.py` | `ChunkingStrategy` Protocol; `Chunker` base with the GR2 token assertion and the GR3 metadata assertion. Token counting via `AutoTokenizer` |
| `P1-T8` | Strategy A | `pipeline/strategies/heading_aware.py` | Section tree → tables first (header row repeated per part) → recursive split on `[. ]` at 160 tokens, 15% overlap → prefix section heading to chunk text |
| `P1-T9` | Strategy B | `pipeline/strategies/fixed_window.py` | 180-token window, 160 stride. Baseline only. Must still not split a table row (GR2 analogue for B: table chunks are protected even in the baseline) |
| `P1-T10` | Strategy C | `pipeline/strategies/atomic_fact.py` | For `overview` + `fees` only. One chunk per labelled fact: `"<label>: <value>. <verbatim sentence>. <section>. <scheme>"`. Record coverage so the gap is visible |
| `P1-T11` | Embedder | `pipeline/embedder.py` | `all-MiniLM-L6-v2`, batch 32, `normalize_embeddings=True`, assert dim 384. Cache by `sha256(model_id + text)` |
| `P1-T12` | Store | `pipeline/store.py` | Collection `mf_facts_v1`. Deterministic ids `f"{scheme_key}:{doc_class}:{hash[:12]}:{chunk_index}"`. `to_chroma_metadata()` — never `None`, use `""`. Model-mismatch guard on init. All Chroma access in this file (GR1) |
| `P1-T13` | Manifest writer | `pipeline/manifest.py` | Emit the `PRD.md` §8.2 schema + build report (doc counts, dedupe count, chunk count, token min/mean/max, per-scheme counts, duration) |
| `P1-T14` | Build CLI | `pipeline/build.py` | `python -m mf_facts.pipeline.build --strategy heading_aware [--refresh] [--limit N]`. Emit manifest + `artifacts/sources.csv` + `artifacts/sources.md`. Hard-fail per arch §6.1 |
| `P1-T15` | Tests | `tests/` | contract: metadata completeness on every chunk; idempotent re-ingest (chunk count stable); table row never split; chunk token cap; no cross-scheme chunk; manifest↔Chroma consistency. Use fixtures, never network |

### Acceptance

```bash
python -m mf_facts.pipeline.build --strategy heading_aware --refresh
python -c "import json;m=json.load(open('artifacts/corpus_manifest.json'));print(m['doc_count'],m['chunk_count'],m['tokens_max'])"
pytest -q -m contract
```

Pass conditions:

- All 5 `scheme_key`s present in `chunks_per_scheme` with non-zero counts
- `tokens_max <= 220`
- `chunk_count` **identical** on a second run with no source change
- `artifacts/sources.csv` has 11 columns and one row per `SourceSpec`, including failed ones
- Spot-check: the expense-ratio and exit-load facts for 2 schemes are present in some chunk's text, unsplit

### Definition of done

- [ ] P1-T1 … P1-T15 complete
- [ ] Manifest matches the collection exactly
- [ ] All four stages individually inspectable (verbose logging per stage; `--stage` flag to run one stage)
- [ ] `sources.csv` regenerated from the run, not hand-written

### Pitfalls

- **Locator-vs-URL conflation** — see §5.0. This is the highest-risk structural mistake in the phase.
- **Chroma `None` metadata** — a `None` value raises at write time. The `to_chroma_metadata()` helper must exist before the first insert.
- **Token counting with `len(text.split())`** — undercounts vs. the WordPiece tokenizer and will let chunks silently exceed 256 and get truncated by the model. Use the model's tokenizer (GR2).
- **PDF tables** — if rows come out ragged, flag and exclude from table chunks rather than emitting a garbled fee slab. A garbled table is worse than a missing one, because it can produce a wrong number.
- **Scope discipline** — no retrieval, no API, no UI. If the agent starts writing `online/`, stop it.

### Cursor prompt block

```text
Implement phase P1 of implementation.md (read implementation.md §5 and architecture.md
§5.1–§5.9, §6.1, §7.1–§7.3, §8.2 first).

Order: (a) seed config/sources.yaml for the 5 schemes from PRD.md §5.1, one SourceSpec per
(scheme_key, doc_class), multiple specs may share a URL with different locators — the page is
fetched once and parsed once per spec; (b) then tasks P1-T1 through P1-T15 in order.

Non-negotiable (implementation.md §2): all Chroma access only in store.py; token counts via
the all-MiniLM-L6-v2 tokenizer, never len(text.split()); every chunk carries all 11 metadata
fields; no chunk from two schemes; tables never split mid-row; page-structure assumptions
only in sources.yaml locators.

If fetching any scheme page fails, do not work around it — stop and report which specs
failed (implementation.md §2.1). Do not implement anything from P2+.

When done run the P1 acceptance commands and paste the output. Report per-scheme chunk
counts and tokens_max explicitly.
```

---

## 6. P2 — Chunking Strategy Decision

**Goal.** Resolve PRD OQ1 with evidence, and write the decision down. This phase is a **gate**, not a code-writing phase.

**Architecture refs.** §5.10, §7.2, §7.6.

**Why it is a separate phase:** retrieval quality is measured against chunks. Tuning retrieval against the wrong chunking wastes P4 and forces a rebuild. Lock the chunking first.

### Tasks

| ID | Task | File |
|---|---|---|
| `P2-T1` | Author `eval/factual.jsonl` — 50 questions: 6 canonical fact categories × 5 schemes, each with `question`, `scheme_key`, `category`, `expected_fact`, `expected_url`, `numeric: bool`. **Write these by reading the built corpus**, not from memory — a question whose answer is not in the corpus is an unanswerable eval item | `eval/` |
| `P2-T2` | Also author `eval/table_facts.jsonl` — 20 questions whose answers are **inside tables** (exit-load slabs, fee schedules, minimum-investment rows). This is the highest-severage failure mode and gets its own metric | `eval/` |
| `P2-T3` | Harness core: `evaluate(collection, eval_set) -> metrics` with `hit@k`, `grounded_accuracy`, `table_fact_accuracy`, `mean/max_chunk_tokens`, `build_time_s` | `eval/harness.py` |
| `P2-T4` | `grounded_accuracy` = generation restricted to the passage, compared to `expected_fact` by normalized string/containment match on the key value. For a numeric answer, compare the extracted number exactly. No LLM judge — it must be deterministic | `eval/harness.py` |
| `P2-T5` | Runner: `python -m mf_facts.eval.harness --strategy {a,b,c} [--only table_facts]` writing a metrics row per strategy | `eval/harness.py` |
| `P2-T6` | Report generator → `docs/chunking_decision.md`: the metric table for A/B/C, the selection, the parameters, the rationale, and **the rejected options with their numbers** | `eval/report.py` |
| `P2-T7` | Lock the winner as `chunking.strategy` in `config/config.yaml`; record the decision in the manifest schema note | config |

### Acceptance

```bash
for s in heading_aware fixed_window atomic_fact; do
  python -m mf_facts.pipeline.build --strategy $s
  python -m mf_facts.eval.harness --strategy $s
done
python -m mf_facts.eval.report
```

- `docs/chunking_decision.md` exists with a filled metric table for all three
- Selection follows the arch §7.6 priority: **`table_fact_accuracy` first**, `hit@5` second, simplicity third
- `atomic_fact`'s coverage gap is documented (it is restricted to `overview` + `fees`)
- The final config value matches the documented selection

### Definition of done

- [ ] P2-T1 … P2-T7 complete
- [ ] Decision documented with numbers, not adjectives
- [ ] **OQ1 closed**

### Pitfalls

- **Do not** hand-tune the eval questions toward one strategy. 50 items are also your regression suite; biasing them invalidates P4.
- **`atomic_fact` is expected to win on numeric precision and lose on coverage.** Both halves belong in the report — that is the trade-off a reviewer needs to see.
- If `fixed_window` ties on `table_fact_accuracy`, prefer `heading_aware` anyway (better coverage of prose facts); the tie-break rule is written in §7.6 step 3.
- Rebuild the winning collection after locking, so the stored corpus matches the documented decision.

### Cursor prompt block

```text
Implement phase P2 of implementation.md (read implementation.md §6 and architecture.md
§5.10, §7.2, §7.6).

Build the corpus and the eval sets first; then implement P2-T3 through P2-T7. Write
eval/factual.jsonl by reading the actual built chunks — every expected_fact must be
verifiable in the corpus, and every expected_url must be a real source_url in the
collection. Then run all three strategies and produce docs/chunking_decision.md with the
filled metric table.

Selection rule, in order: table_fact_accuracy, then hit@5, then build simplicity. Do not
use an LLM judge in the harness — the scoring must be deterministic. Do not change any
code outside eval/ and config (except a chunker bug you find, which you must report
explicitly rather than fix silently).

Report the three-way metric table in your final message.
```

---

## 7. P3 — Query Pipeline: Safety & Routing

**Goal.** PII scanning, query classification, templated refusals, and alias rewriting — with **no generation at all**. A green P3 means every unsafe path is closed before any text is produced.

**Architecture refs.** §5.11–§5.14, §10.4, §7.4.

### Tasks

| ID | Task | File | Key detail |
|---|---|---|---|
| `P3-T1` | `eval/pii.jsonl` — 12 cases: PAN, Aadhaar, 10-digit account, OTP, email, phone, plus PII embedded mid-question | `eval/` | Include the "my PAN XXXXX1234F, what is the expense ratio?" case explicitly |
| `P3-T2` | `eval/refusal.jsonl` — ≥20 opinionated + portfolio cases: "should I buy", "is X safe", "best fund", "which is better", "how much should I invest", "is my portfolio balanced", retirement/child-goal phrasings | `eval/` | Include indirect phrasings, not just keyword-stuffed ones |
| `P3-T3` | `eval/performance.jsonl` — 10 return/CAGR/comparison asks. Assert the response contains **no** numeric return pattern | `eval/` | "Which of these five performed best?" is a `performance` question, not `opinionated` — order matters |
| `P3-T4` | `eval/out_of_scope.jsonl` — 5: a different AMC, a non-MF topic, and 2 with no scheme token at all | `eval/` | |
| `P3-T5` | PII scanner | `online/pii.py` | `PiiScanner.scan(query) -> PiiResult{is_pii, categories, sanitized_query}`. On hit, `sanitized_query == ""`. Value **discarded**, never forwarded (GR7) |
| `P3-T6` | Classifier | `online/classifier.py` | `Classifier.classify(query) -> Classification{cls, rule_id, confidence}`. Stage 1 rules in the arch §5.12 order: pii → performance → opinionated → portfolio_personal → out_of_scope. Stage 2 LLM residual, 3 labels, temperature 0, JSON-only, invalid JSON → `out_of_scope` (fail-closed) |
| `P3-T7` | Educational link registry + refusal composer | `online/refusal.py`, `config/educational_links.yaml` | Templated refusals only (GR8). Structure per arch §5.13: boundary sentence + optional redirect + `Educational link:` + facts-only footer. `performance` links the **scheme's official factsheet** from the registry. Resolve OQ6 URLs with verified targets — an unverified link in a refusal is a defect |
| `P3-T8` | Rewriter | `online/rewriter.py` | `config/aliases.yaml` with every alias row from arch §5.14 for the 5 schemes, plus `hdfc` alone → all 5 keys. `RewrittenQuery{text, scheme_keys, doc_class_hints}`. Hints are **soft ranking boosts only**, never a hard filter |
| `P3-T9` | Pipeline skeleton | `online/ask.py` | Wire pii → classify → refusal terminal branches. For `factual_scheme`/`how_to`, **return a typed `NOT_YET_IMPLEMENTED` marker** — do not call any unbuilt module |
| `P3-T10` | Tests | `tests/` | `test_pii_discards_value` (asserts the raw value never reaches a downstream callable — pass a spy), `test_classifier_rule_priority`, `test_refusal_templates_have_link_and_footer`, `test_no_llm_call_on_refusal_paths` (mock the generator, assert not called), `test_rewriter_alias_coverage` |

### Acceptance

```bash
pytest -q -m adversarial
pytest -q tests/test_pii_discards_value.py tests/test_no_llm_call_on_refusal_paths.py
pytest -q tests/test_no_pii_at_rest.py   # scans repo, caches, logs, artifacts for PII patterns
```

- `pii.jsonl` → 12/12 `class=pii`, 0 stored values
- `refusal.jsonl` → ≥20/20 correct class, all refusals carry exactly one educational link + footer
- `performance.jsonl` → 10/10 `class=performance`, zero numeric return patterns in responses
- `out_of_scope.jsonl` → 5/5
- `test_no_pii_at_rest.py` clean

### Definition of done

- [ ] P3-T1 … P3-T10 complete
- [ ] Refusal correctness 100% on all four eval sets
- [ ] Zero generator invocations on any refusal path (asserted, not assumed)
- [ ] OQ6 closed

### Pitfalls

- **Classifier ordering is a safety property.** A query like "Should I buy HDFC Large Cap for long-term returns?" matches both `opinionated` and `performance`. With `performance` checked first, the response is a factsheet pointer with no numbers — the safer outcome. Changing the order changes the safety profile; if the agent reorders, require a re-run of the adversarial sets.
- **Fail-closed on classifier parse failure.** Defaulting a failed parse to `factual_scheme` would route junk into generation.
- **Aliases must cover real phrasings.** "80C", "tax saver", "elss", "flexicap", "largecap", "baf" — a missing alias is an `out_of_scope` false positive, which the user experiences as "this bot is broken".
- **PII scanner must run before logging the query.** If `pii.py` is imported after the log call, GR7 is violated. Order the imports and the call explicitly.
- **Do not** implement `generator.py` in this phase, even partially. Keeping generation out of P3 is what makes this phase a real gate.

### Cursor prompt block

```text
Implement phase P3 of implementation.md (read implementation.md §7 and architecture.md
§5.11–§5.14, §7.4, §10.4). No generation in this phase — do not create online/generator.py,
online/retriever.py, or online/validator.py.

Execute P3-T1 through P3-T4 first (the four eval JSONL files, written before the code they
test), then P3-T5 through P3-T10.

Non-negotiable: the PII scanner discards the value (sanitized_query is empty on a hit, and
the raw string is never passed to any later component); the classifier's stage-1 rule order
is exactly architecture.md §5.12 and must not be reordered; refusals are templated and
contain no LLM output; the performance class is terminal and never reaches retrieval; the
rewriter's doc_class hints are soft boosts, never filters.

For online/ask.py, the factual_scheme and how_to branches must return a typed
NOT_YET_IMPLEMENTED marker. Do not stub in a fake answer.

When done, run pytest -q -m adversarial and paste the per-file-set pass counts. If any eval
set is below 100%, report which items failed and why — do not loosen the eval data.
```

---

## 8. P4 — Retrieval, Grounding, Generation, Validation

**Goal.** The full answer path. First factual text produced in the project.

**Architecture refs.** §5.15–§5.18, §7.3, §7.5, §7.7, §7.8, §10.1–§10.3.

**Precondition.** P2 green and the collection rebuilt with the winning strategy.

### Tasks

| ID | Task | File | Key detail |
|---|---|---|---|
| `P4-T1` | Retriever | `online/retriever.py` | `retrieve(query_vec, scheme_keys, top_k=12)`. `where={"scheme_key": {"$in": keys}}` — the hard filter (D3). `1 - distance` as similarity |
| `P4-T2` | Reranker | `online/reranker.py` | RRF over (vector order, BM25 order) with `k=60`, then boosts: `doc_class_hint` +0.15, section-keyword +0.10, numeric-question-with-digits +0.10. Return top-4 plus `citation_url` and `last_updated` taken from `passages[0]` metadata |
| `P4-T3` | Grounding | `online/grounding.py` | Both conditions required: relevance ≥0.15 (or hint match + query term present) **and** fact-anchor present. On fail → `refusal(class=grounding_fail)` linking the official document. **Never** fall back to model knowledge |
| `P4-T4` | Prompts | `online/prompts.py` | The 7-rule system prompt from arch §10.1 **verbatim**. Classifier prompt from §10.4. Passages fenced and labelled as data. Generation path has no tools, no network, no filesystem (GR13) |
| `P4-T5` | Generator | `online/generator.py` | Temperature 0, `max_tokens` 180. Emits answer text only — no URL, no stamp. `NO_ANSWER` → `grounding_fail` refusal. Implement `LLMClient` with the provider behind `generation.provider` config, plus a `FakeLLM` for tests so the pipeline is provable without a key (OQ2) |
| `P4-T6` | Citation + stamp attach | `online/ask.py` | Post-generation: `[Source: {label}]({url})` and `Last updated from sources: {ISO}` built **from chunk metadata**, not from model output (GR5) |
| `P4-T7` | Validator | `online/validator.py` | The 8 checks of arch §5.18, in that fixed order, as pure functions of `(response, context)`. Checks 5–7 run on the final string after any repair. Each fail returns a `safe_response` |
| `P4-T8` | Wire the answer path | `online/ask.py` | factual/how_to → rewrite → retrieve → rerank → ground → generate → `NO_ANSWER`? → attach → validate → respond or safe response |
| `P4-T9` | Tests | `tests/` | `test_validator.py` — 8 checks, each with a hand-built failing input, each asserting the correct safe response. `test_grounding.py` — an unanswerable factual question must refuse, not answer. `test_citation_single` — a model emitting 3 URLs yields exactly 1. `test_no_number_not_in_passage` — a model inventing a number is caught. `test_prompt_injection_passage` — a passage saying "ignore previous instructions" changes nothing and leaks no markers |

### Acceptance

```bash
pytest -q -m contract
pytest -q -m adversarial
python -m mf_facts.eval.harness --strategy <locked> --all
```

- Factual eval: `hit@5 ≥ 95%`, `grounded_accuracy ≥ 90%`
- All 8 validator checks demonstrably fire
- `test_grounding.py` proves the "not in corpus" path refuses
- Injection test: no behaviour change, no leaked fence/system markers
- 0 unsupported numbers in any generated answer

### Definition of done

- [x] P4-T1 … P4-T9 complete
- [x] All `PRD.md` §9.2 targets met or deviation documented — see `docs/p4_answer_eval.md` §4 for the two that could not be measured offline (grounded accuracy and end-to-end latency) and why `FakeLLM` cannot stand in for them
- [x] Validator is a pure, independently testable module — it must run without an LLM

### Pitfalls

- **`citation_url` from `passages[0]` only** (after fusion order). Using the union of candidate URLs is the most common way "exactly one link" breaks.
- **The validator must not import the generator.** Independence is what makes it testable and trustworthy (D7). If the agent makes it a method of `generator.py`, reject.
- **Repairs must not reintroduce violations.** Run checks 5–7 last, on the final string.
- **Watch for a sixth `max_tokens` failure mode:** if the model keeps hitting the token cap mid-sentence, sentence-count check 2 will pass while producing a truncated fragment. Assert the last sentence ends with terminal punctuation.
- **`FakeLLM` must not leak into the app path.** It belongs in tests/ or behind an explicit `provider: fake` that the API refuses to serve in the demo.
- **Do not tune retrieval against the eval sets.** If you find yourself editing `top_k` repeatedly against 50 questions, you are overfitting — record the finding in the eval report instead.

### Cursor prompt block

```text
Implement phase P4 of implementation.md (read implementation.md §8 and architecture.md
§5.15–§5.18, §7.3, §7.5, §7.7, §7.8, §10.1–§10.3). The P2 collection must already exist
with the locked strategy — verify that before starting.

Execute P4-T1 through P4-T9 in order. Transcribe the generator system prompt from
architecture.md §10.1 verbatim (all 7 rules); do not paraphrase it.

Non-negotiable: the citation URL comes only from the top-ranked passage's metadata and is
attached after generation, with every model-emitted URL discarded; the validator is a
standalone module of pure functions that never imports the generator; all 8 checks run in
the architecture.md §5.18 order with checks 5–7 last on the final string; the grounding
check fails closed and never falls back to model knowledge; the generation path has no
tools or network access.

Implement an LLMClient whose provider comes from config, plus a FakeLLM used only by tests
(architecture.md OQ2 is still open). Do not hardcode any provider name in code.

When done, run pytest -q -m contract, pytest -q -m adversarial, and the eval harness, and
paste hit@5, grounded_accuracy, and table_fact_accuracy. Report any target missed rather
than adjusting the eval data.
```

---

## 9. P5 — API + Minimal UI

**Goal.** The demoable prototype.

**Architecture refs.** §5.19, §7.7, `PRD.md` FR7.

### Tasks

| ID | Task | File |
|---|---|---|
| `P5-T1` | `GET /api/health` → service status, collection name, manifest version, corpus build time | `api/app.py` |
| `P5-T2` | `GET /api/examples` → the 3 example questions from config | `api/app.py` |
| `P5-T3` | `POST /api/ask` `{question}` → `{answer, citation, last_updated, route, validator_flags, latency_ms}`. No raw exception ever reaches the client (§3.2); 503 `service_unavailable` on Chroma/LLM failure. `route` is for eval/debug, not a headline UI feature | `api/app.py` |
| `P5-T4` | Rate/size guard on `POST /api/ask` — max question length, trivial per-IP limit. No auth, no sessions, no history persistence (Known Limit #6) | `api/app.py` |
| `P5-T5` | Web UI matching the arch §5.19 layout: welcome line, `Facts-only. No investment advice.`, 3 clickable examples, input with the no-PII reminder, message thread showing answer + one citation + stamp, disclaimer in the footer | `ui/` |
| `P5-T6` | Disclaimer rendered on **every** page load from `assets/disclaimer.txt` = the `PRD.md` §10.5 snippet verbatim | `ui/`, `assets/` |
| `P5-T7` | Show the "Last updated from sources" stamp and the citation as a real link on every factual answer; show the refusal footer on every refusal | `ui/` |
| `P5-T8` | Tests | `tests/` | `test_api.py` — health, examples, ask, 503 on LLM failure, no traceback leakage; `test_ui_contract.py` — assert the page contains the disclaimer, the facts-only note, exactly 3 examples, and the no-PII reminder |

### Acceptance

```bash
pytest -q tests/test_api.py tests/test_ui_contract.py
```

Manual: start the service, ask each of the 3 example questions, then ask one opinionated, one performance, and one PII question. Confirm the answer/refusal shape and that no error text leaks.

### Definition of done

- [x] P5-T1 … P5-T8 complete
- [x] `PRD.md` FR7 checklist satisfied line by line
- [x] Zero client-visible stack traces

Notes from the build, for whoever picks this up next:

- **No web framework.** No framework is named in this guide, none is in
  `pyproject.toml`, and none is installed. `api/app.py` is hand-rolled ASGI, so
  it runs under the already-available `uvicorn` (`pip install -e '.[serve]'`) and
  the whole suite tests in-process with `httpx` — which was already a declared
  dependency for the provider client. A framework would have been a new
  dependency for a three-endpoint app.
- **Rate and size limits live in `config/config.yaml`** under `api:`, per §3.4.
  The per-IP limiter keys on the direct peer address, not `X-Forwarded-For`,
  because a forwarded header is client-controlled and would let a caller reset
  their own budget. Revisit if this is ever put behind a proxy.
- **The embedding model is warmed during ASGI startup.** `Embedder.model` is
  lazy and costs tens of seconds; left lazy, the first *uncached* question paid
  it — 38.6s in the manual run, against a §9.2 target of 8s, while a cached
  question beside it took 11ms. `/api/health` reports `embedding_model_loaded`.
- **The 3rd example question refuses.** `docs/corpus_gaps.md` records why: no
  reachable page has statement-download steps, so the question names no scheme,
  resolves no `scheme_key`, and routes `out_of_scope` before retrieval. Correct
  behaviour, poor demo. The examples were left alone deliberately — see the
  pitfall above.
- **Two defects the manual run caught, both fixed:** the cold-start latency above,
  and a duplicated "Facts-only. No investment advice." line in the
  `out_of_scope` refusal, because `assets/scope_note.txt` already ends with that
  sentence. `tests/test_online_p3.py` now asserts the footer appears exactly once
  in every refusal.

### Pitfalls

- **No conversation history.** Tempting to add, but it contradicts Known Limit #6 and would require scheme carry-over in the rewriter. Out of scope.
- **The 3 examples are fixed in config** — do not randomize or rotate them. The demo depends on them being the 3 high-frequency categories.
- **Don't show `route` as the primary user-facing label.** Exposing internal class names invites confusion about what a refusal means.
- **If the demo must be hosted, resolve OQ5 here, not in P6.**

### Cursor prompt block

```text
Implement phase P5 of implementation.md (read implementation.md §9 and architecture.md
§5.19, PRD.md FR7 and §10.5).

Exactly three endpoints: /api/health, /api/examples, /api/ask. No auth, no sessions, no
persisted history. The ask endpoint must never leak a traceback to the client — convert
every failure to a safe response or a 503 with service_unavailable. Rate/size guard on ask.

Build the UI to the architecture.md §5.19 layout: welcome line, the "Facts-only. No
investment advice." note, exactly 3 clickable example questions, an input box carrying the
do not-enter-PAN/Aadhaar/account/OTP/email/phone reminder, a message thread showing the
answer + exactly one citation link + the "Last updated from sources" stamp, and the
disclaimer snippet from PRD.md §10.5 rendered on every page load from assets/disclaimer.txt.

Do not add scheme comparison tables, dashboards, settings, or accounts.

When done run pytest -q tests/test_api.py tests/test_ui_contract.py and paste the output.
```

---

## 10. P6 — Deliverables, Eval Report, Demo

**Goal.** The submission package. Nothing new is built here; everything is generated or written.

**Architecture refs.** `PRD.md` §10, §13; `architecture.md` §16.

### Tasks

| ID | Task | File | Key detail |
|---|---|---|---|
| `P6-T1` | Regenerate `artifacts/sources.csv` + `artifacts/sources.md` from a clean `--refresh` build | `artifacts/` | 11 columns, every spec including failures. Never hand-edited |
| `P6-T2` | Write `README.md` | root | Setup steps · architecture + the 4 RAG stages with the **P2 chunking decision and its numbers** · scope (HDFC AMC + the 5 schemes) · how to rebuild · how to run the UI · how to run the eval harness · **known limits** (all 8 from `PRD.md` §11) · disclaimer |
| `P6-T3` | Generate `artifacts/sample_qa.md` — 5–10 queries with **verbatim** assistant answers, the single link, and the stamp | `artifacts/` | Cover all 6 fact categories, ≥1 how-to, and ≥1 refusal. Generated from real responses, not written by hand |
| `P6-T4` | Full eval report across all six eval sets, with per-set pass counts and a pass/fail column against every `PRD.md` §9.2 target | `artifacts/eval_report.md` | Any missed target gets an explicit deviation note |
| `P6-T5` | `docs/demo_script.md` — the ≤3-minute video beat sheet: problem (10s) · the 4 RAG stages (45s) · 3 example Q&As (45s) · 1 refusal (15s) · 1 PII refusal (10s) · eval numbers (20s) · limits (15s) | `docs/` | |
| `P6-T6` | Walk `PRD.md` §13 and tick every criterion with the artifact that proves it | this document, appendix | Any untickable criterion is escalated, not quietly dropped |
| `P6-T7` | Final sweep: `pytest -q` green; no secrets; no absolute machine paths; no `raw_cache/`, `cache/`, `chroma/` committed; README setup verified on a **clean clone** | repo | Clean-clone verification is the actual gate for `PRD.md` G6 |

### Acceptance

```bash
git clean -xdn            # inspect what is ignored
pytest -q
python -m mf_facts.pipeline.build --refresh
python -m mf_facts.eval.harness --all
```
Then clone to a temp dir, follow `README.md` top to bottom, and confirm the prototype runs and the UI answers.

### Definition of done

- [ ] P6-T1 … P6-T7 complete
- [ ] All `PRD.md` §13 criteria ticked with a proving artifact
- [ ] Prototype link live, or demo video recorded and within 3 minutes (OQ5)

### Pitfalls

- **`sample_qa.md` must be machine-generated.** Hand-written answers drift from the real system and are the easiest thing for a reviewer to catch.
- **Known limits are not optional.** `PRD.md` §11 exists because a reviewer will ask "how fresh is this?" — answer it in the README, not in the demo video.
- **Verify the README on a clean clone.** A README that only works in your working directory fails `PRD.md` G6.
- **Record deviations rather than hiding them.** A missed `hit@5` target with a written explanation is acceptable; a silently edited eval set is not.

### Cursor prompt block

```text
Implement phase P6 of implementation.md (read implementation.md §10, PRD.md §10, §11, §13).

This phase builds no new features. Execute P6-T1 through P6-T7 in order.

artifacts/sample_qa.md must be generated from real pipeline responses (run the actual
queries and capture verbatim output with its citation and stamp) — never hand-written.
README.md must contain all 8 known limits from PRD.md §11 verbatim in substance, the P2
chunking decision with its metric table, and setup steps you have personally verified.

After generating, walk PRD.md §13 criterion by criterion and, for each, name the artifact
that proves it. Print that table in your final message. If a criterion cannot be evidenced,
say so explicitly instead of claiming it.

Finish by running the clean-clone README verification and reporting the result.
```

---

## 11. Phase gate summary

| Gate | Must be true to advance | Fails if |
|---|---|---|
| P0 → P1 | `pytest` green; config loads; redaction provably blocks PII in logs | Redaction is per-callsite rather than formatter-level |
| P1 → P2 | Manifest ↔ Chroma consistent; all 5 schemes indexed; `tokens_max ≤ 220`; re-ingest idempotent | Any scheme silently absent; any chunk over cap |
| P2 → P3 | `docs/chunking_decision.md` with a real 3-way metric table; config locked to the winner | Decision made by assertion, or eval sets tuned toward one strategy |
| P3 → P4 | 100% on refusal/performance/out-of-scope/PII sets; no generator exists yet | Any safety set below 100%; generation implemented early |
| P4 → P5 | `hit@5 ≥ 95%`; all 8 validator checks fire; grounding refuses the unanswerable | Validator not independently testable; grounding falls back to model knowledge |
| P5 → P6 | `PRD.md` FR7 satisfied; no client-visible tracebacks | Disclaimer not on every load; examples ≠ 3 |
| P6 → submit | `PRD.md` §13 fully evidenced; clean-clone README verified | Hand-written sample answers; clean clone untested |

---

## 12. Traceability — `PRD.md` §13 to phase

| `PRD.md` §13 criterion | Phase |
|---|---|
| Four offline stages run, individually inspectable | P1 |
| `corpus_manifest.json` consistent with the collection | P1 |
| Chunking chosen by evaluation, decision documented | P2 |
| All 5 schemes answerably indexed; 6 fact categories retrievable | P1, verified P4 |
| ≤3 sentences, exactly one link, last-updated stamp | P4 |
| Opinion/portfolio/performance refused; factsheet linked, no numbers | P3, verified P4 |
| PII refused, nothing persisted | P0 (redaction), P3 (scanner) |
| UI: welcome line, 3 examples, facts-only note, disclaimer | P5 |
| `sources.csv` + `sources.md` shipped, regenerable | P1, finalized P6 |
| `README.md` with setup, scope, chunking rationale, known limits | P6 |
| `sample_qa.md` with 5–10 Q&A | P6 |
| Prototype link or ≤3-min video | P5, P6 |
| Eval targets met or deviations documented | P2, P4, P6 |

---

## 13. Appendix A — Seeding checklist for the first build

Before writing any fetcher, confirm these exist. A gap here becomes a silent corpus hole.

- [ ] 5 scheme URLs resolve and return parseable content
- [ ] `scheme_key` values match `architecture.md` §5.14 exactly
- [ ] Per scheme, which of the 8 doc classes are actually reachable, recorded in `sources.yaml`
- [ ] Locators identified for `fees`, `faq`, `riskometer`, `statement_guide` sections on the scheme pages
- [ ] Factsheet/KIM/SID PDF locations, or an explicit record that they are unavailable
- [ ] `last_updated` visible on each page, or the fallback decided
- [ ] AMC-vs-aggregator precedence resolved per `(scheme, doc_class)` — OQ3
- [ ] Educational links verified for all 6 refusal classes — OQ6
- [ ] LLM provider decided, or `FakeLLM` accepted for dev — OQ2
- [ ] Hosting decided for the demo link — OQ5

## 14. Appendix B — Glossary

| Term | Meaning here |
|---|---|
| **Stage (S1–S4)** | The four required offline RAG operations: load, chunk, embed, store |
| **Node ([1]–[20])** | A module in the arch §4 container view |
| **Task (P3-T5)** | A numbered unit of work inside a phase |
| **SourceSpec** | Declarative record of one ingestible `(scheme, doc_class)` region |
| **Chunk** | The retrieval unit: text + 11 metadata fields, ≤220 tokens |
| **Refusal rail** | The set of terminal paths in arch §6.2 that end in a templated refusal |
| **Grounding check** | Pre-generation verification that a passage supports an answer |
| **Validator** | Post-generation deterministic gate; the last thing before the user |
| **Safe response** | The templated substitute returned when a check fails |
| **Class** | One of the 7 query routes in `PRD.md` FR3 |
| **Hard guard** | A condition that fails the build rather than degrading it |

## 15. Appendix C — `PRD.md` §13 acceptance walk (P6-T6)

Walked 2026-09-27 against the fresh `--refresh` build and a live generator
(`qwen/qwen3.8-27b`). A criterion is ticked only if the named artifact proves it.

| | `PRD.md` §13 criterion | Proving artifact | Status |
|---|---|---|---|
| [x] | Four offline stages run end-to-end, individually inspectable | `python -m mf_facts.pipeline.build [--stage …]` log; `artifacts/build_report.json` | met |
| [x] | `corpus_manifest.json` consistent with the collection | `artifacts/corpus_manifest.json` (20 chunks) = Chroma `indexed_chunks` 20 (`/api/health`); `tests/test_store_manifest_contract.py` | met |
| [x] | Chunking chosen by evaluation, decision and numbers documented | `docs/chunking_decision.md` | met |
| [ ] | All 5 schemes answerably indexed; **all 6 fact categories retrievable for each** | `artifacts/eval_report.md` §3.2 | **not met: minimum SIP missing (overview locator). Schemes indexed 5/5; statement download unsourced** |
| [x] | Every factual answer ≤3 sentences, one link, stamp | `artifacts/eval_report.md` §1: 81/81 on all three | met |
| [x] | Opinion/portfolio/performance refused; performance links the official page with no numbers | `artifacts/eval_report.md` §2.3: 24/24, 10/10, 0 figures. Links the scheme page, because the factsheet is unsourced (`docs/corpus_gaps.md`) | met, with factsheet substitution |
| [x] | PII set 100% refused, nothing persisted | 12/12; `tests/test_no_pii_at_rest.py`, `tests/test_pii_discards_value.py` | met |
| [x] | UI: welcome line, exactly 3 examples, facts-only note, disclaimer | `tests/test_ui_contract.py`; `assets/disclaimer.txt` | met |
| [x] | `sources.csv` + `sources.md` shipped, regenerable | regenerated by the P6 `--refresh` build | met, with one deviation: 12 columns, not 11 (extra `detail` column carries the failure reason) |
| [x] | `README.md` with setup, scope, chunking rationale, known limits | `README.md` | met. Clean-copy check (P6-T7): install, `--refresh` build, UI and `/api/ask` all work; **the suite needs a built corpus and `HF_HUB_OFFLINE=1`** (21 fail / 9 skip before a build; 18 fail after; 440 pass with both), now stated in the README |
| [x] | `sample_qa.md` with 5–10 entries | `artifacts/sample_qa.md`, 10 entries, generated by `scripts/make_sample_qa.py` | met |
| [ ] | Prototype link live, or ≤3-min demo video | `docs/demo_script.md` beat sheet | **open: OQ5 undecided; video not recorded** |
| [x] | Eval harness run; §9.2 targets met or deviations documented | `artifacts/eval_report.md` | met: factual grounded accuracy 80% < 90%, documented in §3.1 |

**Escalated, not dropped:** minimum-SIP coverage (fixable with a locator edit and
rebuild) and the demo link/video (OQ5, needs the owner).
