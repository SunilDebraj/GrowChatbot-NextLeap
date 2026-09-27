# MF-Facts-Bot

A facts-only Q&A assistant for five HDFC Asset Management mutual fund schemes,
built as a retrieval-augmented generation (RAG) pipeline. It answers questions
about expense ratio, exit load, ELSS lock-in, riskometer, benchmark and fund
size in at most three sentences, with exactly one source link and a
`Last updated from sources:` stamp. It refuses advice, performance, portfolio
and PII-bearing questions with a templated message, and never asks the model
to write one.

> **Facts only — no investment advice.** This assistant answers general factual
> questions about five HDFC Asset Management schemes using publicly available
> official information. It does not recommend, compare, or rate any fund, and it
> does not compute or display returns. It is not investment advice. Verify every
> detail against the linked source and consult a SEBI-registered investment
> adviser before investing. Do not enter PAN, Aadhaar, account numbers, OTPs,
> email addresses, or phone numbers.

| Document | What it is |
|---|---|
| `PRD.md` | Requirements and acceptance criteria |
| `architecture.md` | Component contracts, algorithms, invariants |
| `implementation.md` | Phase-by-phase build runbook (P0–P6) |
| `docs/chunking_decision.md` | The chunking evaluation and decision (PRD OQ1) |
| `docs/corpus_gaps.md` | Which document classes could and could not be sourced |
| `docs/p4_answer_eval.md` | Answer-path evaluation at P4 (offline) |
| `artifacts/eval_report.md` | Final evaluation against every PRD §9.2 target |
| `artifacts/sample_qa.md` | Machine-generated sample questions and verbatim answers |
| `artifacts/sources.csv`, `artifacts/sources.md` | Every source ingested, regenerated per build |

## Scope

One AMC, **HDFC Asset Management**, and five schemes, all Direct Growth:

| Scheme | `scheme_key` |
|---|---|
| HDFC Large Cap Fund | `hdfc_large_cap_direct_growth` |
| HDFC Equity (Flexi Cap) Fund | `hdfc_equity_flexi_cap_direct_growth` |
| HDFC ELSS Tax Saver Fund | `hdfc_elss_tax_saver_direct_growth` |
| HDFC Small Cap Fund | `hdfc_small_cap_direct_growth` |
| HDFC Balanced Advantage Fund | `hdfc_balanced_advantage_direct_growth` |

The source pages are the five `groww.in` scheme pages named in the brief.
Groww republishes AMC data; every chunk records `publisher: aggregator` so the
provenance is visible. The AMC's own site returned HTTP 403 from the build
network, so factsheet, KIM and SID PDFs are not in the corpus — see
`docs/corpus_gaps.md`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve]"
cp .env.example .env                # then set MF_FACTS_LLM_API_KEY in .env
```

The LLM is any OpenAI-compatible `/chat/completions` endpoint, set in
`config/config.yaml` under `generation:` (`base_url`, `model`, `api_key_env`).
The key is read from the environment variable named by `api_key_env`
(default `MF_FACTS_LLM_API_KEY`), or from `.env`, which is gitignored. No key
is needed to run the tests or the offline demo (`--fake`, below).

## Architecture

### Offline: the four RAG stages

```
config/sources.yaml ─► [S1 Load] fetch (content-addressed cache) → HTML parse
                        per locator → normalize → dedupe
                    ─► [S2 Chunk] heading-aware split, ≤220 tokens, tables never split
                    ─► [S3 Embed] all-MiniLM-L6-v2, 384-dim, cached by content hash
                    ─► [S4 Store] ChromaDB collection mf_facts_v1
                    ─► artifacts/corpus_manifest.json, sources.csv, sources.md
```

Each stage is a module in `src/mf_facts/pipeline/` and can be run on its own
with `--stage`. Page-structure assumptions live only in the `locator` fields of
`config/sources.yaml`, so an upstream layout change is a config edit.

### Chunking decision (PRD OQ1)

Three strategies were built against the real corpus and scored with the same
eval sets. Selection order: table-fact accuracy, then hit@5, then simplicity.

| Strategy | factual hit@5 | table-fact accuracy | chunks | max tokens |
|---|---:|---:|---:|---:|
| **A. heading-aware (selected)** | 100% | 100% | 20 | 66 |
| B. fixed window | 100% | 100% | 15 | 124 |
| C. atomic fact | 50% | 45% | 15 | 55 |

A and B tie on the primary metric. A wins the documented tie-break for prose
coverage, and it keeps section headings in each chunk and emits tables as their
own chunks. C drops every riskometer document. The corpus is small (four chunks
per scheme), so the tie reflects the gate's resolution, not proof that chunk
boundaries do not matter. The full table, parameters and limitations are in
`docs/chunking_decision.md`.

### Online: answering a question

```
question ─► PII scan ──hit──► templated PII refusal (value discarded, never logged)
         ─► classify: performance → opinionated/portfolio → out_of_scope → LLM residual
              └─ refusal classes ──► templated refusal + one educational link
         ─► rewrite aliases → scheme_keys (hard filter) + doc_class hints (soft boost)
         ─► retrieve top-12 (Chroma, filtered) ─► rerank (RRF vector+BM25, boosts) → top-4
         ─► grounding check ──fail──► "not in my sources" refusal
         ─► generate (temperature 0, ≤180 tokens, no tools)
         ─► attach citation + stamp from chunk metadata (model URLs discarded)
         ─► validator: 8 deterministic checks ──fail──► safe response
```

The validator is a standalone module that never imports the generator. Nothing
the model writes reaches the user without passing it.

## Rebuilding the corpus

```bash
python -m mf_facts.pipeline.build --strategy heading_aware --refresh
```

`--refresh` re-fetches the pages; without it the content-addressed cache in
`raw_cache/` is reused. The build fails loudly, and publishes nothing, if a
chunk exceeds the token cap, a scheme has no documents, or the embedding model
does not match the collection. It writes `artifacts/corpus_manifest.json`,
`artifacts/build_report.json`, `artifacts/sources.csv` and `artifacts/sources.md`.

## Running the UI

```bash
python -m api                # http://127.0.0.1:8000
python -m api --fake         # offline, extractive FakeLLM; no key needed
```

The page shows the welcome line, the facts-only note, three clickable example
questions, the no-PII reminder, and the disclaimer on every load. The API has
three endpoints: `GET /api/health`, `GET /api/examples`, `POST /api/ask`. It
has no accounts, sessions or history, and a per-IP rate limit. A provider or
Chroma failure returns `503 service_unavailable`, never a traceback.

## Running the tests and the eval harness

**Build the corpus before running the tests.** 18 answer-path tests query the
built `mf_facts_v1` collection and embed questions with the real
`all-MiniLM-L6-v2`. The first build downloads that model into the local
Hugging Face cache. The suite blocks outbound network (GR14), so run it with
`HF_HUB_OFFLINE=1`; otherwise the model's metadata check is blocked even when
the weights are cached. Verified on a clean copy: 440 passed.

```bash
python -m mf_facts.pipeline.build --strategy heading_aware   # once; fetches pages + model
HF_HUB_OFFLINE=1 pytest -q                   # full suite, no network (PowerShell: $env:HF_HUB_OFFLINE=1; pytest -q)
pytest -q -m contract                        # structural promises
pytest -q -m adversarial                     # PII / refusal / performance / out-of-scope sets

python -m mf_facts.eval.harness --strategy heading_aware --all   # retrieval metrics
python -m mf_facts.eval.answer_eval --pace-s 7                   # live answer path
python -m mf_facts.eval.answer_eval --fake                       # offline answer path
python scripts/make_sample_qa.py                                 # regenerate sample_qa.md
```

`--pace-s` spaces requests out and backs off on HTTP 429. It is needed on
free-tier providers with a tokens-per-minute cap. The eval sets are data, in
`eval/*.jsonl`. Results against every PRD §9.2 target are in
`artifacts/eval_report.md`.

## Known limits

1. **Single AMC.** Only HDFC AMC and these five schemes. The assistant knows
   nothing about other AMCs.
2. **Snapshot, not live.** Answers reflect the corpus at build time. Fees and
   terms change. The `Last updated from sources` stamp is the only freshness
   signal, and it can lag the AMC.
3. **Aggregator-hosted pages.** The five source URLs are aggregator pages
   republishing AMC data. An upstream layout change can break ingestion until
   the locators in `config/sources.yaml` are updated.
4. **PDF dependence.** Factsheets, KIM and SID are PDFs, and table extraction
   from PDFs is imperfect. In this build they could not be fetched at all
   (HTTP 403), so the corpus has 3 of the 8 planned document classes. How to
   download a statement is not answerable, and neither is minimum SIP (the
   overview locator does not capture it). The assistant refuses rather than
   guessing. See `docs/corpus_gaps.md` and `artifacts/eval_report.md` §3.
5. **MiniLM trade-off.** `all-MiniLM-L6-v2` is fast and local but weaker than
   larger models on finance-specific phrasing. Expect occasional misses on
   unusual wording.
6. **No memory.** Single-turn only. A follow-up such as "and the exit load?"
   does not resolve the scheme.
7. **No evaluation of fund quality.** Deliberately. Judging quality is out of
   scope by design.
8. **English only, Indian regulatory framing.** No regional languages and no
   non-MF products (bonds, insurance, PMS).
