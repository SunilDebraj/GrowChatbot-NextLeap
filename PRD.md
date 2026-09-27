# PRD — Mutual Fund Facts-Only Q&A Assistant (RAG Chatbot)

| Field | Value |
|---|---|
| Document | Product Requirements Document v1.0 |
| Status | Draft for review |
| Codename | MF-Facts-Bot |
| Type | Retrieval-Augmented Generation (RAG) chatbot |
| Corpus scope | 1 AMC (HDFC Asset Management) · 5 schemes |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Source policy | Public official pages only |

---

## 1. Summary

Build a small, working FAQ assistant that answers **factual** questions about five HDFC Asset Management mutual fund schemes using only official, publicly available source pages.

The assistant answers questions about expense ratio, exit load, minimum SIP/ lump sum, ELSS lock-in, riskometer, benchmark, and how to download statements/tax documents. Every answer is capped at three sentences and carries exactly one source link. The assistant refuses opinionated, predictive, and portfolio questions.

This is a **facts-only** product. It never computes returns, never compares performance, and never recommends a scheme.

---

## 2. Problem Statement

Retail investors comparing HDFC mutual fund schemes repeatedly ask the same factual questions: "What is the expense ratio?", "Is there a lock-in?", "What is the minimum SIP?", "How do I download my capital gains statement?".

Today these questions are answered by:

- Support agents and content teams, manually, one at a time — repetitive and slow.
- Aggregator blogs and forums — which are frequently stale, unverified, opinionated, or both. Using them as sources is explicitly out of scope for this product.

There is no trustworthy, always-available, citation-backed source of scheme facts. Users get either human latency or unreliable answers.

**The gap:** a source-grounded assistant that answers only what the official documents say, cites exactly where it found it, and politely refuses anything that requires judgment.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| # | Goal | Success signal |
|---|---|---|
| G1 | Answer the 6 canonical fact categories with ≥90% accuracy vs. source page | Manual review of 50-question eval set |
| G2 | 100% of factual answers carry exactly one working source link | Automated link-presence check in eval harness |
| G3 | 100% refusal on opinionated / predictive / portfolio questions | Adversarial test set of ≥20 questions |
| G4 | 100% PII refusal — no PAN, Aadhaar, account number, OTP, email, phone accepted or stored | PII test set + input/output logging review |
| G5 | Answers ≤3 sentences, end with a visible "Last updated from sources:" stamp | Output validator in eval harness |
| G6 | Prototype runnable end-to-end on a clean machine | README steps verified on a fresh clone |
| G7 | Full RAG pipeline demonstrable across all stages (load → chunk → embed → store → retrieve → generate) | Architecture doc + demo video walkthrough |

### 3.2 Non-Goals

- **No performance claims.** No return computation, no CAGR, no fund ranking, no "best performing". If asked → link to the official factsheet.
- **No investment advice.** No "should I buy/sell", no allocation, no risk profiling, no goal planning.
- **No PII handling.** The product must never accept or persist sensitive identity or account data.
- **No third-party blogs / news / forums as sources.** Official AMC, SEBI, and AMFI pages only.
- **No user accounts, no personalization, no multi-turn memory across sessions** in v1.
- **No AMC beyond HDFC** in v1.
- **No live NAV, portfolio valuation, or transaction execution.**

---

## 4. Users & Personas

| Persona | Need | Pain today |
|---|---|---|
| **Retail investor (primary)** — comparing schemes, reading facts before investing | Quick, trustworthy scheme facts with a link they can verify themselves | Answers come from blogs of unknown freshness |
| **Support agent** — handling repetitive MF queries on chat/email | A verified answer + citation they can paste into a ticket | Must open 4–5 tabs per question |
| **Content/SEO team** — writing scheme pages and FAQs | Single source of truth for fee/charge/limit facts | Facts drift between documents over time |

---

## 5. Scope

### 5.1 In Scope — Schemes (all HDFC AMC, Direct Growth variants)

| # | Scheme | Category | URL |
|---|---|---|---|
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | HDFC Equity (Flexi Cap) Fund – Direct Growth | Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | HDFC ELSS Tax Saver Fund – Direct Plan Growth | ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth |
| 4 | HDFC Small Cap Fund – Direct Growth | Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | Hybrid / BAF | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Each scheme must be indexed under a **normalized scheme key** (e.g. `hdfc_large_cap_direct_growth`) so that multi-scheme questions can be answered and cited correctly.

### 5.2 In Scope — Document Classes per Scheme

| Class | Purpose | Typical facts extracted |
|---|---|---|
| Scheme overview / landing page | Basic identity, category, benchmark | Fund name, category, benchmark, riskometer |
| Factsheet (current) | Snapshot of terms | Expense ratio, exit load, minimum SIP/lump sum, portfolio mix, AUM |
| Scheme FAQ page | Direct Q&A | Lock-in, transfer, nomination, minor accounts |
| Fee & charges page | Cost facts | TER, exit load slabs, GST, stamp duty, other charges |
| KIM (Key Information Memorandum) | Statutory terms | Investment objective, risk, benchmark, custodian/auditor/RTA |
| SID (Scheme Information Document) | Statutory detail | Scheme particulars, restrictions, taxation note |
| Riskometer note | Risk classification | Riskometer level, category, month of classification |
| Statement / tax-doc guide | How-to | Steps to download capital-gains / tax reports, and from where |

### 5.3 Out of Scope — Source Types

Aggregator blog posts, news articles, forums, YouTube, social media, third-party reviews, and any source that is not operated by the AMC, SEBI, or AMFI.

> **Note on the listed URLs:** the brief's five URLs are on `groww.in`, a broker/aggregator. Groww republishes AMC-published scheme data and is acceptable as the **retrieval surface** for this milestone. Every ingested document must therefore be treated as AMC-derived and must carry its underlying official document reference (factsheet month, KIM/SID version) in chunk metadata. If an AMC-hosted page for a scheme is reachable, it takes precedence over the aggregator copy.

---

## 6. Core Functional Requirements

### FR1 — Corpus Ingestion Pipeline
The system must load, chunk, embed, and persist the corpus in four explicit, independently observable stages.

| Stage | Requirement |
|---|---|
| **S1 Load** | Fetch/ingest every source document for all 5 schemes. Parse HTML/PDF to clean text. Preserve document-level metadata: scheme key, doc class, source URL, publisher, effective date, "last updated" date, page/section. De-duplicate by content hash. |
| **S2 Chunk** | Split into retrieval-sized units using the strategy chosen in §7.2. Every chunk must retain scheme key, doc class, section heading, source URL, and last-updated date. |
| **S3 Embed** | Encode chunks with `all-MiniLM-L6-v2` (384-dim, 256-token limit). Batch encode. Persist alongside chunk text. |
| **S4 Store** | Write vectors + metadata + text to ChromaDB. One collection per corpus version, named with a content/version tag. Persist a `corpus_manifest.json` (doc count, chunk count, per-scheme counts, timestamps, embedding model id) for reproducibility. |

Re-ingestion must be idempotent: re-running the pipeline on unchanged documents must not create duplicate chunks.

### FR2 — Retrieval Pipeline
1. **Query classification** (§6.1) runs before retrieval.
2. **Query rewrite / expansion** — normalize scheme aliases ("ELSS", "tax saver", "80C fund", "large cap") to canonical scheme keys; expand abbreviations.
3. **Metadata filter** — restrict search to the resolved scheme key(s) when the question names a scheme. This is the primary precision lever.
4. **Hybrid vector search** — ChromaDB vector similarity over the filtered subset.
5. **Re-rank** — fuse top-k candidates and select the best passages.
6. **Grounding check** — verify at least one retrieved passage is actually relevant; if not, fall back to refusal (§6.2).
7. **Generation** — constrained prompt producing ≤3 sentences + exactly one link.

### FR3 — Query Classification & Routing

| Class | Definition | Route |
|---|---|---|
| `factual_scheme` | Asks for a documented scheme attribute | RAG answer + 1 citation |
| `how_to` | Asks how to perform an action (download statement, download tax doc, check riskometer) | RAG answer + 1 citation, phrased as steps |
| `opinionated` | "Should I buy X?", "Is X good?", "Best fund?" | Refusal + 1 educational link |
| `portfolio_personal` | "How much should I invest?", "Is 40% equity right for me?", "Is my portfolio balanced?" | Refusal + 1 educational link |
| `performance` | "Returns of X?", "Which performed best?", "CAGR of X?" | **No numbers.** Link official factsheet |
| `pii` | Contains PAN, Aadhaar, account number, OTP, email, phone | Refusal + PII notice; **do not store the value** |
| `out_of_scope` | Anything not about these 5 schemes | Scope refusal + list of covered schemes |

### FR4 — Answer Format Contract
Every successful factual answer MUST contain, in order:

1. The answer body — **≤3 sentences**, factual, present-tense, no hedging filler.
2. Exactly **one** source link, rendered as a clickable citation attached to the claim it supports.
3. A `Last updated from sources: <date>` stamp using the `last_updated` metadata of the supporting chunk.

Forbidden in any answer: computed returns, performance comparisons, rankings, buy/sell language, allocation percentages not present verbatim in the source, and any second source link.

### FR5 — Refusal Behavior
Refusals MUST be polite, one-to-two sentences, state the facts-only boundary, and include one relevant **educational** link (e.g. SEBI investor-education page on mutual fund basics, or an AMC page explaining riskometer/exit load). Refusals must not moralize and must not restate the question.

### FR6 — PII Handling
- Input is scanned before retrieval for PAN-format, Aadhaar-format, 10-digit account-like, 6-digit OTP, email, and phone patterns.
- On detection: return the PII notice, and **discard the raw value** — do not write it to ChromaDB, logs, prompt history, or any file.
- Log only a boolean flag `pii_detected` and the matched category, never the value.
- The UI must show a standing note: "Do not enter PAN, Aadhaar, account numbers, OTPs, email, or phone numbers."

### FR7 — UI (Minimal, per brief)
- A welcome line naming the assistant and its facts-only scope.
- Exactly **3 example questions**, clickable to auto-fill the input.
- A persistent note: **"Facts-only. No investment advice."**
- A disclaimer snippet in the footer or an About panel (§10.5).
- Message thread showing the answer, the single citation link, and the last-updated stamp.
- Input box with the no-PII reminder.
- No accounts, no settings, no dashboards, no scheme comparison table in v1.

### FR8 — Source Transparency
Ship a machine-readable `sources.csv` (and mirrored `sources.md`) listing every URL ingested, with: URL, publisher, doc class, schemes covered, last-updated date, retrieval date, content hash. Must be regenerable from the pipeline.

---

## 7. Architecture

### 7.1 RAG Stages (as required)

```
                    ┌──────────────────────────────────────────────┐
                    │              OFFLINE PIPELINE                │
                    └──────────────────────────────────────────────┘

  [S1 LOADING]  Sources (5 scheme pages + factsheets/KIM/SID/FAQ/
                fee+riskometer/statement guides)
        │       fetcher → HTML/PDF parser → clean text + metadata
        │       dedupe by content hash
        ▼
  [S2 CHUNKING]  Heading/section-aware splitter (strategy §7.2)
        │       emits: text, scheme_key, doc_class, section, url, last_updated
        ▼
  [S3 EMBEDDING] sentence-transformers/all-MiniLM-L6-v2  (384-dim)
        │       batch encode, cache by content hash
        ▼
  [S4 VECTOR STORE]  ChromaDB  (collection = corpus_version)
        │       writes vectors + metadata + documents
        ▼
  corpus_manifest.json


                    ┌──────────────────────────────────────────────┐
                    │             ONLINE (QUERY TIME)              │
                    └──────────────────────────────────────────────┘

  user query
      │
      ▼
  [PII SCAN] ──pii hit──▶ refusal (value discarded)
      │ no
      ▼
  [CLASSIFY]  factual | how_to | opinionated | portfolio |
      │       performance | pii | out_of_scope
      ├── opinionated / portfolio ──▶ refusal + educational link
      ├── performance ─────────────▶ factsheet link, no numbers
      ├── out_of_scope ────────────▶ scope notice
      │
      ▼ (factual | how_to)
  [QUERY REWRITE]  alias → canonical scheme_key
      │
      ▼
  [METADATA FILTER]  scheme_key ∈ {resolved schemes}
      │
      ▼
  [RETRIEVAL]  ChromaDB vector search (top-k)  ──▶  [RERANK]  ──▶  top-n
      │                                              │
      │ no relevant passage                           │
      ▼                                              ▼
  refusal + link                          [GROUNDING CHECK]  ──fail──▶ refusal
                                                     │ pass
                                                     ▼
                                          [GENERATE]  ≤3 sentences + 1 link
                                                     │
                                                     ▼
                                          [FORMAT VALIDATOR]  sentence cap,
                                          exactly 1 link, last-updated stamp
                                                     │
                                                     ▼
                                                   response
```

### 7.2 Chunking Strategy — DECISION REQUIRED

Per the brief, the chunking strategy is to be decided after inspecting the real corpus. The pipeline must therefore implement the strategy behind a single swappable interface so it can be changed without touching retrieval or generation.

**Status: OPEN — to be decided against the ingested corpus.** Recommended starting point to evaluate:

| Candidate | Rationale | Risk |
|---|---|---|
| **A. Heading/section-aware recursive split** — split on scheme-page section boundaries, then recursively by character/token, target ~350–450 tokens, 10–15% overlap, never split a table row or a fee/exit-load slab table | Keeps "Exit load" or "Expense ratio" tables whole; matches how a human locates the answer | Needs the page's heading structure to be reliably extractable |
| B. Fixed-size token windows (256, stride 64) | Trivial, no dependencies | Breaks tables and fee slabs mid-row; hurts exact-number questions |
| C. Per-fact atomic chunks (one fee/charge/limit per chunk) | Highest precision for numeric lookups | Requires structured extraction; brittle when pages change layout |

**Evaluation gate before locking:** run all three over the corpus, build a 50-question eval set (§9), and pick the strategy with the best grounded-answer accuracy. Record the decision, the numbers, and the rationale in the architecture doc.

**Non-negotiable regardless of strategy chosen:**
- `all-MiniLM-L6-v2` truncates at 256 tokens → hard cap every chunk well under 256 tokens (~350–450 tokens is the practical target after accounting for the model's effective window).
- Never split a numeric table, an exit-load slab, or a fee row across a chunk boundary.
- Every chunk carries the full metadata set in §6 FR1/S2 — the metadata filter in FR2 depends on it.
- Chunks from different schemes must never be merged.

### 7.3 Component Choices

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fixed by brief. 384-dim, 256-token cap. |
| Vector DB | ChromaDB | Fixed by brief. Persistent client, one collection per corpus version. |
| LLM | Provider TBD | Must support a system prompt, ≤3-sentence style control, and reliable citation emission. |
| Ingestion | TBD | HTML parser + PDF parser (factsheets/KIM/SID are PDF). |
| Orchestration | TBD | CLI/service for offline build; lightweight web UI for the demo. |
| Eval | Custom harness | Deterministic checks: link present, ≤3 sentences, last-updated stamp, refusal correctness, no-numbers-on-performance. |

---

## 8. Data Model

### 8.1 Chunk Metadata (required on every chunk)

| Field | Type | Purpose |
|---|---|---|
| `scheme_key` | str | Canonical scheme id — drives metadata filtering |
| `scheme_name` | str | Display name |
| `doc_class` | enum | `overview` \| `factsheet` \| `faq` \| `fees` \| `kim` \| `sid` \| `riskometer` \| `statement_guide` |
| `publisher` | str | AMC / SEBI / AMFI / aggregator |
| `source_url` | str | The single citation target |
| `section` | str | Page/section heading the fact came from |
| `effective_date` | str | Document's own effective date, when present |
| `last_updated` | str | Drives the "Last updated from sources:" stamp |
| `content_hash` | str | Idempotency + dedupe |

### 8.2 `corpus_manifest.json`

```json
{
  "corpus_version": "<hash-or-timestamp>",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "chunking_strategy": "<name + params>",
  "collection_name": "mf_facts_v1",
  "doc_count": 0,
  "chunk_count": 0,
  "chunks_per_scheme": {},
  "sources": [],
  "built_at": "",
  "rebuilt_from_hash": ""
}
```

---

## 9. Evaluation & Success Metrics

### 9.1 Eval Sets

| Set | Size | Content |
|---|---|---|
| **Factual** | 50 | The 6 canonical fact categories × 5 schemes (expense ratio, exit load, min SIP, ELSS lock-in, riskometer/benchmark, statement download) |
| **How-to** | 10 | Statement/tax-doc download and navigation steps |
| **Refusal** | ≥20 | Opinionated, predictive, portfolio, and "which is best" phrasings |
| **Performance** | 10 | Return/CAGR/comparison asks — must yield a factsheet link and zero numbers |
| **PII** | 12 | PAN, Aadhaar, account number, OTP, email, phone variants |
| **Out-of-scope** | 5 | Non-HDFC AMC, non-MF topics |

### 9.2 Metrics & Targets

| Metric | Target |
|---|---|
| Grounded factual accuracy (answer matches source) | ≥90% |
| Exactly one valid source link on every factual answer | 100% |
| Answer ≤3 sentences | 100% |
| "Last updated from sources:" stamp present | 100% |
| Refusal correctness on refusal + performance + out-of-scope sets | 100% |
| PII refusal + non-persistence | 100% |
| Unsupported number hallucination | 0 occurrences |
| Retrieval hit@5 (supporting chunk in top-5) | ≥95% |
| End-to-end p95 latency | ≤8s (local, non-streaming) |

### 9.3 Automated Checks
Every generated response must pass a deterministic validator before it is shown: `refusal_or_answer` classification consistency · sentence count ≤3 · exactly one URL on factual answers · last-updated stamp present · no performance numerics on performance questions · no PII pattern echoed in the output. A response that fails validation is replaced with a safe fallback rather than shown.

---

## 10. Deliverables

### 10.1 Working Prototype
A runnable app (or notebook) exposing load → chunk → embed → store → retrieve → generate, plus the minimal UI. A hosted link is preferred; if hosting is not possible, a **≤3-minute demo video** is the accepted substitute.

### 10.2 Source List
`sources.csv` **and** `sources.md` listing the 5 URLs used, with publisher, doc class, schemes covered, last-updated date, retrieval date, and content hash. Must be regenerable from the pipeline, not hand-maintained.

### 10.3 README
Setup steps · architecture and RAG stage walkthrough (with the chunking decision and its rationale) · scope (AMC + the 5 schemes) · how to rebuild the corpus · how to run the UI · eval harness usage · **known limits** (see §11).

### 10.4 Sample Q&A File
`sample_qa.md` — 5–10 representative queries with the assistant's verbatim answers, the single citation link, and the last-updated stamp. Must span all fact categories plus at least one refusal.

### 10.5 Disclaimer Snippet (used verbatim in the UI)
> **Facts only — no investment advice.** This assistant answers general factual questions about five HDFC Asset Management schemes using publicly available official information. It does not recommend, compare, or rate any fund, and it does not compute or display returns. It is not investment advice. Verify every detail against the linked source and consult a SEBI-registered investment adviser before investing. Do not enter PAN, Aadhaar, account numbers, OTPs, email addresses, or phone numbers.

---

## 11. Known Limits (to be stated in the README)

1. **Single AMC.** Only HDFC AMC, 5 schemes. Nothing about other AMCs.
2. **Snapshot, not live.** Answers reflect the corpus at build time. Fees and terms change; the "Last updated from sources" stamp is the only freshness signal, and it can lag the AMC.
3. **Aggregator-hosted pages.** The five listed URLs are aggregator-hosted republishing of AMC data. A page layout change upstream can break ingestion until selectors are updated.
4. **PDF dependence.** Factsheets, KIM, and SID are PDFs; table extraction from them is imperfect and is a known source of chunk quality issues.
5. **MiniLM trade-off.** `all-MiniLM-L6-v2` is fast and local but weaker than larger models on finance-specific phrasing; expect occasional misses on unusual wording.
6. **No memory.** Single-turn only. Follow-up questions like "and the exit load?" will not resolve the scheme.
7. **No evaluation of fund quality.** Deliberately. Quality judgment is out of scope by design.
8. **English only, Indian regulatory framing.** No regional languages; no non-MF products (bonds, insurance, PMS).

---

## 12. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chunking breaks fee/exit-load tables | Wrong numbers — the highest-severity failure | Table-aware splitting; never split a row; numeric questions in the eval gate |
| AMC updates pages, ingestion silently drifts | Stale or empty answers | Content hashes, `last_updated` metadata, visible stamp, periodic rebuild |
| Aggregator page layout change | Pipeline breaks | Selectors isolated in one config; `sources.csv` shows retrieval date |
| Model answers from parametric memory despite grounding | Performance claims leak into answers | Grounding check + format validator + refusal fallback; no-numbers check |
| User submits PAN/account number | PII persistence | Pre-retrieval scan, value discarded, never logged or stored |
| Scope creep into advice | Regulatory exposure | Explicit class routing, hard refusal set, validator gate, disclaimer on every surface |
| MiniLM under-retrieves on finance phrasing | "I don't know" on answerable questions | Query rewrite + alias map + metadata filter; measure hit@5 |

---

## 13. Acceptance Criteria

The milestone is complete when all of the following are demonstrably true:

- [ ] All four offline stages (load, chunk, embed, store) run end-to-end and are individually inspectable.
- [ ] `corpus_manifest.json` present and consistent with the ChromaDB collection.
- [ ] Chunking strategy chosen by evaluating candidates against the real corpus, with the decision and numbers documented.
- [ ] All 5 schemes answerably indexed; all 6 fact categories retrievable for each.
- [ ] Every factual answer: ≤3 sentences, exactly one source link, "Last updated from sources:" stamp.
- [ ] Opinionated, portfolio, and performance questions refused; performance questions link the official factsheet and show no numbers.
- [ ] PII test set 100% refused with no value persisted anywhere.
- [ ] UI shows welcome line, exactly 3 example questions, "Facts-only. No investment advice." note, and the disclaimer snippet.
- [ ] `sources.csv` + `sources.md` shipped and regenerable.
- [ ] `README.md` with setup, scope, chunking rationale, and known limits.
- [ ] `sample_qa.md` with 5–10 Q&A entries.
- [ ] Prototype link live, or ≤3-minute demo video recorded.
- [ ] Eval harness run; all §9.2 targets met or deviations explicitly documented.

---

## 14. Out of Scope for v1 (Backlog)

- Multi-AMC expansion via the same pipeline.
- Scheme-to-scheme factual comparison tables (facts only, still no performance).
- Telugu/Hindi and other regional-language ingestion.
- Streaming responses and conversation memory with scheme carry-over.
- Scheduled re-ingestion with change alerts on fees.
- A governed, human-reviewed answer log for support-agent embedding.

---

## 15. Open Questions

| # | Question | Owner | Needed By |
|---|---|---|---|
| OQ1 | Chunking strategy — pending corpus inspection and A/B/C evaluation (§7.2) | Engineering | Before retrieval is tuned |
| OQ2 | LLM provider and whether it must run fully offline | | Before generation is built |
| OQ3 | Are AMC-hosted pages reachable for all 5 schemes, and do they take precedence? | | Before ingestion is locked |
| OQ4 | ChromaDB host: local persistent client vs. server | | Before store stage |
| OQ5 | Hosting target for the demo link, or confirm video-only submission | | Before submission |
| OQ6 | Which specific educational links to use for each refusal class | Content | Before refusal copy is finalized |
