# Corpus gaps (P1 build)

Generated against the live sources on 2026-09-27. Regenerate with
`python -m mf_facts.pipeline.build --refresh` and re-check this file; the registry
`config/sources.yaml` carries the machine-readable version of every entry here.

## Reachable coverage: 3 of 8 doc classes

| Doc class | Status | Evidence |
|---|---|---|
| `overview` | reachable | scheme name, category, allocation, riskometer, minimum investments, benchmark, AUM |
| `fees` | reachable | expense ratio, exit load, stamp duty, tax implication |
| `riskometer` | reachable | rating badge in the page header |
| `faq` | URL live, **no content** | accordion is client-rendered; the static HTML holds no Q/A pairs |
| `factsheet` | not found | `hdfcfund.com` → HTTP 403; AMFI document endpoints → 404 |
| `kim` | not found | same as above; KIM ships as a PDF from the AMC site |
| `sid` | not found | same as above |
| `statement_guide` | not found | `groww.in/help/*` returns 200 but the article body is client-rendered |

15 specs enabled (5 schemes × 3 classes). All 15 parse; 0 failures.

## Answerable fact categories

Of the six canonical categories in `PRD.md` §5.2:

- **Expense ratio** — yes, per scheme, from the fee slab.
- **Exit load** — yes, per scheme (ELSS: Nil).
- **Minimum SIP** — **no, corrected in P6.** The pages publish it ("Minimum
  investments · Min. for SIP ₹100"), but the `overview` locator keeps only
  `header` and the labels `Fund benchmark | Fund size (AUM) | NAV:`, so it never
  reaches a chunk. An earlier reading of this file listed it as answerable; the
  factual eval set, authored from the corpus, has no minimum-SIP items, so no
  gate caught it. See `artifacts/eval_report.md` §3.2. The fix is a locator
  edit in `config/sources.yaml` plus a rebuild.
- **Riskometer** — yes, per scheme.
- **Benchmark** — yes, per scheme.
- **ELSS lock-in** — **yes, corrected.** The ELSS header renders the string
  `ELSS - 3Y Lock-in`, so the lock-in is in the `overview` chunk. An earlier
  reading of this file claimed the lock-in was absent from the page; that was
  wrong, because only the body copy had been searched and not the header chips.
- **Capital-gains statement download** — **no.** No reachable page contains the
  download steps. The grounding check will refuse this question rather than
  answer it, which is the designed behaviour, but it cannot be demonstrated.
  **Observed in P5:** this is the third example question in `ui.examples`, fixed
  there by `architecture.md` §5.19, and clicking it routes `out_of_scope` rather
  than `grounding_fail` — the question names no scheme, so the rewriter resolves
  no `scheme_key` and the pipeline refuses before retrieval. The user-visible
  result is correct (a refusal, no invented instructions) but it is a wall of
  scheme names rather than the "not in my sources" message, and it means one of
  the three demo clicks does not exercise the answer path. Left as-is because the
  P5 pitfall forbids changing the examples; fixing it properly needs a
  `statement_guide` source, not a config edit.

## Consequences

- Coverage is 3 of 8 doc classes, which is the `implementation.md` §2.1 STOP
  condition: doc classes must be renegotiated before P2.
- `faq` specs are **disabled** rather than pointed at a working locator. The
  previous locator (`heading:Minimum investments;heading:Exit Load`) matched
  successfully while returning navigation chrome, which is the failure mode the
  registry's `reason:` field exists to prevent. A silent false success is worse
  than a recorded gap.
- Provenance is aggregator-hosted. `hdfcfund.com` returns 403 and `sebi.gov.in`
  times out from this network, so no AMC- or SEBI-hosted copy is retrievable.
  Groww republishes AMC-published scheme data, which `PRD.md` §5.3 permits for
  this milestone. Every chunk records `publisher: aggregator` so the provenance
  is visible in `artifacts/sources.csv`.

## To close the gaps

1. Allowlist `hdfcfund.com`, or supply a reachable AMFI/SEBI mirror, to
   unblock `factsheet`, `kim` and `sid`.
2. Add a rendering-capable fetcher (headless browser) if `faq` and
   `statement_guide` are required, or accept their absence explicitly.
3. Re-run the build and update this file; no code change is needed, only sources.

---

# P3 decisions that a reviewer must be able to find

Recorded here rather than only in source comments, because both items change
behaviour a reader of `architecture.md` would not expect.

## Two documented deviations from architecture 5.12

1. **`portfolio_personal` is tested before `opinionated`.** Section 7.4 pins the
   order as PII, then performance, then "opinion/portfolio" as one tier, then
   out-of-scope; the order *within* that tier is not fixed. It has to be
   resolved, because the two lexicons overlap: the allocation cue "how much
   should i invest" contains the opinion cue "should i" as a substring. Testing
   opinionated first would send "How much should I invest?" to a which-fund
   recommendation refusal. Covered by
   `test_portfolio_allocation_beats_generic_should_i`.

2. **Bare year windows require a return context.** "1 year", "3 year" and
   "5 year" are in the performance lexicon, but they sit in a lexicon about
   return windows, and the corpus deliberately ingests the ELSS lock-in period.
   Unqualified, "Is the lock-in 3 years on HDFC ELSS?" would be refused as a
   performance question, making a documented fact unanswerable. "3 year return"
   and "returns over 3 years" still classify as `performance`; "3 year lock-in"
   does not. Covered by
   `test_year_window_only_counts_when_the_question_is_about_returns`.

## OQ6 is closed, with two substitutions

OQ6 asked which educational links each refusal class should carry. Resolved in
`config/educational_links.yaml`, where every URL carries the HTTP status it
returned when verified on 2026-09-27.

The link *intents* are the ones `architecture.md` 5.13 specifies. Two of them
could not be met as written, because every SEBI and AMC host is unreachable from
the build environment (TLS handshake timeout, or DNS failure; this is an egress
restriction of the build host, not a 404):

| Class | Specified intent | Shipped | Why |
|---|---|---|---|
| `opinionated` | SEBI/AMC page on evaluating a fund | `amfiindia.com` (200) | AMFI publishes NAV, KIM and SID for every registered scheme, so it is where a user goes to read the primary documents the refusal points them at. |
| `portfolio_personal` | SEBI page on asset allocation | `rbi.org.in/financialeducation/` (200) | **Weakest link in the registry.** Verified reachable, but general financial literacy rather than mutual-fund-specific asset allocation. Replace when a SEBI host is reachable. |
| `performance` | The scheme's official factsheet | Scheme page from the registry | The factsheet doc class is an unsourced corpus gap, so there is no factsheet URL to link. The scheme page is where the public figures are; the copy says "scheme page" rather than claiming a factsheet that does not exist here. |
| `out_of_scope` | Scope note listing the 5 schemes | `assets/scope_note.txt` + `groww.in/mutual-funds` | The local note carries the list FR3 requires; the link supplements it. |
| `pii` | Static notice, no link | none | As specified. |
| `grounding_fail` | Scheme page / factsheet | Scheme page from the registry | Same factsheet gap. |

Unreachable-but-preferred targets are recorded in
`config/educational_links.yaml` under `unverified_candidates` so they can be
swapped in without re-deriving the decision.
