# Demo script (≤ 3 minutes)

Beat sheet for the PRD §10.1 demo video. Timings sum to 2:40, leaving 20 s of
slack. Commands assume the setup in `README.md` is done and the corpus is built.

| # | Beat | Time | Running total |
|---|---|---:|---:|
| 1 | Problem | 0:10 | 0:10 |
| 2 | The four RAG stages | 0:45 | 0:55 |
| 3 | Three example Q&As | 0:45 | 1:40 |
| 4 | One refusal | 0:15 | 1:55 |
| 5 | One PII refusal | 0:10 | 2:05 |
| 6 | Eval numbers | 0:20 | 2:25 |
| 7 | Known limits | 0:15 | 2:40 |

## 1. Problem (10 s)

**Show:** the `README.md` title and scope table.

**Say:** "Investors ask the same fee and lock-in questions again and again, and
the usual answers come from blogs of unknown freshness. This assistant answers
only from official scheme data, cites exactly one source, and refuses anything
that needs judgment."

## 2. The four RAG stages (45 s)

**Run**, with the terminal visible:

```bash
python -m mf_facts.pipeline.build --strategy heading_aware
```

**Point at**, as the log scrolls:

- **S1 Load:** 15 source specs, one per (scheme, document class), fetched through a
  content-addressed cache.
- **S2 Chunk:** the `[s2]` lines. Heading-aware chunks, 66 tokens maximum
  against a 220 cap. Tables are never split.
- **S3 Embed:** `all-MiniLM-L6-v2`, 20 vectors of dimension 384.
- **S4 Store:** upserted into Chroma collection `mf_facts_v1`, then the
  manifest.

**Show** `docs/chunking_decision.md`: "Chunking was chosen by measurement. Three
strategies, same eval sets. Atomic-fact dropped to 45% table accuracy; heading-aware
won the tie-break against fixed-window."

## 3. Three example Q&As (45 s)

**Run** `python -m api` and open `http://127.0.0.1:8000`. Point briefly at
the welcome line, the "Facts-only. No investment advice." note, the no-PII
reminder and the footer disclaimer.

Click each example question:

1. *What is the exit load on HDFC Large Cap Fund?* Point at the ≤3-sentence
   answer, the single source link, and the `Last updated from sources:` stamp.
2. *Is there a lock-in period on the HDFC ELSS Tax Saver Fund?* The same
   answer shape.
3. *How do I download my capital gains statement?* **Say it straight:** "No
   statement guide could be fetched, so the assistant refuses rather than
   inventing steps. That is the grounding rail doing its job, and it is
   listed as a known limit."

## 4. One refusal (15 s)

**Type:** *Should I invest in HDFC Small Cap Fund?*

**Point at:** the templated refusal, the one educational link, and the facts-only
footer. "This is a template, not model output. The model is never called on a
refusal path, and a test asserts that."

## 5. One PII refusal (10 s)

**Type** a question with a fake PAN, for example *My PAN is ABCDE1234F, what is the
expense ratio of HDFC Large Cap Fund?*

**Say:** "Detected before anything else runs. The value is discarded. It never
reaches retrieval, the model, or the logs."

## 6. Eval numbers (20 s)

**Show** `artifacts/eval_report.md`, the target table.

**Say:** "Every answer has one link, at most three sentences and a stamp: 81 out
of 81. Refusals: 24 of 24. Performance: 10 of 10. PII: 12 of 12. Zero
unsupported numbers. p95 latency under two seconds. One miss: factual grounded
accuracy is 80 against a 90 target. All ten misses are category questions that
the grounding check refused rather than guessed, and minimum SIP is not captured
from the pages yet. Both are in the report."

## 7. Known limits (15 s)

**Show** the README's *Known limits* section. "One AMC, five schemes. A
snapshot, not live data. Aggregator-hosted pages. Three of eight document classes,
because the AMC's PDFs were blocked. Single-turn. English only."
