# Evaluation report (P6-T4)

Final evaluation of MF-Facts-Bot against every `PRD.md` §9.2 target, with a
live generator. Supersedes the offline-only answer numbers in
`docs/p4_answer_eval.md`, which were measured with `FakeLLM` because OQ2 was
still open at the time.

| Field | Value |
|---|---|
| Date | 2026-09-27 |
| Generator | `chat_completions` / `qwen/qwen3.8-27b`, temperature 0, max_tokens 180 |
| Corpus | `mf_facts_v1`, version `2026-09-27T14:09:43+00:00`, 15 docs, 20 chunks, fresh `--refresh` build |
| Chunking | `heading_aware` (see `docs/chunking_decision.md`) |
| Test suite | `pytest -q`: 440 passed |

## 1. Targets

| PRD §9.2 metric | Target | Result | Verdict |
|---|---|---|---|
| Grounded factual accuracy | ≥ 90% | factual **80.0%** (40/50) · table facts 100% (20/20) · how-to 90.5% (19/21) | **missed on factual. See 3.1** |
| Exactly one valid source link, factual answers | 100% | 100% (81/81 answered) | met |
| Answer ≤ 3 sentences | 100% | 100% (81/81 answered) | met |
| `Last updated from sources:` stamp present | 100% | 100% (81/81 answered) | met |
| Refusal correctness: refusal + performance + out-of-scope | 100% | 24/24 · 10/10 · 5/5 | met |
| PII refusal + non-persistence | 100% | 12/12 refused; `test_no_pii_at_rest` clean | met |
| Unsupported number hallucination | 0 | 0 across 91 answer-path items | met |
| Retrieval hit@5 | ≥ 95% | factual 100% · table facts 100% · how-to 100% | met, but weak. See 3.3 |
| End-to-end p95 latency | ≤ 8 s | 1.04–1.72 s per set | met |

Additional `PRD.md` §13 criterion measured here: **"all 6 fact categories
retrievable for each scheme"** — **not met.** Minimum SIP is not in the corpus.
See 3.2.

## 2. Results by eval set

### 2.1 Answer path (live model)

`python -m mf_facts.eval.answer_eval --pace-s 7 --json-out artifacts/answer_eval_live.json`

| Set | n | answered | grounded | fact in top-4 | one link | ≤3 sent | stamped | unsupported nums | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| factual | 50 | 80.0% (40) | 80.0% | 80.0% | 100% | 100% | 100% | 0 | 1687 ms |
| table_facts | 20 | 100% (20) | 100% | 100% | 100% | 100% | 100% | 0 | 1719 ms |
| howto | 21 | 100% (21) | 90.5% | 100% | 100% | 100% | 100% | 0 | 1038 ms |

The output-contract columns are calculated over answered items only; a refusal
is not an answer and is not scored against the answer contract. Latency is timed
per successful call. The waits caused by the provider's free-tier rate limit
(`--pace-s`) are excluded, because they are a property of the account, not of
the pipeline.

### 2.2 Retrieval (deterministic)

`python -m mf_facts.eval.harness --strategy heading_aware --all`

| Set | n | hit@1 | hit@2 | hit@5 |
|---|---:|---:|---:|---:|
| factual | 50 | 50% | 90% | 100% |
| table_facts | 20 | 0% | 85% | 100% |
| howto | 21 | 4.8% | 57.1% | 100% |

### 2.3 Safety sets (through the real pipeline)

| Set | n | Correct class, refused | One link + footer | Return figures in reply | LLM calls |
|---|---:|---:|---:|---:|---:|
| pii | 12 | 12/12 | 12/12 (no link, by design) | 0 | 0 |
| refusal | 24 | 24/24 | 24/24 | 0 | 3 (residual classifier) |
| performance | 10 | 10/10 | 10/10 | 0 | 0 |
| out_of_scope | 5 | 5/5 | 5/5 | 0 | 0 |

- Three refusal items (`ref_08`, `ref_11`, `ref_12`) have no lexicon cue by
  design and reach the stage-2 LLM classifier. The unit tests script that
  classifier's reply, so they prove routing but not the model's judgment. Run
  live, the model labelled all three `opinionated` (`rule_id: llm:residual`).
  The LLM is never used to *generate* a refusal: every refusal is a template.
- "Return figures" counts digits and `%` in the reply. The validator's
  `_RETURN_PATTERN` also matches the word *CAGR*, which the performance refusal
  template itself contains ("I do not quote returns, CAGR…"). Those are not
  figures.
- `pytest -q -m adversarial`: 186 passed.

## 3. Deviations and findings

### 3.1 Factual grounded accuracy is 80%, below the 90% target

All 10 misses are one question shape: *category* and *sub-category* × 5
schemes. Each is refused by the grounding check (`route=refusal`), not answered
wrongly. The chunk says `Equity` / `Large cap` but never uses the words
*category* or *sub-category*, and the grounding check is lexical, so it finds no
anchor for the question. This is the failure direction the design prefers,
refusing rather than answering from memory, and it was already recorded in
`docs/p4_answer_eval.md` §5.1. It is still a recall gap. Of the 40 factual items
the pipeline answered, **40/40 are correct**.

Two ways to close it, neither applied here, because both change the system
under test after the eval ran:
- Add `category` and `sub-category` as labels in the overview locator, so the
  words appear in the chunk.
- Or let a doc_class-hint match stand in for the lexical anchor on overview
  labels.

### 3.2 Minimum SIP is not in the corpus

The scheme pages publish it ("Minimum investments · Min. for SIP ₹100"). The
`overview` locator in `config/sources.yaml` keeps only `header` plus the labels
`Fund benchmark | Fund size (AUM) | NAV:`, so it never reaches a chunk. The
factual eval set was authored from the built corpus, which is why it has no
minimum-SIP items and why hit@5 never exposed the gap. `docs/corpus_gaps.md`
listed minimum SIP as answerable; that was wrong and is corrected there.

Consequence: `What is the minimum SIP amount for HDFC Balanced Advantage Fund?`
is refused as `grounding_fail` (see `artifacts/sample_qa.md` #3). The refusal is
correct, but it fails the PRD §13 criterion that all 6 canonical fact categories
are retrievable for each scheme. The fix is a locator change plus a rebuild and
re-run of this report; no code change is needed.

### 3.3 hit@5 is a weak gate at this corpus size

Each scheme has 4 chunks, and retrieval is hard-filtered to the scheme, so a
top-5 list returns the whole scheme. hit@5 = 100% therefore says little. hit@1
(0–50%) is the informative number. The generator compensates by reading all 4
passages; `fact in top-4` shows the fact was present every time on table facts
and how-to.

### 3.4 How-to: 2 matcher misses that are correct answers

- `hdfc_elss_tax_saver_direct_growth:lockin_period`: the answer is "The lock-in
  period … is 3 years." The item expects `3Y`. The answer is correct; the
  containment matcher does not equate `3Y` with `3 years`.
- `hdfc_elss_tax_saver_direct_growth:hold_period_for_exit_load`: the answer is
  "The exit load … is Nil." That matches the corpus, since ELSS carries no exit
  load. The item expects `1 year`, which looks like an item-authoring error
  carried over from the non-ELSS template.

Neither eval item has been edited. `implementation.md` forbids adjusting eval
data after seeing results. On manual review the how-to set is 21/21 correct.

### 3.5 How-to questions are classified `factual_scheme`

The live residual classifier labels all 21 how-to items `factual_scheme`
(`class_mismatch = 21`). Both classes take the same answer path, so every item
is still retrieved, grounded, cited and stamped. What is lost is the "phrase as
steps" styling that `how_to` would add. With no statement guide in the corpus,
none of these questions actually requires steps.

## 4. Reproducing

```bash
pytest -q                                                    # 440 passed
pytest -q -m adversarial                                     # 186 passed
python -m mf_facts.pipeline.build --strategy heading_aware --refresh
python -m mf_facts.eval.harness --strategy heading_aware --all
python -m mf_facts.eval.answer_eval --pace-s 7 --json-out artifacts/answer_eval_live.json
python scripts/make_sample_qa.py
```

Per-item answers, including every failure, are in `artifacts/answer_eval_live.json`.
