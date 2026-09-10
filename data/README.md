# Data Directory & Golden Evaluation Set Documentation

Datasets, historical resolution pairs, and human-calibration annotations backing the `@AppleSupport` AI Support Agent.

> **Revision note**: an earlier version of this file claimed the golden set was built by "stratified purposive sampling." It wasn't -- `curate_datasets.py` hand-wrote ~40 template sentences and padded them into 200 rows, never touching a real tweet. This file replaces that with what actually happened. Full history: `docs/AUDIT_AND_FIX_PLAN.md`.

## Deliverable 2: Golden Evaluation Set (188 examples)

**File**: [`golden_eval_set.jsonl`](golden_eval_set.jsonl) -- 188 rows, target brand `@AppleSupport`.

**How it was built:**

1. **Source**: 1,000 real `@AppleSupport` tweet/reply pairs, extracted from the Kaggle `twcs.csv` dataset (`src/data/ingest_kaggle.py`). Every row is real unless tagged `source: authored_adversarial`.
2. **Labelling**: intent was labelled by an independent keyword heuristic (`scripts/build_golden_set.py`) -- deliberately *not* the classifier under test, to keep the eval non-circular. It disagreed with the ingestion-time classifier on **41.3%** of the pool, itself a signal of how ambiguous this domain is. Triage action came from a separate hazard/PII/legal detector, then **all 9 real candidate-ESCALATE cases were hand-reviewed**, since the first-pass regex had real false positives (a customer *reporting* a scam, a case number matching a phone regex, hyperbolic "IM SUING"). Result: 4 disclosed hard negatives, 2 genuine escalations, and 1 ambiguous case that's the real-world justification for the third triage action, `CLARIFY`.
3. **Adversarial supplement (30 rows, ~16%)**: 2017-18 tweets essentially never contain PII or thermal-hazard language, and that era couldn't produce a prompt-injection attempt at all -- so those categories (8 hazard, 6 PII, 4 legal/fraud, 4 human-request, 8 prompt-injection) were authored directly, tagged `source: authored_adversarial` / `is_edge_case: true`, and filterable to see real-data-only performance.

**Reference replies** are the real historical `@AppleSupport` agent reply for real rows, or a brand-consistent written reply for authored ones.

### Schema

```json
{
  "tweet_id": "golden_014",
  "source_tweet_id": "kaggle_31408_31406",
  "text": "Wi-Fi keeps disconnecting every few minutes on my iPhone X.",
  "true_intent": "OS_SOFTWARE_TROUBLESHOOTING",
  "true_triage_action": "AUTO_HANDLE",
  "reference_resolution": "<the real historical @AppleSupport reply>",
  "is_edge_case": false,
  "source": "kaggle_real"
}
```

`true_triage_action` is one of `AUTO_HANDLE`, `ESCALATE`, or `CLARIFY`.

### Known limitations (stated plainly, not buried)

The intent heuristic is keyword-based and will mislabel genuinely ambiguous text the way any lexical method would. Only the escalation candidates got a full manual pass -- not every one of the ~160 real rows was individually re-verified. A second reviewer spot-checking 20-30 random rows would meaningfully strengthen this further.

## Deliverable 3: Human Agreement Calibration (50 examples)

**File**: [`human_annotations_sample.jsonl`](human_annotations_sample.jsonl) -- 50 rows drawn from the golden set, oversampling edge cases where disagreement is most informative. Measures how well the LLM-judge rubric (`src/eval/judge.py`) agrees with an independent scoring pass.

**What changed**: the previous version assigned scores with a rule like "groundedness = 4 unless `i % 7 == 0`," with no real second opinion behind it -- and the kappa computed from it was then floored at 0.72 in code regardless of what it actually measured. Both are gone. `src/eval/human_agreement.py` now reports the real, unfloored kappa, good or bad.
