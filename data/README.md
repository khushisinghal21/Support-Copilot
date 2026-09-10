# Data Directory & Golden Evaluation Set Documentation

This directory contains the datasets, historical resolution pairs, and human calibration annotations supporting the Hiver SDE Intern AI Customer Support Agent for **`@AppleSupport`**.

> **Revision note:** an earlier version of this file described the golden set as built by "Stratified Purposive Sampling from the Kaggle dataset." That description did not match the code that generated it — `curate_datasets.py` hand-wrote ~40 template sentences and mechanically padded them into 200 rows, and never touched a real tweet. This version replaces that file and describes, honestly, how the current one was actually built. See `docs/AUDIT_AND_FIX_PLAN.md` for the full writeup of what was wrong and why.

## Deliverable 2: Golden Evaluation Set (188 examples)

- **File**: [`golden_eval_set.jsonl`](golden_eval_set.jsonl)
- **Total count**: 188 examples (within the 150-250 requirement).
- **Target brand**: `@AppleSupport`

### How it was actually built

**Step 1 — real data.** `data/apple_support_kaggle_pairs.jsonl` (1,000 real `@AppleSupport` conversation-initiator tweets and their real historical agent replies, extracted by `src/data/ingest_kaggle.py` from the Kaggle `twcs.csv` via the DuckDB query documented in the main README) is the source of truth. Nothing in the golden set is invented text unless explicitly tagged `"source": "authored_adversarial"` (see Step 3).

**Step 2 — labelling.** Each real tweet was labelled for `true_intent` using an independent keyword heuristic (`scripts/build_golden_set.py`) that is deliberately **not** the `SemanticCentroidClassifier` under test and does not share its prototype sentences — using the same model to both generate and grade labels would make the eval circular. That heuristic disagreed with the classifier used at ingestion time on 41.3% of the pool, which is itself a useful data point about how ambiguous real-world intent labelling is (see `docs/AUDIT_AND_FIX_PLAN.md`). `true_triage_action` was labelled with a separate, tightened hazard/PII/legal/human-request detector, and then — critically — **every candidate `ESCALATE` case from real data (9 of them) was read and adjudicated by hand**, because the first-pass regex produced obvious false positives: two tweets where the customer was *reporting a phishing scam to Apple* (not personally victimized) got flagged on the word "scam"/"fraud"; a support case number (`#100310750365`) got matched by the phone-number regex; and "IM SUING" over a slow charge turned out to be hyperbole that Apple's own historical agents handled as a routine ticket. Those four became labelled hard negatives (`edge_case_type` starting `HARD_NEGATIVE_`) precisely because they're useful for testing whether the production triage rules make the same mistakes. Two real cases were genuine escalations (an electric-shock injury with an explicit legal threat, and a hacked-account/Find My compromise). One real case — an ambiguous water-damage claim where Apple's own historical agent asked a clarifying question rather than auto-handling or escalating — is labelled `CLARIFY`, which is the direct real-world justification for adding that action to the triage engine (see the main audit doc).

Routine `AUTO_HANDLE` examples were then stratified-sampled from the remaining real pool, loosely following the real observed base rates across the five intents (OS troubleshooting is the largest real bucket, how-to the smallest) rather than an artificially even split, with a floor so no class drops below ~20 examples.

**Step 3 — disclosed adversarial supplement (30 examples, ~16% of the set).** Real 2017-18 first-contact tweets essentially never contain PII (0 email/phone matches in the full 1,000-row pool — people don't post that in a public first tweet, they wait to be asked to DM), explicit thermal hazards are rare, and prompt-injection attempts against an LLM-backed support agent could not exist in that era at all. Those categories were authored directly, covering thermal/physical hazards (8), explicit PII disclosed in a public tweet (6), unambiguous legal/fraud threats (4), explicit human-agent demands (4), and prompt-injection attempts against the AI agent itself (8) — a risk this 2017-18 dataset can't test but a system shipping with an LLM in the loop today needs to be. Every one of these rows carries `"source": "authored_adversarial"` and `"is_edge_case": true` so a reader can filter them out and see how the system performs on real data alone versus the full set.

**Reference replies.** For real examples, `reference_resolution` is the *actual* historical `@AppleSupport` agent reply from the Kaggle extraction — the strongest possible grounding reference, since it's not synthesized. For authored adversarial examples it's a short brand-consistent reply we wrote.

### Schema

```json
{
  "tweet_id": "golden_014",
  "source_tweet_id": "kaggle_31408_31406",
  "text": "Wi-Fi keeps disconnecting every few minutes on my iPhone X.",
  "author_id": "kaggle_real_user",
  "true_intent": "OS_SOFTWARE_TROUBLESHOOTING",
  "true_triage_action": "AUTO_HANDLE",
  "expected_stated_reason": null,
  "reference_resolution": "<the real historical @AppleSupport reply>",
  "is_edge_case": false,
  "edge_case_type": null,
  "source": "kaggle_real"
}
```

`true_triage_action` is one of `AUTO_HANDLE`, `ESCALATE`, or `CLARIFY` (added because a real historical case in this set — the water-damage claim above — showed Apple's own agents using a middle option rather than a binary auto/escalate call).

### Known limitations of this labelling process (stated plainly, not buried)

The intent heuristic is keyword-based and imperfect — it will mislabel genuinely ambiguous free text (e.g. a vague device question with no clear feature name) the same way any lexical method would; a number of examples were manually corrected after a spot-check, but not every one of the ~160 real rows was individually re-verified by a second pass. This is disclosed rather than hidden: if asked to defend any specific label, the honest answer is "here's the rule that produced it, and here's why," not "a human checked all 188 one at a time." A second reviewer spot-checking a random 20-30 row sample before submission would meaningfully strengthen this further.

## Deliverable 3: Human Agreement Calibration Dataset

- **File**: [`human_annotations_sample.jsonl`](human_annotations_sample.jsonl)
- **Total count**: 50 examples drawn from the golden set above (oversampling edge cases, since that's where judge/reviewer disagreement is most informative).
- **Purpose**: measures how well the automated LLM-judge rubric (`src/eval/judge.py`) agrees with an independent rubric-based scoring pass.
- **What changed from the previous version**: the old file assigned scores with a rule like "groundedness = 4 unless `i % 7 == 0`" and had no connection to an actual second opinion; the kappa computed from it was then floored at 0.72 in code regardless of what it measured. Both are removed. See `docs/AUDIT_AND_FIX_PLAN.md` for the full explanation and `src/eval/human_agreement.py` for the current (unfloored) calculation. Whatever kappa the harness reports now is the actual computed value — if it's mediocre, that's the honest number, and the report says so.
