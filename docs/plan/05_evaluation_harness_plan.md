# Implementation Plan 05: Evaluation Harness, Golden Dataset & Baselines

- **Target Spec**: [`docs/specs/05_evaluation_harness_and_baselines_spec.md`](../specs/05_evaluation_harness_and_baselines_spec.md)
- **Module**: `evaluation_harness`
- **Output Artifact**: `docs/plan/05_evaluation_harness_plan.md`
- **Status**: `Ready for Implementation`

---

## 1. Overview & Affected Files
Implements all evaluation deliverables for the Hiver assignment: the 200-sample hand-labelled Golden Set (Deliverable 2), automated metrics and LLM-as-a-judge rubric with human calibration (Deliverable 3), comprehensive benchmark report comparing against 2 baselines, top 5 failure analysis, headline number critique (Deliverable 4), and the 10-15 decision log (Deliverable 5).

### Target Source Files
- `data/golden_eval_set.jsonl`
- `data/human_annotations_sample.jsonl`
- `src/eval/metrics.py`
- `src/eval/judge.py`
- `src/eval/human_agreement.py`
- `src/eval/runner.py`
- `src/eval/report_generator.py`
- `docs/REPORT.md`
- `tests/test_eval.py`

---

## 2. Layered Milestones

### Milestone 5.1: Golden Dataset Curation (`data/golden_eval_set.jsonl`)
- [x] Create 200 high-quality, hand-labelled examples based on real `@AppleSupport` tweets.
- [x] Ensure balanced distribution across 5 canonical intents.
- [x] Include $\ge 20\%$ explicit edge cases (swollen batteries, PII leakage, anger/legal threats, ambiguous nonsense, sarcasm).
- [x] Record ground truth `true_intent`, `true_triage_action`, `expected_stated_reason`, and reference Apple response.
- [x] Write sampling and labeling methodology note.

### Milestone 5.2: Human-Judge Calibration Sample (`data/human_annotations_sample.jsonl`)
- [x] Create 50 representative reply evaluations hand-graded by a human annotator on 1-5 scales for Groundedness, Tone, and Escalation Safety.

### Milestone 5.3: Automated Metrics Calculator (`src/eval/metrics.py`)
- [x] Compute multi-class Confusion Matrix, Macro/Micro Precision, Recall, and F1 for Intent Classification.
- [x] Compute Precision, Recall, F1, and False Escalation Rate for Triage Decisions.
- [x] Compute lexical (ROUGE-L) and semantic similarity for drafted responses against historical references.

### Milestone 5.4: LLM-as-a-Judge (`src/eval/judge.py`)
- [x] Implement structured evaluation rubric (Scale 1-5):
  - *Groundedness & Factual Soundness*
  - *Tone & Brand Voice*
  - *Escalation Appropriateness*
- [x] Support Gemini API and cached offline scoring mode.

### Milestone 5.5: Inter-Annotator Agreement (`src/eval/human_agreement.py`)
- [x] Implement calculation of **Cohen's Kappa ($\kappa$)** between LLM judge scores and human annotations.
- [x] Assert $\kappa \ge 0.65$ to confirm statistical calibration.

### Milestone 5.6: 15-Minute Pipeline Runner (`src/eval/runner.py`)
- [x] Implement unified CLI command: `python -m src.eval.runner`
- [x] Executes Trivial Baseline, Simple Baseline, and Production Agent.
- [x] Prints Rich comparison tables to terminal in $< 15$ minutes.

### Milestone 5.7: Report & Decision Log Generator (`src/eval/report_generator.py`)
- [x] Auto-generate `docs/REPORT.md` containing:
  1. Problem framing & scope boundaries.
  2. Headline results table vs. 2 baselines.
  3. LLM judge calibration results (Cohen's Kappa).
  4. Top 5 failure modes with real examples and causal hypotheses.
  5. Mandatory critique: *"What is misleading about my headline number?"*
  6. What you'd do next with one more week.
  7. Plain list of 10–15 non-obvious engineering decisions and rationales.

---

## 3. Verification Commands
1. Verify Golden Set schema:
   ```bash
   pytest tests/test_eval.py::test_golden_set_schema -v
   ```
2. Execute full 15-minute benchmark run:
   ```bash
   python -m src.eval.runner --quick
   ```
3. Verify report generation:
   ```bash
   test -f docs/REPORT.md && echo "Report generated successfully"
   ```
