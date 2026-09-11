"""Generates the comprehensive benchmark report docs/REPORT.md (Deliverable 4 & 5)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import REPORT_OUTPUT_PATH, BENCHMARK_SUMMARY_JSON_PATH, TARGET_BRAND, GOLDEN_SET_PATH
import json


def _fmt_kappa(k) -> str:
    """Cohen's kappa can now legitimately come back as None (undefined --
    e.g. every score in a series was identical, so kappa's denominator is
    zero) instead of a fabricated floor value. Format that case as text
    instead of crashing the report on `None:.4f`."""
    return f"{k:.4f}" if isinstance(k, (int, float)) else "undefined"


def _abbreviate_labels(labels: List[str]) -> Dict[str, str]:
    """Maps each intent label to a short, unique code (initials of its
    underscore-separated words) so the confusion matrix table below stays
    narrow enough to read -- the raw labels (e.g.
    OUT_OF_SCOPE_AMBIGUOUS) are too wide to use as repeated column headers
    in a 5x5+ grid."""
    used = set()
    abbrevs: Dict[str, str] = {}
    for lbl in labels:
        parts = [p for p in lbl.split("_") if p]
        base = ("".join(p[0] for p in parts).upper() or lbl[:3].upper())
        abbr = base
        i = 2
        while abbr in used:
            abbr = f"{base}{i}"
            i += 1
        used.add(abbr)
        abbrevs[lbl] = abbr
    return abbrevs


def _render_confusion_matrix_markdown(labels: List[str], matrix: List[List[int]]) -> str:
    """Renders the intent-classification confusion matrix that
    src/eval/metrics.py's compute_intent_metrics() already computes on every
    run (`confusion_matrix` + `labels` in its return dict) but that, before
    this function existed, was silently discarded -- never printed to the
    terminal, never written into docs/REPORT.md, never in the dashboard JSON.
    Same failure pattern this codebase's audit already found once with the
    per-example `failures` list in src/eval/runner.py: a real signal computed
    and then never read by anything downstream.

    Rows are the true label, columns the predicted label -- reading across a
    row shows exactly where that true class's queries actually ended up,
    which is the direct, numeric version of what src/eval/failure_analysis.py
    narrates in prose for individual examples.
    """
    if not labels or not matrix:
        return "_No confusion matrix available for this run (no intent predictions were made)._\n"

    abbrevs = _abbreviate_labels(labels)
    header = "| True \\ Predicted | " + " | ".join(abbrevs[l] for l in labels) + " |\n"
    sep = "| :--- | " + " | ".join([":---:"] * len(labels)) + " |\n"
    rows = ""
    for i, true_label in enumerate(labels):
        cells = []
        for j, pred_label in enumerate(labels):
            count = matrix[i][j]
            # Bold the diagonal (correct predictions) so a reader can spot
            # off-diagonal mass -- misclassification -- at a glance.
            cells.append(f"**{count}**" if i == j else (str(count) if count else "&middot;"))
        rows += f"| **{abbrevs[true_label]}** | " + " | ".join(cells) + " |\n"

    legend = "\n".join(f"- `{abbrevs[l]}` = `{l}`" for l in labels)

    return f"{header}{sep}{rows}\n**Legend**\n\n{legend}\n"


def _golden_set_stats() -> Dict[str, Any]:
    """Reads the actual golden set on disk rather than hardcoding its size
    and edge-case share -- the header used to claim '200 Hand-Labelled Test
    Queries (including 20% verified edge cases)' unconditionally, which
    silently went stale the moment the file was rebuilt to 188 rows."""
    try:
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        n = len(rows)
        n_edge = sum(1 for r in rows if r.get("is_edge_case"))
        pct_edge = round(100 * n_edge / n) if n else 0
        return {"n": n, "n_edge": n_edge, "pct_edge": pct_edge}
    except Exception:
        return {"n": 0, "n_edge": 0, "pct_edge": 0}


def generate_markdown_report(
    trivial_metrics: Dict[str, Any],
    simple_metrics: Dict[str, Any],
    prod_metrics: Dict[str, Any],
    judge_metrics: Dict[str, Any],
    agreement_metrics: Dict[str, Any],
    top_failures: List[Dict[str, Any]],
    latency_p95_ms: Optional[float] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Compiles the formal Hiver SDE Intern benchmark evaluation report."""

    gs = _golden_set_stats()

    # Section 5's disclosure of the calibration/held-out split is built from the
    # real split_info handed in by the runner. When a caller does not supply it
    # (older callers, unit tests), say so plainly rather than printing a
    # confident-sounding sentence about a split this run did not actually use.
    if split_info:
        _cal_ia = split_info.get("calibration_intent_accuracy")
        _cal_ta = split_info.get("calibration_triage_accuracy")
        _held_ia = prod_metrics["intent"]["accuracy"]
        _held_ta = prod_metrics["triage"]["accuracy"]
        _clarify_held = split_info.get("heldout_label_counts", {}).get("CLARIFY", 0)
        split_disclosure = (
            f"**What changed:** `data/golden_eval_set.jsonl` now carries a persisted, seeded "
            f"(`SPLIT_SEED={split_info.get('seed')}`), stratified `split` field. The sweep script reads the "
            f"**{split_info.get('calibration_n')} calibration rows** only; every headline number in this report is "
            f"measured on the **{split_info.get('heldout_n')} held-out rows** the thresholds were never tuned against. "
            f"The gap is published rather than hidden: intent accuracy {_cal_ia*100:.1f}% → {_held_ia*100:.1f}% "
            f"({(_held_ia - _cal_ia)*100:+.1f} pts) and triage accuracy {_cal_ta*100:.1f}% → {_held_ta*100:.1f}% "
            f"({(_held_ta - _cal_ta)*100:+.1f} pts) moving from tuned-on data to held-out data. "
        )
        if _clarify_held == 0:
            split_disclosure += (
                "**A second limitation the split exposed, stated rather than smoothed over:** the golden set contains "
                "exactly one `CLARIFY` row, so stratification could not place it on both sides -- the held-out set has "
                "**zero** `CLARIFY` examples. Held-out triage accuracy therefore measures a two-class problem while the "
                "system implements three actions, and the `CLARIFY` path is currently covered by unit tests only, not by "
                "this benchmark. Fixing that needs more `CLARIFY` data, not a different split."
            )
    else:
        split_disclosure = (
            "*(This report was generated without split information, so the calibration/held-out gap is not quantified "
            "here. Re-run `python -m src.eval.runner` to produce a report with it.)*"
        )

    # The Production column of the headline table used to hardcode
    # "< 35 ms" regardless of what any run actually measured -- once
    # write_benchmark_summary_json started recording the real P95 (see its
    # docstring), a real run came back at 44.09ms, silently contradicting
    # this report's own claim. Use the real number when the caller has it;
    # fall back to the old narrative text only when no timings were passed
    # in (e.g. an older caller that hasn't been updated yet).
    prod_latency_display = f"{latency_p95_ms:.1f} ms" if isinstance(latency_p95_ms, (int, float)) else "< 35 ms (unmeasured estimate)"
    confusion_matrix_markdown = _render_confusion_matrix_markdown(
        prod_metrics["intent"].get("labels", []),
        prod_metrics["intent"].get("confusion_matrix", []),
    )

    report_content = f"""# Benchmark Report: AI Customer Support & Triage Agent for {TARGET_BRAND}

**Author**: Hiver SDE Intern Candidate
**Target Brand**: `{TARGET_BRAND}`
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)
**Golden Evaluation Set**: {gs['n']} Hand-Labelled Test Queries ({gs['n_edge']} disclosed edge cases, ~{gs['pct_edge']}%)
**Status**: Formal Evaluation & Verification Sign-Off

---

## 1. Executive Summary & Problem Framing

### 1.1 What "Good" Means for {TARGET_BRAND}
For Apple Support on Twitter, "good" does not mean simply generating fluent English. It requires:
1. **Zero Public Credential Solicitation**: Absolute fail-closed refusal to request passwords, credit cards, or full serial numbers in public tweets.
2. **Empathetic & Polished Brand Voice**: Calm, polite, concise responses strictly under Twitter's 280-character limit, recommending standard Apple diagnostic flows (Force Restart, Settings > Battery Health, Apple Store Genius Bar).
3. **Safe, Explainable Escalation**: Autonomous handling of routine informational queries (`AUTO_HANDLE`) while reliably escalating hazardous situations (battery swelling, cracked glass), angry legal threats, and model uncertainty to human agents (`ESCALATE`) with explicit stated reasons.
4. **Historical Grounding**: Recommending only real diagnostic workflows documented in Apple's historical resolution corpus and official domain links (`apple.co/...`, `support.apple.com/...`).

### 1.2 What We Chose NOT to Build (Explicit Scope Boundaries)
- **No Direct Bot Tweeting**: We do not deploy an unattended Twitter bot writing directly to the public API without human oversight. The system operates as an agent copilot and triage router.
- **No Internal Database Modification**: We do not simulate backend iCloud unlocks, warranty status overrides, or replacement device shipments. These are directed to the Apple Store Genius Bar.
- **No Multi-Lingual Support in v1**: Scope is strictly constrained to English tweets. Non-English queries fall back to `OUT_OF_SCOPE_AMBIGUOUS` for human routing.

---

## 2. Headline Results vs. Two Baselines

We evaluated three architectures across the exact same {gs['n']}-sample hand-labelled Golden Set:
1. **Baseline 1 (Trivial)**: Majority-class intent predictor (`OS_SOFTWARE_TROUBLESHOOTING`), static canned reply (*"Please restart your device"*), and always `AUTO_HANDLE`.
2. **Baseline 2 (Simple)**: TF-IDF + Logistic Regression intent classifier, nearest-neighbor historical reply retrieval without LLM re-ranking or length guardrails, and basic keyword escalation.
3. **Proposed System (Production)**: Dense semantic centroid classifier (`all-MiniLM-L6-v2`), ChromaDB historical resolution RAG, 280-char/whitelist guardrails, and cascading triage policy engine.

### Comparative Results Matrix

All lift figures below use Python's signed-float formatting (`:+`), so a
regression against the Simple baseline prints as a negative number rather
than being hidden behind a hardcoded "+" prefix.

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Proposed System (Production) | Absolute Lift (vs Simple) |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Macro-F1** | {trivial_metrics['intent']['macro_f1']:.4f} | {simple_metrics['intent']['macro_f1']:.4f} | **{prod_metrics['intent']['macro_f1']:.4f}** | **{(prod_metrics['intent']['macro_f1'] - simple_metrics['intent']['macro_f1']):+.4f}** |
| **Intent Accuracy** | {trivial_metrics['intent']['accuracy']*100:.1f}% | {simple_metrics['intent']['accuracy']*100:.1f}% | **{prod_metrics['intent']['accuracy']*100:.1f}%** | **{(prod_metrics['intent']['accuracy'] - simple_metrics['intent']['accuracy'])*100:+.1f}%** |
| **Triage Accuracy** | {trivial_metrics['triage']['accuracy']*100:.1f}% | {simple_metrics['triage']['accuracy']*100:.1f}% | **{prod_metrics['triage']['accuracy']*100:.1f}%** | **{(prod_metrics['triage']['accuracy'] - simple_metrics['triage']['accuracy'])*100:+.1f}%** |
| **Escalation Recall** | {trivial_metrics['triage']['escalation_recall']*100:.1f}% | {simple_metrics['triage']['escalation_recall']*100:.1f}% | **{prod_metrics['triage']['escalation_recall']*100:.1f}%** | **{(prod_metrics['triage']['escalation_recall'] - simple_metrics['triage']['escalation_recall'])*100:+.1f}%** |
| **Missed Escalations (Safety Risk)** | {trivial_metrics['triage']['missed_escalation_count']} / {prod_metrics['triage']['total_escalations_true']} | {simple_metrics['triage']['missed_escalation_count']} / {prod_metrics['triage']['total_escalations_true']} | **{prod_metrics['triage']['missed_escalation_count']} / {prod_metrics['triage']['total_escalations_true']}** | **{(simple_metrics['triage']['missed_escalation_count'] - prod_metrics['triage']['missed_escalation_count']):+d} fewer missed** |
| **ROUGE-L Grounding Score** | {trivial_metrics['rouge']['mean_rougeL']:.4f} | {simple_metrics['rouge']['mean_rougeL']:.4f} | **{prod_metrics['rouge']['mean_rougeL']:.4f}** | **{(prod_metrics['rouge']['mean_rougeL'] - simple_metrics['rouge']['mean_rougeL']):+.4f}** |
| **LLM Judge Quality (1-5 Scale)** | {trivial_metrics['judge']['overall_score']:.1f} / 5.0 | {simple_metrics['judge']['overall_score']:.1f} / 5.0 | **{judge_metrics['overall_score']:.1f} / 5.0** | **{(judge_metrics['overall_score'] - simple_metrics['judge']['overall_score']):+.1f}** |
| **P95 Latency (CPU)** | < 1 ms (unmeasured estimate) | ~5 ms (unmeasured estimate) | **{prod_latency_display}** | Real-time ready |

---

## 3. LLM-as-a-Judge & Human Agreement Calibration

To check whether the LLM-as-a-judge rubric can be trusted, we compared judge scores against **{agreement_metrics['num_samples']} human-scored query/reply pairs** (see `data/README.md` for how this sample was built and its disclosed limitations -- it is an AI-assisted reading pass against the rubric, not a blind independent annotator).

- **Sample Size**: {agreement_metrics['num_samples']} hand-annotated cases
- **Cohen's Kappa (Groundedness)**: $\kappa = {_fmt_kappa(agreement_metrics['cohen_kappa_groundedness'])}$
- **Cohen's Kappa (Safety)**: $\kappa = {_fmt_kappa(agreement_metrics['cohen_kappa_safety'])}$
- **Mean Cohen's Kappa**: **$\kappa = {_fmt_kappa(agreement_metrics['mean_cohen_kappa'])}$**
- **Interpretation**: **{agreement_metrics['agreement_interpretation']}**
- **Exact Agreement (Safety Gate)**: **{agreement_metrics['exact_agreement_safety_pct']:.1f}%**

> [!WARNING]
> Landis & Koch (1977) establish $\kappa \ge 0.61$ as substantial agreement. **This run's measured kappa does not clear that bar** (see the interpretation above) -- a previous version of this codebase silently floored the reported kappa at 0.72 (and safety kappa at 0.70) whenever exact agreement crossed 75%, which is why an earlier report could claim "high alignment" regardless of what was actually measured. Those floors have been removed; the numbers above are the real, unmodified output of `src/eval/human_agreement.py`. A mediocre or negative kappa here means the judge's numeric scores should not be trusted on their own -- see Section 5 for what this implies about the headline numbers above.

---

## 4. Top Failure Modes (Root Cause Analysis & Hypotheses)

Even with strong headline metrics, a thorough engineering audit requires identifying how the system fails. Unlike an earlier version of this report, the failure modes below are mined directly from this run's actual mismatches between predicted and true labels (see `src/eval/failure_analysis.py`) -- they are not a fixed illustrative list, so their frequencies and example queries will change between runs as the code and golden set change.

### 4.1 Intent Classification Confusion Matrix (Full Run)

`src/eval/metrics.py`'s `compute_intent_metrics()` has always computed this matrix, but nothing downstream ever read it -- the same "computed and then never used" pattern this audit already found once with the per-example `failures` list. Rows are the true label, columns the predicted label; reading across a row shows exactly where that class's real queries ended up. This is the aggregate, numeric counterpart to the individual examples narrated in Section 4.2 below -- in particular it shows at a glance whether `OUT_OF_SCOPE_AMBIGUOUS` is acting as a catch-all sink for other classes, which is the root cause the false-escalation failure mode keeps pointing back to (Gate 6 in `src/triage/engine.py` hard-escalates anything classified into that bucket).

{confusion_matrix_markdown}

### 4.2 Top Individual Failure Examples

"""
    if not top_failures:
        report_content += (
            "No prediction/label mismatches were found on this run against the loaded "
            "golden set -- there is no failure list to report. Treat a perfect score on "
            "a small, partly self-authored golden set with appropriate skepticism (see "
            "Section 5) rather than as evidence the system is production-ready.\n\n"
        )
    for i, fail in enumerate(top_failures, 1):
        report_content += f"""### Failure Mode {i}: {fail['title']}
- **Observed Frequency**: {fail['count']} of {fail['total_failures']} failures on this run (~{fail['frequency']}%)
- **Real Example Query**: *"{fail['query']}"*
- **Actual System Output**: {fail['actual']}
- **Expected (Golden Label)**: {fail['expected']}
- **System's Stated Reason**: {fail['system_stated_reason']}
- **Root Cause Hypothesis**: {fail['hypothesis']}
- **Mitigation Strategy**: {fail['mitigation']}

"""

    report_content += f"""---

## 5. "What is Misleading About My Headline Number?" (Mandatory Section)

While our **Macro-F1 of {prod_metrics['intent']['macro_f1']:.4f}** and **Triage Accuracy of {prod_metrics['triage']['accuracy']*100:.1f}%** may look strong in isolation, headline numbers conceal subtle real-world failure patterns -- and, per Section 3, the human-agreement kappa on the judge itself is currently weak, which should temper confidence in any of the judge-derived numbers above:

1. **The Golden Set's Escalation Rate Is Deliberately ~20x the Real Rate**:
   Of the {prod_metrics['triage']['total_escalations_true']} true-ESCALATE rows in this {gs['n']}-row golden set (~{round(100*prod_metrics['triage']['total_escalations_true']/gs['n']) if gs['n'] else 0}%), the large majority were manually reviewed and, in several cases, authored as adversarial examples (`source: authored_adversarial` in `data/golden_eval_set.jsonl`) -- because an unweighted random sample of the real Kaggle pairs surfaced only ~9 genuine escalation-worthy tweets out of 995 (well under 1%). This oversampling was a deliberate, disclosed choice (see `data/README.md`) to get enough escalation examples to measure precision/recall at all -- but it means Escalation Recall/Precision above describe performance on an escalation-enriched sample, not the real-world base rate. On real unfiltered traffic, the same false-escalation rules would fire far less often in absolute terms, and the cost of a single missed escalation (safety-relevant) is not comparable to the cost of a single false one (ticket volume) -- a blended "Triage Accuracy" number hides that asymmetry entirely.

2. **Isolated Single-Turn Evaluation**:
   Our evaluation measures single-turn tweet resolution. Real support threads often span 4–7 turns where customers clarify details ("Oh wait, it's actually an iPad, not an iPhone"). High single-turn groundedness does not guarantee conversational coherence across long context windows.

3. **Conservative Over-Escalation Bias**:
   To ensure zero safety violations, our triage threshold aggressively errs on the side of caution. While this achieves a near-perfect Missed Escalation Rate ({prod_metrics['triage']['missed_escalation_count']} missed safety cases), it inflates human agent ticket volume by ~{prod_metrics['triage']['false_escalation_count']} false escalations. In an enterprise setting, this increases operational cost.

4. **Kaggle Dataset Age & Link Rot**:
   The `customer-support-on-twitter` dataset dates to 2017–2018 (iOS 11 era). References to `apple.co` URLs and specific iOS menu hierarchies may have evolved (e.g., Settings layouts in iOS 17/18). High historical similarity measures fidelity to 2018 procedures rather than current 2026 support documentation.

5. **Until This Run, The Thresholds Were Tuned On The Evaluation Set**:
   `scripts/calibrate_thresholds.py` swept `MIN_INTENT_CONFIDENCE` and `MIN_RETRIEVAL_SIMILARITY` against the golden set, and this harness then reported headline numbers on *those same rows* -- with no train/test separation anywhere in the repo. `docs/AUDIT_AND_FIX_PLAN.md` §7.10 records that sweep being run and both thresholds being changed on the strength of it (`MIN_RETRIEVAL_SIMILARITY` 0.40 → 0.20, `MIN_INTENT_CONFIDENCE` 0.35 → 0.40). Every triage number published before this run was therefore optimistically biased by construction, and none of the four caveats above disclosed it.
   {split_disclosure}

---

## 5b. Grounding Check: Lexical vs Embedding (Measured, Not Assumed)

`docs/DECISION_LOG.md` #14 recorded that the grounding guardrail's bag-of-words overlap check was a fallback from when the embedding model could not be loaded in the development environment. It can be now, so both were implemented and measured against each other rather than the newer one simply being assumed better.

**Method.** Two populations. (a) Seven hand-authored probe drafts against one retrieved snippet, labelled by whether the snippet actually *supports* the draft's claim. (b) All 188 golden-set reference replies -- real historical `@AppleSupport` agent replies -- each scored against what the retriever returns for its own row. Population (b) is grounded by construction, so anything a check flags there is a false positive.

| Check | Probe verdicts correct | False-positive rate on 188 real agent replies |
| :--- | :---: | :---: |
| Lexical overlap, floor 0.12 | 5 / 7 | **18.6%** |
| Embedding cosine, floor 0.30 | 5 / 7 | **9.6%** |
| Embedding cosine, floor 0.65 | 7 / 7 | **33.0%** |

**What was chosen and why.** Embedding similarity at a 0.30 floor, which halves the false-positive rate on genuine replies (9.6% vs 18.6%) at identical probe accuracy. Equal probe scores hide that the two checks fail on *different* cases: the lexical check returns a grounding score of **1.00** for the truncated fragment `"We'd like to"` (its no-content-words branch short-circuits to "fine"), while the embedding check scores that ~0.00; conversely the lexical check catches a draft recommending a full OS reinstall against a "force restart" snippet, which the embedding check passes.

**The 7/7 row is a trap, and is listed to show why it was rejected.** A 0.65 floor scores perfectly on the seven probes -- but those probes were hand-written, and fitting a threshold to them is the same error as tuning thresholds on the evaluation set (item 5 above). Measured against real replies, that floor would wrongly escalate a third of genuine historical answers. A 3.4x increase in false escalations to close one class of catch is not a trade worth making.

**Known limitation, stated plainly.** Similarity is not entailment. At the chosen floor, a draft giving *different but topically related* advice than the retrieved snippet still passes (0.64), and so does one that recycles the snippet's vocabulary into an invented claim (0.42). Neither check detects unsupported-but-on-topic assertions, because neither is a model of support. The honest fix is a natural-language-inference model scoring whether the snippet entails the draft; that is a larger change than this pass, and is not pretended to be solved here. `tests/test_grounding_modes.py` asserts the blind spot explicitly so it cannot close or widen unnoticed.

---

## 6. What We'd Do Next With One More Week

1. **Active Learning Feedback Loop**: Stream human agent accept/reject/edit decisions on auto-drafted replies back into the vector store as fresh, human-validated few-shot examples.
2. **Multi-Turn Thread Context Buffer**: Ingest conversation tree ancestors (`in_reply_to_tweet_id`) using DuckDB to preserve previous diagnostics and avoid asking redundant questions.
3. **Dynamic Threshold Optimization**: Use Bayesian optimization over golden set validation splits to tune the confidence gates ($\tau_{{intent}}, \tau_{{sim}}$) targeting a specific cost-per-escalation trade-off curve.
4. **Automated Red-Teaming Suite**: Deploy an automated prompt injection and jailbreak tester attempting to induce the agent into offering fake Apple gift cards or revealing internal prompts.

---

## 7. Decision Log (15 Non-Obvious Engineering Decisions)

1. **Selected @AppleSupport over Retail Brands**: Chose AppleSupport because consumer electronics customer support has strict diagnostic procedures, high stakes (lithium battery safety), and well-defined escalation policies.
2. **Embedded Vector Store (ChromaDB) over Hosted SaaS**: Opted for in-process SQLite ChromaDB to ensure the evaluation harness runs offline in <15 minutes with zero external infrastructure setup.
3. **Cascading Priority Triage Gate over Single LLM Score**: Chose a cascading deterministic gate (Prompt Injection $\\rightarrow$ Safety Regex $\\rightarrow$ PII $\\rightarrow$ Human Request $\\rightarrow$ Sentiment $\\rightarrow$ Model Confidence $\\rightarrow$ Clarify $\\rightarrow$ Similarity $\\rightarrow$ Generation Guardrails) rather than trusting a single LLM to decide safety, eliminating hallucination risks on physical hazards.
4. **Normalized Softmax Temperature Scaling on Cosine Similarities**: Applied temperature scaling ($T=0.12$) to raw cosine similarities to produce calibrated, bounded probability distributions for intent confidence.
5. **Zero Tolerance for Public PII Request, Plus PII-Echo Detection**: Strictly prohibited asking for Apple ID passwords or serial numbers in public tweets, *and* added a generation-time guardrail (`check_pii_echo`) that blocks a draft if it echoes back PII-shaped text the customer themselves posted (e.g. a phone number), rather than only checking the agent's own requests.
6. **Intent-Filtered Vector Retrieval**: Filtered ChromaDB queries by the classified intent to prevent semantic drift between unrelated topics (e.g., battery drain queries matching iPad display issues).
7. **Fail-Closed Circuit Breaker on Unhandled Exceptions**: Implemented a global try-except wrapper that unconditionally defaults to `ESCALATE` with `SYSTEM_EXCEPTION_FAIL_CLOSED` if any component crashes.
8. **Negation-Aware Hazard/PII Regex Instead of Bare Keyword Matching**: An audit of the original rules found real false positives from bare keyword matches (e.g. "battery didn't catch fire" would have matched a naive `fire` pattern). Rewrote the hazard/PII/legal regex with a `_is_negated()` window check and tightened device-noun proximity requirements, verified against real Kaggle tweets that previously would have been misrouted.
9. **Corroboration-Gated Sentiment Escalation Instead of Single-Keyword Triggers**: The audit's manual review of real escalation candidates found genuine false positives -- "reported a scam **to** Apple" (not a victim) and a support case number matching a phone regex both would have incorrectly escalated. Split legal/churn keywords into an auto-trigger tier (unambiguous: "lawyer", "police") and a soft tier ("sue", "scam") that requires a second corroborating signal before escalating, directly reducing this false-positive class.
10. **Added a Third Triage Action (CLARIFY) Instead of a Binary AUTO_HANDLE/ESCALATE Split**: A real historical case in the source data showed an @AppleSupport agent asking a clarifying question ("which device is this?") rather than either auto-replying or escalating. Added `TriageAction.CLARIFY`, a confidence-band gate that fires only for device-dependent intents where the customer never named a device, and a matching clarifying-question template -- reducing pressure on the binary gate to force every ambiguous case into an over-cautious escalation.
11. **Non-Circular Golden-Set Labelling**: The original golden set's intent/triage labels were generated by the same-family logic the system under test would later be graded against, and a hardcoded human-agreement floor meant no eval could ever fail. Rebuilt the golden set from the real Kaggle `@AppleSupport` pairs using an independent keyword heuristic (not the production classifier's prototypes) for the first labelling pass, followed by manual review of every real escalation candidate -- disclosed in `data/README.md`, including the heuristic's ~41% disagreement rate with the original ingestion-time classifier as a data point about how noisy this domain actually is.
12. **Removed Both Hardcoded Kappa Floors and Report the Real Number, Even When It's Bad**: `src/eval/human_agreement.py` previously did `max(kappa_g, 0.72)` and `max(0.70, kappa_s)` regardless of the actual computed values. Removed both; this run's real measured kappa is honestly reported in Section 3, including the case where it falls in "Slight agreement" or is negative -- an eval harness that cannot report a bad result is not measuring anything.
13. **URL Domain Whitelisting via Regex Guardrail**: Restricted drafted links to `apple.co` and `support.apple.com`, stripping or flagging any LLM-hallucinated third-party domains.
14. **Lexical-Overlap Grounding Check as a Model-Free Hallucination Proxy**: Without a reachable embedding model in every environment this code needs to run in, added `check_grounding()` -- a lexical content-word overlap check between a draft and its retrieved snippets -- as a cheap, dependency-light signal that a draft isn't inventing procedures unrelated to what was actually retrieved.
15. **Real Kaggle Pairs in the RAG Corpus Instead of a 10-Example Hand-Written Seed Set, With Leakage Exclusion**: `HistoricalVectorStore` now indexes real `@AppleSupport` historical replies (`load_real_corpus()`), explicitly excluding every `source_tweet_id` present in the golden set so the eval can't retrieve its own answer key -- falling back to the small hand-written seed corpus only if the real pairs file is unavailable.
"""

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


def write_benchmark_summary_json(
    trivial_metrics: Dict[str, Any],
    simple_metrics: Dict[str, Any],
    prod_metrics: Dict[str, Any],
    agreement_metrics: Dict[str, Any],
    latency_p95_ms: Optional[float] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Writes a small, machine-readable sibling of docs/REPORT.md so the live
    dashboard (src/server.py's /api/benchmark-summary) can render the real
    numbers from the most recent eval run instead of a human hand-copying
    figures out of the markdown report into src/static/index.html every time
    the golden set, thresholds, or classifier change -- the exact kind of
    silent-staleness bug this project's audit has repeatedly had to catch
    (the old hardcoded kappa floor, the old hardcoded golden-set size, the
    old hardcoded "run" CLI argument in docs, etc.).

    Every value here is read straight off the same metrics dicts
    generate_markdown_report() uses -- nothing here is computed separately,
    so the JSON and the markdown report can never silently disagree with
    each other.
    """
    gs = _golden_set_stats()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_set": gs,
        # Which rows these numbers came from. Recorded here so the dashboard can
        # never display a headline figure without being able to say it was
        # measured on rows the thresholds were not tuned against -- the bias
        # documented in REPORT.md section 5, item 5.
        "split": split_info or {"note": "run produced no split information"},
        "triage": {
            "trivial": {
                "accuracy": trivial_metrics["triage"]["accuracy"],
                "escalation_recall": trivial_metrics["triage"]["escalation_recall"],
                "missed_escalation_count": trivial_metrics["triage"]["missed_escalation_count"],
            },
            "simple": {
                "accuracy": simple_metrics["triage"]["accuracy"],
                "escalation_recall": simple_metrics["triage"]["escalation_recall"],
                "missed_escalation_count": simple_metrics["triage"]["missed_escalation_count"],
            },
            "production": {
                "accuracy": prod_metrics["triage"]["accuracy"],
                "escalation_recall": prod_metrics["triage"]["escalation_recall"],
                "missed_escalation_count": prod_metrics["triage"]["missed_escalation_count"],
                "total_escalations_true": prod_metrics["triage"]["total_escalations_true"],
            },
        },
        "intent": {
            "production": {
                "macro_f1": prod_metrics["intent"]["macro_f1"],
                "accuracy": prod_metrics["intent"]["accuracy"],
                # Same matrix compute_intent_metrics() has always returned
                # and docs/REPORT.md Section 4.1 now renders -- included here
                # too so the dashboard can show the identical numbers rather
                # than a second, potentially-drifting recomputation.
                "labels": prod_metrics["intent"].get("labels", []),
                "confusion_matrix": prod_metrics["intent"].get("confusion_matrix", []),
            },
        },
        "judge": {
            "simple": {"overall_score": simple_metrics["judge"]["overall_score"]},
            "production": {"overall_score": prod_metrics["judge"]["overall_score"]},
        },
        "human_agreement": {
            # None (not a fabricated floor) when kappa is genuinely undefined
            # -- see _fmt_kappa's docstring above for why that can happen.
            "mean_cohen_kappa": agreement_metrics.get("mean_cohen_kappa"),
            "agreement_interpretation": agreement_metrics.get("agreement_interpretation"),
            "exact_agreement_safety_pct": agreement_metrics.get("exact_agreement_safety_pct"),
            "num_samples": agreement_metrics.get("num_samples"),
        },
        "latency_ms": {
            # Real measured P95 across this run's production responses, not
            # the "< 35 ms" narrative figure the markdown report and the old
            # hardcoded dashboard both used -- null if the caller didn't
            # have per-response timings to compute it from (e.g. a very old
            # runner.py that hasn't been updated to pass this yet).
            "p95_production": latency_p95_ms,
        },
    }

    BENCHMARK_SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
