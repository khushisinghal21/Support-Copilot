"""Empirically calibrates MIN_INTENT_CONFIDENCE and MIN_RETRIEVAL_SIMILARITY
against the CALIBRATION SPLIT of the golden set, instead of the values in
src/config.py being a guess someone typed in once and never revisited.

WHY THIS EXISTS
----------------
The audit (docs/AUDIT_AND_FIX_PLAN.md) found a real doc/code mismatch: the
README claimed the intent-confidence guard fires below tau=0.60, while
config.py at the time set MIN_INTENT_CONFIDENCE=0.35 (it is 0.40 now, changed by
the section 7.10 sweep). Neither number had ever
been checked against data -- they were both just typed in. This script
replaces "pick a threshold and hope" with an actual precision/recall sweep:

  1. Runs the real SemanticCentroidClassifier over every row in the golden
     set and records its confidence score and whether its predicted intent
     matched the true label.
  2. Sweeps candidate MIN_INTENT_CONFIDENCE values and reports, at each
     value, how many genuinely-wrong predictions the threshold would catch
     (escalate instead of confidently-wrong auto-handle) versus how many
     genuinely-correct predictions it would needlessly escalate.
  3. Does the same for MIN_RETRIEVAL_SIMILARITY using the real
     HistoricalRetriever's max similarity per row against whether the
     retrieved historical reply is actually a reasonable match for the
     query's true intent.
  4. Prints a recommended value for each (the knee of the trade-off curve)
     -- but does not silently overwrite config.py, since the "right" choice
     also depends on a cost trade-off (missed safety escalation vs.
     unnecessary human review) that's a product decision, not a purely
     statistical one.

WHY IT ONLY READS THE CALIBRATION SPLIT
----------------------------------------
It used to sweep the entire golden set -- the same rows src/eval/runner.py then
reported headline numbers on. Thresholds tuned on the evaluation data make every
reported triage number optimistically biased, and nothing disclosed that. Since
the split was added (src/eval/splits.py), this script sees the `calibration`
rows only: roughly a third of the set, stratified on triage label and edge-case
status. The held-out majority is never read here, so a threshold chosen from
this output is not chosen on the data it will later be scored against.

RUNNING IT
-----------
This script loads the sentence-transformers embedding model `all-MiniLM-L6-v2`
via SemanticCentroidClassifier and HistoricalRetriever:

    python scripts/calibrate_thresholds.py

An earlier version of this docstring said the script had never been executed,
because huggingface.co was blocked at the network policy layer in the
environment where it was written (a direct curl returned 403, in both the cloud
tool-runner and the local device shell). That is no longer accurate: it has since
been run for real, and docs/AUDIT_AND_FIX_PLAN.md section 7.10 records the sweep
output and the threshold changes made on the strength of it. If you hit the same
network block, point EMBEDDING_MODEL_NAME at a local copy of the model directory
instead of a hub id.

Whatever it reports, it does not silently overwrite src/config.py -- the right
threshold depends on the real cost ratio between a missed safety escalation and
an unnecessary human review, which is a product decision, not a statistical one.
"""

import sys
from pathlib import Path

# Allows running this as a bare script (`python scripts/calibrate_thresholds.py`)
# from anywhere, not just via `python -m scripts.calibrate_thresholds` -- without
# this, Python only puts this file's own directory (scripts/) on sys.path, and
# `from src...` below fails with ModuleNotFoundError. Same pattern already used
# in scripts/build_golden_set.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.drafting.retriever import HistoricalRetriever
from src.eval.splits import CALIBRATION, load_golden_rows
from src.intent.classifier import SemanticCentroidClassifier

# Candidate thresholds to sweep. Fine enough to see the trade-off curve's
# shape without an unreasonable number of classifier calls.
CONFIDENCE_CANDIDATES = [round(0.05 * i, 2) for i in range(2, 16)]  # 0.10 .. 0.75
SIMILARITY_CANDIDATES = [round(0.05 * i, 2) for i in range(2, 16)]  # 0.10 .. 0.75


def _load_golden() -> list[dict]:
    """Calibration rows only -- see this module's docstring. Reading the held-out
    rows here is the bug this function exists to prevent."""
    return load_golden_rows(split=CALIBRATION)


def _collect_confidence_data(rows: list[dict], clf: SemanticCentroidClassifier) -> list[tuple[float, bool]]:
    """Returns (confidence, was_prediction_correct) per row."""
    out = []
    for r in rows:
        res = clf.predict(r["text"])
        correct = res.primary_intent.value == r["true_intent"]
        out.append((res.confidence, correct))
    return out


def _collect_similarity_data(rows: list[dict], retriever: HistoricalRetriever) -> list[tuple[float, bool]]:
    """Returns (max_similarity, is_this_row_a_true_auto_handle) per row --
    a low-grounding escalation should mostly fire on rows that genuinely
    need a human anyway, not on ones the golden set says are safe to
    auto-handle."""
    out = []
    for r in rows:
        rag = retriever.retrieve(query=r["text"], intent=r["true_intent"], k=3)
        is_auto_handle = r["true_triage_action"] == "AUTO_HANDLE"
        out.append((rag.max_similarity, is_auto_handle))
    return out


def _sweep_confidence(data: list[tuple[float, bool]]) -> None:
    print("\n=== MIN_INTENT_CONFIDENCE sweep ===")
    print(f"{'tau':>6} | {'wrong preds caught':>19} | {'correct preds needlessly escalated':>36}")
    total_wrong = sum(1 for _, correct in data if not correct)
    total_correct = sum(1 for _, correct in data if correct)
    best_tau, best_score = None, -1.0
    for tau in CONFIDENCE_CANDIDATES:
        caught_wrong = sum(1 for conf, correct in data if not correct and conf < tau)
        escalated_correct = sum(1 for conf, correct in data if correct and conf < tau)
        recall_wrong = caught_wrong / total_wrong if total_wrong else 0.0
        fp_rate = escalated_correct / total_correct if total_correct else 0.0
        # Youden's J on this 2x2 -- simple, defensible knee-of-curve metric.
        j = recall_wrong - fp_rate
        if j > best_score:
            best_score, best_tau = j, tau
        print(f"{tau:>6.2f} | {caught_wrong:>6}/{total_wrong:<3} ({recall_wrong:>5.1%})   | "
              f"{escalated_correct:>6}/{total_correct:<3} ({fp_rate:>5.1%})")
    print(f"\nSuggested MIN_INTENT_CONFIDENCE (max recall-minus-false-positive-rate): {best_tau}")
    print("Current src/config.py value: MIN_INTENT_CONFIDENCE (see that file)")


def _sweep_similarity(data: list[tuple[float, bool]]) -> None:
    print("\n=== MIN_RETRIEVAL_SIMILARITY sweep ===")
    print(f"{'tau':>6} | {'true-escalate rows below tau':>29} | {'true-auto-handle rows below tau (bad)':>38}")
    total_escalate = sum(1 for _, is_auto in data if not is_auto)
    total_auto = sum(1 for _, is_auto in data if is_auto)
    best_tau, best_score = None, -1.0
    for tau in SIMILARITY_CANDIDATES:
        below_escalate = sum(1 for sim, is_auto in data if not is_auto and sim < tau)
        below_auto = sum(1 for sim, is_auto in data if is_auto and sim < tau)
        recall = below_escalate / total_escalate if total_escalate else 0.0
        fp_rate = below_auto / total_auto if total_auto else 0.0
        j = recall - fp_rate
        if j > best_score:
            best_score, best_tau = j, tau
        print(f"{tau:>6.2f} | {below_escalate:>6}/{total_escalate:<3} ({recall:>5.1%})     | "
              f"{below_auto:>6}/{total_auto:<3} ({fp_rate:>5.1%})")
    print(f"\nSuggested MIN_RETRIEVAL_SIMILARITY (max recall-minus-false-positive-rate): {best_tau}")
    print("Current src/config.py value: MIN_RETRIEVAL_SIMILARITY (see that file)")


def main():
    rows = _load_golden()
    print(
        f"Loaded {len(rows)} CALIBRATION-split rows "
        f"(the held-out rows are deliberately not read by this script)"
    )

    print("Instantiating SemanticCentroidClassifier (downloads all-MiniLM-L6-v2 on first run)...")
    clf = SemanticCentroidClassifier()
    confidence_data = _collect_confidence_data(rows, clf)
    _sweep_confidence(confidence_data)

    print("\nInstantiating HistoricalRetriever (uses the same embedding model + ChromaDB)...")
    retriever = HistoricalRetriever()
    similarity_data = _collect_similarity_data(rows, retriever)
    _sweep_similarity(similarity_data)

    print(
        "\nNOTE: 'best' here maximizes Youden's J (recall - false-positive-rate), a "
        "reasonable statistical default. The actual right threshold also depends on "
        "the real cost ratio between a missed safety escalation and an unnecessary "
        "human review, which is a product decision -- treat this as evidence to bring "
        "to that decision, not an answer to apply blindly."
    )


if __name__ == "__main__":
    main()
