"""Automated statistical evaluation metrics for Intent, Triage, and Reply Quality."""

from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Computes Macro-F1, Accuracy, and per-class metrics for intent classification."""
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": report[cls]["support"],
            }
            for cls in labels
            if cls in report
        },
        "confusion_matrix": cm,
        "labels": labels,
    }


def compute_triage_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Computes accuracy, escalation recall/precision, and false handling rates."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Escalation is the critical safety class
    esc_precision = report.get("ESCALATE", {}).get("precision", 0.0)
    esc_recall = report.get("ESCALATE", {}).get("recall", 0.0)
    esc_f1 = report.get("ESCALATE", {}).get("f1-score", 0.0)

    # Calculate safety-critical missed escalations: True=ESCALATE, Pred=AUTO_HANDLE
    missed_escalations = sum(
        1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == "ESCALATE" and yp == "AUTO_HANDLE"
    )
    false_escalations = sum(
        1 for yt, yp in zip(y_true, y_pred, strict=False) if yt == "AUTO_HANDLE" and yp == "ESCALATE"
    )
    total_escalations_true = sum(1 for yt in y_true if yt == "ESCALATE")

    missed_rate = (missed_escalations / total_escalations_true) if total_escalations_true > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "escalation_precision": round(float(esc_precision), 4),
        "escalation_recall": round(float(esc_recall), 4),
        "escalation_f1": round(float(esc_f1), 4),
        "missed_escalation_count": missed_escalations,
        "missed_escalation_rate": round(float(missed_rate), 4),
        "false_escalation_count": false_escalations,
        # Was hardcoded as a literal "/ 30" in the report/terminal table
        # regardless of how many true escalations were actually in the
        # loaded golden set -- now the real denominator is available to format with.
        "total_escalations_true": total_escalations_true,
        # The full triage confusion, because accuracy alone hid the single largest
        # error class in this system. Adversarial review round 3 computed it by
        # hand and found 23 of the 46 triage errors were AUTO_HANDLE -> CLARIFY --
        # more than the 21 false escalations the report blamed for the low
        # accuracy, and 22% of all genuine AUTO_HANDLE traffic. No document
        # mentioned it, and three of them said the CLARIFY path was "covered by
        # unit tests only, not by this benchmark", which is true only of the
        # LABEL: the benchmark exercises the CLARIFY gate on 23 of 124 rows and it
        # is wrong every time it fires. Computed here so it cannot go unreported
        # again.
        "confusion": {
            f"{yt}->{yp}": sum(
                1 for a, b in zip(y_true, y_pred, strict=False) if a == yt and b == yp
            )
            for yt in sorted(set(y_true) | set(y_pred))
            for yp in sorted(set(y_true) | set(y_pred))
            if sum(1 for a, b in zip(y_true, y_pred, strict=False) if a == yt and b == yp)
        },
        "false_clarify_count": sum(
            1
            for yt, yp in zip(y_true, y_pred, strict=False)
            if yp == "CLARIFY" and yt != "CLARIFY"
        ),
    }


def compute_rouge_similarity(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Computes average ROUGE-1 and ROUGE-L scores against reference resolutions."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores = []
    rl_scores = []

    for ref, hyp in zip(references, hypotheses, strict=False):
        if not hyp:
            continue
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    mean_r1 = float(np.mean(r1_scores)) if r1_scores else 0.0
    mean_rl = float(np.mean(rl_scores)) if rl_scores else 0.0

    return {
        "mean_rouge1": round(mean_r1, 4),
        "mean_rougeL": round(mean_rl, 4),
    }
