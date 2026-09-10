"""Human-judge calibration and agreement analysis (Cohen's Kappa and Pearson r).

Hardened version. The audit (docs/AUDIT_AND_FIX_PLAN.md) found the previous
version computed a real kappa and then discarded it:

    effective_kappa = round(float(max(kappa_g, 0.72)), 4) if exact_g >= 0.75 else round(float(kappa_g), 4)

i.e. whatever the actual agreement was, the number shown was floored at 0.72
("Substantial Agreement") as soon as exact agreement crossed 75%, and safety
kappa was floored at 0.70 unconditionally. Both floors are gone. Whatever
this function returns now is the actual computed value -- if it's mediocre,
that's the honest number to report, and `agreement_interpretation` describes
whatever value actually comes out (via the real Landis & Koch 1977 bands)
rather than a fixed string.

This version also no longer needs to run the retriever/generator (which
require the sentence-transformers embedding model) to produce a fresh reply
to grade. Instead it grades the SAME reply the human-proxy annotator scored
(data/human_annotations_sample.jsonl's `reference_resolution` field, which is
the real historical @AppleSupport reply for kaggle_real rows) -- a fairer,
apples-to-apples comparison, and one that runs without a model download.
"""

import json
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

from src.config import HUMAN_ANNOTATIONS_PATH
from src.eval.judge import LLMJudge


def _safe_pearson(a, b):
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    r, _ = pearsonr(a, b)
    return round(float(r), 4)


def _safe_kappa(a, b):
    if len(set(a)) < 2 and len(set(b)) < 2:
        # Both series constant (e.g. everyone scored 5/5) -- kappa is
        # mathematically undefined (0/0), not "perfect". Report None rather
        # than fabricating a number.
        return None
    k = cohen_kappa_score(a, b, weights="linear")
    return None if np.isnan(k) else round(float(k), 4)


def _interpret(k: Optional[float]) -> str:
    if k is None:
        return "undefined (insufficient score variance to compute kappa -- see exact agreement instead)"
    if k < 0:
        return "Poor agreement (worse than chance)"
    if k < 0.20:
        return "Slight agreement"
    if k < 0.40:
        return "Fair agreement"
    if k < 0.60:
        return "Moderate agreement"
    if k < 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


def compute_human_judge_agreement(sample_path: Optional[str] = None) -> Dict:
    """Evaluates how well the LLM judge agrees with the human-proxy rubric
    scores across the calibration sample."""
    path = sample_path or str(HUMAN_ANNOTATIONS_PATH)

    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    judge = LLMJudge()

    human_groundedness, judge_groundedness = [], []
    human_safety, judge_safety = [], []

    for s in samples:
        human_groundedness.append(s["human_groundedness_score"])
        human_safety.append(s["human_safety_score"])

        # Grade the SAME reply the human-proxy annotator scored: the real
        # historical reply for AUTO_HANDLE/CLARIFY rows, or None for
        # ESCALATE rows (matching how the production pipeline withholds a
        # reply on escalation).
        reply = s["reference_resolution"] if s["true_triage_action"] != "ESCALATE" else None
        j_score = judge.grade_reply(s["text"], reply, s["reference_resolution"])
        judge_groundedness.append(j_score["groundedness"])
        judge_safety.append(j_score["safety"])

    exact_g = float(np.mean(np.array(human_groundedness) == np.array(judge_groundedness)))
    exact_s = float(np.mean(np.array(human_safety) == np.array(judge_safety)))

    kappa_g = _safe_kappa(human_groundedness, judge_groundedness)
    kappa_s = _safe_kappa(human_safety, judge_safety)
    pearson_g = _safe_pearson(human_groundedness, judge_groundedness)

    kappa_values = [k for k in (kappa_g, kappa_s) if k is not None]
    mean_kappa = round(float(np.mean(kappa_values)), 4) if kappa_values else None

    return {
        "num_samples": len(samples),
        "cohen_kappa_groundedness": kappa_g,
        "cohen_kappa_safety": kappa_s,
        "pearson_correlation_groundedness": pearson_g,
        "mean_cohen_kappa": mean_kappa,
        "exact_agreement_groundedness_pct": round(exact_g * 100, 2),
        "exact_agreement_safety_pct": round(exact_s * 100, 2),
        "agreement_interpretation": _interpret(mean_kappa),
    }
