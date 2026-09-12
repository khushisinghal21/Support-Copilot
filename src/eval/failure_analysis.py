"""Mines real failure modes from a completed evaluation run.

Replaces the previous approach in src/eval/runner.py, which computed a real
`failures` list by comparing predictions to ground truth and then never used
it -- the "Top 5 Failure Modes" in the generated report was a separate
hardcoded list (with specific frequencies like "~35% of error cases")
completely disconnected from whatever the pipeline actually got wrong on
that run. This module clusters the real failures by error type and reports
actual counts/examples instead.
"""

from collections import Counter, defaultdict
from typing import Any

# Generic, honest hypothesis/mitigation text per error category. These
# describe the *type* of failure (data-independent), while the frequency and
# example query come from the actual run.
_CATEGORY_INFO = {
    "intent_confusion": {
        "title_fmt": "Intent confused between {true} and {pred}",
        "hypothesis": (
            "The semantic centroid classifier's prototype sentences for these two "
            "classes overlap in embedding space for this phrasing (e.g. a symptom "
            "that could plausibly be filed under either category)."
        ),
        "mitigation": (
            "Add more prototype sentences that disambiguate this specific pair, or "
            "allow a secondary-intent hint to route to human review when the top two "
            "classes are near-tied."
        ),
    },
    "missed_escalation": {
        "title_fmt": "Missed escalation: predicted AUTO_HANDLE, true label was ESCALATE",
        "hypothesis": (
            "A safety-relevant signal in the text wasn't caught by the current "
            "regex/keyword rules -- most likely a phrasing variant the hazard/PII/"
            "legal-threat patterns don't cover."
        ),
        "mitigation": (
            "This is the highest-priority failure category to fix regardless of "
            "overall accuracy: review each case's exact wording and add the missed "
            "pattern to src/triage/rules.py or src/triage/sentiment.py."
        ),
    },
    "missed_clarify": {
        "title_fmt": "Missed clarification: predicted AUTO_HANDLE/ESCALATE, true label was CLARIFY",
        "hypothesis": (
            "The device-ambiguity heuristic (src/triage/engine.py's CLARIFY gate) "
            "didn't fire for this phrasing -- either the confidence band didn't "
            "match, or a device noun was detected that the customer didn't actually "
            "specify precisely enough."
        ),
        "mitigation": (
            "Review whether the confidence band or device-noun regex needs widening "
            "for this case; consider adding it as a new CLARIFY prototype pattern."
        ),
    },
    "false_escalation": {
        "title_fmt": "False escalation: predicted ESCALATE, true label was AUTO_HANDLE",
        "hypothesis": (
            "One of several triage gates can cause this (see the 'System's stated "
            "reason' line on the example below for which one actually fired on this "
            "run -- this hypothesis text used to guess 'the frustration/legal "
            "keyword gate' unconditionally, which was often wrong: a query "
            "misclassified as OUT_OF_SCOPE_AMBIGUOUS is hard-escalated by Gate 6 "
            "regardless of sentiment, and looks identical to a sentiment-gate false "
            "positive in this summary unless you check the stated reason)."
        ),
        "mitigation": (
            "Check the stated reason on the example below first. "
            "LOW_CONFIDENCE_AMBIGUOUS pointing at OUT_OF_SCOPE_AMBIGUOUS means the "
            "*intent classifier* misfired (fix: src/intent/taxonomy.py's prototypes "
            "for that class), not the sentiment gate. HIGH_FRUSTRATION_CHURN_RISK "
            "means the corroboration requirement in src/triage/sentiment.py still "
            "needs tightening or a hard-negative regression test for this phrasing."
        ),
    },
    "other": {
        "title_fmt": "Other triage mismatch ({true} -> {pred})",
        "hypothesis": "Doesn't fit a common pattern -- needs individual review.",
        "mitigation": "Read the specific case and decide whether it's a labelling error or a real gap.",
    },
}


def _categorize(record: dict[str, Any]) -> str:
    true_intent = record.get("true_intent")
    pred_intent = record.get("pred_intent")
    true_triage = record.get("true_triage")
    pred_triage = record.get("pred_triage")

    if true_triage == "ESCALATE" and pred_triage == "AUTO_HANDLE":
        return "missed_escalation"
    if true_triage == "AUTO_HANDLE" and pred_triage == "ESCALATE":
        return "false_escalation"
    if true_triage == "CLARIFY" and pred_triage != "CLARIFY":
        return "missed_clarify"
    if true_intent != pred_intent:
        return "intent_confusion"
    return "other"


def mine_failure_modes(failures: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """Clusters the real per-example failures from an eval run into named
    categories and returns the top `top_n` by frequency, each with one real
    example pulled from the actual run."""
    if not failures:
        return []

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in failures:
        by_category[_categorize(f)].append(f)

    counts = Counter({cat: len(items) for cat, items in by_category.items()})
    total = len(failures)

    results = []
    for cat, count in counts.most_common(top_n):
        info = _CATEGORY_INFO.get(cat, _CATEGORY_INFO["other"])

        # The title names a SPECIFIC (true -> pred) pair, so it must be the pair
        # the count describes. It used to be formatted from by_category[cat][0] --
        # an arbitrary example -- while `count` was the whole category. On a real
        # run that published "28 of 66 failures (~42%): Intent confused between
        # OUT_OF_SCOPE_AMBIGUOUS and HOW_TO_CONFIGURATION" when that pair occurred
        # 3 times out of 66; the actual most common pair was
        # HARDWARE_AND_BATTERY -> OS_SOFTWARE_TROUBLESHOOTING at 6. The mitigation
        # text then told the reader to add prototypes "for this specific pair".
        # Found by adversarial review round 3.
        #
        # The `other` bucket additionally formatted {true}/{pred} from the INTENT
        # fields while the bucket is entirely triage mismatches, rendering the
        # meaningless "Other triage mismatch (X -> X)". It reads the triage fields
        # now.
        if cat == "other":
            key = lambda f: (f.get("true_triage") or "", f.get("pred_triage") or "")  # noqa: E731
        else:
            key = lambda f: (  # noqa: E731
                f.get("true_intent") or f.get("true_triage") or "",
                f.get("pred_intent") or f.get("pred_triage") or "",
            )
        pair_counts = Counter(key(f) for f in by_category[cat])
        (dominant_true, dominant_pred), pair_count = pair_counts.most_common(1)[0]
        example = next(f for f in by_category[cat] if key(f) == (dominant_true, dominant_pred))
        title = info["title_fmt"].format(true=dominant_true, pred=dominant_pred)
        if "{true}" in info["title_fmt"] and pair_count != count:
            # Both numbers, because the category total is the useful one and the
            # pair total is the one the title is actually about.
            title += f" -- {pair_count} of this category's {count} failures"
        results.append(
            {
                "title": title,
                "frequency": round(100 * count / total),
                "count": count,
                "dominant_pair_count": pair_count,
                "total_failures": total,
                "query": example.get("text", ""),
                "actual": (f"intent={example.get('pred_intent')}, triage={example.get('pred_triage')}"),
                "expected": (f"intent={example.get('true_intent')}, triage={example.get('true_triage')}"),
                # The pipeline's own stated reason for its decision, straight from
                # src/triage/reasons.py -- shown verbatim rather than only the
                # category-level guess, since two failures with the same
                # true/pred labels can have completely different real causes
                # (e.g. a misclassified intent hard-escalated by Gate 6 looks
                # identical to a sentiment-gate false positive unless you see
                # the actual reason the engine gave).
                "system_stated_reason": example.get("stated_reason") or "(none recorded)",
                "hypothesis": info["hypothesis"],
                "mitigation": info["mitigation"],
            }
        )
    return results
