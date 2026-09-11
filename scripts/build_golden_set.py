"""Builds a real-data golden evaluation set from apple_support_kaggle_pairs.jsonl.

Methodology (documented honestly in data/README.md alongside this):
1. Load the real extracted @AppleSupport pairs (customer tweet + real historical agent reply).
2. Label true_intent using an INDEPENDENT keyword heuristic (deliberately NOT the
   SemanticCentroidClassifier under test / not the taxonomy prototypes it was built from,
   to avoid circular/self-confirming labels).
3. Label true_triage_action using tightened, hand-reviewed hazard/PII/legal/human-request
   detectors, then MANUALLY REVIEW every candidate ESCALATE case (small number) by hand.
4. Because real 2017-18 Twitter first-contact tweets rarely contain PII, explicit thermal
   hazards, or prompt-injection attempts (a risk that didn't exist for this dataset's era),
   those categories are supplemented with a clearly-tagged `source=authored_adversarial`
   block, distinct from `source=kaggle_real`. This is disclosed, not hidden.
5. reference_resolution for real examples is the ACTUAL historical agent reply (best possible
   grounding reference). For authored examples it's a brand-consistent reply we write.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.heuristic_intent import heuristic_intent

DATA_DIR = REPO_ROOT / "data"
SRC_PAIRS = DATA_DIR / "apple_support_kaggle_pairs.jsonl"
OUT_GOLDEN = DATA_DIR / "golden_eval_set.jsonl"


# ---------------------------------------------------------------------------
# Tightened triage heuristics for LABELLING (independent of src/triage/rules.py,
# and deliberately stricter than the first draft to avoid the "burnt calories"
# style false positives found during the audit).
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

HAZARD_RE = re.compile(
    r"\b(swoll|swell|bulg|puff(ed|ing)?)\w*\s*(battery|phone|device|screen)?|"
    r"\bsmoke\b|\bsmoking\b|\bspark(s|ing)?\b|\bcaught fire\b|\bon fire\b|\bexplod\w*|"
    r"\bburning smell\b|\bburn(ed|t)?\s+(my|the)?\s*(hand|finger|skin)\b|"
    r"\bshattered\b.*\b(glass|screen)\b|\b(glass|screen)\b.*\bshattered\b|"
    r"\bwater damage\b|\bliquid damage\b|\bdropped\b.*\b(in|into)\b.*\b(pool|toilet|sink|bath|ocean|lake)\b",
    re.IGNORECASE,
)
LEGAL_CHURN_RE = re.compile(
    r"\blawyer\b|\battorney\b|\blawsuit\b|\bsu(e|ing)\b|\bfraud\b|\bscam(mer|med)?\b|"
    r"\bhacked\b|\bstolen\b|\bpolice\b|\bbbb\b|\bconsumer protection\b|\bidentity theft\b",
    re.IGNORECASE,
)
HUMAN_REQUEST_RE = re.compile(
    r"\bspeak (to|with) (a )?(human|person|real person|agent|representative)\b|"
    r"\btalk to (a )?(human|person|real person)\b|\breal human\b|\bstop.*(bot|robot)\b|"
    r"\bnot a bot\b",
    re.IGNORECASE,
)


def heuristic_triage(text: str):
    """Returns (action, reason_code, matched_rule) for LABELLING purposes."""
    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        return "ESCALATE", "PII_SECURITY_SENSITIVE", "PII_REGEX"
    if HAZARD_RE.search(text):
        return "ESCALATE", "HARDWARE_PHYSICAL_DAMAGE", "HAZARD_REGEX"
    if HUMAN_REQUEST_RE.search(text):
        return "ESCALATE", "HUMAN_AGENT_REQUESTED", "HUMAN_REQUEST_REGEX"
    if LEGAL_CHURN_RE.search(text):
        return "ESCALATE", "HIGH_FRUSTRATION_CHURN_RISK", "LEGAL_CHURN_REGEX"
    return "AUTO_HANDLE", None, None


def load_real_pairs():
    pairs = []
    with open(SRC_PAIRS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append(d)
    return pairs


def main():
    pairs = load_real_pairs()
    print(f"Loaded {len(pairs)} real Kaggle-extracted @AppleSupport pairs")

    labeled = []
    for p in pairs:
        text = p["customer_text"]
        if len(text) < 20 or len(text) > 260:
            continue
        intent = heuristic_intent(text)
        action, reason, rule = heuristic_triage(text)
        labeled.append(
            {
                "source_tweet_id": p["tweet_id"],
                "text": text,
                "real_agent_reply": p["agent_reply"],
                "heuristic_intent": intent,
                "ingest_time_intent": p.get("intent"),  # from the (circular) classifier, for comparison only
                "triage_action": action,
                "reason_code": reason,
                "matched_rule": rule,
            }
        )

    by_intent = defaultdict(list)
    for item in labeled:
        by_intent[item["heuristic_intent"]].append(item)

    disagreements = sum(1 for it in labeled if it["heuristic_intent"] != it["ingest_time_intent"])
    print(
        f"Heuristic vs ingest-time-classifier intent disagreement: {disagreements}/{len(labeled)}"
        f" ({disagreements / len(labeled) * 100:.1f}%) -- expected, they're independent methods"
    )

    escalate_candidates = [it for it in labeled if it["triage_action"] == "ESCALATE"]
    print(f"\n{len(escalate_candidates)} real-data ESCALATE candidates found by heuristic (manually reviewed below):")
    for it in escalate_candidates:
        print(f"  [{it['reason_code']}] {it['text'][:140]}")

    for intent, items in by_intent.items():
        print(f"{intent}: {len(items)} candidates")

    # Save intermediate labeled pool for manual review / sampling step.
    out_path = DATA_DIR / "_kaggle_labeled_pool.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in labeled:
            f.write(json.dumps(it) + "\n")
    print(f"\nWrote {len(labeled)} labeled candidates to {out_path}")


if __name__ == "__main__":
    main()
