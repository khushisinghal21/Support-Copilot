"""Independent keyword-based intent heuristic.

Deliberately NOT the SemanticCentroidClassifier under test, and does not share
its prototype sentences (src/intent/taxonomy.py). Used for two purposes that
both require a labelling signal independent of the model being evaluated:

1. Labelling the golden evaluation set (scripts/build_golden_set.py), so
   intent-classifier accuracy isn't measured against labels the same model
   (or a near-identical zero-shot method) produced.
2. Re-tagging the real Kaggle-extracted RAG corpus (src/drafting/vector_store.py)
   for intent-filtered retrieval, instead of trusting the tags that were
   assigned at ingestion time by the same centroid classifier -- using a
   model's own output as the ground truth for its supporting data is circular.
"""
from collections import Counter

INTENT_KEYWORDS = {
    "HOW_TO_CONFIGURATION": [
        "how do i", "how to", "how can i", "how do you", "can i put",
        "can i use", "can i connect", "can i pair", "set up", "setup",
        "transfer", "backup", "back up", "restore", "airdrop", "pair", "enable",
        "turn on", "turn off", "configure", "sync", "connect two", "sync them up",
    ],
    "HARDWARE_AND_BATTERY": [
        "battery", "charge", "charging", "charger", "screen", "glass", "crack",
        "speaker", "camera", "button", "port", "overheat", "too hot", "drop",
        "dropped", "broken", "swell", "bulg", "mic", "microphone", "display",
        "shattered", "physical damage",
    ],
    "ACCOUNT_BILLING_ICLOUD": [
        "apple id", "icloud", "password", "subscription", "bill", "billed",
        "charged", "refund", "itunes", "app store", "payment", "2fa",
        "two-factor", "two factor", "verification code", "account", "storage full",
        "purchase", "locked out", "sign in", "signin", "activate", "activation",
        "cellular plan", "carrier", "sim card", "sim ",
    ],
    "OS_SOFTWARE_TROUBLESHOOTING": [
        "crash", "freeze", "freezing", "froze", "bug", "glitch", "update",
        "ios ", "ios11", "ios 11", "wifi", "wi-fi", "bluetooth", "restart",
        "reboot", "stuck", "lag", "laggy", "notification", "app keeps",
        "boot loop", "won't respond", "not responding",
    ],
}


def heuristic_intent(text: str) -> str:
    """Assigns one of the 5 canonical intents using literal keyword matching,
    independent of any embedding model. Falls back to OUT_OF_SCOPE_AMBIGUOUS
    when nothing matches."""
    t = text.lower()
    scores: Counter[str] = Counter()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[intent] += 1
    if not scores:
        return "OUT_OF_SCOPE_AMBIGUOUS"
    top = scores.most_common()
    best_score = top[0][1]
    contenders = [i for i, s in scores.items() if s == best_score]
    if "HOW_TO_CONFIGURATION" in contenders:
        return "HOW_TO_CONFIGURATION"
    return top[0][0]
