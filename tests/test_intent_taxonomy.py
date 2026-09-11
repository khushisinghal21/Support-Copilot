"""Structural tests for src/intent/taxonomy.py's prototype sentences.

These don't need the embedding model (they check the raw prototype data,
not classifier behavior), so they run anywhere -- including this sandbox,
which can't download sentence-transformers. Verifying actual classification
behavior for the new prototypes added below requires the real model and
should be checked by rerunning `python -m src.eval.runner` and re-reading
the confusion matrix in docs/benchmark_summary.json.
"""

from src.intent.taxonomy import INTENT_DESCRIPTIONS, INTENT_PROTOTYPES
from src.models import AppleIntentEnum


def test_every_intent_has_prototypes_and_a_description():
    for intent in AppleIntentEnum:
        assert intent in INTENT_PROTOTYPES, f"{intent} has no prototype sentences"
        assert len(INTENT_PROTOTYPES[intent]) >= 8, f"{intent} has too few prototypes"
        assert intent in INTENT_DESCRIPTIONS, f"{intent} has no description"


def test_no_duplicate_prototypes_within_a_class():
    for intent, prototypes in INTENT_PROTOTYPES.items():
        assert len(prototypes) == len(set(prototypes)), f"{intent} has duplicate prototype sentences"


def test_hardware_and_battery_covers_update_triggered_battery_drain():
    """Real regression: this run's confusion matrix (docs/benchmark_summary.json)
    showed 13 of 47 true HARDWARE_AND_BATTERY rows (28%) misclassified as
    OS_SOFTWARE_TROUBLESHOOTING, because real battery-drain complaints often
    name a software update as the trigger, and every
    OS_SOFTWARE_TROUBLESHOOTING prototype also mentions an update. These
    prototypes must exist so the class centroid captures that pattern."""
    prototypes = INTENT_PROTOTYPES[AppleIntentEnum.HARDWARE_AND_BATTERY]
    assert any("battery" in p.lower() and "update" in p.lower() for p in prototypes), (
        "expected at least one HARDWARE_AND_BATTERY prototype combining battery drain with a software update trigger"
    )


def test_how_to_configuration_covers_subscription_management_via_settings():
    """Real regression: 6 of 20 true HOW_TO_CONFIGURATION rows were
    misclassified as ACCOUNT_BILLING_ICLOUD, because a genuine
    'how do I manage this in settings' subscription question reads like a
    billing dispute when every ACCOUNT_BILLING_ICLOUD prototype is about
    being charged or needing a refund."""
    prototypes = INTENT_PROTOTYPES[AppleIntentEnum.HOW_TO_CONFIGURATION]
    assert any("subscription" in p.lower() and "settings" in p.lower() for p in prototypes), (
        "expected at least one HOW_TO_CONFIGURATION prototype about managing a subscription via device settings"
    )
