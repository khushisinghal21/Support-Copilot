"""Phase 3.1: the remaining uncovered branches in the triage support modules.

Small surface, but these are the strings an operator actually reads and the signals
that decide a frustration escalation, so "uncovered" here is not harmless.
"""

import pytest

from src.models import EscalationReasonCode
from src.triage.reasons import (
    DEFAULT_CLARIFYING_QUESTION,
    format_stated_reason,
    get_clarifying_question,
)
from src.triage.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()


# --------------------------------------------------------------------------
# Stated reasons and clarifying questions
# --------------------------------------------------------------------------


def test_clarifying_question_falls_back_for_none():
    assert get_clarifying_question(None) == DEFAULT_CLARIFYING_QUESTION


def test_clarifying_question_falls_back_for_a_code_with_no_specific_question():
    """Most reason codes have no clarifying question because they escalate rather
    than ask. Those must still return usable text, not None or a KeyError."""
    question = get_clarifying_question(EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE)
    assert question == DEFAULT_CLARIFYING_QUESTION
    assert question.endswith("?")


def test_clarifying_question_for_the_ambiguous_device_case_asks_about_the_device():
    question = get_clarifying_question(EscalationReasonCode.AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION)
    assert question != DEFAULT_CLARIFYING_QUESTION
    assert "?" in question
    assert any(device in question for device in ("iPhone", "iPad", "Mac", "Apple Watch"))


def test_stated_reason_without_custom_detail_still_names_its_code():
    """The bare form -- used wherever a gate has no extra detail to add. The code
    must still appear, because the code is the auditability contract."""
    reason = format_stated_reason(EscalationReasonCode.HUMAN_AGENT_REQUESTED)
    assert reason.startswith(f"[{EscalationReasonCode.HUMAN_AGENT_REQUESTED.value}]")
    assert len(reason) > len(EscalationReasonCode.HUMAN_AGENT_REQUESTED.value) + 3


def test_stated_reason_with_custom_detail_includes_both():
    reason = format_stated_reason(EscalationReasonCode.PII_SECURITY_SENSITIVE, "Triggered rules: ['EMAIL']")
    assert EscalationReasonCode.PII_SECURITY_SENSITIVE.value in reason
    assert "EMAIL" in reason


@pytest.mark.parametrize("code", list(EscalationReasonCode))
def test_every_reason_code_produces_a_non_empty_explanation(code):
    """No code may render as a bare identifier -- an operator reading
    'PII_ECHO_IN_DRAFT' with no sentence after it has to go read the source."""
    reason = format_stated_reason(code)
    assert reason.startswith(f"[{code.value}]")
    assert len(reason.split("]", 1)[1].strip()) > 10, f"{code.value} has no explanatory text"


# --------------------------------------------------------------------------
# Frustration scoring signals
# --------------------------------------------------------------------------


def test_anger_keywords_contribute_to_the_score():
    score, markers = analyzer.compute_frustration("this is absolutely terrible and I am furious about it")
    assert score > 0.0
    assert any("ANGER_KEYWORDS" in m for m in markers)


def test_uppercase_shouting_is_detected_above_the_letter_threshold():
    """Needs >= 15 letters and >= 40% uppercase -- short all-caps words like 'OK'
    must not trip it."""
    score, markers = analyzer.compute_frustration("THIS IS COMPLETELY UNACCEPTABLE SERVICE")
    assert any("UPPERCASE_SHOUTING" in m for m in markers), markers
    assert score > 0.0

    _, short_markers = analyzer.compute_frustration("OK SURE")
    assert not any("UPPERCASE_SHOUTING" in m for m in short_markers), short_markers


def test_excessive_punctuation_is_a_signal_but_not_sufficient_alone():
    score, markers = analyzer.compute_frustration("why won't this work!!!")
    assert any("EXCESSIVE_PUNCTUATION" in m for m in markers), markers
    # One weak signal must not clear the escalation threshold on its own.
    assert score < analyzer.threshold


def test_severe_frustration_requires_corroboration():
    """The gate exists to catch churn risk, not mild annoyance. A single soft signal
    should not escalate; stacked signals should."""
    mild, _, _ = analyzer.is_severe_frustration("this is a bit annoying")
    assert mild is False

    severe, score, markers = analyzer.is_severe_frustration(
        "I am FURIOUS, this is ABSOLUTELY UNACCEPTABLE!!! I'm cancelling and calling my lawyer"
    )
    assert severe is True
    assert score >= analyzer.threshold
    assert len(markers) >= 2, f"escalated on a single signal: {markers}"


def test_neutral_text_scores_zero_with_no_markers():
    score, markers = analyzer.compute_frustration("How do I transfer photos to my PC?")
    assert score == 0.0
    assert markers == []
