"""Unit tests for Triage & Escalation Decision Engine."""

import pytest

from src.models import (
    AppleIntentEnum,
    EscalationReasonCode,
    IntentResult,
    RetrievalResult,
    TriageAction,
    TweetInput,
)
from src.triage.engine import TriageEngine


@pytest.fixture(scope="module")
def triage_engine():
    return TriageEngine()


def test_triage_battery_hazard_escalation(triage_engine):
    tweet = TweetInput(
        tweet_id="t1", text="Help! My iPhone battery is swollen and bulging the screen out!!", author_id="user_1"
    )
    intent = IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.95)
    decision = triage_engine.evaluate(tweet, intent)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE
    assert "HARDWARE_PHYSICAL_DAMAGE" in decision.stated_reason
    assert decision.risk_score == 1.0


def test_triage_pii_email_escalation(triage_engine):
    tweet = TweetInput(
        tweet_id="t2", text="My Apple ID is locked, please email me at john.doe@example.com", author_id="user_2"
    )
    intent = IntentResult(primary_intent=AppleIntentEnum.ACCOUNT_BILLING_ICLOUD, confidence=0.92)
    decision = triage_engine.evaluate(tweet, intent)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.PII_SECURITY_SENSITIVE
    assert "PII_SECURITY_SENSITIVE" in decision.stated_reason


def test_triage_human_request_escalation(triage_engine):
    tweet = TweetInput(
        tweet_id="t3",
        text="Stop with this automated response! I want to speak with a human agent now.",
        author_id="user_3",
    )
    intent = IntentResult(primary_intent=AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS, confidence=0.50)
    decision = triage_engine.evaluate(tweet, intent)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.HUMAN_AGENT_REQUESTED


def test_triage_high_frustration_lawsuit_threat(triage_engine):
    tweet = TweetInput(
        tweet_id="t4",
        text="Apple stole my money and this is a scam!! Getting my lawyer involved right now!!",
        author_id="user_4",
    )
    intent = IntentResult(primary_intent=AppleIntentEnum.ACCOUNT_BILLING_ICLOUD, confidence=0.85)
    decision = triage_engine.evaluate(tweet, intent)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK


def test_triage_low_confidence_escalation(triage_engine):
    tweet = TweetInput(tweet_id="t5", text="I don't know what is wrong with this thing", author_id="user_5")
    intent = IntentResult(primary_intent=AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS, confidence=0.45)
    decision = triage_engine.evaluate(tweet, intent)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS


def test_triage_low_retrieval_similarity_escalation(triage_engine):
    """Similarity is set to half the engine's actual configured threshold,
    not a hardcoded literal -- this test previously hardcoded 0.35, which
    silently stopped testing the behavior it claims to test the moment
    MIN_RETRIEVAL_SIMILARITY was recalibrated away from its old (miscalibrated,
    see docs/AUDIT_AND_FIX_PLAN.md Section 7.10) default of 0.40."""
    tweet = TweetInput(tweet_id="t6", text="Can I cook an egg on my iPhone?", author_id="user_6")
    intent = IntentResult(primary_intent=AppleIntentEnum.HOW_TO_CONFIGURATION, confidence=0.75)
    below_threshold = triage_engine.min_retrieval_similarity / 2
    low_rag = RetrievalResult(snippets=["Random"], similarity_scores=[below_threshold], max_similarity=below_threshold)
    decision = triage_engine.evaluate(tweet, intent, rag_res=low_rag)

    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS


def test_triage_safe_auto_handle_clearance(triage_engine):
    tweet = TweetInput(
        tweet_id="t7", text="How do I transfer my photos from iPhone to my Windows PC?", author_id="user_7"
    )
    intent = IntentResult(primary_intent=AppleIntentEnum.HOW_TO_CONFIGURATION, confidence=0.92)
    good_rag = RetrievalResult(
        snippets=["Connect your iPhone to your PC with a USB cable: apple.co/importphotos"],
        similarity_scores=[0.88],
        max_similarity=0.88,
    )
    decision = triage_engine.evaluate(tweet, intent, rag_res=good_rag, guardrail_passed=True)

    assert decision.action == TriageAction.AUTO_HANDLE
    assert decision.reason_code is None
    assert decision.risk_score <= 0.15
