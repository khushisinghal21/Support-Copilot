"""Unit tests for src/models.py data schemas and contracts."""

import pytest
from pydantic import ValidationError

from src.models import (
    AppleIntentEnum,
    EscalationReasonCode,
    IntentResult,
    RetrievalResult,
    SupportResponse,
    TriageAction,
    TriageDecision,
    TweetInput,
)


def test_tweet_input_valid():
    tweet = TweetInput(
        tweet_id="101",
        text="My iPhone 12 battery is draining fast.",
        author_id="user_abc",
        created_at="2026-09-09T12:00:00Z",
    )
    assert tweet.tweet_id == "101"
    assert tweet.text == "My iPhone 12 battery is draining fast."
    assert tweet.author_id == "user_abc"


def test_tweet_input_rejects_empty_text():
    with pytest.raises(ValidationError):
        TweetInput(tweet_id="102", text="   ", author_id="user_abc")


def test_intent_result_valid():
    intent = IntentResult(
        primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY,
        confidence=0.88,
        secondary_intents=[AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING],
        score_distribution={"HARDWARE_AND_BATTERY": 0.88, "OS_SOFTWARE_TROUBLESHOOTING": 0.12},
    )
    assert intent.primary_intent == AppleIntentEnum.HARDWARE_AND_BATTERY
    assert intent.confidence == 0.88
    assert len(intent.secondary_intents) == 1


def test_intent_result_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        IntentResult(
            primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY,
            confidence=1.5,  # Must be <= 1.0
        )


def test_triage_decision_valid():
    decision = TriageDecision(
        action=TriageAction.ESCALATE,
        stated_reason="Physical battery swelling requires Genius Bar inspection",
        reason_code=EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE,
        risk_score=0.95,
        triggered_rules=["BATTERY_SWELL_REGEX"],
    )
    assert decision.action == TriageAction.ESCALATE
    assert decision.reason_code == EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE


def test_support_response_full_serialization():
    response = SupportResponse(
        tweet_id="999",
        intent=IntentResult(primary_intent=AppleIntentEnum.HOW_TO_CONFIGURATION, confidence=0.92),
        triage=TriageDecision(
            action=TriageAction.AUTO_HANDLE, stated_reason="High confidence resolution grounded in historical data"
        ),
        drafted_reply="You can transfer photos using AirDrop or iCloud. Check: apple.co/airdrop",
        grounding_context=RetrievalResult(
            snippets=["Use AirDrop to share photos between devices."], similarity_scores=[0.85], max_similarity=0.85
        ),
        execution_time_ms=142.5,
    )

    data = response.to_dict()
    assert data["tweet_id"] == "999"
    assert data["triage"]["action"] == "AUTO_HANDLE"
    assert "apple.co/airdrop" in data["drafted_reply"]
    json_str = response.to_json()
    assert isinstance(json_str, str)
