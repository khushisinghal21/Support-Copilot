"""Unit tests for src/pipeline.py."""

from unittest.mock import MagicMock
from src.models import (
    TweetInput,
    TriageAction,
    EscalationReasonCode,
    AppleIntentEnum,
)
from src.pipeline import SupportPipeline


def test_pipeline_single_tweet_default_execution():
    pipeline = SupportPipeline()
    tweet = TweetInput(
        tweet_id="12345",
        text="My iPhone 11 Wi-Fi keeps dropping.",
        author_id="user_1"
    )
    response = pipeline.process(tweet)

    assert response.tweet_id == "12345"
    assert response.intent.primary_intent == AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING
    assert response.triage.action == TriageAction.AUTO_HANDLE
    assert response.drafted_reply is not None
    assert response.execution_time_ms >= 0.0


def test_pipeline_batch_processing():
    pipeline = SupportPipeline()
    tweets = [
        TweetInput(tweet_id=f"tw_{i}", text=f"Sample issue {i}", author_id=f"author_{i}")
        for i in range(3)
    ]
    responses = pipeline.batch_process(tweets)
    assert len(responses) == 3
    assert [r.tweet_id for r in responses] == ["tw_0", "tw_1", "tw_2"]


def test_pipeline_fail_closed_circuit_breaker():
    # Mock a broken intent classifier that crashes
    broken_classifier = MagicMock()
    broken_classifier.predict.side_effect = RuntimeError("Database connection died!")

    pipeline = SupportPipeline(intent_classifier=broken_classifier)
    tweet = TweetInput(
        tweet_id="err_tweet",
        text="Can you help me?",
        author_id="user_err"
    )
    response = pipeline.process(tweet)

    # Must fail closed to ESCALATE
    assert response.triage.action == TriageAction.ESCALATE
    assert response.triage.reason_code == EscalationReasonCode.SYSTEM_EXCEPTION_FAIL_CLOSED
    assert response.drafted_reply is None
    assert "CIRCUIT_BREAKER_FAIL_CLOSED" in response.triage.triggered_rules
