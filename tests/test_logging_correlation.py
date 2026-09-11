"""Phase 2: logging is actually configured, and request lines carry a tweet_id.

Before this, 17 logger call sites existed and nothing configured logging, so
everything below WARNING was discarded and what survived had no timestamp, no
logger name and no way to tell which request produced it.
"""

import logging
from unittest.mock import MagicMock

from src.logging_config import (
    TweetIdFilter,
    get_tweet_id,
    reset_tweet_id,
    set_tweet_id,
    setup_logging,
)
from src.models import AppleIntentEnum, IntentResult, TweetInput
from src.pipeline import SupportPipeline
from src.triage.engine import TriageEngine


def test_setup_logging_is_idempotent():
    setup_logging()
    before = len(logging.getLogger().handlers)
    setup_logging()
    setup_logging()
    assert len(logging.getLogger().handlers) == before, "duplicate handlers would duplicate every line"


def test_filter_supplies_a_placeholder_outside_a_request():
    """Records emitted with no request in scope must still format -- a KeyError in
    a log formatter turns a diagnostic into a second failure."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    assert TweetIdFilter().filter(record) is True
    assert record.tweet_id == "-"


def test_context_var_round_trip():
    assert get_tweet_id() is None
    token = set_tweet_id("tweet-123")
    assert get_tweet_id() == "tweet-123"
    reset_tweet_id(token)
    assert get_tweet_id() is None


def test_pipeline_binds_and_clears_the_correlation_id(caplog):
    """The id must appear on lines logged during processing, and must NOT leak
    afterwards -- in a thread pool a leaked id mislabels the next request."""
    classifier = MagicMock()
    classifier.predict.return_value = IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.92)
    generator = MagicMock()
    generator.generate.return_value = ("A grounded reply about battery health today.", True, [])
    pipeline = SupportPipeline(
        intent_classifier=classifier,
        retriever=MagicMock(),
        reply_generator=generator,
        triage_engine=TriageEngine(),
    )

    seen = {}

    def _capture(tweet, *a, **k):
        seen["during"] = get_tweet_id()
        return IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.92)

    classifier.predict.side_effect = _capture
    pipeline.process(TweetInput(tweet_id="corr-777", text="My battery drains fast lately.", author_id="u"))

    assert seen["during"] == "corr-777", "correlation id was not bound during processing"
    assert get_tweet_id() is None, "correlation id leaked past the request"


def test_correlation_id_is_cleared_even_when_the_pipeline_fails_closed():
    classifier = MagicMock()
    classifier.predict.side_effect = RuntimeError("classifier exploded")
    pipeline = SupportPipeline(
        intent_classifier=classifier,
        retriever=MagicMock(),
        reply_generator=MagicMock(),
        triage_engine=TriageEngine(),
    )
    response = pipeline.process(TweetInput(tweet_id="corr-fail", text="anything at all here", author_id="u"))
    assert response.triage.reason_code.value == "SYSTEM_EXCEPTION_FAIL_CLOSED"
    assert get_tweet_id() is None, "correlation id leaked after a fail-closed exception"
