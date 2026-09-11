"""Phase 2: configuration is validated, and a bad value says which one it is.

Before this, every numeric setting was a bare cast: a malformed value raised
ValueError at import time with a traceback naming float() rather than the
setting, and a well-formed nonsensical value (MIN_INTENT_CONFIDENCE=40, meaning
"40%") was accepted in silence -- a confidence floor of 40.0 against a softmax
probability escalates every ticket, and nothing would have reported it.
"""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_defaults_match_the_documented_values():
    s = Settings()
    assert s.MIN_INTENT_CONFIDENCE == pytest.approx(0.40)
    assert s.MIN_RETRIEVAL_SIMILARITY == pytest.approx(0.20)
    assert s.FRUSTRATION_THRESHOLD == pytest.approx(0.60)
    assert s.RAG_CORPUS_MAX_RECORDS == 800


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("MIN_INTENT_CONFIDENCE", 40.0),  # the silent-acceptance case
        ("MIN_INTENT_CONFIDENCE", -0.1),
        ("MIN_RETRIEVAL_SIMILARITY", 1.5),
        ("FRUSTRATION_THRESHOLD", 2.0),
        ("MIN_GROUNDING_SIMILARITY", -1.5),  # below the valid cosine range
        ("RAG_CORPUS_MAX_RECORDS", 0),  # would index nothing at all
        ("RATE_LIMIT_BURST", 0),  # would reject every first request
        ("DECISION_LOG_QUEUE_MAX", 0),
    ],
)
def test_out_of_range_values_are_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        Settings(**{field: bad_value})


def test_cosine_floor_allows_legitimate_negative_values():
    """Cosine similarity is genuinely negative sometimes; a ge=0 bound here would
    be wrong rather than safe."""
    assert Settings(MIN_GROUNDING_SIMILARITY=-0.4).MIN_GROUNDING_SIMILARITY == pytest.approx(-0.4)


def test_rate_limit_of_zero_is_allowed_because_it_means_disabled():
    assert Settings(RATE_LIMIT_PER_MINUTE=0).RATE_LIMIT_PER_MINUTE == 0


def test_bad_env_var_produces_a_message_naming_the_setting_and_the_value():
    """The actual deliverable of this change is what a misconfigured deploy reads
    in its logs, so assert on that rather than on the exception type."""
    result = subprocess.run(
        [sys.executable, "-c", "import src.config"],
        env={"PATH": "/usr/bin:/bin", "MIN_INTENT_CONFIDENCE": "40", "PYTHONPATH": "."},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "MIN_INTENT_CONFIDENCE" in combined, "error does not name the offending setting"
    assert "'40'" in combined, "error does not quote the offending value"
    assert "less than or equal to 1" in combined, "error does not state the allowed range"


def test_module_level_aliases_still_exist_for_every_setting():
    """~40 call sites do `from src.config import X`. The migration must not have
    turned any of them into an ImportError."""
    import src.config as config

    for name in (
        "MIN_INTENT_CONFIDENCE",
        "MIN_RETRIEVAL_SIMILARITY",
        "FRUSTRATION_THRESHOLD",
        "MIN_GROUNDING_SIMILARITY",
        "RAG_CORPUS_MAX_RECORDS",
        "RATE_LIMIT_PER_MINUTE",
        "RATE_LIMIT_BURST",
        "DECISION_LOG_QUEUE_MAX",
        "MAX_TWEET_CHARS",
        "GROUNDING_MODE",
        "CORS_ALLOW_ORIGINS",
        "API_KEY",
        "TARGET_BRAND",
    ):
        assert hasattr(config, name), f"src.config lost the module-level name {name}"
