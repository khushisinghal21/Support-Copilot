"""Phase 3.2: integration -- real wiring, only externals mocked.

Covers every terminal outcome, fail-closed behaviour at each stage boundary, and
the properties that only show up when the parts are connected.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.models import (
    AppleIntentEnum,
    EscalationReasonCode,
    IntentResult,
    RetrievalResult,
    TriageAction,
    TweetInput,
)
from src.pipeline import SupportPipeline
from src.triage.engine import TriageEngine


def _rag(max_similarity=0.8, replies=("Try a force restart: hold side and volume down.",)):
    return RetrievalResult(
        retrieved_replies=list(replies),
        retrieved_queries=["my phone won't restart"],
        similarities=[max_similarity],
        max_similarity=max_similarity,
    )


def _pipeline(intent=None, rag=None, draft=("A grounded reply about battery health today.", True, [])):
    classifier = MagicMock()
    classifier.predict.return_value = intent or IntentResult(
        primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.93
    )
    retriever = MagicMock()
    retriever.retrieve.return_value = rag if rag is not None else _rag()
    generator = MagicMock()
    generator.generate.return_value = draft
    return SupportPipeline(
        intent_classifier=classifier,
        retriever=retriever,
        reply_generator=generator,
        triage_engine=TriageEngine(),
    )


def _process(pipeline, text="How do I check battery health on my iPhone?"):
    return pipeline.process(TweetInput(tweet_id="int-1", text=text, author_id="u1"))


# --------------------------------------------------------------------------
# Terminal outcomes
# --------------------------------------------------------------------------


def test_auto_handle_returns_the_draft():
    response = _process(_pipeline())
    assert response.triage.action == TriageAction.AUTO_HANDLE
    assert response.drafted_reply == "A grounded reply about battery health today."
    assert response.triage.reason_code is None


def test_escalate_withholds_the_draft_entirely():
    """A withheld reply must be absent, not merely unused. Returning it alongside an
    ESCALATE invites a caller to send it anyway."""
    response = _process(_pipeline(), text="My battery is swollen and leaking fluid")
    assert response.triage.action == TriageAction.ESCALATE
    assert response.drafted_reply is None


def test_clarify_returns_a_question_not_the_draft():
    response = _process(
        _pipeline(intent=IntentResult(primary_intent=AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING, confidence=0.50)),
        text="It keeps disconnecting from Bluetooth when I go running",
    )
    assert response.triage.action == TriageAction.CLARIFY
    assert response.drafted_reply
    assert response.drafted_reply != "A grounded reply about battery health today."
    assert "?" in response.drafted_reply


def test_low_retrieval_similarity_escalates_on_the_output_side():
    response = _process(_pipeline(rag=_rag(max_similarity=0.01)))
    assert response.triage.action == TriageAction.ESCALATE
    assert response.triage.reason_code == EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS


def test_guardrail_violation_escalates_with_the_specific_code():
    response = _process(_pipeline(draft=("a draft", False, ["PII_ECHO: leaked x@y.com"])))
    assert response.triage.action == TriageAction.ESCALATE
    assert response.triage.reason_code == EscalationReasonCode.PII_ECHO_IN_DRAFT


def test_low_intent_confidence_escalates():
    response = _process(
        _pipeline(intent=IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.05))
    )
    assert response.triage.action == TriageAction.ESCALATE
    assert response.triage.reason_code == EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS


# --------------------------------------------------------------------------
# Fail closed, at every stage boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["intent_classifier", "retriever", "reply_generator", "triage_engine"])
def test_exception_at_any_stage_fails_closed_with_no_draft(stage):
    pipeline = _pipeline()
    broken = MagicMock()
    if stage == "intent_classifier":
        broken.predict.side_effect = RuntimeError("boom")
    elif stage == "retriever":
        broken.retrieve.side_effect = RuntimeError("boom")
    elif stage == "reply_generator":
        broken.generate.side_effect = RuntimeError("boom")
    else:
        broken.evaluate_input.side_effect = RuntimeError("boom")
    setattr(pipeline, stage, broken)

    response = _process(pipeline)

    assert response.triage.action == TriageAction.ESCALATE, f"{stage} failure did not escalate"
    assert response.triage.reason_code == EscalationReasonCode.SYSTEM_EXCEPTION_FAIL_CLOSED
    assert response.drafted_reply is None, f"{stage} failure leaked a draft"
    assert response.triage.risk_score == 1.0


def test_fail_closed_response_is_still_a_valid_serialisable_response():
    """A fail-closed path that cannot be serialised is a 500 instead of a safe
    escalation."""
    pipeline = _pipeline()
    pipeline.intent_classifier.predict.side_effect = RuntimeError("boom")
    response = _process(pipeline)
    payload = json.loads(response.model_dump_json())
    assert payload["triage"]["action"] == "ESCALATE"
    assert payload["drafted_reply"] is None


# --------------------------------------------------------------------------
# Batch + audit log
# --------------------------------------------------------------------------


def test_batch_process_handles_a_mixed_batch_independently():
    pipeline = _pipeline()
    tweets = [
        TweetInput(tweet_id="b1", text="How do I check battery health?", author_id="u"),
        TweetInput(tweet_id="b2", text="My battery is swollen and smoking", author_id="u"),
        TweetInput(tweet_id="b3", text="I want to speak to a real human", author_id="u"),
    ]
    responses = pipeline.batch_process(tweets)
    assert [r.triage.action for r in responses] == [
        TriageAction.AUTO_HANDLE,
        TriageAction.ESCALATE,
        TriageAction.ESCALATE,
    ]
    assert [r.tweet_id for r in responses] == ["b1", "b2", "b3"]


def test_every_decision_is_written_to_the_audit_log(tmp_path, monkeypatch):
    import src.pipeline as pipeline_mod

    log = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(pipeline_mod, "DECISION_LOG_PATH", log)

    pipeline = _pipeline()
    _process(pipeline)
    _process(pipeline, text="My battery is swollen and smoking")
    assert pipeline_mod.flush_decision_log(timeout=5.0), "audit writer did not drain"

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert {r["triage_action"] for r in rows} == {"AUTO_HANDLE", "ESCALATE"}
    for row in rows:
        assert row["tweet_id"]
        assert row["execution_time_ms"] is not None
