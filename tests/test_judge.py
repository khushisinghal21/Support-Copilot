"""Unit tests for the LLM-as-a-Judge module (src/eval/judge.py).

There was no test coverage for this module at all before this file --  a
real gap, since grade_reply()'s LLM branch used to fail completely silently
(a bare `except Exception: pass` with no logging), meaning a broken judge
call could have been substituting the deterministic fallback on every
single benchmark run with zero visible trace. These tests exercise both the
deterministic fallback path (no network/API key needed) and the
thinking-token-truncation defense added when the module was migrated from
the deprecated `google-generativeai` SDK to `google-genai`.
"""

from src.eval.judge import LLMJudge


class _FakeFinishReason:
    def __init__(self, name):
        self.name = name


class _FakeCandidate:
    def __init__(self, finish_reason_name):
        self.finish_reason = _FakeFinishReason(finish_reason_name)


class _FakeGeminiResponse:
    def __init__(self, text, finish_reason_name="STOP"):
        self.text = text
        self.candidates = [_FakeCandidate(finish_reason_name)]


class _FakeModels:
    def __init__(self, response):
        self._response = response

    def generate_content(self, model=None, contents=None, config=None):
        return self._response


class _FakeLLM:
    """Mirrors google.genai.Client's shape without touching the network."""

    def __init__(self, response):
        self.models = _FakeModels(response)


def test_judge_escalated_ticket_no_reply():
    judge = LLMJudge(api_key="")
    result = judge.grade_reply("some customer complaint", None, "reference reply")
    assert result["safety"] == 5
    assert "escalated" in result["reasoning"].lower()


def test_judge_deterministic_fallback_when_no_api_key():
    """With no API key, _llm stays None and grade_reply must use the
    deterministic rule-based judge rather than raising or returning nothing."""
    judge = LLMJudge(api_key="")
    assert judge._llm is None
    result = judge.grade_reply(
        "My battery drains fast",
        "We'd like to help. Please check support.apple.com/batteryhealth.",
        "Check battery health in Settings.",
    )
    assert result["groundedness"] == 5  # official domain link present
    assert result["reasoning"] == "Evaluated via calibrated brand rules and link validation."


def test_judge_discards_max_tokens_truncated_verdict():
    """Real regression pattern (same root cause as the generator bug): a
    judge call asking for strict JSON is especially exposed to
    thinking-token truncation, since a partial JSON blob fails json.loads()
    outright. When finish_reason is MAX_TOKENS, the judge must not attempt
    to parse the partial text -- it should fall back to the deterministic
    judge instead."""
    judge = LLMJudge(api_key="")
    judge._llm = _FakeLLM(_FakeGeminiResponse('{"groundedness": 5, "to', finish_reason_name="MAX_TOKENS"))
    result = judge.grade_reply(
        "My battery drains fast",
        "We'd like to help. Please check support.apple.com/batteryhealth.",
        "Check battery health in Settings.",
    )
    # Fell back to the deterministic judge rather than crashing or returning
    # a verdict built from truncated JSON.
    assert result["reasoning"] == "Evaluated via calibrated brand rules and link validation."


def test_judge_parses_complete_llm_verdict():
    """A clean, complete JSON verdict should be parsed and used as-is."""
    judge = LLMJudge(api_key="")
    fake_json = '{"groundedness": 5, "tone": 4, "safety": 5, "reasoning": "Accurate and grounded."}'
    judge._llm = _FakeLLM(_FakeGeminiResponse(fake_json, finish_reason_name="STOP"))
    result = judge.grade_reply(
        "My battery drains fast",
        "We'd like to help. Please check support.apple.com/batteryhealth.",
        "Check battery health in Settings.",
    )
    assert result["groundedness"] == 5
    assert result["tone"] == 4
    assert result["safety"] == 5
    assert result["overall"] == round((5 + 4 + 5) / 3.0, 2)
    assert result["reasoning"] == "Accurate and grounded."
