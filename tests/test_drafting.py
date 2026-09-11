"""Unit tests for Grounded Reply Drafting and Guardrails."""

import pytest

from src.drafting.generator import GroundedReplyGenerator
from src.drafting.guardrails import OutputGuardrail
from src.drafting.retriever import HistoricalRetriever
from src.models import RetrievalResult


@pytest.fixture(scope="module")
def retriever():
    return HistoricalRetriever()


@pytest.fixture(scope="module")
def guardrail():
    return OutputGuardrail()


def test_retriever_returns_relevant_battery_resolutions(retriever):
    """Relevance is asserted on the retrieved QUERY and the similarity score, not
    on the reply text.

    The original assertion was `any("battery" in s.lower() for s in res.snippets)`
    -- i.e. the retrieved *agent reply* had to contain the word "battery". That
    passed only because the golden set's own source rows were in the RAG corpus:
    it was retrieving the row's own reference answer, which naturally echoed the
    question. With the leakage guard actually working (it excluded zero rows
    before -- see src/drafting/vector_store.py), the retriever returns real
    @AppleSupport replies, and real replies rarely repeat the problem noun
    ("Send us a DM and let us know what version of iOS you're running").

    So the old assertion was measuring leakage, not relevance. Similarity is
    still high (0.74 here), and the matched customer query is still about
    battery -- which is what "relevant retrieval" actually means.
    """
    res = retriever.retrieve("My iPhone battery dies in two hours", intent="HARDWARE_AND_BATTERY", k=2)
    assert len(res.snippets) == 2
    assert res.max_similarity > 0.50
    matched = [c.customer_text.lower() for c in res.citations]
    assert any(any(word in q for word in ("battery", "charge", "power", "die", "drain")) for q in matched), (
        f"no retrieved customer query looks battery-related: {matched}"
    )


def test_retriever_returns_wifi_resolutions(retriever):
    res = retriever.retrieve("Wi-Fi keeps dropping on iOS update", intent="OS_SOFTWARE_TROUBLESHOOTING", k=2)
    assert len(res.snippets) == 2
    assert res.max_similarity > 0.50


def test_guardrail_length_pass_and_fail(guardrail):
    short_text = "We'd like to help. Check apple.co/batteryhealth."
    assert guardrail.check_length(short_text) is True

    long_text = "A" * 285
    assert guardrail.check_length(long_text) is False


def test_guardrail_catches_pii_solicitation(guardrail):
    bad_draft = "Please DM us your password and Apple ID security code so we can reset it."
    assert guardrail.check_pii_solicitation(bad_draft) is False

    clean_draft = "You can reset your password securely at iforgot.apple.com."
    assert guardrail.check_pii_solicitation(clean_draft) is True


def test_guardrail_url_whitelisting(guardrail):
    valid_text = "Visit apple.co/forcerestart or support.apple.com/iphone for help."
    passed, invalid = guardrail.validate_urls(valid_text)
    assert passed is True
    assert len(invalid) == 0

    phishing_text = "Click here to claim your free iPhone: http://free-apple-scam.xyz/claim"
    passed, invalid = guardrail.validate_urls(phishing_text)
    assert passed is False
    assert len(invalid) > 0


def test_generator_drafts_safe_reply(retriever):
    gen = GroundedReplyGenerator(provider="mock")
    ret_res = retriever.retrieve("How do I back up my iPhone?", intent="HOW_TO_CONFIGURATION")
    reply, passed, violations = gen.generate("How do I back up my iPhone?", "HOW_TO_CONFIGURATION", ret_res)

    assert reply is not None
    assert len(reply) <= 280
    assert passed is True
    assert len(violations) == 0


def test_guardrail_rejects_short_incomplete_draft(guardrail):
    """Real regression: a 3-word truncated Gemini output ("We'd like to")
    passed every guardrail because check_length allows anything under 280
    chars and check_grounding trivially passes when the draft has zero
    content words to compare. check_minimum_content is the fix -- it must
    reject a fragment like this on its own, independent of any other check."""
    assert guardrail.check_minimum_content("We'd like to") is False
    passed, violations = guardrail.evaluate("We'd like to")
    assert passed is False
    assert any(v.startswith("INCOMPLETE_DRAFT") for v in violations)


def test_guardrail_accepts_normal_length_reply(guardrail):
    real_reply = "We'd like to help. Have you tried restarting your device? Let us know which iOS version you have."
    assert guardrail.check_minimum_content(real_reply) is True


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
    """Stands in for a google.genai.Client without touching the network --
    this sandbox can't reach the real Gemini API, so the truncation-handling
    logic in GroundedReplyGenerator.generate() is exercised directly against
    a scripted fake response instead. Mirrors the real client's shape
    (`client.models.generate_content(model=, contents=, config=)`)."""

    def __init__(self, response):
        self.models = _FakeModels(response)


def _fake_retrieval_result():
    # Built directly rather than via HistoricalRetriever().retrieve() so
    # these tests exercise only the generator's truncation-handling logic
    # and don't require downloading the sentence-transformers embedding
    # model (unavailable in network-restricted sandboxes; confirmed present
    # and working in the real project environment).
    return RetrievalResult(
        snippets=["We'd like to help. Have you tried a force restart? Let us know your iOS version."],
        citations=[],
        similarity_scores=[0.83],
        max_similarity=0.83,
    )


def test_generator_discards_max_tokens_truncated_output():
    """Real regression, observed twice in production across two different
    model generations (gemini-2.5-flash, then gemini-3.6-flash): Gemini
    spent its whole output-token budget on internal "thinking" before
    writing the visible answer, and the response came back truncated. The
    generator has since migrated to the google-genai SDK with
    thinking_config=ThinkingConfig(thinking_budget=0) to prevent this at the
    source (see generator.py), but this finish_reason check remains as a
    defense-in-depth backstop: whenever finish_reason is MAX_TOKENS for any
    reason, the generator must discard the partial text and fall back to
    the retrieved historical snippet instead of using it."""
    gen = GroundedReplyGenerator(provider="mock")
    gen._llm = _FakeLLM(_FakeGeminiResponse("We'd like to", finish_reason_name="MAX_TOKENS"))
    ret_res = _fake_retrieval_result()
    reply, passed, _violations = gen.generate("My iPhone battery dies in two hours", "HARDWARE_AND_BATTERY", ret_res)

    assert reply != "We'd like to"
    assert reply in ret_res.snippets
    assert passed is True


def test_generator_discards_suspiciously_short_stop_output():
    """Defense in depth: even a clean STOP finish reason shouldn't be trusted
    if the visible text is too short to be a real answer."""
    gen = GroundedReplyGenerator(provider="mock")
    gen._llm = _FakeLLM(_FakeGeminiResponse("We'd like to", finish_reason_name="STOP"))
    ret_res = _fake_retrieval_result()
    reply, _passed, _violations = gen.generate("My iPhone battery dies in two hours", "HARDWARE_AND_BATTERY", ret_res)

    assert reply != "We'd like to"
    assert reply in ret_res.snippets


def test_generator_accepts_complete_llm_output():
    """A full, well-formed answer with a normal STOP finish should be used
    as-is rather than discarded."""
    gen = GroundedReplyGenerator(provider="mock")
    full_reply = "We'd like to help with that. Please check Settings > Battery > Battery Health for details."
    gen._llm = _FakeLLM(_FakeGeminiResponse(full_reply, finish_reason_name="STOP"))
    ret_res = _fake_retrieval_result()
    reply, _passed, _violations = gen.generate("My iPhone battery dies in two hours", "HARDWARE_AND_BATTERY", ret_res)

    assert reply == full_reply
