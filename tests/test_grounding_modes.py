"""Phase 1.5: the two grounding implementations, including what each one misses.

The point of this file is not to show the new check is better everywhere -- it
isn't. It is to pin down, in executable form, exactly where each one succeeds and
fails, so neither can silently regress and neither gets oversold.

Measured on 188 real historical agent replies scored against what the retriever
returns for their own row (references are grounded by construction, so anything
flagged there is a false positive):

    lexical   @ 0.12 -> 18.6% of genuine replies wrongly flagged ungrounded
    embedding @ 0.30 ->  9.6%
    embedding @ 0.65 -> 33.0%   (rejected: see test docstring below)
"""

from unittest.mock import patch

import pytest

from src.drafting.guardrails import OutputGuardrail

SNIPPETS = ["Try a force restart: hold the side button and volume down."]


def test_invalid_grounding_mode_is_rejected_loudly():
    with pytest.raises(ValueError):
        OutputGuardrail(grounding_mode="semantic-ish")


def test_embedding_check_accepts_a_faithful_paraphrase():
    """The lexical check's structural weakness: a correct answer in different
    words. Measured 0.44 lexical vs 0.88 embedding."""
    guardrail = OutputGuardrail(grounding_mode="embedding")
    ok, score = guardrail.check_grounding(
        draft="Please perform a forced reboot by holding the side button together with volume down.",
        retrieved_snippets=SNIPPETS,
    )
    assert ok is True
    assert score > 0.6


def test_embedding_check_catches_a_truncated_draft_that_lexical_scores_perfect():
    """The lexical check returns (True, 1.00) for a draft with no content words --
    a *perfect* grounding score for a fragment. INCOMPLETE_DRAFT catches this
    separately, but a grounding measure reporting 1.00 for "We'd like to" is
    actively misleading. The embedding check scores it ~0.00."""
    lexical_ok, lexical_score = OutputGuardrail(grounding_mode="lexical").check_grounding(
        draft="We'd like to", retrieved_snippets=SNIPPETS
    )
    assert (lexical_ok, lexical_score) == (True, 1.0)

    embedding_ok, embedding_score = OutputGuardrail(grounding_mode="embedding").check_grounding(
        draft="We'd like to", retrieved_snippets=SNIPPETS
    )
    assert embedding_ok is False
    assert embedding_score < 0.2


def test_embedding_grounding_known_blind_spot():
    """KNOWN LIMITATION, asserted so it cannot regress silently.

    Embedding similarity measures topicality, not entailment. A draft giving
    DIFFERENT (and riskier) advice on the same subject as the retrieved snippet
    scores 0.64 -- above the 0.30 floor -- and passes.

    The threshold is not the fix. Raising it to 0.65 makes this case fail, but
    the measurement on 188 real agent replies shows that floor would falsely
    reject 33.0% of genuinely grounded replies (vs 9.6% at 0.30). Trading a 3.4x
    false-escalation rate for one class of catch is not a trade worth making.
    The real fix is an entailment/NLI model, which is out of scope here and is
    recorded as such in docs/REPORT.md.
    """
    ok, score = OutputGuardrail(grounding_mode="embedding").check_grounding(
        draft="You should try reinstalling the entire operating system from scratch tonight.",
        retrieved_snippets=SNIPPETS,
    )
    assert ok is True, "blind spot closed -- update this test and the report if so"
    assert 0.4 < score < 0.8


def test_embedding_check_rejects_unrelated_content():
    ok, score = OutputGuardrail(grounding_mode="embedding").check_grounding(
        draft="Your AppleCare subscription has been cancelled and a refund of $249 was issued.",
        retrieved_snippets=SNIPPETS,
    )
    assert ok is False
    assert score < 0.3


def test_embedding_mode_degrades_to_lexical_when_no_encoder_is_available():
    """CI and the offline suite run with no model. Failing every draft would turn
    a missing optional dependency into a total outage; passing every draft would
    silently disable a safety check. It must fall back instead."""
    guardrail = OutputGuardrail(grounding_mode="embedding")
    with patch("src.embeddings.get_encoder", side_effect=RuntimeError("no model here")):
        ok, score = guardrail.check_grounding(
            draft="You should try reinstalling the entire operating system from scratch tonight.",
            retrieved_snippets=SNIPPETS,
        )
    # i.e. it produced the LEXICAL verdict for this draft, not a crash and not a pass.
    assert ok is False
    assert score == pytest.approx(0.1111, abs=0.01)


def test_verbatim_snippet_is_grounded_by_construction_in_both_modes():
    for mode in ("lexical", "embedding"):
        ok, score = OutputGuardrail(grounding_mode=mode).check_grounding(draft=SNIPPETS[0], retrieved_snippets=SNIPPETS)
        assert ok is True, mode
        assert score == 1.0, mode


def test_violation_message_names_the_measure_actually_used():
    """An UNGROUNDED violation used to say 'lexical overlap' unconditionally. If
    the embedding check produced it, that string was simply wrong."""
    _, violations = OutputGuardrail(grounding_mode="embedding").evaluate(
        text="Your AppleCare subscription has been cancelled and a refund of $249 was issued.",
        retrieved_snippets=SNIPPETS,
    )
    ungrounded = [v for v in violations if v.startswith("UNGROUNDED")]
    assert ungrounded, f"expected an UNGROUNDED violation, got {violations}"
    assert "embedding similarity" in ungrounded[0]
