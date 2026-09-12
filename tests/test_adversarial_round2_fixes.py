"""Round 2 of the adversarial review, pinned as executable assertions.

Every test here failed before the round-2 commit. They exist so that a later
refactor that reopens one of these holes fails the suite instead of shipping.
Each docstring states the observed behaviour before the fix, measured, not
recalled -- see docs/ADVERSARIAL_REVIEW.md for the full reproductions.
"""

import json
import time

import pytest

from src.drafting.guardrails import URL_EXTRACTOR, OutputGuardrail
from src.eval.splits import assign_splits
from src.triage.rules import CREDIT_CARD_REGEX, RuleMatcher


# ---------------------------------------------------------------------------
# R2-1: emphatic negation, without the comma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "No joke, my iPhone battery is swollen",
        "No joke my iPhone battery is swollen",
        "I'm not kidding, my phone is swollen",
        "I'm not kidding my phone is swollen",
        "no lie my iphone battery is bulging",
        "I'm not making this up my macbook battery is swollen",
        # These two sit the idiom INSIDE the device-noun-to-hazard-word match
        # span rather than before it. The first version of the round-2 fix
        # blanked only the lookback and left this bypass open; found by
        # re-reading the patch before committing it.
        "my battery no joke is swollen",
        "my iphone battery not kidding is swollen",
    ],
)
def test_emphatic_insistence_never_reads_as_a_denial(text):
    """Round 1 fixed this by truncating the negation lookback at punctuation.
    Dropping the comma -- what someone typing in a hurry about a swelling battery
    actually does -- restored the identical bypass: the highest-severity gate in
    the system returned (False, []) on a lithium hazard report."""
    assert RuleMatcher().detect_hardware_hazard(text)[0] is True


@pytest.mark.parametrize(
    "text",
    [
        "my battery is not swollen",
        "my battery is not really swollen",
        "my iphone battery isn't swollen, just warm",
        "the phone is not on fire",
        "there is no smoke coming from it",
        "my battery was never swollen",
    ],
)
def test_real_denials_still_suppress(text):
    """The other half of the trade: the negation rule must not stop suppressing
    genuine denials. A queue flooded with "it is NOT swollen" tickets is a real
    cost.

    NOTE ON A CASE REMOVED FROM THIS LIST. It originally asserted that
    "no my battery is swollen" suppresses. That assertion was wrong and I wrote
    it: "no my battery is swollen" is almost certainly "No, my battery IS
    swollen" -- a customer answering a question, reporting a hazard. The round-3
    negation rewrite escalates it, which is correct, and the test was encoding my
    own heuristic's behaviour as though it were the requirement. Removed rather
    than edited quietly, because a test that certifies a false negative on a
    lithium hazard is worse than no test."""
    assert RuleMatcher().detect_hardware_hazard(text)[0] is False


# ---------------------------------------------------------------------------
# R2-2: URL extractor -- false positives and quadratic backtracking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Let us know how it goes.Thanks for your patience",
        "Sorry about that.We will help.",
        "Update to iOS 17.4.1 and try again.",
    ],
)
def test_a_missing_space_after_a_full_stop_is_not_a_url(text):
    """`goes.Thanks` and `that.We` were extracted as URLs, failed the whitelist,
    and escalated ordinary support copy as UNAUTHORIZED_URL. A missing space
    after a period is commonplace in real replies, so this was not a rare edge."""
    assert URL_EXTRACTOR.findall(text) == []
    assert OutputGuardrail().validate_urls(text) == (True, [])


def test_the_phishing_case_the_scheme_less_branch_exists_for_still_fires():
    """Guards the fix against over-tightening: the whole reason the scheme-less
    branch was added in round 1 was that a bare phishing domain extracted to
    nothing and silently passed the whitelist."""
    ok, invalid = OutputGuardrail().validate_urls("Verify your identity at appleid-verify.com/unlock")
    assert ok is False
    assert invalid == ["appleid-verify.com/unlock"]


def test_url_extraction_is_linear_not_quadratic():
    """Measured on the old pattern: 16KB of "a-a-a-..." took 1.08s and 100KB took
    roughly 47s, inside a guardrail that runs on a public HTTP endpoint. The
    threshold below is deliberately loose (a slow CI box must not flake it) while
    still being ~100x under the old pattern's time at this length."""
    payload = "a-" * 50_000 + "a!"
    start = time.perf_counter()
    URL_EXTRACTOR.findall(payload)
    assert time.perf_counter() - start < 2.0


def test_the_endpoint_bounds_its_input():
    """The regex above is fixed, but the control that survives the next pattern
    someone adds is refusing a 100KB body at the edge."""
    from pydantic import ValidationError

    from src.server import QueryRequest

    QueryRequest(text="a" * 4000)
    with pytest.raises(ValidationError):
        QueryRequest(text="a" * 4001)


# ---------------------------------------------------------------------------
# R2-3: "overheat" alone was a risk-1.0 physical-damage hazard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "My MacBook overheats when I run Final Cut Pro",
        "my iphone overheats while charging overnight",
    ],
)
def test_an_ordinary_thermal_complaint_is_not_physical_damage(text):
    """`\\boverheat\\w*\\b` alone escalated the single most common Mac complaint
    there is at risk_score 1.0 under HARDWARE_PHYSICAL_DAMAGE -- wrong action and
    a false stated reason. Deliberate recall-for-precision trade, recorded in
    docs/DECISION_LOG.md."""
    assert RuleMatcher().detect_hardware_hazard(text)[0] is False


@pytest.mark.parametrize(
    "text",
    [
        "my iPhone is overheating and the back is scorched",
        "the macbook overheats so badly it shuts down and smells like burning",
        "phone overheating, too hot to touch",
    ],
)
def test_corroborated_overheating_still_escalates(text):
    """The half of the trade that must not be lost: thermal runaway with any
    severity signal is still a hazard."""
    assert RuleMatcher().detect_hardware_hazard(text)[0] is True


# ---------------------------------------------------------------------------
# R2-4: a 15-digit IMEI reported as CREDIT_CARD_DETECTED
# ---------------------------------------------------------------------------
def test_an_imei_is_not_reported_as_a_credit_card():
    """Escalating was arguably the safe call; calling it a credit card was a false
    stated reason, and a true stated reason is this system's whole contract.
    IMEIs carry a Luhn check digit too, so only issuer prefixes separate them."""
    assert RuleMatcher().detect_pii("My IMEI is 356938035643809") == (False, [])


@pytest.mark.parametrize(
    "number",
    [
        "4111111111111111",  # Visa
        "5500000000000004",  # Mastercard
        "340000000000009",  # Amex, 15 digits
        "6011000000000004",  # Discover
    ],
)
def test_real_separator_free_pans_are_still_caught(number):
    """The regression this guards: tightening to issuer prefixes must not undo
    the round-1 fix, which existed because a full PAN in a public tweet was being
    auto-handled and then echoed back by the draft."""
    assert CREDIT_CARD_REGEX.search(f"my card {number}") is not None
    assert RuleMatcher().detect_pii(f"my card {number}")[0] is True


def test_the_support_case_number_false_positive_stays_fixed():
    assert RuleMatcher().detect_pii("case number #100310750365") == (False, [])


# ---------------------------------------------------------------------------
# R2-5: stratified split collapsed rows sharing a tweet_id
# ---------------------------------------------------------------------------
def test_duplicate_tweet_ids_do_not_collide():
    """LATENT at the time of the fix: the shipped golden set has 188 distinct
    non-null ids, so no reported number was ever affected. The mechanism was real
    though -- a tweet_id-keyed assignment dict meant the last stratum processed
    overwrote an earlier stratum's decision for the same id, silently defeating
    stratification. Two rows with id None stringify to "None" and collide the
    same way."""
    rows = [
        {"tweet_id": "X", "true_triage_action": "ESCALATE", "is_edge_case": True},
        {"tweet_id": "X", "true_triage_action": "AUTO_HANDLE", "is_edge_case": False},
        {"tweet_id": None, "true_triage_action": "ESCALATE", "is_edge_case": False},
        {"tweet_id": None, "true_triage_action": "AUTO_HANDLE", "is_edge_case": False},
    ]
    out = assign_splits(rows)
    assert all(r["split"] in ("calibration", "heldout") for r in out)
    # Every row got its OWN decision -- no KeyError, and no row silently taking
    # the split computed for a different stratum.
    assert len(out) == 4


def test_assign_splits_is_still_deterministic():
    rows = [
        {"tweet_id": f"t{i}", "true_triage_action": "ESCALATE" if i % 3 == 0 else "AUTO_HANDLE", "is_edge_case": i % 2}
        for i in range(30)
    ]
    first = [r["split"] for r in assign_splits([dict(r) for r in rows])]
    second = [r["split"] for r in assign_splits([dict(r) for r in rows])]
    assert first == second


# ---------------------------------------------------------------------------
# R2-6: section 5b's numbers are produced by a script, not typed
# ---------------------------------------------------------------------------
def test_grounding_mode_numbers_come_from_a_generated_file():
    """The 18.6 / 9.6 / 33.0 in section 5b were typed into the report template
    from a one-off measurement taken BEFORE the leakage guard was repaired -- i.e.
    measured against a corpus that still contained the golden set's own answers.
    Real measurements of the wrong thing. They now come from
    docs/grounding_modes.json, which scripts/measure_grounding_modes.py writes."""
    from src.config import PROJECT_ROOT

    path = PROJECT_ROOT / "docs" / "grounding_modes.json"
    assert path.exists(), "run: python scripts/measure_grounding_modes.py"
    data = json.loads(path.read_text())
    assert data["population_n"] > 0
    rates = {(m["mode"], m["floor"]): m["false_positive_rate_pct"] for m in data["modes"]}
    assert set(rates) == {("lexical", 0.12), ("embedding", 0.3), ("embedding", 0.65)}
    # The conclusion the report draws must hold in the data the report quotes.
    assert rates[("embedding", 0.3)] < rates[("lexical", 0.12)]
