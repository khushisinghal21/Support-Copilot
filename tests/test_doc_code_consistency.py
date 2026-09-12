"""Phase 1.3: documented constants must equal live ones, enforced mechanically.

Doc/code drift is not a typo class, it is a trust class. Every instance found in
this repo was individually plausible and collectively corrosive: DECISION_LOG
claimed temperature scaling at T=0.12 while the classifier used 0.08; the README
advertised "82 test functions" when there were 91; a script's docstring said it
had never been run while AUDIT_AND_FIX_PLAN quoted its output. A reader who
catches one of those stops believing the rest of the document, which is the real
cost.

These tests parse the claims out of the docs and compare them to the imported
values, so the next divergence fails here instead of being discovered by whoever
is reading the docs to decide whether to trust the project.

DELIBERATELY NOT CHECKED: docs/specs/*. Those are pre-implementation design
intent and are allowed -- expected -- to differ from what shipped. Each
difference is instead enumerated in an "Implementation Status vs This
Specification" table inside the spec itself; test_spec_status_blocks_exist below
asserts those tables are present so the differences cannot quietly disappear.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------


def test_decision_log_temperature_matches_the_classifier():
    import inspect

    from src.intent.classifier import SemanticCentroidClassifier

    live = inspect.signature(SemanticCentroidClassifier.__init__).parameters["temperature"].default
    claimed = re.search(r"temperature scaling \(T=([0-9.]+)\)", _read("docs/DECISION_LOG.md"))
    assert claimed, "DECISION_LOG no longer states a temperature -- update this test if that is deliberate"
    assert float(claimed.group(1)) == pytest.approx(live), (
        f"DECISION_LOG says T={claimed.group(1)}, classifier default is {live}"
    )


def test_readme_test_count_matches_the_suite():
    """The README advertises a test count as a reproducibility signal; a wrong one
    is worse than none. Counted by collecting test functions, the same way a
    reader would verify it."""
    claimed = re.search(r"Runs the test suite \((\d+) test functions\)", _read("README.md"))
    assert claimed, "README no longer advertises a test count"

    actual = 0
    for path in (REPO_ROOT / "tests").glob("test_*.py"):
        tree_src = path.read_text(encoding="utf-8")
        actual += len(re.findall(r"^\s*def (test_\w+)", tree_src, re.MULTILINE))

    # Parametrised cases make the collected total exceed the function count, so
    # compare against functions and allow a small drift window rather than
    # pretending to an exactness that would fail on every new test.
    assert abs(int(claimed.group(1)) - actual) <= 5, (
        f"README claims {claimed.group(1)} test functions, found {actual} definitions"
    )


def test_readme_headline_numbers_match_the_last_real_eval_run():
    """The README table is supposed to come from docs/benchmark_summary.json, not
    a human's memory of a previous run. If benchmark_summary.json is absent this
    skips rather than fails -- a fresh clone has not run the eval yet."""
    import json

    summary_path = REPO_ROOT / "docs" / "benchmark_summary.json"
    if not summary_path.exists():
        pytest.skip("no eval run in this checkout yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    readme = _read("README.md")
    prod_triage = summary["triage"]["production"]["accuracy"]
    prod_recall = summary["triage"]["production"]["escalation_recall"]

    for label, value in (("triage accuracy", prod_triage), ("escalation recall", prod_recall)):
        rendered = f"{value * 100:.1f}%"
        assert rendered in readme, (
            f"README does not contain the measured {label} ({rendered}); "
            f"regenerate the table from docs/benchmark_summary.json"
        )


def test_readme_states_which_split_the_numbers_come_from():
    """A headline number without its population is not a claim, it is a vibe."""
    readme = _read("README.md")
    assert "held-out" in readme.lower(), "README must say the headline numbers are held-out only"


# ---------------------------------------------------------------------------
# Behavioural claims
# ---------------------------------------------------------------------------


def test_readme_flow_diagram_puts_the_safety_gate_before_generation():
    """The diagram claimed this while the code did the opposite for the whole
    life of the project until the gate split. Now that the code matches, keep it
    matching: assert the diagram's gate appears before its generator."""
    readme = _read("README.md")
    gate_idx = readme.find("Safety Triage Gate")
    gen_idx = readme.find("Generator")
    assert gate_idx != -1 and gen_idx != -1, "flow diagram labels changed"
    assert gate_idx < gen_idx, "diagram shows generation before the safety gate"


def test_pipeline_really_calls_input_gates_before_the_generator():
    """The executable half of the claim above -- source order in pipeline.py."""
    src = _read("src/pipeline.py")
    gate_idx = src.find("evaluate_input")
    gen_idx = src.find("reply_generator.generate")
    assert gate_idx != -1 and gen_idx != -1
    assert gate_idx < gen_idx, "generation is invoked before the input-side triage gates"


def test_spec_status_blocks_exist_where_the_specs_diverge():
    """specs 02 and 04 state requirements the shipped system does not meet. The
    divergence is acceptable; hiding it is not."""
    for rel in ("docs/specs/02_intent_classification_spec.md", "docs/specs/04_triage_escalation_spec.md"):
        content = _read(rel)
        assert "Implementation Status vs This Specification" in content, f"{rel} lost its status table"
        assert "src/config.py" in content, f"{rel} status table should cite the live values"


def test_no_doc_claims_the_calibration_script_was_never_run():
    """That docstring contradicted AUDIT_AND_FIX_PLAN 7.10, which quotes its
    output. Whichever was wrong, both could not be true."""
    assert "WHY IT HASN'T BEEN RUN YET" not in _read("scripts/calibrate_thresholds.py")


def test_committed_report_agrees_with_the_generator_on_every_measured_number():
    """The gap that let a stale docs/REPORT.md ship through two review rounds.

    This file parsed DECISION_LOG.md, README.md and docs/specs/* and never read
    docs/REPORT.md -- which is the headline deliverable. So the report kept
    `T=0.12` (the classifier's default has been 0.08), kept the three grounding
    false-positive rates that had been measured against a leaking corpus, and kept
    "the exact same 188-sample Golden Set" while the harness evaluated 124 rows --
    all after the generator had been fixed, because nobody regenerated the output
    and nothing checked.

    Asserting byte-identity against a fresh generation would require a full eval
    run in CI. Instead this pins the specific values a reader acts on: if the
    committed report disagrees with the live code or the measured artifacts, it is
    stale and must be regenerated with `python -m src.eval.runner`.
    """
    import inspect as _inspect
    import json as _json

    report = (REPO_ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")

    from src.intent.classifier import SemanticCentroidClassifier

    temperature = _inspect.signature(SemanticCentroidClassifier.__init__).parameters["temperature"].default
    assert f"$T={temperature}$" in report, (
        f"REPORT.md does not state the live classifier temperature ({temperature}). "
        "Regenerate it: python -m src.eval.runner"
    )

    grounding_path = REPO_ROOT / "docs" / "grounding_modes.json"
    if grounding_path.exists():
        data = _json.loads(grounding_path.read_text(encoding="utf-8"))
        for mode in data["modes"]:
            rate = f"{mode['false_positive_rate_pct']}%"
            assert rate in report, (
                f"REPORT.md section 5b does not contain the measured rate {rate} for "
                f"{mode['mode']}@{mode['floor']}. Regenerate it: python -m src.eval.runner"
            )
        # And must NOT contain the invalidated pre-leak-fix figures as live values.
        for stale in ("| 5 / 7 | **18.6%**", "| 5 / 7 | **9.6%**", "| 7 / 7 | **33.0%**"):
            assert stale not in report, (
                f"REPORT.md still publishes the pre-leakage-fix grounding rate {stale!r} "
                "as a live measurement. Regenerate it: python -m src.eval.runner"
            )


def test_report_discloses_the_false_clarify_rate():
    """The largest single triage error class went unreported until round 3.

    Accuracy alone hid it, and three documents said CLARIFY was "covered by unit
    tests only, not by this benchmark" -- true of the label, misleading about the
    gate. If a future change drops this disclosure, the report goes back to
    blaming over-escalation for an error budget that is mostly something else.
    """
    report = (REPO_ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")
    assert "CLARIFY" in report
    assert "False `CLARIFY`" in report or "false `CLARIFY`" in report, (
        "REPORT.md section 5 no longer discloses the false-CLARIFY rate. "
        "Regenerate it: python -m src.eval.runner"
    )
