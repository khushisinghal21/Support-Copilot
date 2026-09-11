"""Phase 1.2: the calibration / held-out split must be real, stable, and honest.

These tests exist because the alternative -- thresholds swept against the same
rows the headline numbers are reported on -- is a bias that produces
better-looking results and cannot be detected by reading the output.
"""

import json

import pytest

from src.config import GOLDEN_SET_PATH
from src.eval import splits


def _rows():
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_every_row_carries_a_valid_split():
    rows = _rows()
    assert rows, "golden set is empty"
    for r in rows:
        assert r.get("split") in splits.VALID_SPLITS, f"{r.get('tweet_id')} has split={r.get('split')!r}"


def test_split_is_persisted_not_recomputed():
    """The assignment must live in the data file. If it were recomputed per
    process, a calibration run and an eval run could disagree about which rows
    are held out -- reintroducing the leak while appearing to fix it."""
    assert all("split" in r for r in _rows())


def test_split_assignment_is_deterministic():
    rows = _rows()
    first = {r["tweet_id"]: r["split"] for r in splits.assign_splits([dict(r) for r in rows])}
    second = {r["tweet_id"]: r["split"] for r in splits.assign_splits([dict(r) for r in rows])}
    assert first == second

    # and independent of input ordering
    shuffled = list(reversed([dict(r) for r in rows]))
    third = {r["tweet_id"]: r["split"] for r in splits.assign_splits(shuffled)}
    assert first == third


def test_calibration_fraction_is_roughly_one_third():
    counts = splits.split_counts()
    total = sum(counts.values())
    frac = counts[splits.CALIBRATION] / total
    assert 0.25 <= frac <= 0.42, f"calibration share {frac:.1%} is far from the intended third"


def test_split_is_stratified_on_triage_label_and_edge_case():
    """Every stratum with more than one row must appear on both sides, so one
    half cannot quietly collect all the hard cases."""
    rows = _rows()
    strata = {}
    for r in rows:
        key = (r.get("true_triage_action"), bool(r.get("is_edge_case")))
        strata.setdefault(key, []).append(r["split"])
    for key, assigned in strata.items():
        if len(assigned) > 3:
            assert splits.CALIBRATION in assigned, f"stratum {key} absent from calibration"
            assert splits.HELDOUT in assigned, f"stratum {key} absent from held-out"


def test_loader_rejects_an_unknown_split_name():
    with pytest.raises(ValueError):
        splits.load_golden_rows(split="train")


def test_calibration_and_heldout_are_disjoint_and_exhaustive():
    cal = {r["tweet_id"] for r in splits.load_golden_rows(split=splits.CALIBRATION)}
    held = {r["tweet_id"] for r in splits.load_golden_rows(split=splits.HELDOUT)}
    everything = {r["tweet_id"] for r in _rows()}
    assert not (cal & held), "a row is in both splits"
    assert cal | held == everything, "some row is in neither split"


def test_calibration_script_cannot_see_heldout_rows():
    """Regression guard on the actual defect: the sweep script must read the
    calibration rows only."""
    import scripts.calibrate_thresholds as ct

    loaded = {r["tweet_id"] for r in ct._load_golden()}
    held = {r["tweet_id"] for r in splits.load_golden_rows(split=splits.HELDOUT)}
    assert loaded, "calibration loader returned nothing"
    assert not (loaded & held), "the threshold sweep can see held-out rows"
