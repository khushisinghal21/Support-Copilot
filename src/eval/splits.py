"""Calibration / held-out split for the golden evaluation set.

WHY THIS EXISTS
----------------
`scripts/calibrate_thresholds.py` sweeps MIN_INTENT_CONFIDENCE and
MIN_RETRIEVAL_SIMILARITY against the golden set, and `src/eval/runner.py` then
reported headline numbers on that same golden set. docs/AUDIT_AND_FIX_PLAN.md
section 7.10 records that sweep being run for real and both thresholds being
changed on the strength of it. With no train/test separation anywhere in the
repo, every reported triage number was therefore tuned on the data it was scored
against -- optimistically biased by construction -- and docs/REPORT.md section 5
("What is misleading about my headline number?") did not disclose it.

The split is:
  * deterministic and seeded, so it is reproducible run to run;
  * stratified on (true_triage_action, is_edge_case), so both halves keep the
    escalation mix and the adversarial/edge-case mix rather than one half
    accidentally collecting all the hard rows;
  * persisted INTO data/golden_eval_set.jsonl as a per-row `split` field, so the
    assignment is auditable in version control and cannot drift between a
    calibration run and an eval run -- which is exactly the failure mode a
    recomputed-on-each-import split would reintroduce.

CALIBRATION_FRACTION is one third: enough rows to see the shape of a threshold
sweep, while leaving the majority of the set untouched for reporting. A 50/50
split would have produced a noisier held-out number on a 188-row set.
"""

import json
import random
from collections import defaultdict

from src.config import GOLDEN_SET_PATH

# Fixed so the split is reproducible. Changing this re-partitions the data and
# invalidates comparability with every previously reported number -- don't.
SPLIT_SEED = 20260911
CALIBRATION_FRACTION = 1.0 / 3.0

CALIBRATION = "calibration"
HELDOUT = "heldout"
VALID_SPLITS = (CALIBRATION, HELDOUT)


def _stratum_key(row: dict) -> str:
    """Rows are stratified on the two attributes that actually matter for a
    threshold sweep: the triage label being predicted, and whether the row is an
    authored/edge case (which behaves very differently from organic traffic)."""
    return f"{row.get('true_triage_action')}|{bool(row.get('is_edge_case'))}"


def assign_splits(rows: list[dict]) -> list[dict]:
    """Returns the rows with a deterministic stratified `split` field added.

    Pure function of (row contents, SPLIT_SEED) -- it sorts by tweet_id inside
    each stratum before shuffling so the result does not depend on the order the
    file happened to be written in.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_stratum_key(row)].append(row)

    assignment: dict[str, str] = {}
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda r: str(r.get("tweet_id")))
        rng = random.Random(f"{SPLIT_SEED}:{key}")
        rng.shuffle(members)
        # round() rather than int(): with int(), a stratum of 2 rows would put
        # zero rows in calibration, quietly dropping that stratum from the sweep.
        n_cal = max(1, round(len(members) * CALIBRATION_FRACTION)) if members else 0
        for i, row in enumerate(members):
            assignment[str(row.get("tweet_id"))] = CALIBRATION if i < n_cal else HELDOUT

    for row in rows:
        row["split"] = assignment[str(row.get("tweet_id"))]
    return rows


def load_golden_rows(split: str | None = None, limit: int | None = None) -> list[dict]:
    """Loads golden-set rows, optionally restricted to one split.

    Falls back to computing the split in memory if the on-disk file predates
    `scripts/add_golden_split.py` having been run, so an older checkout still
    works -- but the persisted field is the source of truth when present.
    """
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if any("split" not in r for r in rows):
        rows = assign_splits(rows)

    if split is not None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        rows = [r for r in rows if r["split"] == split]

    return rows[:limit] if limit else rows


def split_counts(rows: list[dict] | None = None) -> dict[str, int]:
    """{split: n} for the whole golden set, for reporting."""
    rows = rows if rows is not None else load_golden_rows()
    counts: dict[str, int] = dict.fromkeys(VALID_SPLITS, 0)
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts
