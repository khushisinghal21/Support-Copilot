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
import logging
import random
from collections import defaultdict

from src.config import GOLDEN_SET_PATH

logger = logging.getLogger(__name__)

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

    # Keyed by id(), not by tweet_id. Keying on tweet_id meant two rows sharing an
    # id -- or two rows whose tweet_id is None, which stringify to the same "None"
    # -- collapsed to one dict entry, and the LAST stratum processed silently
    # overwrote the earlier one's assignment. A row in the ESCALATE stratum could
    # therefore receive the split computed for an AUTO_HANDLE row, defeating the
    # stratification this function exists to provide, with no error.
    #
    # LATENT, NOT ACTIVE: the current data/golden_eval_set.jsonl has 188 rows with
    # 188 distinct non-null tweet_ids, so no assignment was ever wrong in any
    # reported number. This closes the trapdoor before a future append opens it.
    # Found by adversarial review round 2; the collision is reproduced in
    # tests/test_splits.py::test_duplicate_tweet_ids_do_not_collide.
    assignment: dict[int, str] = {}
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda r: (str(r.get("tweet_id")), id(r)))
        rng = random.Random(f"{SPLIT_SEED}:{key}")
        rng.shuffle(members)
        # round() rather than int(): with int(), a stratum of 2 rows would put
        # zero rows in calibration, quietly dropping that stratum from the sweep.
        n_cal = max(1, round(len(members) * CALIBRATION_FRACTION)) if members else 0
        for i, row in enumerate(members):
            assignment[id(row)] = CALIBRATION if i < n_cal else HELDOUT

    for row in rows:
        row["split"] = assignment[id(row)]
    return rows


def load_golden_rows(split: str | None = None, limit: int | None = None) -> list[dict]:
    """Loads golden-set rows, optionally restricted to one split.

    Falls back to computing the split in memory if the on-disk file predates
    `scripts/add_golden_split.py` having been run, so an older checkout still
    works -- but the persisted field is the source of truth when present.
    """
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    missing = [r for r in rows if "split" not in r or r.get("split") not in VALID_SPLITS]
    if missing:
        # Assign ONLY the rows that lack a split, keeping every persisted
        # assignment exactly as it is. The previous version recomputed the whole
        # partition whenever a single row was missing one, which silently moved
        # 53 of 188 already-persisted rows across the boundary -- turning
        # previously held-out rows into calibration rows the thresholds were
        # tuned on, i.e. quietly reintroducing the exact bias the split exists to
        # remove, with no warning. Found by adversarial review.
        logger.warning(
            f"{len(missing)} golden row(s) have no persisted split; assigning those only. "
            f"Run `python scripts/add_golden_split.py` to persist them."
        )
        # Zipped by POSITION, not keyed by tweet_id: assign_splits() returns the
        # copies in the order it was given them, and a tweet_id-keyed lookup here
        # would reintroduce the same collision the function itself just stopped
        # making (duplicate or null ids collapsing to one entry).
        assigned = assign_splits([dict(r) for r in rows])
        by_identity = {id(row): a["split"] for row, a in zip(rows, assigned, strict=True)}
        for row in missing:
            row["split"] = by_identity[id(row)]

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
