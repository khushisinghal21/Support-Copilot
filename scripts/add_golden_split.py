"""One-shot (idempotent) writer for the golden set's `split` field.

Run after the golden set changes:

    python scripts/add_golden_split.py

It rewrites data/golden_eval_set.jsonl in place, adding (or refreshing) a
deterministic, stratified `split` of "calibration" or "heldout" per row. The
assignment logic lives in src/eval/splits.py -- see that module's docstring for
why the split is persisted in the data file instead of recomputed at import time.
Running it twice produces a byte-identical file.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import GOLDEN_SET_PATH
from src.eval.splits import assign_splits, split_counts


def main() -> None:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    before = sum(1 for r in rows if "split" in r)
    rows = assign_splits(rows)

    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = split_counts(rows)
    total = sum(counts.values())
    print(f"{GOLDEN_SET_PATH}: {total} rows ({before} already carried a split field)")
    for name, n in sorted(counts.items()):
        print(f"  {name:>12}: {n:>4} ({n / total:.1%})")

    print("\nPer-stratum breakdown (true_triage_action | is_edge_case):")
    strata = {}
    for r in rows:
        key = f"{r.get('true_triage_action')}|{bool(r.get('is_edge_case'))}"
        strata.setdefault(key, {"calibration": 0, "heldout": 0})[r["split"]] += 1
    for key in sorted(strata):
        c, h = strata[key]["calibration"], strata[key]["heldout"]
        print(f"  {key:<28} calibration={c:<4} heldout={h:<4}")


if __name__ == "__main__":
    main()
