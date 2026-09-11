"""Builds data/human_annotations_sample.jsonl.

Replaces the previous version, which assigned scores with a rule like
"groundedness = 4 unless i % 7 == 0" and had no connection to an actual
second opinion. This version's scores were produced by reading each
customer tweet and its (mostly real, historical) reference reply in full and
applying the 1-5 rubric from src/eval/judge.py's JUDGE_SYSTEM_PROMPT --
disclosed here as an AI-assisted review pass, not a blind independent human
annotator (see data/README.md's "Known limitations" section for what that
means and what would strengthen it further: a second, genuinely independent
reviewer scoring the same 50 rows before submission).

Critically, this scoring pass does NOT reuse src/eval/judge.py's own
keyword-matching fallback logic -- it's a materially different method
(contextual reading vs. "does the string contain apple.co/"), which is what
makes comparing the two non-circular.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GOLDEN_PATH = DATA_DIR / "golden_eval_set.jsonl"
OUT_PATH = DATA_DIR / "human_annotations_sample.jsonl"

# golden_id -> (groundedness, tone, safety), scored by reading each
# customer tweet + reference reply against the rubric.
SCORES = {
    "golden_025": (5, 5, 5),
    "golden_169": (5, 5, 5),
    "golden_060": (4, 5, 5),
    "golden_137": (4, 5, 5),
    "golden_083": (3, 4, 5),
    "golden_161": (4, 5, 5),
    "golden_005": (5, 5, 5),
    "golden_076": (4, 4, 5),
    "golden_185": (5, 4, 5),
    "golden_171": (5, 5, 5),
    "golden_164": (4, 4, 5),
    "golden_108": (5, 5, 5),
    "golden_030": (5, 4, 5),
    "golden_184": (5, 4, 5),
    "golden_150": (5, 5, 5),
    "golden_048": (3, 4, 5),
    "golden_040": (3, 4, 5),
    "golden_039": (5, 5, 5),
    "golden_142": (3, 4, 5),
    "golden_067": (5, 5, 5),
    "golden_114": (3, 4, 4),
    "golden_186": (5, 5, 5),
    "golden_146": (5, 5, 5),
    "golden_003": (5, 4, 5),
    "golden_126": (4, 4, 5),
    "golden_118": (4, 4, 5),
    "golden_182": (3, 4, 5),
    "golden_050": (5, 5, 5),
    "golden_121": (5, 5, 5),
    "golden_157": (5, 5, 5),
    "golden_063": (5, 4, 5),
    "golden_168": (5, 5, 5),
    "golden_143": (5, 5, 5),
    "golden_178": (3, 4, 5),
    "golden_006": (3, 4, 5),
    "golden_031": (5, 5, 5),
    "golden_112": (5, 5, 5),
    "golden_105": (2, 4, 5),
    "golden_084": (5, 5, 5),
    "golden_073": (4, 4, 5),
    "golden_147": (5, 4, 5),
    "golden_092": (4, 5, 5),
    "golden_099": (5, 4, 5),
    "golden_102": (5, 4, 5),
    "golden_165": (5, 5, 5),
    "golden_045": (3, 4, 5),
    "golden_038": (3, 4, 5),
    "golden_055": (5, 4, 5),
    "golden_037": (5, 5, 5),
    "golden_156": (4, 4, 5),
}


def main():
    by_id = {}
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_id[d["tweet_id"]] = d

    rows = []
    for golden_id, (g, t, s) in SCORES.items():
        d = by_id[golden_id]
        rows.append(
            {
                "tweet_id": d["tweet_id"],
                "text": d["text"],
                "true_intent": d["true_intent"],
                "true_triage_action": d["true_triage_action"],
                "reference_resolution": d["reference_resolution"],
                "human_groundedness_score": g,
                "human_tone_score": t,
                "human_safety_score": s,
                "annotator_notes": (
                    "AI-assisted review: scored by reading the full tweet + reference reply "
                    "against the judge rubric, independently of src/eval/judge.py's own "
                    "keyword-matching logic. Not a blind independent human pass -- see "
                    "data/README.md limitations."
                ),
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} human-proxy annotations to {OUT_PATH}")


if __name__ == "__main__":
    main()
