"""Measures the grounding check's false-positive rate, for docs/REPORT.md section 5b.

WHY THIS IS A SCRIPT AND NOT A PROSE PARAGRAPH
-----------------------------------------------
The numbers in section 5b used to be typed into the report template by hand from
a one-off measurement. That measurement was taken *before* the RAG leakage guard
was repaired, i.e. against a corpus that still contained the golden set's own
reference replies. Every reply therefore retrieved itself, grounding similarity
was ~1.0 by construction, and the false-positive rates that fell out of it
(18.6 / 9.6 / 33.0) described a broken corpus rather than the system. They were
real measurements of the wrong thing, which is the most dangerous kind of number
to leave in a report.

Recomputing by hand would only move the problem. This script writes
docs/grounding_modes.json, which the report reads; re-running the eval after any
change to the corpus, the retriever or the thresholds refreshes the report's
numbers with it, and a stale file is visible as a stale timestamp.

POPULATION
----------
All golden-set reference replies -- real historical @AppleSupport agent replies --
each scored against what the retriever returns for that row's own customer text.
A real agent's real reply to a ticket is grounded by definition, so anything a
grounding check flags in this population is a false positive. This is a
false-positive measurement only: it says nothing about how often each check
catches a genuinely ungrounded draft, and must not be quoted as accuracy.
"""

import json
import time

from src.config import PROJECT_ROOT
from src.drafting.guardrails import OutputGuardrail
from src.drafting.retriever import HistoricalRetriever
from src.eval.splits import load_golden_rows

MODES = [("lexical", 0.12), ("embedding", 0.30), ("embedding", 0.65)]


def measure() -> dict:
    rows = load_golden_rows()
    retriever = HistoricalRetriever()
    # Retrieve once per row and reuse across all three modes: the retrieval is
    # identical for each, and doing it three times would triple the runtime
    # while inviting the three rates to be measured against different draws.
    cases = []
    for row in rows:
        reply = (row.get("reference_resolution") or "").strip()
        text = (row.get("text") or "").strip()
        if not reply or not text:
            continue
        result = retriever.retrieve(text, intent=None, k=3)
        cases.append((reply, list(result.snippets)))

    results = []
    for mode, floor in MODES:
        kwargs = {"min_grounding_overlap": floor} if mode == "lexical" else {"min_grounding_similarity": floor}
        guardrail = OutputGuardrail(grounding_mode=mode, **kwargs)  # type: ignore[arg-type]
        flagged = sum(1 for reply, snippets in cases if not guardrail.check_grounding(reply, snippets)[0])
        results.append(
            {
                "mode": mode,
                "floor": floor,
                "n": len(cases),
                "flagged": flagged,
                "false_positive_rate_pct": round(100.0 * flagged / len(cases), 1) if cases else None,
            }
        )
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "population_n": len(cases), "modes": results}


if __name__ == "__main__":
    data = measure()
    out = PROJECT_ROOT / "docs" / "grounding_modes.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
