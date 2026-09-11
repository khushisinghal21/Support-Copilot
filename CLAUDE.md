# Notes for agents working in this repository

Read `CONTRIBUTING.md` first; it has the commands. This file is the short list of
things that are easy to break without noticing, because each one has been broken
here before.

## Architecture invariants

**Input gates before generation.** `SupportPipeline.process` runs
`TriageEngine.evaluate_input()` (gates 1-5: prompt injection, hardware hazard, PII,
human request, frustration) immediately after classification and returns early if
any fires -- no retrieval, no LLM call, `drafted_reply=None`. Gates 6-9 run after
drafting via `evaluate_output()`. The pipeline used to generate first, which meant
the prompt-injection gate could not protect the model it was guarding. Do not
reorder this; `tests/test_pipeline_gate_order.py` will fail.

**Fail closed, always.** Any unhandled exception returns ESCALATE with
`SYSTEM_EXCEPTION_FAIL_CLOSED` and no draft. An exception must never produce an
auto-handled reply.

**Gate order and reason codes are a contract.** Reason codes appear in the audit
log (`data/decision_log.jsonl`) and tell a human operator why a ticket reached
them. Adding a code is fine; renaming or reordering is not.

**One encoder, loaded lazily.** Both the classifier and the retriever take their
`SentenceTransformer` from `src/embeddings.py`'s cached `get_encoder()`. Never
construct one directly: two instances means two copies of the model in a process
whose deploy target has 512MB. Heavy imports (`sentence_transformers`, `chromadb`)
stay **inside** functions -- a module-level import once delayed startup enough to
blow past a platform port-scan timeout, because `import` runs at load time whether
or not the object is ever constructed.

## Evaluation invariants

**No fabricated metrics, ever.** Every number in the docs traces to
`docs/benchmark_summary.json`. This repo has a history of the opposite: a kappa
floored at 0.72 in code regardless of measurement, a hardcoded list of "top failure
modes" unconnected to the failures actually computed, typed-in baseline judge
scores. Those were removed. Do not reintroduce the pattern in any form, including
a plausible-looking fallback constant.

**Held-out reporting.** Thresholds are calibrated on the `calibration` split;
headline numbers come from `heldout`. `scripts/calibrate_thresholds.py` must not
read a held-out row.

**Leakage exclusion.** `load_real_corpus()` excludes every golden-set
`source_tweet_id` from the RAG corpus. Removing that makes the eval meaningless
while making the numbers better, which is the dangerous direction.

**Disclose, don't smooth.** `docs/REPORT.md` section 5 exists to say what is
misleading about the headline numbers. When you find a new limitation, it goes
there. Known open ones: the held-out set contains zero `CLARIFY` rows (the golden
set has exactly one), the embedding grounding check cannot tell on-topic-but-
unsupported advice from supported advice, and the judge's kappa is 0.07.

## Documentation invariants

`docs/REPORT.md` is generated -- change `src/eval/report_generator.py`.
`docs/specs/*` are pre-implementation intent and are deliberately NOT edited to
match the code; where they diverge, each carries an "Implementation Status vs This
Specification" table. `tests/test_doc_code_consistency.py` enforces that documented
constants equal live ones, so a doc change without a code change (or vice versa)
fails CI.
