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

**No fabricated metrics, ever.** Every number in the docs must trace to a
generated artifact -- `docs/benchmark_summary.json`, `docs/grounding_modes.json`,
or `docs/baselines/*.txt` -- or to a committed script that reproduces it. This
repo has a long history of the opposite: a kappa floored at 0.72 in code
regardless of measurement, a hardcoded list of "top failure modes" unconnected to
the failures actually computed, typed-in baseline judge scores, a fabricated
"46/188" in a commit message, grounding false-positive rates measured against a
leaking corpus, and a failure-mode title naming a class pair that occurred 3 times
above a count of 28. Every one of those was found by someone re-deriving the
number, not by reading the prose.

This invariant previously said "traces to `docs/benchmark_summary.json`" and named
only that file, which was itself false: the report's ROUGE-L row and the
trivial-baseline judge score are not in that JSON at all. The rule is "traces to
something re-runnable", and the way to honour it is to render the number from the
artifact rather than typing it.

**Held-out reporting.** Thresholds are calibrated on the `calibration` split;
headline numbers come from `heldout`. `scripts/calibrate_thresholds.py` must not
read a held-out row.

**Leakage exclusion.** `load_real_corpus()` excludes golden-set rows from the RAG
corpus on **three** independent keys: collapsed `source_tweet_id`, exact
`customer_text`, and exact `agent_reply`. Removing any of them makes the eval
meaningless while making the numbers better, which is the dangerous direction.

This paragraph said "excludes every golden-set `source_tweet_id`" -- the id-only
guard -- and that guard excluded **zero rows** for the life of the project,
because the two id formats could never be equal. One key is not enough precisely
because an id is a formatting convention that can drift silently; identical text
cannot. The guard logs a WARNING when it excludes nothing, and
`HistoricalVectorStore` stamps the persisted index with a fingerprint of the
corpus content, the exclusion sets, the record cap and the model, rebuilding when
any of them changes -- because the first version of this fix did not apply to
indexes that already existed on disk, which is how a repaired guard stayed a no-op
on every machine that had ever run the eval. See `docs/ADVERSARIAL_REVIEW.md`.

**Disclose, don't smooth.** `docs/REPORT.md` section 5 exists to say what is
misleading about the headline numbers. When you find a new limitation, it goes
there. Known open ones: the held-out set contains zero `CLARIFY` *labels* (the golden
set has exactly one) **while the CLARIFY gate fires on 23 of 124 held-out rows and
is wrong every time -- the largest single triage error class, and undisclosed until
adversarial review round 3 computed it**; the embedding grounding check cannot tell
on-topic-but-unsupported advice from supported advice; and the judge's kappa is
0.07. "Covered by unit tests only, not by this benchmark" is true of the label and
misleading about the gate -- do not write it that way again.

## Documentation invariants

`docs/REPORT.md` is generated -- change `src/eval/report_generator.py`.
`docs/specs/*` are pre-implementation intent and are deliberately NOT edited to
match the code; where they diverge, each carries an "Implementation Status vs This
Specification" table. `tests/test_doc_code_consistency.py` checks a
specific, enumerated set of documented constants and counts against live ones --
it is not a general guarantee, and saying it was one is how `docs/REPORT.md` kept
a wrong classifier temperature and three invalidated grounding rates through two
review rounds: that file parsed `DECISION_LOG.md`, `README.md` and `docs/specs/*`
and never read the report at all. It now also asserts the committed `REPORT.md`
agrees with what the generator currently produces. When you add a documented
constant, add it there too; the test is a list, not a net.
