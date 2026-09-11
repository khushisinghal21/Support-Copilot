# Hardening Plan

Every change made in the hardening pass, as *defect → fix → verification*. Read this file
alone and you should know what happened and why. Items are grouped by the phase that
produced them. Anything **declined** is recorded here too, with the reason.

Baseline captured before any change: [`docs/baselines/before.txt`](baselines/before.txt)
(`pytest tests/ -v` + `python -m src.eval.runner`).

---

## Phase 1 — Correctness defects

### 1.1 The pipeline contradicted its own architecture diagram

**Defect (verified).** `src/pipeline.py` ran classify (L71) → retrieve (L74) → **generate (L81)**
→ triage (L88). `README.md`'s flow diagram showed the Safety Triage Gate *before* the retriever
and generator, and claimed of the hacked-iCloud example: *"no auto-reply is drafted at all."*
Both were false. Three consequences, all real:

* A prompt-injection tweet was interpolated into the Gemini prompt and reached the model
  *before* Gate 1 (the gate that exists to stop exactly that).
* Every hazard / PII / human-request / frustration ticket paid a full LLM call whose output was
  then discarded by triage.
* `src/drafting/guardrails.py`'s own module docstring conceded it: *"drafting currently runs
  before triage in the pipeline."*

**Fix.** `TriageEngine.evaluate` split into two phases, gate order and every reason code
preserved byte-for-byte:

* `evaluate_input(tweet)` → `Optional[TriageDecision]` — Gates 1–5 (prompt injection, hardware
  hazard, PII, human request, frustration/fraud). These depend only on the raw tweet text.
  Returns `None` when nothing fires, meaning "safe to spend retrieval + generation on".
* `evaluate_output(...)` → `TriageDecision` — Gates 6, 6b, 7, 8, 9 (intent confidence floor,
  CLARIFY band, retrieval similarity, generation guardrails, AUTO_HANDLE clearance).
* `evaluate(...)` retained as a thin backward-compatible wrapper that calls input-then-output,
  so the existing 91 tests and any external caller keep working unchanged.

`SupportPipeline.process` now calls `evaluate_input` immediately after classification and
short-circuits on a hit: no retrieval, no generator call, `drafted_reply=None`,
`grounding_context=None`.

**Verification.** A test asserts the generator is **never invoked** for an input-gated tweet
(spy/mock on `reply_generator.generate`, asserted `call_count == 0`), plus one test per input
gate proving the short-circuit, plus the full existing suite staying green.

**Deliberate non-change (noted, not done).** Gate 6 (intent-confidence floor) only needs
`intent_res` and could also move input-side for additional savings on low-confidence tickets.
The master spec puts Gates 6–9 in the output phase and says to keep ordering identical, so it
stays there; moving it would change which tickets pay for retrieval, which is a product
decision, not a refactor.

### 1.2 Thresholds were calibrated on the evaluation set

**Defect (verified).** `scripts/calibrate_thresholds.py` sweeps `MIN_INTENT_CONFIDENCE` and
`MIN_RETRIEVAL_SIMILARITY` against `GOLDEN_SET_PATH` — the same 188 rows `src/eval/runner.py`
then reports headline numbers on. `docs/AUDIT_AND_FIX_PLAN.md` §7.10 records that sweep being
run for real and both thresholds being changed on the strength of it (`MIN_RETRIEVAL_SIMILARITY`
0.40 → 0.20, `MIN_INTENT_CONFIDENCE` 0.35 → 0.40). There is no train/validation/test split
anywhere in the repo, so every reported triage number is optimistically biased — and
`docs/REPORT.md` §5 ("What is misleading about my headline number?") listed four caveats, none
of which was this one.

**Fix.**

* A deterministic, seeded (`random.Random(20260911)`), stratified split written **into the data
  file itself** as a `split` field per row (`"calibration"` ≈ ⅓ / `"heldout"` ≈ ⅔), stratified
  on `(true_triage_action, is_edge_case)` so both halves keep the escalation and edge-case mix.
  Persisted rather than recomputed so it is reproducible and auditable.
* `scripts/calibrate_thresholds.py` reads `calibration` rows only, and says so in its output.
* `src/eval/runner.py` reports headline numbers on `heldout` only, and prints the
  calibration-split numbers beside them so the generalization gap is visible rather than hidden.
* `docs/REPORT.md` §5 gains a new numbered item disclosing the previous bias and what changed.

**Verification.** Re-run `python -m src.eval.runner` and record the new numbers in
`docs/baselines/after.txt`. **The expectation is that they get worse; they are reported anyway.**
A test asserts every golden row carries a valid `split` value and that the stratified
proportions hold.

### 1.3 Documentation / code drift

**Defects (each verified against the code).**

| # | Claim | Where | Reality | Which side is wrong |
|---|---|---|---|---|
| a | temperature scaling `T=0.12` | `docs/DECISION_LOG.md` #4 | `src/intent/classifier.py` uses `temperature=0.08` | doc |
| b | "82 test functions" | `README.md` commands table | 91 test functions | doc |
| c | "WHY IT HASN'T BEEN RUN YET" | `scripts/calibrate_thresholds.py` docstring | `docs/AUDIT_AND_FIX_PLAN.md` §7.10 records it run for real, with output | code comment |
| d | "config.py actually set MIN_INTENT_CONFIDENCE=0.35" | same docstring | config.py now sets 0.40 | code comment |
| e | "drafting currently runs before triage" | `src/drafting/guardrails.py` docstring | false after 1.1 | code comment |
| f | triage gate drawn before retriever/generator | `README.md` flow diagram | false before 1.1, true after | resolved by 1.1 |
| g | "no auto-reply is drafted at all" (hacked-iCloud example) | `README.md` | false before 1.1, true after | resolved by 1.1 |

The full sweep of every numeric/behavioural claim in `README.md`, `docs/REPORT.md`,
`docs/DECISION_LOG.md`, `docs/specs/*`, and `data/README.md` against the code is recorded in
Phase 4, with each fix applied to whichever side was actually wrong.

**Fix.** Correct each side as marked above. Then add an automated guard
(`tests/test_doc_code_consistency.py`) that parses the documented constants out of the docs and
asserts they equal the live values imported from `src.config` / the classifier default, so this
class of drift fails CI instead of surviving to a reader.

**Verification.** The guard test fails when a constant is changed in code without the doc being
updated (demonstrated by a deliberate local edit, then reverted).

### 1.4 Concurrency and server hygiene (`src/server.py`)

| Defect (verified) | Fix | Verification |
|---|---|---|
| `get_pipeline()` is an unguarded global singleton — two concurrent cold requests build two `SupportPipeline`s and load two encoders, the exact OOM class the Render deploy notes fought on a 512MB tier | double-checked locking with a module-level `threading.Lock` | concurrency test: N parallel first-requests construct **exactly one** pipeline |
| `CORSMiddleware` with `allow_origins=["*"]` **and** `allow_credentials=True` — rejected by browsers, unsafe as written | origin list from `CORS_ALLOW_ORIGINS` env (comma-separated), sane default, credentials only enabled when the list is not `*` | test asserting the combination can never be `*` + credentials |
| `/api/process` has no auth and no rate limit, and spends Gemini money per call | in-process token-bucket rate limiter (configurable, generous default) + optional `X-API-Key` check, **off by default** so local dev is unaffected; both documented | tests for 429 on burst and for key-required mode |
| `_append_decision_log` does blocking, unbounded append I/O inside the request path | bounded, non-blocking write (background single-writer queue) keeping the existing "never break the pipeline" guarantee | test asserting a write failure still returns a normal response, and that the call does not block the response path |

### 1.5 Grounding check is lexically weak

**Defect (verified).** `OutputGuardrail.check_grounding()` is a bag-of-content-words overlap
ratio with a `0.12` floor. A fluent hallucination that reuses retrieved vocabulary passes; a
correct paraphrase fails. `docs/DECISION_LOG.md` #14 is already honest that this was a fallback
from when embeddings could not reliably be loaded — they can now (`src/embeddings.py`).

**Fix.** Add an embedding-similarity grounding check reusing the cached
`get_encoder()` (no second model load), selected explicitly, with the lexical check retained as
the offline/no-model fallback.

**Verification.** Measure both on the golden set, report which catches more genuinely ungrounded
drafts, and put that comparison in `docs/REPORT.md` — including if the embedding version wins by
less than expected.

### 1.6 Repository hygiene

**Defect (verified).** `git ls-files` shows `data/chroma_db/<uuid>/{data_level0,header,length,link_lists}.bin`,
`data/chroma_db/chroma.sqlite3`, and `data/decision_log.jsonl` all tracked — a built vector index
and runtime output committed to source control.

**Fix.** `git rm -r --cached` those paths, add them to `.gitignore`, confirm the code recreates
both on demand (`CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)` in `src/config.py` and
`_ensure_seed_data()` in `src/drafting/vector_store.py` already do). **History is not rewritten.**
Secrets audit reported explicitly.

**Verification.** Fresh clone + `pytest` + eval with the index absent, proving it rebuilds.

---

## Phase 2 — Engineering best practices

ruff (lint + format) and mypy configured in `pyproject.toml`, mypy strict on the safety-critical
core (`src/models.py`, `src/triage/`, `src/pipeline.py`) with the intended path to repo-wide
strict recorded; a pinned lockfile beside the loose `pyproject.toml` ranges; offline CI
(`.github/workflows/ci.yml`) running ruff + mypy + pytest-with-coverage + a fast eval subset with
no API key and no network; `Dockerfile` + `docker-compose.yml` (CPU-only torch, following
`render.yaml`'s single-pip-call reasoning); `print()` → `logging` with a `tweet_id` correlation id
on every request-path line; `src/config.py` migrated to a validated Pydantic `BaseSettings` with
bounds, **every existing explanatory comment preserved verbatim**; `.pre-commit-config.yaml`
(ruff + mypy + secrets scan); `CONTRIBUTING.md`; root `CLAUDE.md` recording the invariants an
agent must not break.

## Phase 3 — Test suite

Target ≥85% line coverage on `src/`, ≥95% on `src/triage/` and `src/drafting/guardrails.py`,
every test offline and deterministic. Unit (every regex including the documented historical
false positives as permanent regression tests; per-gate fire/not-fire plus ordering tests;
every guardrail check; classifier determinism and abstention; kappa **not** floored; config
bounds), integration (every terminal outcome, the 1.1 input-gate invariant, fail-closed at each
stage, FastAPI `TestClient`, vector-store leakage, eval end-to-end on a fixture, concurrency),
and a new adversarial/red-team suite (injection paraphrases the current regex misses, PII
leakage, unsafe-advice elicitation, link fabrication). **Where the system fails a red-team
test, the failure is recorded in `docs/REPORT.md` as a known limitation rather than the test
being weakened.**

## Phase 4 — Document reconciliation

`docs/REPORT.md` regenerated from a real eval run via `src/eval/report_generator.py` (the
generator is changed, never the generated output by hand); README flow diagram corrected to the
true post-1.1 order; a new decision-log entry per Phase 1–3 decision in the existing
*chosen-over-what, with evidence* style — including the entries where the honest split made the
numbers worse.

## Phase 5 — Adversarial review

Up to four rounds, three hostile subagents per round (staff engineer / adversary / skeptical
reviewer), every finding reproduced before being acted on, CONFIRMED findings fixed, declined
findings reasoned in `docs/DECISION_LOG.md`, and the whole loop — including the rejected and
unfixable findings — written up unsanitised in `docs/ADVERSARIAL_REVIEW.md`.

## Phase 6 — Interview-prep book

A published Artifact teaching the project end to end, with company/interview research cited and
*verified* sources separated from *inferred* ones.
