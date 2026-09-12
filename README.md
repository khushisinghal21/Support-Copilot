# Hiver Take-Home: AI Support & Triage Agent for @AppleSupport

An AI agent that reads a customer's tweet, decides what they need, drafts a reply grounded in how `@AppleSupport` has actually resolved similar issues before, and decides whether to send that reply automatically or hand the ticket to a human -- with a stated reason either way.

> *Confident when it's right. Honest when it isn't. Never fabricates, never leaks, never pretends to be sure.*

**Live demo**: [hiver-ai-support-agent.onrender.com](https://hiver-ai-support-agent.onrender.com/) -- hosted on Render's free tier, so the *first* query after a period of inactivity can take 30-60s (cold start: the instance spins down when idle, then has to load the embedding model and index the historical corpus from scratch). Every query after that is fast. Not a bug, just the honest cost of the free tier.

Built and evaluated on real `@AppleSupport` conversations from Kaggle's `customer-support-on-twitter` dataset.

---

## How it works

```
Customer tweet
      │
      ▼
Intent Classifier ───────────► one of 5 intents (semantic similarity, no LLM)
      │
      ▼
Safety Triage Gate  ─────────► hazard / PII / fraud / human-request / prompt-injection?
      │
      ├── YES ───────────────► ESCALATE to a human, with a specific reason code. Stop here.
      │
      └── NO
           │
           ▼
      Retriever ──────────────► pulls the closest real historical @AppleSupport resolution(s)
           │
           ▼
      Generator ──────────────► drafts a reply grounded in that history
           │
           ▼
      Output Guardrails ──────► length, PII echo, unsafe advice, grounding, live link check
           │
           ├── FAIL ──────────► ESCALATE, with the specific guardrail that failed
           └── PASS ──────────► AUTO_HANDLE (reply goes out)
```

The safety gate genuinely sits where the diagram puts it: gates 1-5 (prompt
injection, hardware hazard, PII, human request, frustration/fraud) run on the raw
tweet in `TriageEngine.evaluate_input()` *before* any retrieval or LLM call, and
gates 6-9 run after drafting in `evaluate_output()`. Until the hardening pass the
code ran generate-then-triage and contradicted this picture -- which meant the
prompt-injection gate could not protect the model it was guarding, and 15.4% of
tickets paid for a reply that was then thrown away. `tests/test_pipeline_gate_order.py`
asserts the generator is never invoked for an input-gated tweet, so the diagram
cannot drift from the code again.

Two examples end to end:

**"My iPhone battery drains really fast since the last update. How can I check which apps are using the most battery?"**
→ Intent: `HARDWARE_AND_BATTERY` (74.2% confidence -- classification is deterministic, so this number is reproducible) → retrieves 3 real historical resolutions on battery usage → drafts a reply grounded in them → passes all guardrails → **AUTO_HANDLE**.

   *Without a `GEMINI_API_KEY` -- the mode every published number in this repo was produced in -- the drafter returns a template/snippet reply and cites no URL. An earlier version of this line said "78% confidence" and "citing a real `support.apple.com` article"; neither reproduced. Corrected after adversarial review round 3 checked it by running the example.*

**"Someone hacked my iCloud and bought 100 gift cards, cancel this now!"**
→ Triage gate fires immediately on account-compromise + churn signals → **ESCALATE** (`HIGH_FRUSTRATION_CHURN_RISK`) → no auto-reply is drafted at all; routed straight to a human specialist.

---

## Running it

```bash
git clone https://github.com/khushisinghal21/Support-Copilot.git
cd Support-Copilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # optional: add a real GEMINI_API_KEY here
```

| Command | What it does |
| :--- | :--- |
| `python -m src.eval.runner` | Runs the full benchmark (188 examples, 2 baselines, LLM judge, human-agreement check) in under 15 minutes -- usually well under 1. |
| `pytest tests/ -v` | Runs the test suite (267 test functions). |
| `./run.sh` | Starts the dashboard + API at `http://localhost:8000`. |
| `python -m src.cli process --text "..."` | Processes one query from the terminal. |

Without a `GEMINI_API_KEY`, generation and judging fall back to deterministic/template behavior -- the eval still runs and produces real metrics, just not LLM-graded ones.

---

## What this delivers

| Deliverable | Where |
| :--- | :--- |
| Runnable repo, reproducible in <15 min | this README, `run.sh` |
| Golden evaluation set (188 hand-labelled examples) | [`data/golden_eval_set.jsonl`](data/golden_eval_set.jsonl), [`data/README.md`](data/README.md) (sampling/labelling methodology) |
| Evaluation harness + LLM judge + human agreement | [`src/eval/`](src/eval/), [`docs/benchmark_summary.json`](docs/benchmark_summary.json) |
| Report (problem framing, baselines, failure analysis, "what's misleading about my number") | [`docs/REPORT.md`](docs/REPORT.md) |
| Decision log (43 non-obvious decisions) | [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) |
| Adversarial review: every round, every finding, including the rejected ones | [`docs/ADVERSARIAL_REVIEW.md`](docs/ADVERSARIAL_REVIEW.md) |

Datasets/models/libraries borrowed and cited: [`docs/REPORT.md`](docs/REPORT.md) Section 8. Full audit trail of what was found broken and fixed along the way: [`docs/AUDIT_AND_FIX_PLAN.md`](docs/AUDIT_AND_FIX_PLAN.md).

---

## Honest headline numbers

Regenerate anytime with `python -m src.eval.runner` -- these come straight from `docs/benchmark_summary.json`, not hand-typed.

**Measured on the 124 held-out rows only.** The two triage thresholds were tuned against the other 64 (`calibration`) rows, so reporting on those would flatter the system. See `docs/REPORT.md` Section 5, item 6.

> **The RAG corpus used to contain this evaluation set's own answers.** A guard that claimed to exclude every golden-set row from the retrieval corpus compared two id formats that could never be equal, so it excluded **nothing** — **80 of the 124** held-out rows could retrieve, verbatim, the exact reply they were scored against (123 of the 162 distinct reference replies sat in the indexed corpus). This README said otherwise for the life of the project. Fixed, with the full finding and reproduction in [`docs/ADVERSARIAL_REVIEW.md`](docs/ADVERSARIAL_REVIEW.md); grounding and ROUGE-L figures published before 2026-09-11 were inflated by an unknown amount.

| Metric | Trivial baseline | Simple (TF-IDF) baseline | Production |
| :--- | :---: | :---: | :---: |
| Intent accuracy | 29.8% | 40.3% | **63.7%** |
| Triage accuracy | 83.1% | 75.8% | **63.7%** ⚠️ |
| Escalation recall | 0.0% | 33.3% | **95.2%** |
| Missed escalations (of 21) | 21 | 14 | **1** |
| Human-judge kappa | -- | -- | **0.07** ("Slight agreement") |

Triage *accuracy* looks worse for the production system than either baseline -- that's expected, not a bug: the trivial baseline "wins" on accuracy only because it never escalates anything, which also means it misses 100% of real safety hazards. Escalation *recall* (95.2% vs 0%) is the number that actually matters for a safety system.

**What actually drives the 63.7%, stated because it is not what you would guess:** of 45 triage errors, 23 are the system asking a clarifying question instead of answering (22% of all genuine `AUTO_HANDLE` traffic), 21 are false escalations, and 1 is a missed escalation. The false-`CLARIFY` class is the largest and was undisclosed until the third round of adversarial review computed the confusion matrix. `docs/REPORT.md` Section 5 item 3.

Measured on all 188 rows, with thresholds tuned on those same rows, this code scores 61.7% / 64.9% / 93.8% (`docs/baselines/before.txt`). Moving to held-out reporting cost **2.0 points of triage accuracy and 3.3 points of escalation recall**, while intent accuracy rose.

   *This paragraph previously compared against 62.2% / 60.1% / 93.8% and then asserted a 2.0-point triage loss -- but 60.1% → 62.9% is a gain of 2.8. The 60.1% figure came from the pre-hardening README and was itself never a measurement of this code; splicing two baselines into one sentence produced a delta that contradicted its own numbers. Found by adversarial review round 3.* Full breakdown, including why the human-judge kappa is weak and what that implies, is in `docs/REPORT.md` Section 5.

---

## Architecture

* `src/intent/` -- semantic centroid classifier over 5 data-derived intents, no training data required.
* `src/drafting/` -- retrieval-augmented generation over real historical `@AppleSupport` replies, plus output guardrails (length, PII, unsafe advice, grounding, and live link verification -- `src/drafting/link_checker.py` actually fetches any cited URL rather than trusting a domain whitelist alone).
* `src/triage/` -- a 9-gate deterministic cascade (prompt injection → hazards → PII → human request → frustration/fraud → confidence → clarify → grounding → generation guardrails) that decides `AUTO_HANDLE`, `ESCALATE`, or `CLARIFY`, always with a stated reason code.
* `src/eval/` -- the benchmark harness, LLM-as-judge, human-agreement calibration, and failure-mode mining.

Diagrams, full gate-by-gate spec, and citations: `docs/specs/`.

---

## What I decided not to build

Full reasoning for each: [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md).

* **Rejected by design** -- a live per-ticket **debate agent** (multiple LLM "roles" arguing out each triage decision before it's made), LLM self-reported confidence, a fine-tuned classifier, a learned meta-model over signals. All add complexity/cost without a clear safety win over the current deterministic cascade.
* **Tested and rejected, with evidence** -- an agentic retry loop for intent classification made accuracy *worse* (48.9-50.5% vs. 52.2% baseline); a 3-role **debate agent** as the LLM judge cost 3x the API calls with no accuracy gain over a single-call judge, so it was simplified back down.
* **Safety-driven scope boundaries** -- no tool-calling/real actions (draft-and-recommend only), no persistent memory (stateless by design), no data-poisoning defense (the corpus is static, not a live vector store -- that attack surface doesn't exist yet).
* **Known limitations, deferred** -- semantic-similarity fallback for synonyms, full multilingual embeddings, real sentence-transformer retrieval (sandbox-blocked, not abandoned), and softmax recalibration (the real bug was data leakage, not temperature) -- real gaps, disclosed rather than hidden.

---

## Future scope

Full detail: [`docs/REPORT.md`](docs/REPORT.md) Section 6 ("What We'd Do Next With One More Week").

* **Active learning** -- feed human accept/reject/edit decisions on auto-drafted replies back into the vector store as fresh, human-validated examples.
* **Multi-turn thread context** -- use conversation history (`in_reply_to_tweet_id`) so the agent doesn't re-ask what's already known.
* **Bayesian threshold tuning** -- optimize the confidence-gate thresholds against a target cost-per-escalation trade-off instead of hand-picked values.
* **Automated red-teaming** -- a prompt-injection/jailbreak tester that tries to make the agent leak internal prompts or offer fake gift cards.
