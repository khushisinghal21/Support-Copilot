# Hiver Take-Home: AI Support & Triage Agent for @AppleSupport

An AI agent that reads a customer's tweet, decides what they need, drafts a reply grounded in how `@AppleSupport` has actually resolved similar issues before, and decides whether to send that reply automatically or hand the ticket to a human -- with a stated reason either way.

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

Two examples end to end:

**"My iPhone battery drains really fast since the last update. How can I check which apps are using the most battery?"**
→ Intent: `HARDWARE_AND_BATTERY` (78% confidence) → retrieves 3 real historical resolutions on battery usage → drafts a grounded reply citing a real `support.apple.com` article → passes all guardrails → **AUTO_HANDLE**.

**"Someone hacked my iCloud and bought 100 gift cards, cancel this now!"**
→ Triage gate fires immediately on account-compromise + churn signals → **ESCALATE** (`HIGH_FRUSTRATION_CHURN_RISK`) → no auto-reply is drafted at all; routed straight to a human specialist.

---

## Running it

```bash
git clone https://github.com/nanthitha25/hiver_assignment.git
cd hiver_assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # optional: add a real GEMINI_API_KEY here
```

| Command | What it does |
| :--- | :--- |
| `python -m src.eval.runner` | Runs the full benchmark (188 examples, 2 baselines, LLM judge, human-agreement check) in under 15 minutes -- usually well under 1. |
| `pytest tests/ -v` | Runs the test suite (82 test functions). |
| `./run.sh` | Starts the dashboard + API at `http://localhost:8000`. |
| `python -m src.cli --query "..."` | Processes one query from the terminal. |

Without a `GEMINI_API_KEY`, generation and judging fall back to deterministic/template behavior -- the eval still runs and produces real metrics, just not LLM-graded ones.

---

## What this delivers

| Deliverable | Where |
| :--- | :--- |
| Runnable repo, reproducible in <15 min | this README, `run.sh` |
| Golden evaluation set (188 hand-labelled examples) | [`data/golden_eval_set.jsonl`](data/golden_eval_set.jsonl), [`data/README.md`](data/README.md) (sampling/labelling methodology) |
| Evaluation harness + LLM judge + human agreement | [`src/eval/`](src/eval/), [`docs/benchmark_summary.json`](docs/benchmark_summary.json) |
| Report (problem framing, baselines, failure analysis, "what's misleading about my number") | [`docs/REPORT.md`](docs/REPORT.md) |
| Decision log (15 non-obvious decisions) | [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) |

Datasets/models/libraries borrowed and cited: [`docs/REPORT.md`](docs/REPORT.md) Section 8. Full audit trail of what was found broken and fixed along the way: [`docs/AUDIT_AND_FIX_PLAN.md`](docs/AUDIT_AND_FIX_PLAN.md).

---

## Honest headline numbers

Regenerate anytime with `python -m src.eval.runner` -- these come straight from `docs/benchmark_summary.json`, not hand-typed.

| Metric | Trivial baseline | Simple (TF-IDF) baseline | Production |
| :--- | :---: | :---: | :---: |
| Intent accuracy | 29.3% | 52.1% | **62.2%** |
| Triage accuracy | 82.5% | 75.0% | **60.1%** ⚠️ |
| Escalation recall | 0.0% | 25.0% | **93.8%** |
| Missed escalations (of 32) | 32 | 24 | **2** |
| Human-judge kappa | -- | -- | **0.07** ("Slight agreement") |

Triage *accuracy* looks worse for the production system than either baseline -- that's expected, not a bug: the trivial baseline "wins" on accuracy only because it never escalates anything, which also means it misses 100% of real safety hazards. Escalation *recall* (93.8% vs 0%) is the number that actually matters for a safety system. Full breakdown, including why the human-judge kappa is weak and what that implies, is in `docs/REPORT.md` Section 5.

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

* **Rejected by design** -- live per-ticket debate routing, LLM self-reported confidence, a fine-tuned classifier, a learned meta-model over signals. All add complexity/cost without a clear safety win over the current deterministic cascade.
* **Tested and rejected, with evidence** -- an agentic retry loop for intent classification made accuracy *worse* (48.9-50.5% vs. 52.2% baseline); a 3-role debate judge cost 3x with no accuracy gain over a single-call judge.
* **Safety-driven scope boundaries** -- no tool-calling/real actions (draft-and-recommend only), no persistent memory (stateless by design), no data-poisoning defense (the corpus is static, not a live vector store -- that attack surface doesn't exist yet).
* **Known limitations, deferred** -- semantic-similarity fallback for synonyms, full multilingual embeddings, real sentence-transformer retrieval (sandbox-blocked, not abandoned), and softmax recalibration (the real bug was data leakage, not temperature) -- real gaps, disclosed rather than hidden.
