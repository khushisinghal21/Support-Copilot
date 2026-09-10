# Hiver SDE Intern Take-Home: AI Customer Support & Triage Agent

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-82%20test%20functions-blue.svg)](tests/)
[![Reproduction Time](https://img.shields.io/badge/reproduction-<15min-success.svg)](#-quickstart-reproduce-headline-results-in-under-15-minutes)
[![Human Agreement](https://img.shields.io/badge/Cohen's%20Kappa-κ%20=%200.07%20(Slight)-orange.svg)](#-llm-judge-human-agreement)
[![Escalation Safety Recall](https://img.shields.io/badge/Safety%20Recall-93.8%25%20(2%20Misses%20of%2032)-yellow.svg)](#-headline-benchmark-results-matrix)

An autonomous, data-grounded AI customer support agent and triage router engineered for **`@AppleSupport`** using real-world Twitter customer conversations from Kaggle.

> *"What we are testing: whether you can turn a messy real-world dataset into a working AI system and prove it works. The proof is worth more than the system."*

---

## 🎯 1. The Problem We Are Solving

### 1.1 The Real-World Business Context
Global consumer tech brands like **`@AppleSupport`** receive tens of thousands of customer support inquiries daily on public social media (Twitter/X). These inquiries span a wide spectrum of customer intent and severity:
* **Routine How-Tos**: *"How do I transfer photos from my iPhone to my Windows PC?"*
* **OS / Software Bugs**: *"Wi-Fi keeps dropping every few minutes since updating to iOS 11."*
* **Physical Safety Emergencies**: *"Smoke came out of my iPad charging port when I plugged it in!"*
* **Security & Account Fraud**: *"Someone hacked my iCloud and bought 100 gift cards, cancel this now!"*
* **Public PII Leaks**: *"My phone is locked, here is my email and phone number..."*
* **Outraged Churn Threats**: *"3 weeks and Apple stole my money, getting my lawyer involved."*

### 1.2 The "LLM Trap" (Why Naive AI Fails in Production)
Placing a generic, unconstrained Large Language Model directly in front of public customer inquiries leads to four catastrophic failure modes:
1. **Hallucinated Policies & Unauthorized Promises**: Naive LLMs invent warranty policies, promise free hardware replacements, or advise risky steps (e.g., unauthorized battery puncturing or unofficial jailbreaks).
2. **Deadly False-Positive Auto-Handling**: When a customer reports a swelling battery or smoking charger, a naive bot might cheerfully reply: *"We'd love to help! Have you tried restarting your device?"* That is a physical safety liability and a public PR disaster.
3. **Public Privacy (PII) Leaks**: A model may solicit customer Apple ID passwords, phone numbers, or credit card numbers in public tweet threads.
4. **Tone-Deaf Responses to Outraged Customers**: Emitting canned corporate greetings to a customer threatening legal action inflames customer outrage and drives public churn.

### 1.3 The Solution: Three Production-Grade Pillars
To make an autonomous agent trustworthy enough to deploy in production, our architecture enforces three distinct capabilities:
1. **Intent Classification**: Maps messy, colloquial tweets into an operational 5-class taxonomy derived empirically from data, safely routing ambiguous cases to human specialists.
2. **Grounded Reply Drafting (RAG)**: Retrieves historical resolution pairs from `@AppleSupport` and drafts replies strictly grounded in verified brand history (under 280 characters, official `apple.co` URLs only).
3. **Deterministic Safety Triage & Escalation Gate**: Cascading rule-based gates designed to catch physical hazards, thermal risks, liquid immersion, fraud, PII, and customer aggression before auto-handling. Measured at **93.8% escalation recall (2 missed cases out of 32)** on the current golden set -- not a guaranteed zero, and reported honestly rather than rounded up (see Section 3 below and `docs/REPORT.md`).

![Hiver AI Support Pipeline Workflow](./docs/assets/pipeline_workflow.png)

---

## 🚀 2. Quickstart: Reproduce Headline Results in Under 15 Minutes

The entire benchmark evaluation (188 hand-labelled examples, 2 baselines, LLM-as-a-judge rubric, and human-agreement calibration) runs in well under 15 minutes on standard CPU. **Run it yourself and read the numbers it prints** -- do not trust the table in Section 3 below (or `docs/REPORT.md`) without regenerating it first, since both are only as current as the last time someone ran `python -m src.eval.runner`.

### Step 1: Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/nanthitha25/hiver_assignment.git
cd hiver_assignment

# Create Python 3.11 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Headline Benchmark Runner (< 20 Seconds)
```bash
# Runs Baseline 1, Baseline 2, and Proposed System across all 200 Golden Set queries
python -m src.eval.runner
```

### Step 3: Run Automated Test Suite (All 33 Tests)
```bash
pytest tests/ -v
```

### Step 4: Launch Interactive Web Dashboard & Backend API
```bash
# Start FastAPI backend and interactive browser UI on port 8000
./run.sh
```
* **Web UI Dashboard**: Open [**`http://localhost:8000`**](http://localhost:8000) in your browser.
* **Interactive Swagger Docs**: [**`http://localhost:8000/docs`**](http://localhost:8000/docs)

---

## 📊 3. Headline Benchmark Results Matrix

> ⚠️ **These are the real, current numbers from `docs/benchmark_summary.json`, regenerated by `python -m src.eval.runner`.** An earlier version of this README showed different, higher numbers (87.5% intent accuracy, 93.0% triage accuracy, 100% escalation recall, κ = 0.72) from before an audit found those figures were partly hardcoded and floored rather than measured -- see `docs/AUDIT_AND_FIX_PLAN.md` for the full account. **Do not restore the old numbers without re-running the harness and confirming them yourself.**

Evaluated across the same **188-sample hand-labelled Golden Set** (~20% disclosed edge cases -- safety hazards, PII leaks, fraud, prompt injection):

| Evaluation Metric | Baseline 1 (Trivial Majority) | Baseline 2 (Simple TF-IDF) | Proposed System (Production) | Absolute Lift (vs Simple) |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Macro-F1** | 0.0905 | 0.5112 | **0.5587** | **+0.0475** |
| **Intent Accuracy** | 29.3% | 52.1% | **62.2%** | **+10.1%** |
| **Triage Accuracy** | 82.5% | 75.0% | **60.1%** | **-14.9% ⚠️** |
| **Escalation Recall** | 0.0% | 25.0% | **93.8%** | **+68.8%** |
| **Missed Escalations (of 32 true)** | 32 | 24 | **2** | **30 fewer missed** |
| **LLM Judge Quality (1–5 Scale)** | — | 3.95 | **4.55** | **+0.60** |
| **Human Agreement ($\kappa$)** | N/A | N/A | **$\kappa = 0.0716$ ("Slight agreement")** | **⚠️ weak -- see below** |
| **P95 Latency (CPU)** | < 1 ms | ~5 ms | **44.09 ms** | Real-time ready |

### Read this table correctly, not just at face value:
* **Triage Accuracy is *lower* for the production system than for either baseline.** That looks bad out of context, and it's exactly why raw accuracy is the wrong headline metric for a safety system: the Trivial baseline scores 82.5% by classifying almost everything `AUTO_HANDLE`, which also means it misses **100% of the 32 real safety escalations** in this set. The production system trades some accuracy for **93.8% escalation recall** (missing only 2 of 32 dangerous cases) because a false escalation costs a human reviewer a few seconds while a missed one is a safety failure -- see `docs/REPORT.md` Section 5, *"What is Misleading About My Headline Number?"*, for the full discussion.
* **Human-judge agreement ($\kappa = 0.07$) is weak.** Read the LLM judge's 1-5 scores above as a rough signal, not validated ground truth. This is the real, unfloored number -- a previous version of this codebase silently reported $\kappa = 0.72$ regardless of what was actually measured (removed; see `docs/AUDIT_AND_FIX_PLAN.md` Section 7.2 and `docs/REPORT.md` Section 3).
* **These numbers will drift** every time the golden set, thresholds, or code change. Regenerate with `python -m src.eval.runner` before citing any of this out loud.

---

## 🏗️ 4. Architecture, Use Case, Class & Sequence Diagrams

### 4.1 System Architecture Diagram

![System Architecture Diagram](./docs/assets/architecture_diagram.png)

<details>
<summary>📝 View Mermaid Code</summary>

```text
flowchart TD
    A["Customer Tweet"] --> B["Text Normalizer and PII Sanitizer"]
    B --> C["Intent Classifier"]
    C --> D{"Safety Gate"}
    D -->|"Hazard or PII or Fraud"| E["Triage Engine"]
    D -->|"Routine Support"| F["ChromaDB Vector Store"]
    F --> G["Grounded Reply Drafter"]
    G --> E
    E -->|"Escalate"| H["Tier-2 Human Specialist"]
    E -->|"Auto-Handle"| I["Auto-Reply Dispatcher"]
```

</details>

---

### 4.2 Use Case Diagram

![Use Case Diagram](./docs/assets/usecase_diagram.png)

<details>
<summary>📝 View Mermaid Code</summary>

```text
flowchart LR
    Customer["Customer"]
    Agent["Tier-2 Agent"]
    Auditor["Evaluator"]

    subgraph System["Hiver AI Support System"]
        UC1["UC1: Submit Customer Inquiry"]
        UC2["UC2: Classify Support Intent"]
        UC3["UC3: Deterministic Triage Gate"]
        UC4["UC4: Retrieve Historical Solutions"]
        UC5["UC5: Review Escalated Tickets"]
        UC6["UC6: Run Evaluation Harness"]
    end

    Customer --> UC1
    Customer --> UC2
    Customer --> UC4
    UC5 --> Agent
    UC6 --> Auditor
```

</details>

---

### 4.3 Class Diagram

![Class Diagram](./docs/assets/class_diagram.png)

<details>
<summary>📝 View Mermaid Code</summary>

```text
classDiagram
    class TweetInput {
        +string tweet_id
        +string text
        +string author_id
        +string created_at
    }

    class IntentResult {
        +string primary_intent
        +float confidence
        +string secondary_intents
        +string score_distribution
    }

    class TriageDecision {
        +string action
        +string stated_reason
        +string reason_code
        +float risk_score
    }

    class SupportResponse {
        +string tweet_id
        +IntentResult intent
        +TriageDecision triage
        +string drafted_reply
        +float execution_time_ms
    }

    class SupportPipeline {
        -string classifier_model
        -string vector_store
        -string triage_engine
        +process(tweet) SupportResponse
    }

    SupportPipeline ..> TweetInput : processes
    SupportPipeline --> SupportResponse : returns
    SupportResponse *-- IntentResult : contains
    SupportResponse *-- TriageDecision : contains
```

</details>

---

### 4.4 Sequence Diagram

![Sequence Diagram](./docs/assets/sequence_diagram.png)

<details>
<summary>📝 View Mermaid Code</summary>

```text
sequenceDiagram
    autonumber
    actor Customer
    participant Pipeline as SupportPipeline
    participant Classifier as IntentClassifier
    participant Triage as TriageEngine
    participant RAG as GroundedDrafter
    actor Agent as HumanAgent

    Customer->>Pipeline: Submit Customer Tweet
    Pipeline->>Classifier: Predict Intent
    Classifier-->>Pipeline: Return Intent and Confidence
    Pipeline->>Triage: Evaluate Safety and Intent

    alt Hazard or Sensitive PII or Fraud Detected
        Triage->>Agent: Route Ticket to Escalation Queue
        Triage-->>Pipeline: Return Escalate Decision
    else Safe Routine Support Query
        Pipeline->>RAG: Retrieve Historical Resolution
        RAG-->>Pipeline: Return Grounded Draft Reply
        Pipeline->>Triage: Validate Draft Safety
        Triage-->>Pipeline: Return Auto-Handle Decision
    end

    Pipeline-->>Customer: Return Support Response JSON
```

</details>

---

## ⚙️ 5. Deep-Dive: The Three Production Engines

### Engine 1: Intent Classification (`src/intent/`)
* **Taxonomy**: 5 data-derived canonical classes:
  * `OS_SOFTWARE_TROUBLESHOOTING`: iOS/macOS update glitches, crashing apps, Wi-Fi drops.
  * `HARDWARE_AND_BATTERY`: Battery drain, broken displays, charging port faults.
  * `ACCOUNT_BILLING_ICLOUD`: Apple ID lockouts, subscription charges, iCloud storage.
  * `HOW_TO_CONFIGURATION`: Device setup, photo transfer, AirDrop, Apple Pay.
  * `OUT_OF_SCOPE_AMBIGUOUS`: Non-Apple queries, vague rants, unparseable fragments.
* **Architecture**: Hybrid semantic centroid classifier using `sentence-transformers/all-MiniLM-L6-v2`.
* **Zero False-Positive Confidence Guard**: Queries with top intent confidence below threshold ($\tau < 0.40$, `MIN_INTENT_CONFIDENCE` in `src/config.py`) automatically fall back to `OUT_OF_SCOPE_AMBIGUOUS`, triggering safe human escalation instead of making confident errors.
  > **Revision note**: this section originally stated $\tau < 0.60$ while `config.py` actually set `0.35` -- a doc/code mismatch found during an audit (see `docs/AUDIT_AND_FIX_PLAN.md`). Both `MIN_INTENT_CONFIDENCE` (0.35 -> 0.40) and `MIN_RETRIEVAL_SIMILARITY` (0.40 -> 0.20, see Engine 2 below) were then empirically recalibrated with `scripts/calibrate_thresholds.py` against the real golden set and the real embedding model. The retrieval-similarity threshold in particular needed to come *down*, not up: it had been silently force-escalating ~61% of legitimate queries once the RAG corpus was switched from a tiny near-duplicated seed set to the real, diverse Kaggle corpus (Engine 2), which naturally produces lower genuine similarity scores. Full sweep data and reasoning in `docs/AUDIT_AND_FIX_PLAN.md` Section 7.10.

### Engine 2: Grounded Historical Reply RAG (`src/drafting/`)
* **Vector Store**: Embedded `chromadb` indexing genuine `@AppleSupport` customer-agent conversation pairs extracted from the Kaggle dataset (`src/drafting/vector_store.py`'s `load_real_corpus()`), explicitly excluding every `source_tweet_id` present in the golden set so the evaluation can't retrieve its own answer key. Falls back to a small hand-written seed corpus only if the real pairs file is missing.
* **Retrieval**: Intent-filtered semantic search ($k=3$) retrieving how real Apple agents historically resolved identical issues.
* **Guardrails**:
  * **Length Enforcement**: Strict $\le 280$ characters for Twitter single-tweet compatibility.
  * **Domain Whitelisting**: Strict regex whitelisting permits only official Apple links (`apple.co/...` or `support.apple.com/...`). Any hallucinated third-party link is rejected.
  * **PII Redaction**: Forbids requesting private credentials (Apple ID password, full serial numbers, credit cards) publicly, *and* blocks a draft that echoes back PII-shaped text the customer themselves posted (`check_pii_echo`).
  * **Unsafe-Advice Block**: Blocks drafts recommending physically unsafe self-repair (e.g. opening a swollen battery) via `check_unsafe_advice`.
  * **Grounding Check**: `check_grounding` requires lexical content-word overlap between a draft and its retrieved snippets as a cheap, model-free proxy for "this isn't inventing a procedure unrelated to what was actually retrieved."

### Engine 3: Deterministic Triage & Escalation Gate (`src/triage/`)
Evaluates queries through a cascading sequence of priority gates (real order, from `src/triage/engine.py`):
1. **Gate 1 (Prompt Injection)**: Detects attempts to hijack the AI drafting step via the customer's own text, independent of tone (`PROMPT_INJECTION_SUSPECTED`).
2. **Gate 2 (Thermal & Physical Hazards)**: Catches battery swelling, smoke, sparks, fire, shattered glass, and liquid immersion, with negation-aware matching so "didn't catch fire" doesn't trigger this gate (`HARDWARE_PHYSICAL_DAMAGE`).
3. **Gate 3 (PII & Security)**: Detects emails, phone numbers, SSNs, credit cards (`PII_SECURITY_SENSITIVE`).
4. **Gate 4 (Human Demand)**: Detects requests to speak with a human agent (`HUMAN_AGENT_REQUESTED`).
5. **Gate 5 (Corroborated Frustration & Fraud)**: Auto-triggers on unambiguous account-compromise/legal keywords; requires a second corroborating signal for softer keywords like "scam" or "sue" to reduce false escalations on hyperbole or reports where the customer isn't the victim (`HIGH_FRUSTRATION_CHURN_RISK`).
6. **Gate 6 (Intent Uncertainty, hard floor)**: Intent confidence below `MIN_INTENT_CONFIDENCE` forces escalation (`LOW_CONFIDENCE_AMBIGUOUS`).
7. **Gate 6b (Clarify)**: Moderate confidence on a device-dependent intent (OS/Hardware) with no device named in the text asks a clarifying question instead of guessing or escalating (`AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION`, `TriageAction.CLARIFY`) -- modeled on a real historical case where an @AppleSupport agent asked which device before answering.
8. **Gate 7 (Retrieval Grounding)**: Low historical retrieval similarity forces escalation (`LOW_CONFIDENCE_AMBIGUOUS`).
9. **Gate 8 (Generation Guardrail Failure)**: Routed to a specific reason (`PII_ECHO_IN_DRAFT`, `UNSAFE_ADVICE_BLOCKED`, `UNGROUNDED_GENERATION`) rather than one generic bucket, so the stated reason describes what actually failed.
10. **Gate 9 (Auto-Handle)**: Everything else clears to `AUTO_HANDLE`.

---

## 🛠️ 6. How to Give Input (4 Supported Interfaces)

### Method 1: Interactive Web Dashboard (Browser)
1. Run `./run.sh` and open **`http://localhost:8000`**.
2. Type any customer query into the **"Test Customer Tweet"** text box and click **"Process Query"**.
3. Or click any of the 1-click verified test scenarios on the left panel:
   * **Routine Auto-Handle**: *"How do I transfer photos from my iPhone to my Windows PC?"* $\rightarrow$ **`AUTO_HANDLE APPROVED`**
   * **Thermal Hazard**: *"Smoke came out of my iPad charging port when I plugged it in!"* $\rightarrow$ **`ESCALATE TO HUMAN AGENT`** (`BATTERY_THERMAL_HAZARD`)
   * **Liquid Immersion**: *"Dropped my phone in the pool and now it won't power on at all."* $\rightarrow$ **`ESCALATE TO HUMAN AGENT`** (`PHYSICAL_DAMAGE_INSPECTION_REQUIRED`)
   * **Security/Fraud**: *"Someone hacked my iCloud and bought 100 gift cards, cancel this now!"* $\rightarrow$ **`ESCALATE TO HUMAN AGENT`** (`HIGH_FRUSTRATION_CHURN_RISK`)
   * **Public PII**: *"My Apple ID is locked, here is my email test.user@icloud.com and phone 415-555-0199."* $\rightarrow$ **`ESCALATE TO HUMAN AGENT`** (`PII_SECURITY_SENSITIVE`)
   * **Human Demand**: *"Stop sending me automated bot replies! I want to speak to a real human person right now."* $\rightarrow$ **`ESCALATE TO HUMAN AGENT`** (`HUMAN_AGENT_REQUESTED`)

### Method 2: Terminal CLI
```bash
# Process a single customer inquiry:
python -m src.cli --query "My iPhone battery is swollen and warm to touch"

# Run in interactive loop mode:
python -m src.cli --interactive
```

### Method 3: REST API (curl / HTTP)
```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My iPhone battery dies within 2 hours after updating to iOS 11",
    "author_id": "customer_123"
  }'
```

### Method 4: Python Programmatic SDK
```python
from src.models import TweetInput
from src.pipeline import SupportPipeline

pipeline = SupportPipeline()
tweet = TweetInput(
    tweet_id="tw_001",
    text="How do I turn on AirDrop on my iPhone?",
    author_id="user_alex",
)
response = pipeline.process(tweet)

print("Intent:", response.intent.primary_intent.value)
print("Action:", response.triage.action.value)
print("Drafted Reply:", response.drafted_reply)
```

---

## 📦 7. Kaggle Dataset Ingestion & Curation

* **Source**: Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`, ~3M tweets).
* **Ingestion Script**: [`src/data/ingest_kaggle.py`](src/data/ingest_kaggle.py)
* **DuckDB Streaming Query**:
  ```sql
  SELECT 
      inbound.tweet_id AS inbound_tweet_id,
      outbound.tweet_id AS outbound_tweet_id,
      inbound.text AS customer_text,
      outbound.text AS agent_reply
  FROM 'data/twcs.csv' AS inbound
  JOIN 'data/twcs.csv' AS outbound 
    ON inbound.tweet_id = outbound.in_response_to_tweet_id
  WHERE outbound.author_id = 'AppleSupport'
    AND inbound.inbound = true
    AND inbound.in_response_to_tweet_id IS NULL  -- Isolates pure conversation initiators
    AND length(inbound.text) > 35
    AND length(outbound.text) > 35
  ```
* **Noise Reduction**: Filtering `in_response_to_tweet_id IS NULL` eliminates noisy mid-thread fragments ("yes", "tried that", bare links), preserving self-contained customer problem statements and high-quality reference solutions.

---

## 📂 8. Repository Structure & Deliverables Mapping

```text
hiver_assignment/
├── README.md                           # Main quickstart & system guide (< 15 min reproduction)
├── run.sh                              # Single-command launcher for Web UI + Backend API
├── pyproject.toml                      # Project metadata & dependency definitions
├── requirements.txt                    # Pinned Python package dependencies
│
├── src/                                # Core Application Source Code
│   ├── config.py                       # Central thresholds, model paths, brand constants
│   ├── models.py                       # Strict Pydantic v2 schemas (TweetInput, SupportResponse, etc.)
│   ├── pipeline.py                     # SupportPipeline orchestrator coordinating all 3 stages
│   ├── cli.py                          # Typer interactive CLI interface
│   ├── server.py                       # FastAPI REST backend & scenario dispatcher
│   ├── static/
│   │   └── index.html                  # Interactive browser dashboard (TailwindCSS)
│   ├── intent/                         # Intent Classification Engine
│   │   ├── taxonomy.py                 # 5-class canonical intent taxonomy enum
│   │   ├── classifier.py               # SemanticCentroidClassifier (MiniLM embeddings)
│   │   └── baselines.py                # Trivial (Majority) & Simple (TF-IDF) intent baselines
│   ├── drafting/                       # Grounded Reply Drafting Engine (RAG)
│   │   ├── vector_store.py             # ChromaDB vector store manager
│   │   ├── retriever.py                # Contextual semantic retriever
│   │   ├── generator.py                # Grounded reply generator
│   │   ├── guardrails.py               # Output guardrail (280 chars, URL whitelist, PII check)
│   │   ├── prompts.py                  # Apple Support tone system prompts
│   │   └── historical_data.py          # Seed resolution pairs
│   ├── triage/                         # Safety Triage & Escalation Engine
│   │   ├── engine.py                   # Cascading priority gate (AUTO_HANDLE vs ESCALATE)
│   │   ├── rules.py                    # Deterministic regex safety rules (battery, liquid, PII)
│   │   ├── sentiment.py                # Customer frustration, churn, and fraud detector
│   │   └── reasons.py                  # Structured explainable reason formatters
│   ├── eval/                           # Evaluation Harness & Metrics
│   │   ├── runner.py                   # 15-minute benchmark evaluation runner (< 20s execution)
│   │   ├── judge.py                    # LLM-as-a-Judge rubric (Groundedness, Tone, Safety)
│   │   ├── human_agreement.py          # Cohen's Kappa calculator (κ = 0.72)
│   │   ├── metrics.py                  # Classification reports, Macro-F1, confusion matrices
│   │   ├── report_generator.py         # Automated REPORT.md generator
│   │   └── curate_datasets.py          # Golden set generator utility
│   └── data/
│       └── ingest_kaggle.py            # DuckDB streaming Kaggle customer support extractor
│
├── data/                               # Evaluation & Historical Corpora
│   ├── README.md                       # Deliverable 2: Sampling & Labelling Methodology Guide
│   ├── golden_eval_set.jsonl           # Deliverable 2: 200 hand-labelled test queries
│   ├── human_annotations_sample.jsonl  # Deliverable 3: 50 human-annotated pairs for judge calibration
│   └── apple_support_kaggle_pairs.jsonl# Extracted Kaggle customer support pairs
│
├── docs/                               # Assignment Documentation & Audit Artifacts
│   ├── REPORT.md                       # Deliverable 4: Technical report (problem framing, results vs 2 baselines, failure analysis, misleading-number section, next steps)
│   ├── DECISION_LOG.md                 # Deliverable 5: Standalone 15-item architectural decision log
│   ├── specs/                          # Formal Specifications (00 through 05)
│   │   ├── 00_index.md
│   │   ├── 01_system_architecture_spec.md
│   │   ├── 02_intent_classification_spec.md
│   │   ├── 03_grounded_reply_drafting_spec.md
│   │   ├── 04_triage_escalation_spec.md
│   │   └── 05_evaluation_harness_and_baselines_spec.md
│   └── plan/                           # Implementation Plans (00 through 05)
│       ├── 00_master_execution_plan.md
│       └── ...
│
└── tests/                              # Automated Pytest Suite (33 Tests)
    ├── test_models.py                  # Schema validation & fail-closed contracts
    ├── test_intent.py                  # Taxonomy, classifier, and baseline tests
    ├── test_drafting.py                # RAG retrieval, guardrails, and character constraints
    ├── test_triage.py                  # Safety rules, PII detection, and escalation triggers
    ├── test_pipeline.py                # End-to-end pipeline integration & circuit breaker
    └── test_eval.py                    # Golden dataset integrity & Cohen's Kappa calibration
```

### Deliverables Traceability Matrix

| Assignment Deliverable | Repository Artifact | Notes & Compliance |
| :--- | :--- | :--- |
| **Deliverable 1: Runnable Pipeline** | [`README.md`](README.md), [`src/pipeline.py`](src/pipeline.py), [`run.sh`](run.sh) | Reproduces all headline results in ~20 seconds (< 15 min requirement). |
| **Deliverable 2: Golden Evaluation Set** | [`data/golden_eval_set.jsonl`](data/golden_eval_set.jsonl), [`data/README.md`](data/README.md) | 188 examples sampled and labelled from real `@AppleSupport` tweets, ~20% disclosed edge cases (adversarial + hard negatives), full sampling/labelling methodology note including known limitations. |
| **Deliverable 3: Evaluation Harness & Judge** | [`src/eval/runner.py`](src/eval/runner.py), [`src/eval/judge.py`](src/eval/judge.py), [`data/human_annotations_sample.jsonl`](data/human_annotations_sample.jsonl) | Automated metrics + LLM-as-a-judge rubric, honestly calibrated against 50 human-scored samples -- see `docs/REPORT.md` Section 3 for the real measured kappa (no hardcoded floor). |
| **Deliverable 4: Technical Report** | [`docs/REPORT.md`](docs/REPORT.md) | Full report covering Problem Framing, Results vs 2 Baselines, Top Failure Modes (mined from the actual run), *"What is Misleading About My Headline Number?"*, and Next Steps. Regenerate with `python -m src.eval.runner` before relying on it -- see that file. |
| **Deliverable 5: Architectural Decision Log** | [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) | Standalone plain list of 15 non-obvious engineering decisions and trade-off rationales. |
| **Ground Rule: Tooling Citations** | Section 10 in [`README.md`](README.md), Section 8 in [`docs/REPORT.md`](docs/REPORT.md) | Full accounting and citations of all borrowed datasets, embeddings, models, and libraries. |

---

## 🧪 9. Running the Automated Tests

The test suite validates schema enforcement, model calibration, RAG retrieval quality, deterministic safety regexes, and evaluation metrics:

```bash
pytest tests/ -v
```

Expected output:
```text
tests/test_drafting.py::test_retriever_returns_relevant_battery_resolutions PASSED
tests/test_drafting.py::test_retriever_returns_wifi_resolutions PASSED
tests/test_drafting.py::test_guardrail_length_pass_and_fail PASSED
tests/test_drafting.py::test_guardrail_catches_pii_solicitation PASSED
tests/test_drafting.py::test_guardrail_url_whitelisting PASSED
tests/test_drafting.py::test_generator_drafts_safe_reply PASSED
tests/test_eval.py::test_golden_dataset_schema PASSED
tests/test_eval.py::test_human_judge_agreement_calibration PASSED
tests/test_eval.py::test_report_file_generation PASSED
tests/test_intent.py::test_majority_baseline_classifier PASSED
tests/test_intent.py::test_simple_tfidf_classifier PASSED
tests/test_intent.py::test_semantic_classifier_hardware_intent PASSED
tests/test_intent.py::test_semantic_classifier_account_intent PASSED
tests/test_intent.py::test_semantic_classifier_software_intent PASSED
tests/test_intent.py::test_semantic_classifier_how_to_intent PASSED
tests/test_intent.py::test_semantic_classifier_ambiguous_fallback PASSED
tests/test_intent.py::test_semantic_classifier_latency PASSED
tests/test_models.py::test_tweet_input_valid PASSED
tests/test_models.py::test_tweet_input_rejects_empty_text PASSED
tests/test_models.py::test_intent_result_valid PASSED
tests/test_models.py::test_intent_result_rejects_invalid_confidence PASSED
tests/test_models.py::test_triage_decision_valid PASSED
tests/test_models.py::test_support_response_full_serialization PASSED
tests/test_pipeline.py::test_pipeline_single_tweet_default_execution PASSED
tests/test_pipeline.py::test_pipeline_batch_processing PASSED
tests/test_pipeline.py::test_pipeline_fail_closed_circuit_breaker PASSED
tests/test_triage.py::test_triage_battery_hazard_escalation PASSED
tests/test_triage.py::test_triage_pii_email_escalation PASSED
tests/test_triage.py::test_triage_human_request_escalation PASSED
tests/test_triage.py::test_triage_high_frustration_lawsuit_threat PASSED
tests/test_triage.py::test_triage_low_confidence_escalation PASSED
tests/test_triage.py::test_triage_low_retrieval_similarity_escalation PASSED
tests/test_triage.py::test_triage_safe_auto_handle_clearance PASSED

============================= 33 passed in 48.38s ==============================
```

---

## 📚 10. Citations & Borrowed Tooling

In accordance with the assignment instructions (*"Cite anything you borrowed. Borrowing is fine; not knowing what you borrowed is not"*), here is the comprehensive attribution of all external models, frameworks, and datasets used across this system:

1. **Kaggle Customer Support Dataset**:
   - **Source**: `thoughtvector/customer-support-on-twitter` (Kaggle, CC BY-NC-SA 4.0).
   - **Usage**: Used for isolating authentic inbound customer queries and outbound `@AppleSupport` agent resolutions to form our historical vector index and evaluation datasets.
2. **Sentence Transformers (`all-MiniLM-L6-v2`)**:
   - **Citation**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
   - **Usage**: Dense 384-dimensional semantic embeddings for zero-shot centroid intent classification and vector search retrieval, running entirely on CPU in $< 35\text{ms}$.
3. **ChromaDB**:
   - **Citation**: Chroma Core Team. (2023). *Chroma: The open-source embedding database* (Apache 2.0).
   - **Usage**: In-process local SQLite vector index storing `@AppleSupport` historical resolution pairs for intent-filtered retrieval-augmented drafting.
4. **DuckDB**:
   - **Citation**: Raasveldt, M., & Mühleisen, H. (2019). *DuckDB: an Embeddable Analytical Database*. In Proceedings of the 2019 International Conference on Management of Data (SIGMOD).
   - **Usage**: High-throughput out-of-core SQL streaming engine in `src/data/ingest_kaggle.py` for isolating conversation initiators without loading the entire 3-million-tweet CSV into memory.
5. **Scikit-Learn**:
   - **Citation**: Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
   - **Usage**: TF-IDF vectorizer and Logistic Regression for the statistical baseline (Baseline 2), and `cohen_kappa_score` for inter-rater agreement calibration.
6. **Pydantic v2**:
   - **Citation**: Colvin, S. et al. (2023). *Pydantic: Data validation using Python type hints*.
   - **Usage**: Runtime strict validation schemas ensuring fail-closed safety and type integrity across API boundaries.
7. **FastAPI & Uvicorn**:
   - **Citation**: Ramírez, S. et al. (2018). *FastAPI: High performance, easy to learn, fast to code, ready for production*.
   - **Usage**: Async ASGI web server powering the interactive dashboard and REST verification endpoints.
8. **Inter-Annotator Agreement Rubric**:
   - **Citation**: Landis, J. R., & Koch, G. G. (1977). *The measurement of observer agreement for categorical data*. Biometrics, 33(1), 159-174.
   - **Usage**: Benchmark scale for interpreting Cohen's Kappa ($\kappa = 0.7200$, establishing Substantial Agreement).
