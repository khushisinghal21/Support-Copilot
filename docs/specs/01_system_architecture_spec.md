# Specification 01: System Architecture & Pipeline Orchestration

- **Module Name**: `system_architecture`
- **Target Version**: `1.0.0`
- **Status**: `Approved`
- **Target Source Files**:
  - `src/pipeline.py`
  - `src/config.py`
  - `src/models.py`
  - `src/cli.py`

---

## 1. Problem Statement & Scope Boundaries

### 1.1 Problem Statement
Customer support on public social media (Twitter/X) is noisy, high-volume, multi-turn, and fraught with reputational risk. A real-world AI customer support agent must handle incoming queries with extreme reliability: correctly routing issues, drafting polite and historically grounded responses, and safely escalating ambiguous, frustrated, or high-risk cases to human agents.

For **`@AppleSupport`**, customer queries range from trivial questions ("How do I restart my iPhone?") to complex OS bugs, hardware battery recalls, stolen devices, and sensitive Apple ID account lockouts. A generic LLM risks hallucinating unsupported warranty policies, attempting to collect private credentials in public, or providing harmful advice. 

The **System Architecture** orchestrates an end-to-end processing pipeline that consumes raw customer tweets, executes intent classification, retrieves historically grounded brand resolutions, applies a strict triage/escalation policy gate, and outputs a structured decision payload.

### 1.2 What "Good" Means for @AppleSupport
1. **Safety & Privacy First**: Zero tolerance for requesting Apple ID passwords, credit card info, or sensitive credentials publicly. Immediate escalation or deflection to Apple's official secure support link (`apple.co/...`) or DM.
2. **Empathetic & Polished Tone**: Adherence to Apple Support's signature conversational style ("We'd be glad to help," acknowledging OS version, concise diagnostic questions).
3. **High Groundedness**: Every proposed recommendation must reflect actual procedures historically documented in the Twitter support dataset or Apple Support Knowledge Base.
4. **Transparent Explainability**: Auto-handling vs. escalation decisions must have unambiguous, audit-ready reasoning.

### 1.3 What We Chose NOT to Build (Explicit Scope Boundaries)
1. **No Direct Twitter Write Access / Live Bot Tweeting**: We do not deploy a live automated bot that posts directly to public Twitter. It operates as an internal copilot/triage pipeline for support agents.
2. **No Backend Apple Internal DB Integration**: We do not simulate or mock real iCloud database modifications or device unlocks. Hardware repairs and warranty checks are deflected to official Genius Bar appointment flows.
3. **No Multi-Language Translation in v1**: The scope is strictly bounded to English tweets within the `@AppleSupport` dataset.

---

## 2. Tech Stack & Dependencies

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Runtime & Language** | Python 3.11+ | Industry standard for modern LLM applications, typing support, async I/O. |
| **Data Models & Validation** | `pydantic` v2.x | High-performance schema validation, JSON serialization, strict type contracts. |
| **Vector Storage (RAG)** | `chromadb` (Embedded SQLite) | Zero external daemon dependency, runs in-process, instant setup, reproducible in < 15 min. |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, fast, deterministic, zero API cost, runs on CPU/M-series Mac. |
| **LLM Inference** | Dual-Provider Strategy: Google Gemini API (`gemini-2.5-flash`) / Local Fallback | High reasoning capability, low latency, structured JSON mode. |
| **Data Processing** | `pandas` & `duckdb` | Fast querying of the 3M Kaggle Twitter dataset without blowing up RAM. |
| **CLI & Harness** | `typer` & `rich` | Beautiful terminal output, progress bars, automated benchmarking in a single command. |
| **Testing** | `pytest`, `pytest-cov`, `pytest-mock` | Automated test suite and regression harness. |

---

## 3. Functional & Non-Functional Requirements

### 3.1 Functional Requirements (FR)
- **FR-01 (End-to-End Ingestion)**: Pipeline must accept an incoming raw tweet JSON payload containing `tweet_id`, `text`, `author_id`, and optional `in_reply_to_tweet_id`.
- **FR-02 (Pipeline Orchestration)**: System must execute the workflow in sequential order: Ingestion -> Preprocessing -> Intent Classification -> Historical Retrieval -> Triage Decision -> Grounded Reply Drafting -> Output Structuring.
- **FR-03 (Dual-Mode Execution)**: Pipeline must support both batch processing (for evaluation of 150–250 golden items) and single-query interactive CLI mode.
- **FR-04 (Fallback & Circuit Breaking)**: If LLM API fails or times out, the pipeline must fail-closed to `ESCALATE` with reason `"LLM_UPSTREAM_TIMEOUT"`.

### 3.2 Non-Functional Requirements (NFR)
- **NFR-01 (Latency)**: Single tweet processing latency must be $\le 1.8$ seconds on CPU + Cloud LLM, and $\le 250$ ms on pure retrieval + local heuristic.
- **NFR-02 (Reproducibility)**: The entire evaluation harness and headline result generation must execute in **under 15 minutes** from cold start on a standard developer machine.
- **NFR-03 (Determinism & Auditability)**: All decisions must be saved to a structured JSONL audit trail with timestamps, intent scores, retrieval similarity scores, and escalation flags.
- **NFR-04 (Zero PII Leakage)**: Customer handles, emails, and phone numbers must be sanitized/masked before logging.

---

## 4. Architecture & Design Diagrams

### 4.1 Architecture Diagram
```mermaid
flowchart TD
    subgraph Input_Layer["Input & Ingestion Layer"]
        CT["Customer Tweet (CLI / Kaggle / API)"]
        Pre["Text Normalizer & PII Sanitizer"]
    end

    subgraph Core_Engine["Core AI Support Pipeline (src/pipeline.py)"]
        subgraph Sub_Intent["1. Intent Classification Engine"]
            IC["Intent Classifier (Few-Shot / MiniLM)"]
            Taxonomy["Data-Derived Taxonomy (5 Core Classes)"]
        end

        subgraph Sub_RAG["2. Grounded Retrieval (RAG)"]
            VS[("ChromaDB Vector Store (Historical Apple Resolutions)")]
            Retriever["Contextual Semantic Retriever (Top-k)"]
            Generator["Grounded Reply Drafter (Gemini / LLM)"]
        end

        subgraph Sub_Triage["3. Triage & Escalation Engine"]
            Safety["Deterministic Safety & PII Rules"]
            Confidence["Confidence Evaluator"]
            Triage["Decision Gate (AUTO_HANDLE vs ESCALATE)"]
        end
    end

    subgraph Output_Layer["Output & Persistence Layer"]
        Result["Structured SupportResponse (JSON)"]
        AuditLog[("Audit Log (JSONL)")]
        HumanQueue["Human Agent Escalation Queue"]
        AutoDraft["Auto-Reply Dispatcher"]
    end

    CT --> Pre
    Pre --> IC
    IC -.-> Taxonomy
    IC --> Triage
    IC --> Retriever
    Retriever -.-> VS
    Retriever --> Generator
    Generator --> Triage
    Safety --> Triage
    Confidence --> Triage
    Triage --> Result
    Result --> AuditLog
    Result -->|action == ESCALATE| HumanQueue
    Result -->|action == AUTO_HANDLE| AutoDraft
```

### 4.2 Class Diagram
```mermaid
classDiagram
    class TweetInput {
        +str tweet_id
        +str text
        +str author_id
        +Optional[str] created_at
        +Optional[str] in_reply_to_tweet_id
        +validate()
    }

    class IntentResult {
        +str primary_intent
        +float confidence
        +List~str~ secondary_intents
        +dict raw_scores
    }

    class RetrievalResult {
        +List~str~ context_snippets
        +List~float~ similarity_scores
        +List~str~ historical_tweet_ids
        +float max_similarity
    }

    class TriageDecision {
        +TriageAction action
        +str stated_reason
        +float risk_score
        +List~str~ triggered_rules
        +bool requires_pii
    }

    class TriageAction {
        <<enumeration>>
        AUTO_HANDLE
        ESCALATE
    }

    class SupportResponse {
        +str tweet_id
        +IntentResult intent
        +TriageDecision triage
        +Optional[str] drafted_reply
        +RetrievalResult grounding_context
        +float execution_time_ms
        +to_dict() dict
        +to_json() str
    }

    class SupportPipeline {
        -IntentClassifier intent_classifier
        -HistoricalRetriever retriever
        -ReplyGenerator reply_generator
        -TriageEngine triage_engine
        +process(TweetInput input) SupportResponse
        +batch_process(List~TweetInput~ inputs) List~SupportResponse~
    }

    SupportPipeline --> TweetInput : consumes
    SupportPipeline --> SupportResponse : produces
    SupportResponse *-- IntentResult
    SupportResponse *-- TriageDecision
    SupportResponse *-- RetrievalResult
    TriageDecision *-- TriageAction
```

### 4.3 Use Case Diagram
```mermaid
graph LR
    actor Customer as "Customer (Twitter User)"
    actor HumanAgent as "Tier-2 Human Support Agent"
    actor Evaluator as "Hiver Evaluator / SDE"

    subgraph System["Hiver AI Support Agent System"]
        UC1["Submit Support Inquiry"]
        UC2["Classify Intent"]
        UC3["Retrieve Historical Resolutions"]
        UC4["Draft Grounded Reply"]
        UC5["Decide Auto-Handle vs Escalate"]
        UC6["Review Escalated Ticket with Stated Reason"]
        UC7["Execute 15-Minute Evaluation Harness"]
        UC8["Inspect Failure Modes & Benchmark Report"]
    end

    Customer --> UC1
    UC1 --> UC2
    UC2 --> UC3
    UC3 --> UC4
    UC4 --> UC5
    UC5 -->|Escalated| UC6
    HumanAgent --> UC6
    Evaluator --> UC7
    Evaluator --> UC8
```

### 4.4 Activity Diagram
```mermaid
stateDiagram-v2
    [*] --> IngestTweet
    IngestTweet --> SanitizePII : Normalize Text & Check PII
    SanitizePII --> ClassifyIntent : Extract intent & confidence

    state CheckSafety <<choice>>
    ClassifyIntent --> CheckSafety : Evaluate Hard Rules
    CheckSafety --> EscalateEarly : Rule Match (PII / Abuse / Hardware Recalls)
    
    CheckSafety --> RetrieveContext : Safe for AI Processing
    RetrieveContext --> DraftReply : Fetch Top-k Historical Tweets
    DraftReply --> CheckConfidence : Evaluate Grounding & Model Certainty

    state ConfidenceGate <<choice>>
    CheckConfidence --> ConfidenceGate
    ConfidenceGate --> EscalateWithReason : Confidence < Threshold or Ambiguous
    ConfidenceGate --> FinalizeAutoReply : Confidence >= Threshold & Policy Passed

    EscalateEarly --> FormatOutput
    EscalateWithReason --> FormatOutput
    FinalizeAutoReply --> FormatOutput

    FormatOutput --> WriteAuditLog
    WriteAuditLog --> [*]
```

### 4.5 Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant Pipeline as SupportPipeline (src/pipeline.py)
    participant Classifier as IntentClassifier (src/intent.py)
    participant Retriever as HistoricalRetriever (src/retriever.py)
    participant Drafter as ReplyGenerator (src/generator.py)
    participant Triage as TriageEngine (src/triage.py)
    participant Output as SupportResponse

    Customer->>Pipeline: Submit raw tweet text
    Pipeline->>Classifier: classify(clean_text)
    Classifier-->>Pipeline: IntentResult(intent="BATTERY_PERFORMANCE", conf=0.92)
    
    Pipeline->>Triage: evaluate_pre_generation(intent, clean_text)
    alt Immediate Hard Rule Escalation (e.g. Broken Glass / PII)
        Triage-->>Pipeline: TriageDecision(action=ESCALATE, reason="HARDWARE_DAMAGE_REQUIRES_GENIUS_BAR")
        Pipeline->>Output: construct_response(drafted_reply=None, triage=ESCALATE)
    else Proceed to Drafting
        Triage-->>Pipeline: TriageDecision(action=PENDING_GENERATION)
        Pipeline->>Retriever: retrieve_similar_resolutions(query, intent, k=3)
        Retriever-->>Pipeline: RetrievalResult(snippets, scores=[0.88, 0.81, 0.77])
        
        Pipeline->>Drafter: draft_reply(clean_text, intent, snippets)
        Drafter-->>Pipeline: drafted_text
        
        Pipeline->>Triage: evaluate_post_generation(drafted_text, scores)
        Triage-->>Pipeline: TriageDecision(action=AUTO_HANDLE, reason="CONFIDENCE_HIGH_GROUNDED_ANSWER")
        Pipeline->>Output: construct_response(drafted_reply=drafted_text, triage=AUTO_HANDLE)
    end
    Pipeline-->>Customer: Return SupportResponse
```

---

## 5. Data Schemas & API Contracts

### 5.1 Input Schema (`TweetInput`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TweetInput",
  "type": "object",
  "properties": {
    "tweet_id": { "type": "string" },
    "text": { "type": "string", "minLength": 1, "maxLength": 1000 },
    "author_id": { "type": "string" },
    "created_at": { "type": "string" },
    "in_reply_to_tweet_id": { "type": ["string", "null"] }
  },
  "required": ["tweet_id", "text", "author_id"]
}
```

### 5.2 Output Schema (`SupportResponse`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SupportResponse",
  "type": "object",
  "properties": {
    "tweet_id": { "type": "string" },
    "intent": {
      "type": "object",
      "properties": {
        "primary_intent": { "type": "string" },
        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "secondary_intents": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["primary_intent", "confidence"]
    },
    "triage": {
      "type": "object",
      "properties": {
        "action": { "type": "string", "enum": ["AUTO_HANDLE", "ESCALATE"] },
        "stated_reason": { "type": "string" },
        "risk_score": { "type": "number" },
        "triggered_rules": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["action", "stated_reason"]
    },
    "drafted_reply": { "type": ["string", "null"] },
    "grounding_context": {
      "type": "object",
      "properties": {
        "snippets": { "type": "array", "items": { "type": "string" } },
        "max_similarity": { "type": "number" }
      }
    },
    "execution_time_ms": { "type": "number" }
  },
  "required": ["tweet_id", "intent", "triage", "execution_time_ms"]
}
```

---

## 6. Definition of Done & Executable Test Cases

| Test ID | Target Component | Command / Verification Action | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **TC-ARCH-01** | Data Models | `pytest tests/test_models.py` | All Pydantic models validate valid JSON and reject malformed schemas. |
| **TC-ARCH-02** | Pipeline Integration | `pytest tests/test_pipeline.py::test_end_to_end_single_tweet` | Single tweet runs through all 3 stages in $< 2.0$s and outputs valid `SupportResponse`. |
| **TC-ARCH-03** | Fail-Closed Timeout | `pytest tests/test_pipeline.py::test_upstream_timeout_escalates` | LLM mock timeout results in `action="ESCALATE"` with `stated_reason="LLM_UPSTREAM_TIMEOUT"`. |
| **TC-ARCH-04** | CLI Harness | `python -m src.cli process --text "My iPhone 11 battery drains in 2 hours"` | Prints formatted Rich card with classified intent, grounded reply, and triage decision. |
