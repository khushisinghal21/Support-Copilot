# Specification 03: Grounded Reply Drafting Engine

- **Module Name**: `grounded_reply_drafting`
- **Target Version**: `1.0.0`
- **Status**: `Approved`
- **Target Source Files**:
  - `src/drafting/generator.py`
  - `src/drafting/retriever.py`
  - `src/drafting/vector_store.py`
  - `src/drafting/prompts.py`
  - `src/drafting/guardrails.py`

---

## 1. Problem Statement & Scope Boundaries

### 1.1 Problem Statement
When LLMs draft customer support replies without grounding, they hallucinate policies, invent discount codes, promise non-existent hardware replacements, or advise risky steps (such as unauthorized jailbreaking or unofficial repair shops). 

For **`@AppleSupport`**, an acceptable draft must:
1. Mirror Apple's exact historical resolution tone: calm, empathetic, professional, and concise (under 280 characters when formatted for Twitter).
2. Propose verified, safe diagnostic steps (e.g., *Settings > Battery > Battery Health*, Force Restart, updating via iTunes/Finder).
3. Use only verified Apple domain link patterns (`apple.co/...`, `support.apple.com/...`) or invite the user to direct message (DM) when account-specific diagnostics are required.
4. Strictly ground its guidance in how real Apple Support agents have historically resolved similar issues in the Kaggle dataset.

### 1.2 Scope Boundaries
- The engine uses **Retrieval-Augmented Generation (RAG)** over an indexed corpus of historical Apple Support conversation pairs `(customer_inquiry, apple_agent_response)`.
- If no historically relevant resolution is found above a similarity threshold ($\text{cosine similarity} < 0.65$), the engine refuses to guess and signals the Triage Engine to escalate to a human agent.
- Generation is constrained by a post-generation **Grounding Guardrail** that flags any hallucinated URLs, external links, or unverified claims.

---

## 2. Tech Stack & Dependencies

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Vector Store** | `chromadb` (In-Memory / Local SQLite) | Embedded, zero server dependencies, instant setup, reproducible in $< 15$ min. |
| **Embedding Model** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast 384-dimensional dense embeddings matching the Intent Classifier. |
| **LLM Inference** | Google Gemini 2.5 Flash / Claude / OpenAI API with fallback | Fast generation, strict adherence to system instructions, JSON mode support. |
| **Prompt Engineering** | Few-Shot Grounded Template Engine | Dynamically injects top-$k$ historical resolutions and tone constraints. |
| **Evaluation Metrics** | ROUGE-L, BERTScore, Lexical Overlap, Grounding Faithfulness | Measures adherence to historical brand replies. |

---

## 3. Functional & Non-Functional Requirements

### 3.1 Functional Requirements (FR)
- **FR-DRAFT-01 (Historical Retrieval)**: The system must query the vector index for the top-$k$ ($k=3$) most semantically relevant historical resolution pairs for the classified intent.
- **FR-DRAFT-02 (Contextual Prompt Construction)**: The generator must inject the customer tweet, the classified intent, and the retrieved historical resolutions into a strict system prompt.
- **FR-DRAFT-03 (Length Constraint)**: Generated replies must not exceed 280 characters to adhere to Twitter's single-tweet format.
- **FR-DRAFT-04 (Anti-Hallucination Guardrail)**: The output must be scanned for URLs. Any URL not matching `^https?://(apple\.co|support\.apple\.com)/` must be stripped or flagged.
- **FR-DRAFT-05 (Groundedness Score)**: The engine must compute a lexical/semantic overlap score with the retrieved context snippets.

### 3.2 Non-Functional Requirements (NFR)
- **NFR-DRAFT-01 (Generation Latency)**: Retrieval + Generation must complete in $< 1.5$ seconds per query.
- **NFR-DRAFT-02 (Brand Tone Adherence)**: $\ge 90\%$ of generated drafts must receive a score of $\ge 4/5$ on the LLM-as-a-judge "Brand Voice & Grounding" rubric.
- **NFR-DRAFT-03 (Security)**: The prompt must explicitly forbid the model from asking for Apple ID passwords, credit card numbers, or full serial numbers in public tweets.

---

## 4. Architecture & Design Diagrams

### 4.1 Architecture Diagram
```mermaid
flowchart TD
    subgraph Input_Query
        Query["Clean Customer Tweet + Classified Intent"]
    end

    subgraph Retrieval_Subsystem["Historical RAG Subsystem"]
        Embedder["SentenceTransformer (MiniLM)"]
        ChromaStore[("ChromaDB Index (10,000 Apple Support Pairs)")]
        Filter["Intent-Filtered k-NN Search (k=3)"]
    end

    subgraph Prompt_Orchestrator["Prompt Engine"]
        ContextFormatter["Format Top-3 (Query -> Apple Resolution) Pairs"]
        SystemPrompt["Apply Apple Brand Tone & Constraint System Prompt"]
    end

    subgraph Generation_Engine["Generation & Guardrails"]
        LLM["Gemini 2.5 Flash (Temperature = 0.2)"]
        Guard["Post-Gen Guardrail (Length, PII, Link Validator)"]
        DraftResult["Validated DraftedReply"]
    end

    Query --> Embedder
    Embedder --> Filter
    ChromaStore -.-> Filter
    Filter --> ContextFormatter
    Query --> ContextFormatter
    ContextFormatter --> SystemPrompt
    SystemPrompt --> LLM
    LLM --> Guard
    Guard --> DraftResult
```

### 4.2 Class Diagram
```mermaid
classDiagram
    class HistoricalResolutionRecord {
        +str tweet_id
        +str customer_text
        +str agent_response
        +str intent
        +float similarity_score
    }

    class HistoricalRetriever {
        -ChromaDBClient client
        -SentenceTransformer embedder
        -str collection_name
        +retrieve(query: str, intent: str, k: int) List~HistoricalResolutionRecord~
        +index_historical_pairs(data_path: str) int
    }

    class GroundedReplyGenerator {
        -HistoricalRetriever retriever
        -LLMClient llm_client
        -PromptBuilder prompt_builder
        -OutputGuardrail guardrail
        +generate_reply(tweet_text: str, intent: str) DraftedReply
    }

    class DraftedReply {
        +str text
        +List~HistoricalResolutionRecord~ grounding_sources
        +float grounding_confidence
        +bool passed_guardrails
        +List~str~ guardrail_violations
    }

    class OutputGuardrail {
        +check_length(text: str) bool
        +check_pii_solicitation(text: str) bool
        +validate_links(text: str) bool
        +evaluate(text: str) Tuple~bool, List~str~~
    }

    GroundedReplyGenerator --> HistoricalRetriever
    GroundedReplyGenerator --> OutputGuardrail
    GroundedReplyGenerator --> DraftedReply : produces
    HistoricalRetriever --> HistoricalResolutionRecord : returns
```

### 4.3 Use Case Diagram
```mermaid
graph LR
    actor Pipeline as "Support Pipeline Orchestrator"
    actor Agent as "Human Reviewer"
    
    subgraph DraftingEngine["Grounded Reply Drafting Engine"]
        UC1["Retrieve Top-k Historical Resolutions"]
        UC2["Synthesize Grounded Draft Reply"]
        UC3["Apply Anti-Hallucination Guardrails"]
        UC4["Validate URL Safety & Domain Whitelist"]
        UC5["Provide Retrieval Citations for Audit"]
    end

    Pipeline --> UC1
    Pipeline --> UC2
    Pipeline --> UC3
    Pipeline --> UC4
    Agent --> UC5
```

### 4.4 Activity Diagram
```mermaid
stateDiagram-v2
    [*] --> IngestInquiry
    IngestInquiry --> EmbedInquiry : MiniLM encode
    EmbedInquiry --> QueryChromaDB : Top-3 cosine similarity (intent-filtered)
    
    state SimilarityCheck <<choice>>
    QueryChromaDB --> SimilarityCheck
    SimilarityCheck --> FlagLowGrounding : Max Cosine < 0.65
    SimilarityCheck --> AssemblePrompt : Max Cosine >= 0.65
    
    AssemblePrompt --> CallLLM : Send grounded prompt to Gemini API
    CallLLM --> ValidateOutput : Inspect generated response
    
    state GuardrailCheck <<choice>>
    ValidateOutput --> GuardrailCheck
    GuardrailCheck --> RejectDraft : URL violation / PII solicitation / >280 chars
    GuardrailCheck --> ApproveDraft : Passes all guardrails
    
    FlagLowGrounding --> EscalateToTriage : Insufficient grounding context
    RejectDraft --> EscalateToTriage : Guardrail safety violation
    ApproveDraft --> ReturnDraftedReply
    
    EscalateToTriage --> [*]
    ReturnDraftedReply --> [*]
```

### 4.5 Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    participant Pipeline as SupportPipeline
    participant Gen as GroundedReplyGenerator
    participant Ret as HistoricalRetriever
    participant DB as ChromaDB
    participant LLM as Gemini API
    participant Guard as OutputGuardrail

    Pipeline->>Gen: generate_reply(tweet="iPhone screen won't turn on", intent="HARDWARE_AND_BATTERY")
    Gen->>Ret: retrieve("iPhone screen won't turn on", intent="HARDWARE_AND_BATTERY", k=3)
    Ret->>DB: query(embedding, n_results=3, where={"intent": "HARDWARE_AND_BATTERY"})
    DB-->>Ret: 3 historical resolution records (scores: 0.87, 0.82, 0.79)
    Ret-->>Gen: List[HistoricalResolutionRecord]
    
    Gen->>Gen: build_grounded_prompt(tweet, history)
    Gen->>LLM: generate_text(prompt, max_tokens=100, temp=0.2)
    LLM-->>Gen: "We'd like to help. Have you tried a force restart? Follow these steps: apple.co/forcerestart"
    
    Gen->>Guard: evaluate(draft_text)
    Guard-->>Gen: (passed=True, violations=[])
    Gen-->>Pipeline: DraftedReply(text="...", grounding_confidence=0.87, passed=True)
```

---

## 5. Data Schemas & Contracts

### 5.1 Drafted Reply Model
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class GroundingCitation(BaseModel):
    tweet_id: str
    historical_customer_text: str
    historical_agent_reply: str
    similarity_score: float

class DraftedReply(BaseModel):
    text: str = Field(..., max_length=280)
    grounding_citations: List[GroundingCitation]
    grounding_confidence: float = Field(ge=0.0, le=1.0)
    passed_guardrails: bool
    violations: List[str] = []
```

---

## 6. Definition of Done & Executable Test Cases

| Test ID | Target Component | Command / Verification Action | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **TC-DRAFT-01** | Vector Retrieval Recall | `pytest tests/test_drafting.py::test_retriever_top_k` | Returns top-3 records with similarity $\ge 0.70$ for standard queries. |
| **TC-DRAFT-02** | Length Guardrail | `pytest tests/test_drafting.py::test_length_guardrail` | Strings $>280$ characters fail validation or are truncated cleanly. |
| **TC-DRAFT-03** | URL Whitelist Guardrail | `pytest tests/test_drafting.py::test_url_whitelist` | Non-Apple domains (`bit.ly`, `phishing.com`) trigger immediate rejection. |
| **TC-DRAFT-04** | Grounding Consistency | `python -m src.drafting.benchmark_grounding` | $\ge 85\%$ lexical/semantic overlap with historical knowledge base. |
