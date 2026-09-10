# Implementation Plan 03: Grounded Reply Drafting Engine (RAG)

- **Target Spec**: [`docs/specs/03_grounded_reply_drafting_spec.md`](../specs/03_grounded_reply_drafting_spec.md)
- **Module**: `grounded_reply_drafting`
- **Output Artifact**: `docs/plan/03_grounded_reply_drafting_plan.md`
- **Status**: `Ready for Implementation`

---

## 1. Overview & Affected Files
Implements the vector database indexing historical `@AppleSupport` resolutions, intent-filtered semantic retrieval, prompt assembly with brand voice guidelines, LLM generation, and anti-hallucination guardrails.

### Target Source Files
- `src/drafting/vector_store.py`
- `src/drafting/retriever.py`
- `src/drafting/prompts.py`
- `src/drafting/generator.py`
- `src/drafting/guardrails.py`
- `tests/test_drafting.py`

---

## 2. Layered Milestones

### Milestone 3.1: Historical Resolution Vector Store (`src/drafting/vector_store.py`)
- [x] Initialize embedded ChromaDB client under `data/chroma_db`
- [x] Build indexer method `index_pairs(records: List[dict])` mapping historical customer inquiries to official `@AppleSupport` responses
- [x] Store metadata: `intent`, `author_id`, `tweet_id`

### Milestone 3.2: Contextual Semantic Retriever (`src/drafting/retriever.py`)
- [x] Implement `HistoricalRetriever.retrieve(query: str, intent: str, k: int = 3)`
- [x] Filter retrieval by classified intent to maximize relevance
- [x] Extract `similarity_scores` and return structured `RetrievalResult`

### Milestone 3.3: Apple Brand Prompt Engine (`src/drafting/prompts.py`)
- [x] Craft system prompt encoding brand persona, grounding, 280-char limit, privacy bans

### Milestone 3.4: LLM Drafter & Fallback (`src/drafting/generator.py`)
- [x] Implement `GroundedReplyGenerator.generate(tweet: str, intent: str, retrieval: RetrievalResult)`
- [x] Support Gemini API with offline cached template fallback mode

### Milestone 3.5: Output Guardrail (`src/drafting/guardrails.py`)
- [x] Implement length validation (<= 280 chars)
- [x] Implement URL domain whitelist check (`apple.co/*` and `support.apple.com/*`)
- [x] Implement PII solicitation detection

---

## 3. Verification Commands
1. Run drafting tests:
   ```bash
   pytest tests/test_drafting.py -v
   ```
2. Test retrieval on sample query:
   ```bash
   python -c "from src.drafting.retriever import HistoricalRetriever; print('Retriever initialized')"
   ```
