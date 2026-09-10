# Implementation Plan 01: System Scaffolding & Pipeline Orchestration

- **Target Spec**: [`docs/specs/01_system_architecture_spec.md`](../specs/01_system_architecture_spec.md)
- **Module**: `system_architecture`
- **Output Artifact**: `docs/plan/01_system_architecture_plan.md`
- **Status**: `Ready for Implementation`

---

## 1. Overview & Affected Files
Builds the foundational runtime environment, strict Pydantic v2 data models, pipeline configuration, logging, and the core `SupportPipeline` orchestrator coordinating the sequential stages.

### Target Source Files
- `pyproject.toml` / `requirements.txt`
- `src/models.py`
- `src/config.py`
- `src/pipeline.py`
- `src/cli.py`
- `tests/test_models.py`
- `tests/test_pipeline.py`

---

## 2. Layered Milestones

### Milestone 1.1: Environment & Dependencies
- [x] Create `requirements.txt` with locked versions
- [x] Create `src/__init__.py` and configure module paths.

### Milestone 1.2: Core Data Contracts & Schemas (`src/models.py`)
- [x] Implement `TweetInput` model
- [x] Implement `IntentResult` model
- [x] Implement `RetrievalResult` model
- [x] Implement `TriageAction` enum
- [x] Implement `TriageDecision` model
- [x] Implement `SupportResponse` model

### Milestone 1.3: Central Configuration (`src/config.py`)
- [x] Define global thresholds
- [x] Configure ChromaDB persistence path
- [x] Configure embedding model name

### Milestone 1.4: Pipeline Orchestration (`src/pipeline.py`)
- [x] Implement `SupportPipeline` class with `process(tweet: TweetInput) -> SupportResponse`
- [x] Integrate fail-closed circuit breaker
- [x] Add batch processing helper `batch_process(tweets: List[TweetInput]) -> List[SupportResponse]`

### Milestone 1.5: Interactive CLI (`src/cli.py`)
- [x] Build Typer CLI command: `python -m src.cli process --text "..."`

---

## 3. Verification Commands
1. Model schema validation:
   ```bash
   pytest tests/test_models.py -v
   ```
2. Pipeline integration test:
   ```bash
   pytest tests/test_pipeline.py -v
   ```
3. CLI smoke test:
   ```bash
   python -m src.cli --help
   ```
