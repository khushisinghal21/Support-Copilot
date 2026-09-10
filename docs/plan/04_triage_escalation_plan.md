# Implementation Plan 04: Triage & Escalation Decision Engine

- **Target Spec**: [`docs/specs/04_triage_escalation_spec.md`](../specs/04_triage_escalation_spec.md)
- **Module**: `triage_escalation`
- **Output Artifact**: `docs/plan/04_triage_escalation_plan.md`
- **Status**: `Ready for Implementation`

---

## 1. Overview & Affected Files
Implements the multi-stage, cascading triage decision gate that determines whether an incoming query can be safely auto-handled or must be escalated to a human agent, supplying a defensible, standardized stated reason.

### Target Source Files
- `src/triage/rules.py`
- `src/triage/sentiment.py`
- `src/triage/reasons.py`
- `src/triage/engine.py`
- `tests/test_triage.py`

---

## 2. Layered Milestones

### Milestone 4.1: Deterministic Safety Rules (`src/triage/rules.py`)
- [x] Implement regex detectors for PII, physical hardware danger, explicit human agent requests

### Milestone 4.2: Customer Frustration & Urgency Analyzer (`src/triage/sentiment.py`)
- [x] Implement sentiment, legal action, churn threat, anger detector

### Milestone 4.3: Standardized Stated Reasons (`src/triage/reasons.py`)
- [x] Define `EscalationReasonCode` enum and detailed human-readable explanations

### Milestone 4.4: Cascading Triage Engine (`src/triage/engine.py`)
- [x] Implement cascading priority decision gate (Safety -> Human Request -> Frustration -> Confidence -> Guardrails -> Default Auto-Handle)

---

## 3. Verification Commands
1. Run triage unit tests:
   ```bash
   pytest tests/test_triage.py -v
   ```
2. Verify battery swelling escalation:
   ```bash
   python -c "from src.triage.engine import TriageEngine; print('TriageEngine loaded')"
   ```
