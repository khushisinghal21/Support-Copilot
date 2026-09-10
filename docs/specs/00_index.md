# Hiver SDE Intern Assignment: AI Support Agent — Specification Index

## 1. Project Overview & Scope
- **Project Name**: Hiver AI Support & Triage Agent (`hiver-ai-support-agent`)
- **Target Brand**: `@AppleSupport` (Selected from Kaggle `thoughtvector/customer-support-on-twitter` dataset for rich technical troubleshooting, high volume, structured resolution patterns, and strict privacy/hardware boundaries).
- **Core Objective**: Build a production-grade, highly reliable, data-grounded customer support pipeline that:
  1. Classifies incoming tweets into data-grounded intents.
  2. Drafts responses strictly grounded in historical brand resolutions.
  3. Decides whether to auto-handle or escalate to a human agent with an explicit, stated reason.
  4. Proves system reliability via an evaluation harness, golden evaluation dataset (150–250 hand-labelled examples), two baselines, LLM-as-a-judge rubric, human-judge calibration, and failure analysis report.

---

## 2. Specification Index & Status

| Spec ID | Module Name | Primary Responsibility | Assignment Deliverable Mapping | Status |
| :--- | :--- | :--- | :--- | :--- |
| [**01_system_architecture_spec.md**](./01_system_architecture_spec.md) | System Architecture & End-to-End Pipeline | Overall orchestrator, tech stack, data flow, API & CLI design, non-functional requirements | Deliverable 1: Runnable Pipeline (< 15 min reproduction) | **Approved** |
| [**02_intent_classification_spec.md**](./02_intent_classification_spec.md) | Intent Classification Engine | Data-derived intent taxonomy, zero-shot/few-shot embedding classifier, confidence calibration | Requirement 1: Intent Classification | **Approved** |
| [**03_grounded_reply_drafting_spec.md**](./03_grounded_reply_drafting_spec.md) | Grounded Reply Drafting Engine | Historical resolution vector index (RAG), brand voice grounding, hallucination guardrails | Requirement 2: Grounded Reply Drafting | **Approved** |
| [**04_triage_escalation_spec.md**](./04_triage_escalation_spec.md) | Triage & Escalation Decision Engine | Rule-based safety gate, sentiment/urgency detector, auto-handle vs. escalate decision with stated reason | Requirement 3: Triage & Escalation Decision | **Approved** |
| [**05_evaluation_harness_and_baselines_spec.md**](./05_evaluation_harness_and_baselines_spec.md) | Evaluation Harness, Baselines & Reporting | Golden dataset (150–250 hand-labelled), 2 baselines, LLM-as-a-judge, human agreement, report generator | Deliverables 2, 3, 4, 5: Golden Set, Harness, Report, Decision Log | **Approved** |

---

## 3. Assignment Requirements Traceability Matrix

```mermaid
graph TD
    subgraph Hiver_Take_Home_Requirements["Hiver Assignment Core Requirements"]
        R1["1. Intent Classification"]
        R2["2. Grounded Reply Drafting"]
        R3["3. Triage & Escalation with Stated Reason"]
        D1["Deliverable 1: Runnable Pipeline < 15 min"]
        D2["Deliverable 2: Golden Set (150-250 hand-labelled)"]
        D3["Deliverable 3: Eval Harness + LLM Judge + Human Agreement"]
        D4["Deliverable 4: Report (Baselines, Top 5 Failures, Headline Limits)"]
        D5["Deliverable 5: Decision Log (10-15 Decisions)"]
    end

    subgraph Technical_Specifications["Modular Technical Specifications"]
        S01["01_system_architecture_spec.md"]
        S02["02_intent_classification_spec.md"]
        S03["03_grounded_reply_drafting_spec.md"]
        S04["04_triage_escalation_spec.md"]
        S05["05_evaluation_harness_and_baselines_spec.md"]
    end

    R1 --> S02
    R2 --> S03
    R3 --> S04
    D1 --> S01
    D2 --> S05
    D3 --> S05
    D4 --> S05
    D5 --> S01
    D5 --> S05
```

---

## 4. Next Phase: Implementation Planning
Following the completion of these specifications, the **SDLC Spec Kit** (`sdlc-plan`) will be utilized to generate `docs/plan/` artifacts detailing step-by-step tasks, file impacts, and atomic verification commands before code execution.
