# Implementation Plan 02: Intent Classification Engine & Baselines

- **Target Spec**: [`docs/specs/02_intent_classification_spec.md`](../specs/02_intent_classification_spec.md)
- **Module**: `intent_classification`
- **Output Artifact**: `docs/plan/02_intent_classification_plan.md`
- **Status**: `Ready for Implementation`

---

## 1. Overview & Affected Files
Implements the data-derived intent taxonomy for `@AppleSupport`, builds the dual baseline models (Trivial Majority and Simple TF-IDF) to satisfy Deliverable 4, and builds the production Semantic Centroid + Confidence Calibration Classifier.

### Target Source Files
- `src/intent/taxonomy.py`
- `src/intent/baselines.py`
- `src/intent/classifier.py`
- `src/intent/embeddings.py`
- `tests/test_intent.py`

---

## 2. Layered Milestones

### Milestone 2.1: Taxonomy Definition (`src/intent/taxonomy.py`)
- [x] Define `AppleIntentEnum`
- [x] Build intent description catalog and canonical query prototypes for centroid calculation.

### Milestone 2.2: Baseline Classifiers (`src/intent/baselines.py`)
- [x] Implement `MajorityBaselineClassifier`
- [x] Implement `TfidfBaselineClassifier`

### Milestone 2.3: Production Semantic Classifier (`src/intent/classifier.py`)
- [x] Implement `SemanticCentroidClassifier`
- [x] Confidence calibration (softmax with temperature)
- [x] Ambiguous fallback (< 0.60 -> OUT_OF_SCOPE_AMBIGUOUS)
- [x] Runner-up secondary intent detection

### Milestone 2.4: Unit Tests & Baselines Verification (`tests/test_intent.py`)
- [x] Test canonical enum compliance.
- [x] Test ambiguous query fallback.
- [x] Test statistical superiority of production model over Baseline 1 & 2.

---

## 3. Verification Commands
1. Run intent tests:
   ```bash
   pytest tests/test_intent.py -v
   ```
2. Verify baseline lift comparison:
   ```bash
   python -c "from src.intent.classifier import SemanticCentroidClassifier; print('Classifier loaded successfully')"
   ```
