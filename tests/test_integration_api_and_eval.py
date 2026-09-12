"""Phase 3.2: the HTTP surface, the vector store's leakage guard, and the eval
harness end to end on a fixture golden set.

All offline. The encoder and generator are stubbed; nothing here reaches a network.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models import AppleIntentEnum, IntentResult, SupportResponse, TriageAction, TriageDecision

# ===========================================================================
# HTTP endpoints
# ===========================================================================


def _ok_response():
    return SupportResponse(
        tweet_id="api-1",
        intent=IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.9),
        triage=TriageDecision(
            action=TriageAction.AUTO_HANDLE,
            stated_reason="auto-handled in test fixture",
            reason_code=None,
            risk_score=0.1,
            triggered_rules=[],
        ),
        drafted_reply="A grounded reply about battery health.",
        grounding_context=None,
        execution_time_ms=2.0,
    )


@pytest.fixture
def client():
    import src.server as server

    server._pipeline = MagicMock()
    server._pipeline.process.return_value = _ok_response()
    with server._bucket_lock:
        server._buckets.clear()
    yield TestClient(server.app), server
    server._pipeline = None


def test_health_reports_readiness_and_the_security_posture(client):
    c, _ = client
    body = c.get("/api/health").json()
    assert body["status"] == "healthy"
    assert "pipeline_ready" in body
    # Added during hardening: an operator should be able to see whether the
    # deployment they are looking at has auth and rate limiting on.
    assert "rate_limit_per_minute" in body
    assert "auth_required" in body


def test_scenarios_endpoint_returns_usable_demo_rows(client):
    c, _ = client
    rows = c.get("/api/scenarios").json()
    assert rows and isinstance(rows, list)
    for row in rows:
        assert {"category", "title", "text", "expected"} <= set(row)


def test_benchmark_summary_reports_unavailable_rather_than_crashing(client, monkeypatch, tmp_path):
    """A fresh clone has not run the eval. The honest answer is available:false, not
    a 500 and not a fabricated number."""
    import src.server as server

    monkeypatch.setattr(server, "BENCHMARK_SUMMARY_JSON_PATH", tmp_path / "nope.json")
    c, _ = client
    body = c.get("/api/benchmark-summary").json()
    assert body["available"] is False
    assert "reason" in body


def test_benchmark_summary_serves_a_real_run_and_marks_it_available(client, monkeypatch, tmp_path):
    import src.server as server

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"triage": {"production": {"accuracy": 0.629}}}))
    monkeypatch.setattr(server, "BENCHMARK_SUMMARY_JSON_PATH", path)
    c, _ = client
    body = c.get("/api/benchmark-summary").json()
    assert body["available"] is True
    assert body["triage"]["production"]["accuracy"] == 0.629


def test_process_rejects_an_empty_body_with_422_not_500(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 0, raising=False)
    assert c.post("/api/process", json={}).status_code == 422
    assert c.post("/api/process", json={"text": ""}).status_code == 422


def test_process_happy_path_returns_the_full_response_shape(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 0, raising=False)
    body = c.post("/api/process", json={"text": "How do I check battery health?"}).json()
    assert body["triage"]["action"] == "AUTO_HANDLE"
    assert body["drafted_reply"]
    assert body["tweet_id"]


# ===========================================================================
# Vector store: the leakage guard
# ===========================================================================


def test_rag_corpus_excludes_every_golden_set_source_row():
    """Without this exclusion an eval row can retrieve -- and be scored against --
    the exact historical reply it is supposed to be predicting.

    THIS TEST USED TO PASS WHILE THE GUARD EXCLUDED NOTHING. The old version
    asserted only that the exclusion set and the corpus were both non-empty, then
    that no corpus row's id was in the exclusion set -- which is trivially true
    when the two id namespaces cannot intersect at all. Golden rows carried
    doubled ids ("kaggle_kaggle_187962_187961") against corpus ids
    ("kaggle_187962_187961"), so the guard removed 0 rows (of the 997 it scanned to fill an 800-record index) for the life of
    the project while README, REPORT and CLAUDE.md all claimed it worked.

    The assertions below are written so that cannot happen again: the guard must
    demonstrably REMOVE rows, and no golden answer may survive in the corpus by
    any key.
    """
    import json

    from src.config import GOLDEN_SET_PATH, KAGGLE_PAIRS_PATH
    from src.drafting.vector_store import load_real_corpus

    raw_corpus = [json.loads(line) for line in open(KAGGLE_PAIRS_PATH, encoding="utf-8") if line.strip()]
    golden = [json.loads(line) for line in open(GOLDEN_SET_PATH, encoding="utf-8") if line.strip()]
    corpus = load_real_corpus(max_records=10**6)
    assert corpus, "corpus is empty; this test would pass vacuously"

    # 1. The guard must actually remove something. A guard that removes nothing
    #    is indistinguishable from a working one unless you check this.
    assert len(corpus) < len(raw_corpus), f"leakage guard removed 0 of {len(raw_corpus)} rows -- it is a no-op again"

    # 2. No golden reference answer may be reachable, by any key.
    golden_replies = {(g.get("reference_resolution") or "").strip() for g in golden if g.get("reference_resolution")}
    golden_texts = {(g.get("text") or "").strip() for g in golden if g.get("text")}
    leaked_replies = [r for r in corpus if r["agent_reply"].strip() in golden_replies]
    leaked_texts = [r for r in corpus if r["customer_text"].strip() in golden_texts]
    assert not leaked_replies, f"{len(leaked_replies)} golden reference answers are retrievable"
    assert not leaked_texts, f"{len(leaked_texts)} golden customer tweets are in the corpus"


def test_leakage_guard_survives_an_id_format_change():
    """The specific failure mode: ids drift and the guard silently stops matching.
    _id_variants must collapse a repeated prefix."""
    from src.drafting.vector_store import _id_variants

    assert "kaggle_187962_187961" in _id_variants("kaggle_kaggle_187962_187961")
    assert "kaggle_1_2" in _id_variants("kaggle_kaggle_kaggle_1_2")
    # An ordinary id is returned unchanged rather than mangled.
    assert _id_variants("kaggle_1_2") == {"kaggle_1_2"}
    assert _id_variants("adv_001") == {"adv_001"}


def test_corpus_respects_the_configured_record_cap():
    from src.drafting.vector_store import load_real_corpus

    assert len(load_real_corpus(max_records=7)) <= 7


def test_corpus_retags_intent_with_the_independent_heuristic():
    """Trusting the ingestion-time tag would mean the retriever is filtered by the
    same classifier being evaluated."""
    from src.data.heuristic_intent import heuristic_intent
    from src.drafting.vector_store import load_real_corpus

    for record in load_real_corpus(max_records=25):
        assert record["intent"] == heuristic_intent(record["customer_text"])


# ===========================================================================
# Eval harness, end to end on a fixture golden set
# ===========================================================================


def test_eval_harness_writes_both_artifacts_on_a_fixture_set(tmp_path, monkeypatch):
    """Proves the harness still loads a golden set, honours the split, and writes
    the report plus the JSON the dashboard reads -- the class of breakage no unit
    test sees."""
    from src.eval import report_generator, splits

    golden = tmp_path / "golden.jsonl"
    rows = []
    for i in range(12):
        escalate = i % 4 == 0
        rows.append(
            {
                "tweet_id": f"fx_{i:03d}",
                "source_tweet_id": f"src_{i}",
                "text": "My battery is swollen and smoking" if escalate else "How do I transfer photos?",
                "author_id": "u",
                "true_intent": "HARDWARE_AND_BATTERY" if escalate else "HOW_TO_CONFIGURATION",
                "true_triage_action": "ESCALATE" if escalate else "AUTO_HANDLE",
                "reference_resolution": "We can help with that, please DM us.",
                "is_edge_case": escalate,
                "edge_case_type": None,
                "source": "fixture",
            }
        )
    golden.write_text("\n".join(json.dumps(r) for r in rows))

    report_path = tmp_path / "REPORT.md"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(splits, "GOLDEN_SET_PATH", golden)
    monkeypatch.setattr(report_generator, "GOLDEN_SET_PATH", golden)
    monkeypatch.setattr(report_generator, "REPORT_OUTPUT_PATH", report_path)
    monkeypatch.setattr(report_generator, "BENCHMARK_SUMMARY_JSON_PATH", summary_path)

    loaded = splits.load_golden_rows()
    assert len(loaded) == 12
    assert all(r["split"] in splits.VALID_SPLITS for r in loaded)

    metrics = {
        "intent": {"accuracy": 0.6, "macro_f1": 0.55, "per_class": {}, "confusion_matrix": [[1]], "labels": ["X"]},
        "triage": {
            "accuracy": 0.62,
            "escalation_recall": 0.9,
            "escalation_precision": 0.5,
            "missed_escalation_count": 1,
            "false_escalation_count": 3,
            "total_escalations_true": 3,
        },
        "rouge": {"mean_rougeL": 0.4},
        "judge": {"overall_score": 4.0},
    }
    agreement = {
        "mean_cohen_kappa": 0.0716,
        "cohen_kappa_groundedness": 0.0716,
        "cohen_kappa_safety": -0.04,
        "agreement_interpretation": "Slight agreement",
        "exact_agreement_safety_pct": 42.0,
        "exact_agreement_groundedness_pct": 48.0,
        "n_samples": 50,
        "num_samples": 50,
    }
    split_info = {
        "heldout_n": 8,
        "calibration_n": 4,
        "seed": splits.SPLIT_SEED,
        "calibration_intent_accuracy": 0.7,
        "calibration_triage_accuracy": 0.68,
        "calibration_escalation_recall": 1.0,
        "heldout_label_counts": {"AUTO_HANDLE": 6, "ESCALATE": 2, "CLARIFY": 0},
    }

    content = report_generator.generate_markdown_report(
        trivial_metrics=metrics,
        simple_metrics=metrics,
        prod_metrics=metrics,
        judge_metrics=metrics["judge"],
        agreement_metrics=agreement,
        top_failures=[],
        latency_p95_ms=61.2,
        split_info=split_info,
    )
    summary = report_generator.write_benchmark_summary_json(
        trivial_metrics=metrics,
        simple_metrics=metrics,
        prod_metrics=metrics,
        agreement_metrics=agreement,
        latency_p95_ms=61.2,
        split_info=split_info,
    )

    assert report_path.exists() and summary_path.exists()
    assert "held-out" in content.lower()
    assert "0.0716" in content or "0.07" in content
    assert summary["split"]["heldout_n"] == 8
    # The dashboard reads these exact keys.
    assert {"triage", "golden_set", "generated_at", "split"} <= set(summary)


def test_report_degrades_gracefully_when_split_info_is_incomplete(tmp_path, monkeypatch):
    """The latent crash mypy caught: a truthy but partial split_info used to raise
    TypeError after a full eval run had already been paid for."""
    from src.eval import report_generator

    monkeypatch.setattr(report_generator, "REPORT_OUTPUT_PATH", tmp_path / "R.md")
    metrics = {
        "intent": {"accuracy": 0.6, "macro_f1": 0.55, "per_class": {}, "confusion_matrix": [[1]], "labels": ["X"]},
        "triage": {
            "accuracy": 0.62,
            "escalation_recall": 0.9,
            "escalation_precision": 0.5,
            "missed_escalation_count": 1,
            "false_escalation_count": 3,
            "total_escalations_true": 3,
        },
        "rouge": {"mean_rougeL": 0.4},
        "judge": {"overall_score": 4.0},
    }
    content = report_generator.generate_markdown_report(
        trivial_metrics=metrics,
        simple_metrics=metrics,
        prod_metrics=metrics,
        judge_metrics=metrics["judge"],
        agreement_metrics={
            "mean_cohen_kappa": 0.0716,
            "cohen_kappa_groundedness": 0.0716,
            "cohen_kappa_safety": -0.04,
            "agreement_interpretation": "Slight agreement",
            "exact_agreement_safety_pct": 42.0,
            "exact_agreement_groundedness_pct": 48.0,
            "n_samples": 50,
            "num_samples": 50,
        },
        top_failures=[],
        latency_p95_ms=None,
        split_info={"heldout_n": 8},  # deliberately missing the accuracy keys
    )
    # The property under test is that it produced a report at all rather than
    # raising TypeError mid-generation; the wording is the generator's fallback text.
    assert "without split information" in content.lower()
    assert "calibration/held-out gap is not quantified" in content.lower()
