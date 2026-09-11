"""Phase 1.4: server hygiene -- concurrency, CORS, rate limit, auth, audit log.

Each test here corresponds to a specific defect, not a hypothetical:
  * two concurrent cold requests used to build two pipelines (two embedding
    models in a 512MB process);
  * CORS was allow_origins=["*"] with allow_credentials=True, which browsers
    reject outright;
  * /api/process had no rate limit and no auth while spending money per call;
  * the audit log was written with a blocking append inside the request path.
"""

import importlib
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models import AppleIntentEnum, IntentResult, SupportResponse, TriageAction, TriageDecision


# --------------------------------------------------------------------------
# Singleton construction
# --------------------------------------------------------------------------

def test_concurrent_cold_requests_construct_exactly_one_pipeline():
    """The regression guard for the OOM class: N threads racing get_pipeline()
    on a cold process must produce one SupportPipeline, not N."""
    import src.server as server

    server._pipeline = None
    constructed = []

    def _slow_pipeline():
        # A real cold start takes seconds (model load); without that delay the
        # race this test exists for would almost never be observable.
        threading.Event().wait(0.05)
        instance = MagicMock(name=f"pipeline-{len(constructed)}")
        constructed.append(instance)
        return instance

    with patch.object(server, "SupportPipeline", side_effect=_slow_pipeline):
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(server.get_pipeline()))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(constructed) == 1, f"built {len(constructed)} pipelines concurrently"
    assert len({id(r) for r in results}) == 1, "callers received different pipeline instances"
    server._pipeline = None


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------

def test_cors_never_pairs_wildcard_origin_with_credentials(monkeypatch):
    """'*' plus credentials is rejected by browsers; it must not be reachable
    from any configuration."""
    import src.config as config
    import src.server as server

    for origins in ("*", "", "https://a.example,https://b.example"):
        monkeypatch.setattr(config, "CORS_ALLOW_ORIGINS", origins, raising=False)
        reloaded = importlib.reload(server)
        cors = [m for m in reloaded.app.user_middleware if "CORS" in str(m.cls)]
        assert cors, "CORS middleware missing"
        kwargs = cors[0].kwargs
        if kwargs.get("allow_origins") == ["*"]:
            assert kwargs.get("allow_credentials") is False, f"wildcard + credentials for {origins!r}"

    monkeypatch.setattr(config, "CORS_ALLOW_ORIGINS", "*", raising=False)
    importlib.reload(server)


# --------------------------------------------------------------------------
# Rate limiting / auth
# --------------------------------------------------------------------------

def _stub_pipeline_response():
    return SupportResponse(
        tweet_id="t1",
        intent=IntentResult(primary_intent=AppleIntentEnum.HARDWARE_AND_BATTERY, confidence=0.9),
        triage=TriageDecision(
            action=TriageAction.AUTO_HANDLE,
            stated_reason="stub",
            reason_code=None,
            risk_score=0.1,
            triggered_rules=[],
        ),
        drafted_reply="Stubbed grounded reply about battery health.",
        grounding_context=None,
        execution_time_ms=1.0,
    )


@pytest.fixture
def client_with_stub_pipeline():
    import src.server as server

    server._pipeline = MagicMock()
    server._pipeline.process.return_value = _stub_pipeline_response()
    with server._bucket_lock:
        server._buckets.clear()
    yield TestClient(server.app), server
    server._pipeline = None
    with server._bucket_lock:
        server._buckets.clear()


def test_rate_limit_returns_429_with_retry_after(client_with_stub_pipeline, monkeypatch):
    client, server = client_with_stub_pipeline
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 60, raising=False)
    monkeypatch.setattr(server, "RATE_LIMIT_BURST", 3, raising=False)

    codes = [client.post("/api/process", json={"text": "hello there friend"}).status_code for _ in range(8)]

    assert 200 in codes, "every request was rejected; the limiter is too strict"
    assert 429 in codes, "burst was never limited"
    limited = client.post("/api/process", json={"text": "hello there friend"})
    if limited.status_code == 429:
        assert "Retry-After" in limited.headers


def test_rate_limit_disabled_when_zero(client_with_stub_pipeline, monkeypatch):
    client, server = client_with_stub_pipeline
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 0, raising=False)
    codes = [client.post("/api/process", json={"text": "hello there friend"}).status_code for _ in range(12)]
    assert all(c == 200 for c in codes), f"limiter fired while disabled: {codes}"


def test_api_key_is_off_by_default(client_with_stub_pipeline, monkeypatch):
    client, server = client_with_stub_pipeline
    monkeypatch.setattr(server, "API_KEY", "", raising=False)
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 0, raising=False)
    assert client.post("/api/process", json={"text": "hello there friend"}).status_code == 200


def test_api_key_enforced_when_configured(client_with_stub_pipeline, monkeypatch):
    client, server = client_with_stub_pipeline
    monkeypatch.setattr(server, "API_KEY", "secret-value", raising=False)
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 0, raising=False)

    assert client.post("/api/process", json={"text": "hello there friend"}).status_code == 401
    ok = client.post(
        "/api/process",
        json={"text": "hello there friend"},
        headers={"X-API-Key": "secret-value"},
    )
    assert ok.status_code == 200


def test_health_and_scenarios_are_unauthenticated(client_with_stub_pipeline, monkeypatch):
    """Cheap endpoints stay open: the dashboard needs them, and they neither
    spend money nor touch the pipeline."""
    client, server = client_with_stub_pipeline
    monkeypatch.setattr(server, "API_KEY", "secret-value", raising=False)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/scenarios").status_code == 200


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

def test_decision_log_write_does_not_block_the_request_path():
    """The enqueue must return promptly even when the writer is wedged."""
    import src.pipeline as pipeline

    response = _stub_pipeline_response()
    with patch("builtins.open", side_effect=lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))):
        start = threading.Event()
        start.set()
        import time as _time

        t0 = _time.perf_counter()
        for _ in range(50):
            pipeline._append_decision_log(response)
        elapsed = _time.perf_counter() - t0

    assert elapsed < 1.0, f"enqueueing 50 audit rows took {elapsed:.2f}s -- that is in the request path"


def test_decision_log_failure_never_raises_into_the_pipeline():
    import src.pipeline as pipeline

    response = _stub_pipeline_response()
    with patch.object(pipeline, "_log_queue") as q:
        q.put_nowait.side_effect = RuntimeError("queue exploded")
        pipeline._append_decision_log(response)  # must not raise


def test_decision_log_queue_is_bounded():
    import src.pipeline as pipeline

    assert pipeline._log_queue.maxsize > 0, "an unbounded audit queue can grow without limit"
