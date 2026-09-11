"""FastAPI Web Server & Interactive Dashboard for Hiver AI Customer Support Agent."""

import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import (
    API_KEY,
    BENCHMARK_SUMMARY_JSON_PATH,
    CORS_ALLOW_ORIGINS,
    RATE_LIMIT_BURST,
    RATE_LIMIT_PER_MINUTE,
    TARGET_BRAND,
)
from src.logging_config import setup_logging
from src.models import SupportResponse, TweetInput
from src.pipeline import SupportPipeline

# Configure logging before anything else can emit. Without this the 17 logger
# call sites across src/ fell through to Python's last-resort handler: nothing
# below WARNING appeared at all, and what did appear had no timestamp, no logger
# name, and no request correlation.
setup_logging()

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Support Copilot",
    description="Autonomous customer support triage and grounded reply drafting system for @AppleSupport",
    version="1.0.0",
)

# CORS. The previous configuration was allow_origins=["*"] WITH
# allow_credentials=True, which the CORS spec forbids: a browser rejects a
# wildcard origin on a credentialed response outright. So that pairing did not
# grant broad credentialed access, it silently broke the credentialed case while
# still exposing the API to every origin for ordinary reads. Origins are now
# configurable (CORS_ALLOW_ORIGINS, comma-separated), and credentials are
# enabled only when an explicit list is supplied -- never alongside "*".
# Derive the wildcard flag from the EFFECTIVE list, after the empty-config
# fallback, not from the parsed one. An earlier version of this fix computed
# `_wildcard = _origins == ["*"]` before falling back, so CORS_ALLOW_ORIGINS=""
# produced allow_origins=["*"] with allow_credentials=True -- the same forbidden
# pairing being fixed here, reachable through a different input. Caught by
# tests/test_server_hardening.py, which is why that test parametrises over the
# empty string rather than only the obvious "*".
_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()] or ["*"]
_wildcard = "*" in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Pipeline is genuinely lazy: constructed on the FIRST real request, not at
# import or startup. This used to be undermined by an @app.on_event("startup")
# handler below that called get_pipeline() eagerly -- which loads two full
# SentenceTransformer models (intent classifier + retrieval encoder) before
# the app finishes starting up. Locally that's a barely-noticeable ~2-5s
# delay, but on a low-CPU host (e.g. Render's free tier, 0.1 vCPU) it was
# slow enough that the ASGI server's port never opened before the platform's
# port-scan timeout, and the deploy failed with "no open ports detected" --
# a deploy failure that had nothing to do with Render config and everything
# to do with this eager call contradicting its own "lazily" comment. Removed
# the startup hook; the first real request now pays a one-time model-load
# cost instead, and the port binds immediately regardless of host CPU.
# Construction is also guarded by a lock. Without one, two concurrent cold
# requests both saw _pipeline is None and both built a SupportPipeline -- which
# means two SentenceTransformer loads in a process whose memory ceiling is 512MB
# on the deploy target. That is precisely the out-of-memory class the deploy
# notes in render.yaml and src/embeddings.py were already fighting; an unguarded
# singleton quietly reintroduced it under the one condition (a cold start taking
# real time) where concurrent first requests are most likely.
_pipeline: SupportPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> SupportPipeline:
    """Thread-safe, double-checked lazy construction. The first check avoids
    taking the lock on the overwhelmingly common warm path; the second makes the
    cold path correct."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = SupportPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Rate limiting and optional auth for /api/process
#
# That endpoint spends money per call (a Gemini request) and had neither. On a
# public URL, one loop is an API bill. This is an in-process token bucket per
# client IP: adequate and honest for a single-instance deployment, and
# deliberately not presented as more than that -- it resets on restart and does
# not coordinate across instances, so a multi-instance deployment would need a
# shared store (Redis) instead.
# ---------------------------------------------------------------------------
_buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill_ts)
_bucket_lock = threading.Lock()


def _rate_limit_check(client_ip: str) -> tuple[bool, float]:
    """Returns (allowed, retry_after_seconds). A limit of 0 disables the check."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True, 0.0
    refill_per_sec = RATE_LIMIT_PER_MINUTE / 60.0
    capacity = float(max(RATE_LIMIT_BURST, 1))
    now = time.monotonic()
    with _bucket_lock:
        tokens, last = _buckets.get(client_ip, (capacity, now))
        tokens = min(capacity, tokens + (now - last) * refill_per_sec)
        if tokens >= 1.0:
            _buckets[client_ip] = (tokens - 1.0, now)
            return True, 0.0
        _buckets[client_ip] = (tokens, now)
        return False, max(1.0, (1.0 - tokens) / refill_per_sec)


def _require_api_key(provided: str | None) -> None:
    """No-op unless API_KEY is configured, so local dev and the existing public
    demo are unaffected by this existing."""
    if not API_KEY:
        return
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer tweet text")
    author_id: str = Field(default="customer_web_user", description="Author handle or identifier")
    tweet_id: str | None = Field(None, description="Optional custom tweet ID")


SAMPLE_SCENARIOS = [
    {
        "category": "Routine Auto-Handle",
        "title": "Photo Transfer (How-To)",
        "text": "How do I transfer photos from my iPhone to my Windows PC?",
        "expected": "AUTO_HANDLE",
    },
    {
        "category": "Routine Auto-Handle",
        "title": "AirDrop Not Working",
        "text": "My AirDrop isn't showing up on my Mac from my iPhone after the iOS update.",
        "expected": "AUTO_HANDLE",
    },
    {
        "category": "Safety Hazard Escalation",
        "title": "Thermal / Smoke Hazard",
        "text": "Smoke came out of my iPad charging port when I plugged it in! Is it safe?",
        "expected": "ESCALATE (HARDWARE_PHYSICAL_DAMAGE)",
    },
    {
        "category": "Hardware Damage Escalation",
        "title": "Liquid Immersion in Pool",
        "text": "Dropped my phone in the pool and now it won't power on at all. Any advice @AppleSupport?",
        "expected": "ESCALATE (HARDWARE_PHYSICAL_DAMAGE)",
    },
    {
        "category": "Security & Fraud Escalation",
        "title": "Hacked Account / Fraud",
        "text": "Someone hacked my iCloud and bought 100 gift cards, cancel this now!",
        "expected": "ESCALATE (HIGH_FRUSTRATION_CHURN_RISK)",
    },
    {
        "category": "PII Security Escalation",
        "title": "Public PII Sharing",
        "text": "My Apple ID is locked, here is my email test.user@icloud.com and phone 415-555-0199.",
        "expected": "ESCALATE (PII_SECURITY_SENSITIVE)",
    },
    {
        "category": "Explicit Human Request",
        "title": "Human Agent Demand",
        "text": "Stop sending me automated bot replies! I want to speak to a real human person right now.",
        "expected": "ESCALATE (HUMAN_AGENT_REQUESTED)",
    },
]


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "brand": TARGET_BRAND,
        "pipeline_ready": _pipeline is not None,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "auth_required": bool(API_KEY),
    }


@app.get("/api/scenarios")
def get_scenarios():
    return SAMPLE_SCENARIOS


@app.get("/api/benchmark-summary")
def get_benchmark_summary():
    """Serves docs/benchmark_summary.json, written by `python -m src.eval.runner`
    (see src/eval/report_generator.py's write_benchmark_summary_json).

    The dashboard used to have the last benchmark's headline numbers
    hand-copied into src/static/index.html, which silently went stale every
    time the eval reran with a different golden set or recalibrated
    thresholds. This endpoint instead serves whatever the most recent real
    run actually produced, with `available: False` (not a crash, and not a
    fabricated number) when no run has happened yet in this checkout.
    """
    if not BENCHMARK_SUMMARY_JSON_PATH.exists():
        return {
            "available": False,
            "reason": "No benchmark run yet. Run `python -m src.eval.runner` to generate docs/benchmark_summary.json.",
        }
    try:
        with open(BENCHMARK_SUMMARY_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception as e:
        return {"available": False, "reason": f"Failed to read benchmark summary: {e}"}


@app.post("/api/process", response_model=SupportResponse)
def process_tweet(
    req: QueryRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _require_api_key(x_api_key)

    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = _rate_limit_check(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). Retry in {retry_after:.0f}s.",
            headers={"Retry-After": str(int(retry_after))},
        )

    pipeline = get_pipeline()
    tw_id = req.tweet_id or f"web_{int(time.time() * 1000)}"
    tweet = TweetInput(
        tweet_id=tw_id,
        text=req.text,
        author_id=req.author_id,
    )
    return pipeline.process(tweet)


@app.get("/", response_class=FileResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)
