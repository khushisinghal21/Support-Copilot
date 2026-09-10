"""FastAPI Web Server & Interactive Dashboard for Hiver AI Customer Support Agent."""

import json
import time
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models import TweetInput, SupportResponse, TriageAction
from src.pipeline import SupportPipeline
from src.config import TARGET_BRAND, BENCHMARK_SUMMARY_JSON_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Support Copilot",
    description="Autonomous customer support triage and grounded reply drafting system for @AppleSupport",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
_pipeline: Optional[SupportPipeline] = None


def get_pipeline() -> SupportPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SupportPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer tweet text")
    author_id: str = Field(default="customer_web_user", description="Author handle or identifier")
    tweet_id: Optional[str] = Field(None, description="Optional custom tweet ID")


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
        return {"available": False, "reason": "No benchmark run yet. Run `python -m src.eval.runner` to generate docs/benchmark_summary.json."}
    try:
        with open(BENCHMARK_SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception as e:
        return {"available": False, "reason": f"Failed to read benchmark summary: {e}"}


@app.post("/api/process", response_model=SupportResponse)
def process_tweet(req: QueryRequest):
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
