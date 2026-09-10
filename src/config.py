"""Central configuration for Hiver AI Support & Triage Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.jsonl"
HUMAN_ANNOTATIONS_PATH = DATA_DIR / "human_annotations_sample.jsonl"
KAGGLE_PAIRS_PATH = DATA_DIR / "apple_support_kaggle_pairs.jsonl"
DECISION_LOG_PATH = DATA_DIR / "decision_log.jsonl"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "REPORT.md"
# Machine-readable sibling of REPORT.md, written by the same eval run
# (src/eval/runner.py -> src/eval/report_generator.py). Exists so the live
# web dashboard (src/server.py's /api/benchmark-summary) can display real
# benchmark numbers without a human manually copying figures out of the
# markdown report into src/static/index.html every time the eval reruns --
# that manual-sync step is exactly the kind of "static system" this project
# was built to avoid.
BENCHMARK_SUMMARY_JSON_PATH = PROJECT_ROOT / "docs" / "benchmark_summary.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds & Constraints
#
# Calibrated against the real golden set with `scripts/calibrate_thresholds.py`
# (see docs/AUDIT_AND_FIX_PLAN.md Section 7.10 for the full sweep output and
# reasoning) after wiring the real Kaggle corpus into the RAG store. That
# change lowered genuine cosine-similarity scores across the board (a diverse
# real corpus just doesn't produce the artificially high similarity a tiny
# near-duplicated hand-written seed set did), so MIN_RETRIEVAL_SIMILARITY's
# old value of 0.40 was silently force-escalating ~61% of legitimate
# AUTO_HANDLE queries by the time it was actually measured against real data.
#
# MIN_INTENT_CONFIDENCE = 0.40: modest bump from 0.35 -- catches more genuine
# misclassifications (16.7% -> 26.4%) for a small false-escalation cost
# (4.3% -> 8.6%).
#
# MIN_RETRIEVAL_SIMILARITY = 0.20: lowered, not raised, from 0.40. The sweep
# showed this signal barely discriminates escalate-worthy from routine
# queries at all (Youden's J stays near zero across most of the range), so it
# should be treated as a last-resort "we found nothing even remotely
# relevant" backstop rather than a primary safety gate -- that job already
# belongs to the dedicated hazard/PII/human-request/frustration gates earlier
# in src/triage/engine.py's cascade. At 0.20 this gate's false-escalation
# rate on true-AUTO_HANDLE rows drops from 60.6% to 8.4%.
MIN_INTENT_CONFIDENCE: float = float(os.getenv("MIN_INTENT_CONFIDENCE", "0.40"))
MIN_RETRIEVAL_SIMILARITY: float = float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.20"))
FRUSTRATION_THRESHOLD: float = float(os.getenv("FRUSTRATION_THRESHOLD", "0.60"))
MAX_TWEET_CHARS: int = 280

# Live link verification (src/drafting/link_checker.py): actually fetches a
# drafted reply's URL(s) instead of trusting the domain-whitelist regex
# alone. Motivated by a real, manually-found case: an LLM-drafted reply cited
# "apple.co/directmessage", which matches the whitelist regex (right domain)
# but 302-redirects to Apple's generic homepage because that specific page
# never existed -- a plausible-looking, fabricated URL a regex can't catch.
# Defaults to OFF (opt in via ENABLE_LIVE_LINK_CHECK=true in .env) because
# src/eval/runner.py's bulk benchmark run reuses this same GroundedReplyGenerator
# / OutputGuardrail construction path -- if this defaulted on, cloning this repo
# fresh and running `python -m src.eval.runner` over 150-250 golden-set rows
# would fire that many live HTTP requests per run, breaking the "runs offline
# in well under 15 minutes" guarantee (see docs/DECISION_LOG.md #2) and making
# the harness flaky in any network-restricted environment (this project has
# already hit a hard outbound network block once; see
# docs/AUDIT_AND_FIX_PLAN.md's huggingface.co note). Turn it on in your own
# .env for live single-query use (dashboard/CLI/API) -- checking the one or
# two links in a single real customer-facing reply is cheap and directly
# prevents shipping a dead or fabricated URL; just don't leave it on for a
# full eval run unless you're prepared for it to take much longer and depend
# on network access.
ENABLE_LIVE_LINK_CHECK: bool = os.getenv("ENABLE_LIVE_LINK_CHECK", "false").lower() == "true"

# Embedding & LLM Configuration
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "mock"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# Real regression (Sept 2026): a freshly-created API key hit
# "404 ... models/gemini-2.5-flash is no longer available to new users" --
# Google is rolling out new-user restrictions on 2.x-generation models ahead
# of their official deprecation dates (confirmed as a live, actively-discussed
# issue on Google's own AI developer forum, not specific to this project).
# The 404 itself named the replacement: models/gemini-3.6-flash, which
# real-time lookup against ai.google.dev/gemini-api/docs/models confirms is a
# current, stable, generally-available model ID. Defaulting to it here so a
# fresh checkout of this project (or a fresh API key on an existing checkout)
# doesn't immediately hit the same wall. If your key predates this rollout
# and still has 2.5-flash access, you can override via the GEMINI_MODEL_NAME
# env var / .env file -- nothing else in the code assumes a specific model.
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

# Target Brand
TARGET_BRAND: str = "@AppleSupport"
