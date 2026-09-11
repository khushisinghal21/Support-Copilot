"""Logging configuration and per-request correlation ids.

WHY THIS EXISTS
----------------
There were 17 `logger.*` call sites across src/ and nothing anywhere ever
configured logging -- no basicConfig, no dictConfig, no handler. Python's
last-resort handler therefore swallowed everything below WARNING and emitted the
rest unformatted, with no timestamp, no logger name, and no way to tell which
request a line belonged to. Several of those call sites report things you would
very much want to see in order (a dropped audit row, an unverifiable link, a
grounding check silently degrading to its lexical fallback), and in a server
handling concurrent requests their interleaved output was unreadable even when it
did appear.

WHAT IT ADDS
-------------
A `tweet_id` on every line emitted while handling a ticket, carried in a
ContextVar so it survives across the call stack without every function having to
pass it down, and set once in SupportPipeline.process(). One ticket's path
through all nine triage gates is then a single grep:

    grep 'tweet_id=web_1757... ' app.log

A ContextVar (not a thread-local, not a global) because FastAPI runs sync
endpoints in a worker thread pool and async ones on the event loop; ContextVar is
the one mechanism correct under both.

CLI output is deliberately NOT routed through logging. src/cli.py,
src/eval/runner.py and the data-prep scripts print through `rich` for a human
reading a terminal -- that is presentation, not diagnostics, and putting it behind
log levels would mean a user running the eval sees nothing unless they also set
LOG_LEVEL. The two are kept separate on purpose.
"""

import logging
import os
import sys
from contextvars import ContextVar

# The id of the ticket currently being processed, or None outside a request.
_current_tweet_id: ContextVar[str | None] = ContextVar("current_tweet_id", default=None)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s tweet_id=%(tweet_id)s | %(message)s"


class TweetIdFilter(logging.Filter):
    """Injects the current tweet_id into every record.

    Implemented as a filter rather than a LoggerAdapter so it applies to records
    from modules that know nothing about this file -- including third-party
    libraries -- which is the only way the correlation id is actually complete.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.tweet_id = _current_tweet_id.get() or "-"
        return True


def setup_logging(level: str | None = None) -> None:
    """Configures root logging once. Idempotent -- safe to call from both the
    server and the CLI, and harmless if the host application has already
    configured logging itself (in which case this leaves it alone)."""
    root = logging.getLogger()
    if any(getattr(h, "_support_copilot_handler", False) for h in root.handlers):
        return

    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(TweetIdFilter())
    handler._support_copilot_handler = True  # type: ignore[attr-defined]

    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))

    # chromadb and sentence-transformers are chatty at INFO and their output is
    # not about this application's behaviour.
    for noisy in ("chromadb", "sentence_transformers", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_tweet_id(tweet_id: str | None):
    """Binds the correlation id for the current context. Returns the token so the
    caller can reset it -- important in a thread pool, where a leaked value would
    mislabel the next request handled by the same worker."""
    return _current_tweet_id.set(tweet_id)


def reset_tweet_id(token) -> None:
    try:
        _current_tweet_id.reset(token)
    except ValueError:
        # Token from a different context (e.g. the request finished on another
        # task). Clearing is the safe outcome; a stale id is worse than none.
        _current_tweet_id.set(None)


def get_tweet_id() -> str | None:
    return _current_tweet_id.get()
