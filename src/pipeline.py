"""End-to-end AI Customer Support & Triage Pipeline orchestrator."""

import json
import logging
import queue
import threading
import time

from src.config import DECISION_LOG_PATH, DECISION_LOG_QUEUE_MAX
from src.drafting.generator import GroundedReplyGenerator
from src.drafting.retriever import HistoricalRetriever
from src.intent.classifier import SemanticCentroidClassifier
from src.logging_config import reset_tweet_id, set_tweet_id
from src.models import (
    AppleIntentEnum,
    EscalationReasonCode,
    IntentResult,
    RetrievalResult,
    SupportResponse,
    TriageAction,
    TriageDecision,
    TweetInput,
)
from src.triage.engine import TriageEngine
from src.triage.reasons import get_clarifying_question

logger = logging.getLogger(__name__)


# Single background writer for the decision audit log.
#
# The log itself is the seed of an active-learning / recalibration loop: with
# real traffic, a human agent's accept/override of each decision could be
# appended to the same row and used to re-tune triage thresholds against actual
# outcomes instead of a fixed one-time golden-set snapshot.
#
# It used to be written with a blocking open/append *inside* the request path.
# That kept the "never break the pipeline" guarantee (the whole thing was
# wrapped in try/except) but not "never slow the pipeline": on a slow, contended
# or full disk, that write latency is paid by the customer waiting for a reply,
# for a line nobody reads in real time. It was also unbounded -- nothing capped
# how much work could pile up.
#
# Now: a bounded queue drained by one daemon thread. Bounded rather than
# unbounded on purpose -- if the writer wedges, the cost is a fixed amount of
# memory and some counted, logged, dropped audit lines, instead of the process.
# Both guarantees hold: a logging failure cannot raise into the request path,
# and a logging slowdown cannot extend it.
_log_queue: "queue.Queue[dict | None]" = queue.Queue(maxsize=DECISION_LOG_QUEUE_MAX)
_log_writer_thread: threading.Thread | None = None
_log_writer_lock = threading.Lock()
_dropped_log_rows = 0


def _drain_decision_log() -> None:
    """Background writer loop. Opens the file per batch rather than holding a
    handle forever, so an external log-rotate still works."""
    while True:
        row = _log_queue.get()
        if row is None:  # shutdown sentinel
            _log_queue.task_done()
            return
        try:
            with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:  # pragma: no cover - logging must never break the pipeline
            logger.warning(f"Failed to write decision audit log: {e}")
        finally:
            _log_queue.task_done()


def _ensure_log_writer() -> None:
    global _log_writer_thread
    if _log_writer_thread is not None and _log_writer_thread.is_alive():
        return
    with _log_writer_lock:
        if _log_writer_thread is not None and _log_writer_thread.is_alive():
            return
        _log_writer_thread = threading.Thread(
            target=_drain_decision_log,
            name="decision-log-writer",
            daemon=True,
        )
        _log_writer_thread.start()


def flush_decision_log(timeout: float = 5.0) -> bool:
    """Blocks until queued audit rows have been written. For tests and for a
    clean shutdown -- the request path never calls this."""
    _ensure_log_writer()
    # queue.empty() goes True the instant the writer calls get() -- BEFORE the row
    # reaches disk -- so the previous implementation could return True with
    # nothing written, and a test asserting on that return value then read an
    # empty file. unfinished_tasks only drops when the writer calls task_done(),
    # which it does after the write. Found by adversarial review.
    deadline = time.perf_counter() + timeout
    while _log_queue.unfinished_tasks and time.perf_counter() < deadline:
        time.sleep(0.01)
    return not _log_queue.unfinished_tasks


def _append_decision_log(response: SupportResponse) -> None:
    """Enqueues one audit row. Non-blocking, bounded, and incapable of raising
    into the caller -- see the module-level note above for why all three
    matter."""
    global _dropped_log_rows
    try:
        row = {
            "tweet_id": response.tweet_id,
            "intent": response.intent.primary_intent.value,
            "intent_confidence": response.intent.confidence,
            "triage_action": response.triage.action.value,
            "triage_reason_code": response.triage.reason_code.value if response.triage.reason_code else None,
            "risk_score": response.triage.risk_score,
            "execution_time_ms": response.execution_time_ms,
        }
        _ensure_log_writer()
        try:
            _log_queue.put_nowait(row)
        except queue.Full:
            _dropped_log_rows += 1
            # Logged, counted, and visible -- a silently dropped audit row would
            # be worse than a slow one.
            logger.warning(
                f"Decision audit log queue full (max {DECISION_LOG_QUEUE_MAX}); "
                f"dropped {_dropped_log_rows} row(s) so far"
            )
    except Exception as e:  # pragma: no cover - logging must never break the pipeline
        logger.warning(f"Failed to enqueue decision audit log row: {e}")


class SupportPipeline:
    """Orchestrates the pipeline in safety-first order: classification, input-side
    triage gates, and only then RAG retrieval, reply generation, and the
    output-side gates. See src/triage/engine.py for why the split exists."""

    def __init__(
        self,
        intent_classifier: SemanticCentroidClassifier | None = None,
        retriever: HistoricalRetriever | None = None,
        reply_generator: GroundedReplyGenerator | None = None,
        triage_engine: TriageEngine | None = None,
    ):
        self.intent_classifier = intent_classifier or SemanticCentroidClassifier()
        self.retriever = retriever or HistoricalRetriever()
        self.reply_generator = reply_generator or GroundedReplyGenerator()
        self.triage_engine = triage_engine or TriageEngine()

    def process(self, tweet: TweetInput) -> SupportResponse:
        """Processes a single incoming customer tweet through the end-to-end pipeline."""
        start_time = time.perf_counter()

        # Bind the correlation id for every log line emitted below this point,
        # including ones from modules that know nothing about it. Reset in the
        # finally block: in a thread pool a leaked id would mislabel the next
        # request handled by the same worker, which is worse than no id at all.
        token = set_tweet_id(tweet.tweet_id)
        try:
            # 1. Intent Classification
            intent_res: IntentResult = self.intent_classifier.predict(tweet.text)

            # 2. Input-side safety gates (triage gates 1-5), BEFORE any retrieval
            #    or generation. This ordering is the point, not an optimisation:
            #    gate 1 exists to stop a prompt-injection tweet from reaching the
            #    model, and it cannot do that if the model has already been called.
            #    It also means a hazard / PII / human-request / frustration ticket
            #    no longer pays for an LLM call whose output is then discarded.
            #    This is the order README.md's flow diagram always claimed; until
            #    this change the code ran classify -> retrieve -> generate -> triage
            #    and contradicted it.
            input_decision: TriageDecision | None = self.triage_engine.evaluate_input(tweet)
            if input_decision is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                response = SupportResponse(
                    tweet_id=tweet.tweet_id,
                    intent=intent_res,
                    triage=input_decision,
                    # No draft exists and none should: nothing was generated.
                    # grounding_context is None for the same reason -- claiming a
                    # retrieval result here would be inventing one.
                    drafted_reply=None,
                    grounding_context=None,
                    execution_time_ms=round(duration_ms, 2),
                )
                _append_decision_log(response)
                return response

            # 3. Historical Retrieval (RAG)
            rag_res: RetrievalResult = self.retriever.retrieve(
                query=tweet.text,
                intent=intent_res.primary_intent.value,
                k=3,
            )

            # 4. Grounded Reply Drafting
            drafted_reply, guardrail_passed, violations = self.reply_generator.generate(
                tweet=tweet.text,
                intent=intent_res.primary_intent.value,
                retrieval_result=rag_res,
            )

            # 5. Output-side gates (triage gates 6-9) on the draft that now exists.
            triage_res: TriageDecision = self.triage_engine.evaluate_output(
                tweet=tweet,
                intent_res=intent_res,
                rag_res=rag_res,
                drafted_reply=drafted_reply,
                guardrail_passed=guardrail_passed,
                guardrail_violations=violations,
            )

            # If triage decided to escalate, withhold the auto-drafted reply
            # to prevent a risky auto-send. If it decided to clarify, send a
            # short clarifying question instead of either the raw draft or
            # silence.
            if triage_res.action == TriageAction.AUTO_HANDLE:
                final_reply = drafted_reply
            elif triage_res.action == TriageAction.CLARIFY:
                final_reply = get_clarifying_question(triage_res.reason_code)
            else:
                final_reply = None

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            response = SupportResponse(
                tweet_id=tweet.tweet_id,
                intent=intent_res,
                triage=triage_res,
                drafted_reply=final_reply,
                grounding_context=rag_res,
                execution_time_ms=round(duration_ms, 2),
            )
            _append_decision_log(response)
            return response

        except Exception as e:
            # Non-negotiable Engineering Rule: Fail-Closed on any unhandled exception
            logger.error(f"SupportPipeline encountered error processing tweet {tweet.tweet_id}: {e}", exc_info=True)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            failed_response = SupportResponse(
                tweet_id=tweet.tweet_id,
                intent=IntentResult(
                    primary_intent=AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS,
                    confidence=0.0,
                ),
                triage=TriageDecision(
                    action=TriageAction.ESCALATE,
                    stated_reason=f"Pipeline exception occurred, failing closed to protect customer: {e!s}",
                    reason_code=EscalationReasonCode.SYSTEM_EXCEPTION_FAIL_CLOSED,
                    risk_score=1.0,
                    triggered_rules=["CIRCUIT_BREAKER_FAIL_CLOSED"],
                ),
                drafted_reply=None,
                grounding_context=None,
                execution_time_ms=round(duration_ms, 2),
            )
            # Audit the fail-closed escalation too. Every other exit path wrote a
            # row and this one did not -- so the ticket an operator most needs
            # context for (the system broke, a human is now handling it) arrived
            # with no audit trail at all. Found by adversarial review.
            _append_decision_log(failed_response)
            return failed_response
        finally:
            reset_tweet_id(token)

    def batch_process(self, tweets: list[TweetInput]) -> list[SupportResponse]:
        """Processes a batch of tweets sequentially."""
        return [self.process(tweet) for tweet in tweets]
