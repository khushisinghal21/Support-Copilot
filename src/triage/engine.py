"""Cascading Triage & Escalation Engine implementing fail-closed policy gates.

Hardened version -- adds (in priority order): a prompt-injection gate on
customer input, a CLARIFY path for genuinely ambiguous device-dependent
queries (see AppleIntentEnum/TriageAction docstrings for the real historical
case that motivated it), and reason-code-specific handling of guardrail
violations (PII echoed back, unsafe advice, ungrounded generation) instead of
a single generic GENERATION_GUARDRAIL_FAILED bucket.
"""

from typing import List, Optional
from src.models import (
    TweetInput,
    IntentResult,
    RetrievalResult,
    TriageDecision,
    TriageAction,
    EscalationReasonCode,
    AppleIntentEnum,
)
from src.config import MIN_INTENT_CONFIDENCE, MIN_RETRIEVAL_SIMILARITY
from src.triage.rules import RuleMatcher
from src.triage.sentiment import SentimentAnalyzer
from src.triage.reasons import format_stated_reason

# Intents where the correct advice genuinely depends on which device is
# involved (a Bluetooth dropout means different steps on iPhone vs Apple
# Watch). Used by the CLARIFY gate.
DEVICE_DEPENDENT_INTENTS = {
    AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING,
    AppleIntentEnum.HARDWARE_AND_BATTERY,
}

# Width of the "moderate confidence" band above the hard escalation floor in
# which an unnamed device triggers a clarifying question instead of a guess.
CLARIFY_CONFIDENCE_BAND = 0.25

# Guardrail violation prefixes mapped to a specific, more informative reason
# code than the generic GENERATION_GUARDRAIL_FAILED bucket.
_GUARDRAIL_REASON_MAP = [
    ("PII_ECHO", EscalationReasonCode.PII_ECHO_IN_DRAFT),
    ("UNSAFE_ADVICE", EscalationReasonCode.UNSAFE_ADVICE_BLOCKED),
    ("UNGROUNDED", EscalationReasonCode.UNGROUNDED_GENERATION),
    ("UNVERIFIED_LINK", EscalationReasonCode.UNVERIFIED_LINK_IN_DRAFT),
]


def _map_guardrail_reason(violations: List[str]) -> EscalationReasonCode:
    for prefix, code in _GUARDRAIL_REASON_MAP:
        if any(v.startswith(prefix) for v in violations):
            return code
    return EscalationReasonCode.GENERATION_GUARDRAIL_FAILED


class TriageEngine:
    """Evaluates customer inquiries through cascading deterministic and probabilistic gates."""

    def __init__(
        self,
        min_intent_confidence: float = MIN_INTENT_CONFIDENCE,
        min_retrieval_similarity: float = MIN_RETRIEVAL_SIMILARITY,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity
        self.rule_matcher = RuleMatcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def evaluate(
        self,
        tweet: TweetInput,
        intent_res: IntentResult,
        rag_res: Optional[RetrievalResult] = None,
        drafted_reply: Optional[str] = None,
        guardrail_passed: bool = True,
        guardrail_violations: Optional[List[str]] = None,
    ) -> TriageDecision:
        """Executes cascading priority gates to determine AUTO_HANDLE / ESCALATE / CLARIFY."""
        text = tweet.text
        guardrail_violations = guardrail_violations or []

        # Gate 1: Prompt injection targeting the AI drafting step. Checked
        # first and independent of tone/sentiment -- an injection attempt can
        # read as perfectly calm text, and must never reach generation.
        is_injection, injection_rules = self.rule_matcher.detect_prompt_injection(text)
        if is_injection:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.PROMPT_INJECTION_SUSPECTED,
                    f"Triggered rules: {injection_rules}"
                ),
                reason_code=EscalationReasonCode.PROMPT_INJECTION_SUSPECTED,
                risk_score=0.90,
                triggered_rules=injection_rules,
            )

        # Gate 2: Physical hardware hazard (battery swelling, thermal hazard, liquid damage)
        is_hazard, hazard_rules = self.rule_matcher.detect_hardware_hazard(text)
        if is_hazard:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE,
                    f"Triggered safety rules: {hazard_rules}"
                ),
                reason_code=EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE,
                risk_score=1.0,
                triggered_rules=hazard_rules,
            )

        # Gate 3: Sensitive PII in public tweet
        has_pii, pii_rules = self.rule_matcher.detect_pii(text)
        if has_pii:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.PII_SECURITY_SENSITIVE,
                    f"Customer posted sensitive PII: {pii_rules}"
                ),
                reason_code=EscalationReasonCode.PII_SECURITY_SENSITIVE,
                risk_score=0.95,
                triggered_rules=pii_rules,
            )

        # Gate 4: Explicit Human Agent Request
        if self.rule_matcher.detect_human_request(text):
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HUMAN_AGENT_REQUESTED,
                    "Customer explicitly asked to speak with a human support agent"
                ),
                reason_code=EscalationReasonCode.HUMAN_AGENT_REQUESTED,
                risk_score=0.75,
                triggered_rules=["HUMAN_REQUEST_REGEX"],
            )

        # Gate 5: Corroborated Frustration, Account Compromise, or Legal/Churn Threats
        is_frustrated, frustration_score, sentiment_markers = self.sentiment_analyzer.is_severe_frustration(text)
        if is_frustrated:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK,
                    f"Frustration score {frustration_score:.2f} exceeded threshold {self.sentiment_analyzer.threshold:.2f} ({sentiment_markers})"
                ),
                reason_code=EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK,
                risk_score=frustration_score,
                triggered_rules=sentiment_markers,
            )

        # Gate 6: Intent Uncertainty / Ambiguous Topic (hard floor)
        if intent_res.primary_intent == AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS or intent_res.confidence < self.min_intent_confidence:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                    f"Intent '{intent_res.primary_intent.value}' has low confidence ({intent_res.confidence:.2f} < {self.min_intent_confidence:.2f})"
                ),
                reason_code=EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                risk_score=0.70,
                triggered_rules=["INTENT_CONFIDENCE_THRESHOLD_UNMET"],
            )

        # Gate 6b: Ambiguous device mention -- moderate confidence on a
        # device-dependent intent, but no device named. Real historical
        # precedent: Apple's own agents asked a clarifying question in an
        # analogous ambiguous case rather than guessing or hard-escalating
        # (see data/README.md). This directly targets a documented failure
        # mode ("It keeps disconnecting from Bluetooth when I go running" ->
        # generic iPhone advice given with no idea it might be a Watch).
        moderate_confidence = (
            self.min_intent_confidence <= intent_res.confidence < self.min_intent_confidence + CLARIFY_CONFIDENCE_BAND
        )
        if (
            intent_res.primary_intent in DEVICE_DEPENDENT_INTENTS
            and moderate_confidence
            and not self.rule_matcher.mentions_device(text)
        ):
            return TriageDecision(
                action=TriageAction.CLARIFY,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION,
                    f"Intent '{intent_res.primary_intent.value}' at moderate confidence "
                    f"({intent_res.confidence:.2f}) with no device named in the text"
                ),
                reason_code=EscalationReasonCode.AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION,
                risk_score=0.35,
                triggered_rules=["DEVICE_NOT_NAMED_MODERATE_CONFIDENCE"],
            )

        # Gate 7: Low Historical Grounding / Retrieval Similarity
        if rag_res and rag_res.max_similarity < self.min_retrieval_similarity:
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                    f"Historical grounding similarity ({rag_res.max_similarity:.2f} < {self.min_retrieval_similarity:.2f}) insufficient for auto-handling"
                ),
                reason_code=EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS,
                risk_score=0.65,
                triggered_rules=["GROUNDING_SIMILARITY_THRESHOLD_UNMET"],
            )

        # Gate 8: Generation Guardrail Failure -- routed to a specific reason
        # code (PII echo / unsafe advice / ungrounded / generic) rather than
        # one bucket, so the stated reason actually describes what failed.
        if not guardrail_passed:
            reason_code = _map_guardrail_reason(guardrail_violations)
            return TriageDecision(
                action=TriageAction.ESCALATE,
                stated_reason=format_stated_reason(
                    reason_code,
                    f"Safety violations: {guardrail_violations}"
                ),
                reason_code=reason_code,
                risk_score=0.85,
                triggered_rules=guardrail_violations,
            )

        # Gate 9: Safe Auto-Handle Clearance
        return TriageDecision(
            action=TriageAction.AUTO_HANDLE,
            stated_reason="High confidence standard resolution grounded in historical brand data",
            reason_code=None,
            risk_score=0.10,
            triggered_rules=[],
        )
