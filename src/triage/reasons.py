"""Standardized escalation reason codes and explainable justification strings."""

from src.models import EscalationReasonCode

REASON_EXPLANATIONS: dict[EscalationReasonCode, str] = {
    EscalationReasonCode.HARDWARE_PHYSICAL_DAMAGE: (
        "Physical damage or battery safety hazard detected (e.g., swelling battery, shattered glass, "
        "liquid immersion, smoke, electric shock). Requires hands-on inspection and reservation at an "
        "Apple Store Genius Bar."
    ),
    EscalationReasonCode.PII_SECURITY_SENSITIVE: (
        "Sensitive customer PII or authentication credentials detected in public tweet (e.g. email, "
        "phone number, credit card). Escalated to secure private channel to safeguard account privacy."
    ),
    EscalationReasonCode.HIGH_FRUSTRATION_CHURN_RISK: (
        "Corroborated customer frustration, account compromise, or legal/churn threat detected. "
        "Routing directly to a Tier-2 human specialist for empathetic conflict de-escalation."
    ),
    EscalationReasonCode.HUMAN_AGENT_REQUESTED: (
        "Customer explicitly requested to interact with a human agent or representative. "
        "Honoring user preference by routing ticket to support queue."
    ),
    # NOTE: this used to hardcode the string "tau < 0.65" regardless of the
    # actual configured threshold, which meant the stated reason could
    # misdescribe the decision whenever MIN_INTENT_CONFIDENCE/
    # MIN_RETRIEVAL_SIMILARITY were changed. format_stated_reason() now
    # interpolates the real value via custom_detail instead.
    EscalationReasonCode.LOW_CONFIDENCE_AMBIGUOUS: (
        "Classification or retrieval confidence fell below the configured safety threshold. "
        "Failing closed to human support to avoid risk of generating hallucinated or inaccurate advice."
    ),
    EscalationReasonCode.GENERATION_GUARDRAIL_FAILED: (
        "Drafted response violated an output quality/policy guardrail (length constraint, unverified "
        "external link, PII solicitation, or a truncated/incomplete draft). Withheld automated reply "
        "and routed to human review -- a quality-control catch, not necessarily a hazard."
    ),
    EscalationReasonCode.SYSTEM_EXCEPTION_FAIL_CLOSED: (
        "An unexpected pipeline or upstream model exception occurred. System failed closed to human "
        "routing to maintain continuous service reliability."
    ),
    EscalationReasonCode.PROMPT_INJECTION_SUSPECTED: (
        "Customer message matched a prompt-injection pattern targeting the AI drafting step "
        "(e.g. 'ignore your instructions', 'reveal your system prompt'). Never auto-handled -- "
        "routed to a human regardless of how benign the rest of the message reads."
    ),
    EscalationReasonCode.PII_ECHO_IN_DRAFT: (
        "The drafted reply echoed PII the customer posted (e.g. repeating their email/phone back). "
        "Withheld to avoid amplifying an already-public privacy exposure."
    ),
    EscalationReasonCode.UNSAFE_ADVICE_BLOCKED: (
        "The drafted reply matched an unsafe-DIY-advice pattern (e.g. puncturing a battery, jailbreaking, "
        "opening the device casing). Withheld regardless of source (LLM or retrieved snippet)."
    ),
    EscalationReasonCode.UNGROUNDED_GENERATION: (
        "The drafted reply had insufficient lexical overlap with the retrieved historical resolutions -- "
        "likely hallucinated rather than grounded. Withheld and routed to human review."
    ),
    EscalationReasonCode.UNVERIFIED_LINK_IN_DRAFT: (
        "A link cited in the drafted reply was live-checked (src/drafting/link_checker.py) and found dead "
        "or redirecting to a generic landing page instead of the specific page it claimed -- the same "
        "pattern found in a real 'apple.co/directmessage' case that a domain-whitelist check alone missed. "
        "Withheld and routed to human review rather than shipping an unverifiable or fabricated URL."
    ),
    EscalationReasonCode.AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION: (
        "The query is plausibly routine but doesn't name a device, and multiple Apple products share this "
        "symptom (e.g. Bluetooth dropouts on both iPhone and Apple Watch). Asking a one-line clarifying "
        "question instead of guessing or forcing an escalation."
    ),
}


CLARIFYING_QUESTIONS: dict[EscalationReasonCode, str] = {
    EscalationReasonCode.AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION: (
        "Thanks for reaching out! Just to confirm which device this is on "
        "(iPhone, iPad, Mac, or Apple Watch) so we can point you to the right fix?"
    ),
}
DEFAULT_CLARIFYING_QUESTION = "Thanks for reaching out -- could you share a bit more detail so we can help?"


def get_clarifying_question(code: EscalationReasonCode | None) -> str:
    if code is None:
        return DEFAULT_CLARIFYING_QUESTION
    return CLARIFYING_QUESTIONS.get(code, DEFAULT_CLARIFYING_QUESTION)


def format_stated_reason(code: EscalationReasonCode, custom_detail: str | None = None) -> str:
    """Returns a structured, human-readable stated explanation for the escalation decision."""
    base_explanation = REASON_EXPLANATIONS.get(code, "Ticket escalated to human specialist.")
    if custom_detail:
        return f"[{code.value}] {custom_detail}. {base_explanation}"
    return f"[{code.value}] {base_explanation}"
