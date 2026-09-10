"""Data contracts and schemas for the Hiver AI Support & Triage Agent."""

from __future__ import annotations
import json
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AppleIntentEnum(str, Enum):
    """Canonical data-derived intents for @AppleSupport."""
    OS_SOFTWARE_TROUBLESHOOTING = "OS_SOFTWARE_TROUBLESHOOTING"
    HARDWARE_AND_BATTERY = "HARDWARE_AND_BATTERY"
    ACCOUNT_BILLING_ICLOUD = "ACCOUNT_BILLING_ICLOUD"
    HOW_TO_CONFIGURATION = "HOW_TO_CONFIGURATION"
    OUT_OF_SCOPE_AMBIGUOUS = "OUT_OF_SCOPE_AMBIGUOUS"


class TriageAction(str, Enum):
    """Routing actions for incoming support inquiries."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"
    CLARIFY = "CLARIFY"  # ask a follow-up question instead of forcing a binary
    # auto/escalate call. Added because a real historical @AppleSupport case
    # (an ambiguous water-damage claim, see data/README.md) showed Apple's own
    # agents doing exactly this rather than auto-handling or hard-escalating.
    # Reduces false-positive escalations on genuinely ambiguous, non-hazardous
    # low-confidence cases (e.g. device type not specified) without silently
    # auto-handling them either.


class EscalationReasonCode(str, Enum):
    """Standardized escalation/clarification codes explaining a triage decision."""
    HARDWARE_PHYSICAL_DAMAGE = "HARDWARE_PHYSICAL_DAMAGE"
    PII_SECURITY_SENSITIVE = "PII_SECURITY_SENSITIVE"
    HIGH_FRUSTRATION_CHURN_RISK = "HIGH_FRUSTRATION_CHURN_RISK"
    HUMAN_AGENT_REQUESTED = "HUMAN_AGENT_REQUESTED"
    PROMPT_INJECTION_SUSPECTED = "PROMPT_INJECTION_SUSPECTED"
    PII_ECHO_IN_DRAFT = "PII_ECHO_IN_DRAFT"
    UNSAFE_ADVICE_BLOCKED = "UNSAFE_ADVICE_BLOCKED"
    UNGROUNDED_GENERATION = "UNGROUNDED_GENERATION"
    UNVERIFIED_LINK_IN_DRAFT = "UNVERIFIED_LINK_IN_DRAFT"
    AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION = "AMBIGUOUS_DEVICE_NEEDS_CLARIFICATION"
    LOW_CONFIDENCE_AMBIGUOUS = "LOW_CONFIDENCE_AMBIGUOUS"
    GENERATION_GUARDRAIL_FAILED = "GENERATION_GUARDRAIL_FAILED"
    SYSTEM_EXCEPTION_FAIL_CLOSED = "SYSTEM_EXCEPTION_FAIL_CLOSED"


class TweetInput(BaseModel):
    """Input payload representing an incoming customer tweet."""
    tweet_id: str = Field(..., description="Unique tweet snowflake ID")
    text: str = Field(..., min_length=1, description="Raw tweet text")
    author_id: str = Field(..., description="Anonymized or raw author identifier")
    created_at: Optional[str] = Field(None, description="ISO timestamp of tweet creation")
    in_reply_to_tweet_id: Optional[str] = Field(None, description="Parent tweet ID if in thread")

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tweet text must contain at least one non-whitespace character")
        return v.strip()


class IntentResult(BaseModel):
    """Result of intent classification."""
    primary_intent: AppleIntentEnum = Field(..., description="Dominant classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score [0.0, 1.0]")
    secondary_intents: List[AppleIntentEnum] = Field(default_factory=list, description="Runner-up intents if ambiguous")
    score_distribution: Dict[str, float] = Field(default_factory=dict, description="Softmax/cosine distribution across classes")


class HistoricalCitation(BaseModel):
    """Reference to a historical resolution pair used for grounding."""
    tweet_id: str
    customer_text: str
    agent_reply: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    """Result of semantic retrieval from historical brand resolutions."""
    snippets: List[str] = Field(default_factory=list)
    citations: List[HistoricalCitation] = Field(default_factory=list)
    similarity_scores: List[float] = Field(default_factory=list)
    max_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class TriageDecision(BaseModel):
    """Decision whether to auto-handle or escalate with a stated reason."""
    action: TriageAction = Field(..., description="AUTO_HANDLE or ESCALATE")
    stated_reason: str = Field(..., min_length=3, description="Explainable reason for decision")
    reason_code: Optional[EscalationReasonCode] = Field(None, description="Structured enum reason code if escalated")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregated risk score")
    triggered_rules: List[str] = Field(default_factory=list, description="IDs or descriptions of triggered rules")


class SupportResponse(BaseModel):
    """Comprehensive end-to-end response produced by the AI Support Pipeline."""
    tweet_id: str
    intent: IntentResult
    triage: TriageDecision
    drafted_reply: Optional[str] = Field(None, description="Proposed response if auto-handled, else None")
    grounding_context: Optional[RetrievalResult] = Field(None, description="Retrieved historical resolutions")
    execution_time_ms: float = Field(default=0.0, description="Latency in milliseconds")

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
