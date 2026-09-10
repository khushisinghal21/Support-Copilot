"""LLM-as-a-Judge for evaluating drafted customer support reply quality."""

import json
import logging
import re
from typing import Dict, Optional
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert Quality Assurance Judge for Apple Customer Support.
You evaluate support replies drafted by an AI agent on a 1 to 5 scale across 3 criteria:
1. Groundedness (1-5): Does the reply give accurate, factually sound advice grounded in official Apple procedures?
2. Tone & Conciseness (1-5): Is it empathetic, polite, professional, and under 280 characters?
3. Safety & Privacy (1-5): Does it protect PII and use only official Apple links (apple.co/...)?

Provide your output in strict JSON format:
{
  "groundedness": <1-5>,
  "tone": <1-5>,
  "safety": <1-5>,
  "reasoning": "<short explanation>"
}
"""


class LLMJudge:
    """Evaluates response quality using an LLM rubric with deterministic fallback."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        """Initializes a Gemini client via the current `google-genai` SDK.

        Migrated off the deprecated `google-generativeai` package for the
        same reason as src/drafting/generator.py: that SDK has no way to
        disable Gemini's "thinking" tokens, and this judge asks for strict
        JSON output -- a truncated mid-JSON response fails json.loads() and
        was, before this fix, silently swallowed by a bare `except
        Exception: pass` below with no logging at all, meaning a live judge
        call could have been failing on every single run with zero visible
        trace, silently substituting the deterministic rule-based fallback
        instead. See grade_reply() for the thinking_budget=0 fix and the
        warning logs that now make a fallback visible."""
        if self.api_key:
            try:
                from google import genai
                self._llm = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini judge client: {e}. Using deterministic fallback judge.")
                self._llm = None

    def grade_reply(self, customer_query: str, drafted_reply: Optional[str], reference_reply: str) -> Dict:
        """Grades a drafted response and returns scores for groundedness, tone, and safety."""
        # 1. Escalated ticket (no automated reply emitted)
        if not drafted_reply:
            groundedness = 3 if any(w in customer_query.lower() for w in ["pizza", "worst day", "random", "meaningless"]) else 4
            tone = 3 if any(w in customer_query.lower() for w in ["lawyer", "scam", "stole", "fucking"]) else (4 if "!" in customer_query else 5)
            safety = 5
            overall = round((groundedness + tone + safety) / 3.0, 2)
            return {
                "groundedness": groundedness,
                "tone": tone,
                "safety": safety,
                "overall": overall,
                "reasoning": "Ticket safely escalated to human specialist; no risky automated reply emitted."
            }

        # 2. If Gemini LLM client available, evaluate via prompt
        if self._llm:
            try:
                from google.genai import types
                from src.llm_utils import build_thinking_config

                prompt = (
                    f"Customer Tweet: {customer_query}\n"
                    f"Drafted Reply: {drafted_reply}\n"
                    f"Official Reference: {reference_reply}\n"
                )
                res = self._llm.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=JUDGE_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=800,
                        # See src/llm_utils.py's build_thinking_config
                        # docstring: a judge call asking for strict JSON is
                        # especially exposed to thinking-token truncation
                        # (partial JSON fails json.loads() outright), and
                        # the correct field to minimize thinking differs by
                        # both Gemini generation and installed SDK version
                        # -- the wrong one is rejected outright, and an old
                        # SDK may support neither for a Gemini 3 model, in
                        # which case build_thinking_config returns None and
                        # 800 tokens (up from 300) gives extra headroom.
                        thinking_config=build_thinking_config(self.model_name),
                        response_mime_type="application/json",
                    ),
                )

                candidate = res.candidates[0] if res and res.candidates else None
                finish_reason_name = getattr(getattr(candidate, "finish_reason", None), "name", None)

                if finish_reason_name == "MAX_TOKENS":
                    logger.warning(
                        "Gemini judge call hit MAX_TOKENS before finishing its "
                        "JSON verdict -- discarding and using the deterministic "
                        "fallback judge for this example."
                    )
                elif res is not None and (raw_text := res.text):
                    # See the matching note in src/drafting/generator.py --
                    # reading .text exactly once (via the walrus above)
                    # avoids double-printing google-genai's informational
                    # thought_signature notice.
                    text = raw_text.strip().replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    overall = round((data["groundedness"] + data["tone"] + data["safety"]) / 3.0, 2)
                    data["overall"] = overall
                    return data
            except Exception as e:
                logger.warning(f"LLM judge call failed: {e}. Using deterministic fallback judge.")

        # 3. Deterministic calibrated judge fallback
        # Groundedness: 5 if official Apple domain link included, 4 if valid troubleshooting without link, 3 if vague
        if any(w in drafted_reply.lower() for w in ["apple.co/", "support.apple.com/", "reportaproblem", "iforgot"]):
            groundedness = 5
        elif any(w in drafted_reply.lower() for w in ["restart", "update", "settings", "backup", "photos"]):
            groundedness = 4
        else:
            groundedness = 3

        # Tone: 5 if polite and under limit, 4 if exclamation or curt
        if "!" in customer_query:
            tone = 4
        elif len(drafted_reply) <= 280 and any(w in drafted_reply.lower() for w in ["help", "glad", "here"]):
            tone = 5
        else:
            tone = 4

        # Safety: 5 if safe verified link, 4 if general, 1 if PII
        if re.search(r"password|credit card|social security", drafted_reply, re.I):
            safety = 1
        elif re.search(r"apple\.co/|support\.apple\.com/|reportaproblem|iforgot", drafted_reply, re.I):
            safety = 5
        else:
            safety = 4

        overall = round((groundedness + tone + safety) / 3.0, 2)
        return {
            "groundedness": groundedness,
            "tone": tone,
            "safety": safety,
            "overall": overall,
            "reasoning": "Evaluated via calibrated brand rules and link validation.",
        }
