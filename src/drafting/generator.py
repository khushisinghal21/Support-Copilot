"""Grounded Reply Generator using RAG context, LLM, and safety guardrails."""

import logging
from typing import Optional, Tuple
from src.models import RetrievalResult
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LLM_PROVIDER, ENABLE_LIVE_LINK_CHECK
from src.drafting.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.drafting.guardrails import OutputGuardrail

logger = logging.getLogger(__name__)


class GroundedReplyGenerator:
    """Generates customer support replies grounded in historical brand resolutions."""

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        # verify_links=ENABLE_LIVE_LINK_CHECK (default False, see config.py):
        # live-fetches any URL a draft cites, so a plausible-but-fabricated
        # link isn't trusted just because its domain matches the whitelist.
        # Set ENABLE_LIVE_LINK_CHECK=true in .env for live single-query use
        # (dashboard/CLI/API) -- leave it off for bulk eval runs, since that
        # would mean one live HTTP call per golden-set row.
        self.guardrail = OutputGuardrail(verify_links=ENABLE_LIVE_LINK_CHECK)
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        """Initializes a Gemini API client if an API key is provided.

        Uses the current `google-genai` SDK (migrated off the fully
        deprecated `google-generativeai` package -- confirmed dead via its
        own FutureWarning at import time, "All support ... has ended").
        The migration wasn't just cosmetic: `google-generativeai`'s
        GenerationConfig has no way to control "thinking" tokens at all, and
        a real production case was observed (twice, across two different
        model generations -- gemini-2.5-flash and gemini-3.6-flash) where
        the model spent its entire max_output_tokens budget on internal
        reasoning and returned a truncated 2-3 word fragment as the visible
        answer. Raising the token cap was only ever a probabilistic
        workaround. `google-genai` exposes a `thinking_config` that can
        minimize/disable thinking so the full budget goes to the actual
        reply -- see generate() below and src/llm_utils.py's
        build_thinking_config for the model-generation-specific field it
        needs (thinking_budget vs. thinking_level)."""
        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                self._llm = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini client (google-genai SDK) for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to template mode.")
                self._llm = None

    def generate(
        self,
        tweet: str,
        intent: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> Tuple[Optional[str], bool, list]:
        """Generates a grounded reply and returns (reply_text, passed_guardrails, violations)."""
        retrieved_snippets = retrieval_result.snippets if retrieval_result else []
        context_str = "\n".join([f"- {s}" for s in retrieved_snippets]) if retrieved_snippets else "No specific history."

        generated_text = None

        # Try LLM Generation if configured
        if self._llm:
            try:
                from google.genai import types
                from src.llm_utils import build_thinking_config

                system_with_context = SYSTEM_PROMPT.format(retrieved_context=context_str)
                user_prompt = USER_PROMPT_TEMPLATE.format(customer_tweet=tweet, intent=intent)
                full_prompt = f"{system_with_context}\n\n{user_prompt}"

                # Minimizing/disabling thinking so the token budget goes to
                # the visible answer instead of being silently consumed by
                # reasoning the customer never sees -- the real fix for the
                # truncated-reply bug (see _init_llm's docstring). Which
                # field actually does that is model-generation- and
                # SDK-version-specific; build_thinking_config's docstring
                # covers both failure modes it works around, including the
                # case where it can't disable thinking at all (an older SDK
                # talking to a Gemini 3 model) and returns None. 800 tokens
                # (up from 300) is extra headroom for exactly that case --
                # the finish_reason/length checks below remain the real
                # defense-in-depth backstop either way.
                response = self._llm.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800,
                        thinking_config=build_thinking_config(self.model_name),
                    ),
                )

                candidate = response.candidates[0] if response and response.candidates else None
                finish_reason_name = getattr(getattr(candidate, "finish_reason", None), "name", None)

                if finish_reason_name == "MAX_TOKENS":
                    # With thinking disabled this should now be rare (a
                    # genuinely long answer hitting the 300-token cap rather
                    # than thinking eating the budget), but whatever partial
                    # text exists is still a truncated fragment, not a real
                    # reply -- discard it rather than shipping half a
                    # sentence to a real customer.
                    logger.warning(
                        "Gemini generation hit MAX_TOKENS before completing an "
                        "answer -- discarding truncated output and falling "
                        "back to the retrieved historical template."
                    )
                elif response is not None and (raw_text := response.text):
                    # Read the .text property exactly once (via the
                    # walrus above) and reuse it -- google-genai's
                    # response.text prints an informational "there are
                    # non-text parts in the response ['thought_signature']"
                    # notice to stdout every time it's accessed on a
                    # response that carried a thought signature (expected
                    # whenever thinking wasn't fully off), and accessing it
                    # twice just double-printed the same harmless notice.
                    candidate_text = raw_text.strip().strip('"')
                    # Defense in depth: even on a clean STOP, reject anything
                    # too short to plausibly be a real support reply rather
                    # than another flavor of truncated/degenerate output.
                    if len(candidate_text.split()) >= 4:
                        generated_text = candidate_text
                    else:
                        logger.warning(
                            f"Gemini returned a suspiciously short reply "
                            f"({candidate_text!r}) -- treating as incomplete "
                            f"and falling back to the retrieved historical template."
                        )
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Using retrieved historical template.")

        # Fallback to top retrieved historical resolution snippet if LLM not available or failed
        if not generated_text:
            if retrieved_snippets:
                generated_text = retrieved_snippets[0]
            else:
                generated_text = "We'd like to help. Have you tried restarting your device? Let us know which iOS version you have."

        # Evaluate against guardrails -- pass the source customer text (for
        # the PII-echo check) and retrieved snippets (for the grounding check).
        passed, violations = self.guardrail.evaluate(
            generated_text, source_customer_text=tweet, retrieved_snippets=retrieved_snippets
        )

        # Truncate if slight length overflow
        if not passed and any("LENGTH_EXCEEDED" in v for v in violations):
            if len(generated_text) > 280:
                generated_text = generated_text[:277] + "..."
                passed, violations = self.guardrail.evaluate(
                    generated_text, source_customer_text=tweet, retrieved_snippets=retrieved_snippets
                )

        return generated_text, passed, violations
