"""Output guardrails for drafted replies: length, link safety, privacy, unsafe
advice, and grounding.

Extends the original three checks (length / PII solicitation / URL whitelist)
with three safety-motivated additions from the audit:

  - PII echo: don't repeat the customer's own PII back in the draft. Belt-
    and-suspenders on top of the input-side PII gate in the triage engine --
    useful because drafting currently runs before triage in the pipeline
    (src/pipeline.py), and because a source of PII the input regex misses
    could still leak if it ends up quoted in the reply.
  - Unsafe advice: block DIY instructions (puncturing a battery, jailbreaking,
    opening the device casing) regardless of whether they came from an LLM
    or a retrieved historical snippet.
  - Grounding: when historical snippets were actually retrieved, a generated
    reply with near-zero lexical overlap with them is more likely
    hallucinated than grounded, and should be treated as a guardrail failure
    rather than silently shipped. Skipped when nothing was retrieved (the
    retrieval-similarity gate in the triage engine already handles that case)
    and skipped when the draft IS one of the retrieved snippets verbatim (the
    no-LLM fallback path, which is grounded by construction).
  - Link reachability (opt-in, see check_link_reachability): the URL
    whitelist above only proves a link's DOMAIN is official Apple -- it
    can't tell a real page from a fabricated one on that domain. A manually
    verified live reply cited "apple.co/directmessage", which passed the
    whitelist but actually 302-redirects to Apple's generic homepage because
    that page doesn't exist. src/drafting/link_checker.py actually fetches
    the URL to catch that; see its module docstring for why this is opt-in
    and not run during the bulk eval harness.
"""

import logging
import re
from typing import List, Optional, Tuple
from src.config import MAX_TWEET_CHARS
from src.triage.rules import RuleMatcher, EMAIL_REGEX, PHONE_REGEX

logger = logging.getLogger(__name__)

# Allowed official Apple domain prefixes and Twitter official link wrapper (t.co)
WHITELISTED_URL_PATTERN = re.compile(
    r"^https?://(apple\.co|support\.apple\.com|iforgot\.apple\.com|reportaproblem\.apple\.com|t\.co)/",
    re.IGNORECASE,
)

# Regex to detect dangerous solicitation of sensitive data in public tweets
PII_SOLICITATION_PATTERN = re.compile(
    r"(send|dm|give|reply with|share)\s+(us\s+)?(your\s+)?(password|passcode|credit card|cvv|security code|social security|ssn)",
    re.IGNORECASE,
)

URL_EXTRACTOR = re.compile(r"https?://\S+|apple\.co/\S+|reportaproblem\.apple\.com\S*|iforgot\.apple\.com\S*")

_STOPWORDS = {
    "the", "a", "an", "to", "and", "or", "is", "are", "we", "you", "your",
    "us", "our", "this", "that", "it", "for", "on", "in", "of", "at",
    "please", "let", "with", "can", "will", "be", "has", "have", "help",
    "would", "like", "here", "check", "out", "we'd", "we're", "let's",
}


def _content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class OutputGuardrail:
    """Validates generated drafts against strict brand safety and formatting constraints."""

    def __init__(
        self,
        max_chars: int = MAX_TWEET_CHARS,
        min_grounding_overlap: float = 0.12,
        verify_links: bool = False,
        link_check_timeout: float = 3.0,
    ):
        self.max_chars = max_chars
        self.min_grounding_overlap = min_grounding_overlap
        self.rule_matcher = RuleMatcher()
        self.verify_links = verify_links
        self.link_check_timeout = link_check_timeout

    def check_length(self, text: str) -> bool:
        """Returns True if within Twitter length limit."""
        return len(text.strip()) <= self.max_chars

    def check_minimum_content(self, text: str) -> bool:
        """Returns True if the draft is long enough to plausibly be a real
        answer rather than a truncated/degenerate fragment.

        Real regression: a generation-layer bug (gemini-2.5-flash spending
        its output-token budget on internal "thinking" before writing the
        visible answer -- see the note in src/drafting/generator.py) produced
        a 3-word fragment ("We'd like to") that every other check here
        passed trivially: check_length allows anything under 280 chars, and
        check_grounding explicitly returns "ok" when a draft has zero
        content words to compare against (nothing to flag as ungrounded).
        Nothing required the draft to actually contain a real answer before
        it shipped to AUTO_HANDLE. This is a deliberate second, independent
        line of defense: even if a future generation-layer bug reintroduces
        truncated output, this guardrail still catches it before send."""
        return len(text.split()) >= 4

    def check_pii_solicitation(self, text: str) -> bool:
        """Returns True if no sensitive PII solicitation is detected."""
        return not bool(PII_SOLICITATION_PATTERN.search(text))

    def validate_urls(self, text: str) -> Tuple[bool, List[str]]:
        """Returns True if all included URLs belong to whitelisted Apple domains."""
        urls = URL_EXTRACTOR.findall(text)
        invalid_urls = []
        for url in urls:
            normalized = url if url.startswith("http") else f"https://{url}"
            if not WHITELISTED_URL_PATTERN.match(normalized):
                invalid_urls.append(url)
        return len(invalid_urls) == 0, invalid_urls

    def check_pii_echo(self, draft: str, source_customer_text: Optional[str]) -> Tuple[bool, List[str]]:
        """Returns (ok, echoed_values) -- False if the draft repeats an email
        or phone number that appeared in the customer's own message."""
        if not source_customer_text:
            return True, []
        echoed = []
        for match in EMAIL_REGEX.findall(source_customer_text):
            if match.lower() in draft.lower():
                echoed.append(match)
        for match in PHONE_REGEX.findall(source_customer_text):
            if match and match in draft:
                echoed.append(match)
        return len(echoed) == 0, echoed

    def check_unsafe_advice(self, text: str) -> Tuple[bool, List[str]]:
        is_unsafe, rules = self.rule_matcher.detect_unsafe_advice(text)
        return not is_unsafe, rules

    def check_grounding(self, draft: str, retrieved_snippets: Optional[List[str]]) -> Tuple[bool, float]:
        """Lexical-overlap grounding check. Returns (ok, overlap_ratio).
        Skipped (returns ok=True) when there's nothing to ground against, or
        when the draft is verbatim one of the retrieved snippets (the no-LLM
        fallback path, grounded by construction)."""
        if not retrieved_snippets:
            return True, 1.0
        if draft.strip() in [s.strip() for s in retrieved_snippets]:
            return True, 1.0
        draft_words = _content_words(draft)
        context_words = set()
        for s in retrieved_snippets:
            context_words |= _content_words(s)
        if not draft_words or not context_words:
            return True, 1.0
        overlap = len(draft_words & context_words) / len(draft_words)
        return overlap >= self.min_grounding_overlap, overlap

    def check_link_reachability(self, text: str) -> Tuple[bool, List[str]]:
        """Live-fetches any URL(s) in the draft and flags ones that are dead
        or bounce to the bare domain root instead of the specific page they
        claim to be. No-op (always passes) unless verify_links=True -- see
        this class's docstring and src/drafting/link_checker.py for why this
        is opt-in rather than always-on."""
        if not self.verify_links:
            return True, []
        from src.drafting.link_checker import verify_all_urls

        problems = []
        for result in verify_all_urls(text, timeout=self.link_check_timeout):
            if result.status in ("broken", "suspicious_redirect"):
                problems.append(f"{result.url} ({result.status}: {result.detail})")
            elif result.status == "inconclusive":
                # Network/DNS/timeout -- don't fail-closed on every reply
                # just because this environment has no route out right now.
                logger.warning(f"Could not verify link {result.url}: {result.detail}")
        return len(problems) == 0, problems

    def evaluate(
        self,
        text: str,
        source_customer_text: Optional[str] = None,
        retrieved_snippets: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Evaluates all guardrail checks."""
        violations = []

        if not self.check_length(text):
            violations.append(f"LENGTH_EXCEEDED: Draft is {len(text)} characters (max {self.max_chars})")

        if not self.check_minimum_content(text):
            violations.append(
                f"INCOMPLETE_DRAFT: Draft is only {len(text.split())} word(s) -- "
                f"too short to plausibly be a complete answer, likely truncated generation"
            )

        if not self.check_pii_solicitation(text):
            violations.append("PII_SOLICITATION: Draft requests sensitive credentials or passwords")

        url_ok, invalid_urls = self.validate_urls(text)
        if not url_ok:
            violations.append(f"UNAUTHORIZED_URL: Draft contains non-whitelisted URLs: {invalid_urls}")

        pii_echo_ok, echoed = self.check_pii_echo(text, source_customer_text)
        if not pii_echo_ok:
            violations.append(f"PII_ECHO: Draft repeats customer PII back publicly: {echoed}")

        advice_ok, advice_rules = self.check_unsafe_advice(text)
        if not advice_ok:
            violations.append(f"UNSAFE_ADVICE: Draft contains unsafe DIY instructions: {advice_rules}")

        grounding_ok, overlap = self.check_grounding(text, retrieved_snippets)
        if not grounding_ok:
            violations.append(
                f"UNGROUNDED: Draft has only {overlap:.0%} lexical overlap with retrieved "
                f"historical resolutions (min {self.min_grounding_overlap:.0%})"
            )

        link_ok, link_problems = self.check_link_reachability(text)
        if not link_ok:
            violations.append(f"UNVERIFIED_LINK: Draft cites a link that appears dead or fabricated: {link_problems}")

        return len(violations) == 0, violations
