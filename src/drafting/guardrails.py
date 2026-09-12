"""Output guardrails for drafted replies: length, link safety, privacy, unsafe
advice, and grounding.

Extends the original three checks (length / PII solicitation / URL whitelist)
with three safety-motivated additions from the audit:

  - PII echo: don't repeat the customer's own PII back in the draft. Belt-
    and-suspenders on top of the input-side PII gate in the triage engine.
    The original reason given here was that "drafting currently runs before
    triage in the pipeline" -- that is no longer true (src/pipeline.py now runs
    triage gates 1-5, PII included, before retrieval or generation; see
    src/triage/engine.py's module docstring). The check stays, for the reason
    that always mattered more: a source of PII the input regex misses can still
    end up quoted in the reply, and an input gate cannot catch what it did not
    recognise. Defence in depth, not redundancy.
  - Unsafe advice: block DIY instructions (puncturing a battery, jailbreaking,
    opening the device casing) regardless of whether they came from an LLM
    or a retrieved historical snippet.
  - Grounding: when historical snippets were actually retrieved, a generated
    reply with near-zero overlap with them is more likely hallucinated than
    grounded, and should be treated as a guardrail failure rather than silently
    shipped. Skipped when nothing was retrieved (the retrieval-similarity gate in
    the triage engine already handles that case) and skipped when the draft IS
    one of the retrieved snippets verbatim (the no-LLM fallback path, which is
    grounded by construction).

    Two implementations, and which one runs matters:

      * EMBEDDING (default when the encoder is available): cosine similarity
        between the draft and the retrieved snippets using the same cached
        all-MiniLM-L6-v2 encoder the rest of the system already holds in memory
        (src/embeddings.py's get_encoder -- no second model load).
      * LEXICAL (fallback): bag-of-content-words overlap ratio.

    The lexical check was never the intended design. docs/DECISION_LOG.md #14 is
    explicit that it was a fallback from when the embedding model could not
    reliably be loaded in the development environment. Its two failure modes are
    structural, not tunable: a fluent hallucination that recycles retrieved
    vocabulary passes, and a correct paraphrase that uses different words fails.
    Word overlap cannot distinguish meaning from vocabulary; embeddings at least
    attempt to. The lexical path is retained deliberately -- it is the only thing
    that works with no model present, which is the state CI and the offline test
    suite run in -- and is selected explicitly rather than by accident.
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

from src.config import GROUNDING_MODE, MAX_TWEET_CHARS, MIN_GROUNDING_SIMILARITY
from src.triage.rules import CREDIT_CARD_REGEX, EMAIL_REGEX, PHONE_REGEX, SSN_REGEX, RuleMatcher

logger = logging.getLogger(__name__)

# WHY THE LINK-CHECK OUTCOMES ARE COUNTED
# ---------------------------------------
# Live link verification is opt-in (ENABLE_LIVE_LINK_CHECK, default off) and it
# moves the headline triage number, because the Kaggle corpus is 2017-2018 and its
# `t.co` links are now dead: a dead link is a real guardrail violation, so the same
# code scores differently depending on whether the machine running it can reach
# the internet.
#
# Two real runs of identical code, same commit:
#   * sandbox, egress-proxied  -> every check "inconclusive" (which PASSES)
#                              -> triage 63.7%, P95 36ms
#   * laptop, real internet    -> dead links actually detected
#                              -> triage 58.9%, P95 2188ms
#
# Neither number is wrong. The report quoting one of them without saying which
# configuration produced it is. That is the same failure as the judge-mode banner
# describing configuration rather than outcome -- except here the report recorded
# NEITHER. Counted so it can state what link verification actually did, including
# the case where it was switched on and achieved nothing because the network was
# blocked.
LINK_CHECK_OUTCOMES: dict[str, int] = {}


def reset_link_check_outcomes() -> None:
    """Zeroes the per-run link-check tally."""
    LINK_CHECK_OUTCOMES.clear()


def link_check_outcomes() -> dict[str, int]:
    """{status: count} for the current run, e.g. {"ok": 12, "broken": 30}."""
    return dict(LINK_CHECK_OUTCOMES)

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

# Scheme-less domains must be extracted too. The previous pattern only matched
# "https?://..." plus three hardcoded Apple hosts, so a draft containing
# "Verify your identity at appleid-verify.com/unlock" yielded ZERO extracted
# URLs -- validate_urls() then iterated an empty list and returned OK. The
# whitelist was bypassed simply by omitting the scheme, and Twitter auto-links
# bare domains, so the result is a clickable phishing link auto-tweeted from
# the brand account. Found by a red-team pass.
#
# The TLD is required to be >= 2 alphabetic characters, which keeps version
# strings ("iOS 17.4.1"), decimals and "e.g." out of the match.
# The scheme-less branch had two defects, both found by adversarial review
# round 2.
#
# 1. FALSE POSITIVES ON MISSING SPACES AFTER A FULL STOP. "Let us know how it
#    goes.Thanks for your patience" yielded the "URL" `goes.Thanks`, which is not
#    whitelisted, so an entirely ordinary reply was escalated as
#    UNAUTHORIZED_URL. A missing space after a period is one of the most common
#    things in real support copy, so this was not a rare edge. Requiring the TLD
#    to come from a fixed list of real TLDs removes the class: `Thanks` and `We`
#    are not TLDs.
#
# 2. QUADRATIC BACKTRACKING (ReDoS). `[a-z0-9](?:[a-z0-9-]*[a-z0-9])?` is
#    ambiguous about where the inner group starts and stops, so a long
#    hyphen-run with no dot made the engine try every split. Measured on the old
#    pattern: 16KB of "a-a-a-..." took 1.08s and 100KB took ~47s, in a guardrail
#    that runs on a public HTTP endpoint. The rewrite bounds each label with a
#    single quantifier ({0,62}, the DNS label limit), so work per start position
#    is capped and total time is linear in input length.
#
# THREE DEFECTS, THREE ROUNDS. This pattern's history, because the shape of the
# mistake matters more than the current regex:
#
#   r1  matched only "https?://..." plus three hardcoded Apple hosts, so
#       "appleid-verify.com/unlock" extracted NOTHING and the whitelist was
#       bypassed by omitting the scheme. Fixed by adding a scheme-less branch.
#   r2  that branch matched any dotted token, so "how it goes.Thanks" extracted
#       "goes.Thanks" and escalated ordinary support copy; and its nested
#       quantifier backtracked quadratically (100KB -> ~47s). Fixed by bounding
#       the label and requiring a TLD from a list.
#   r3  both halves of the r2 fix were wrong in the same way -- too clever by
#       half, and claimed more than they did:
#
#       (a) THE REDOS WAS NOT FIXED, only moved. Bounding the label left the
#           OUTER `(?:label\.)+` unbounded, so a dotted payload still backtracks
#           over every label count at every start position. Measured on the r2
#           pattern: 4000 chars of "a." -> 1.42s, 16KB -> 20.5s, 32KB -> 83.6s.
#           Worse, the r2 regression test asserted linearity using ONLY
#           "a-" * 50000 -- the single shape the rewrite did fix -- so the suite
#           certified a property the pattern did not have. Both the label and the
#           repetition are bounded now, and the test parametrises over all three
#           payload shapes.
#
#       (b) THE TLD LIST CREATED A PHISHING HOLE. A listed TLD that is a PREFIX
#           of an unlisted one matched, then failed its trailing \b:
#           "support.apple.company/verify-now" tried "com", hit "pany", and
#           extracted NOTHING -- so validate_urls() never consulted the
#           whitelist and a far more convincing phish than anything in the r2
#           notes shipped clean. Same for .community, .network, .delivery,
#           .services, .coop. Fixed by matching a generic TLD shape and
#           REJECTING sentence-boundary matches instead of enumerating TLDs: the
#           thing that distinguishes "goes.Thanks" from a domain is not its TLD,
#           it is that a scheme-less domain in real copy carries a path, a www.,
#           or a known Apple host. Requiring one of those removes the false
#           positives without an allowlist that can be prefix-matched.
#
# Residual limit, stated: a scheme-less bare domain with no path and no www.
# ("go to appleid-verify.net") is not extracted. The whitelist still covers every
# scheme-ful URL and every scheme-less one with a path, which is what a phishing
# link needs in order to land the victim anywhere useful.
URL_EXTRACTOR = re.compile(
    r"https?://\S+"
    # www.host[/path] -- www. is itself the signal that this is a hostname
    r"|www\.(?:[a-z0-9][a-z0-9-]{0,62}\.){0,6}[a-z]{2,24}(?:/\S*)?"
    # host/path -- the path is what distinguishes a domain from a full stop
    r"|(?:[a-z0-9][a-z0-9-]{0,62}\.){1,6}[a-z]{2,24}/\S*",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "or",
    "is",
    "are",
    "we",
    "you",
    "your",
    "us",
    "our",
    "this",
    "that",
    "it",
    "for",
    "on",
    "in",
    "of",
    "at",
    "please",
    "let",
    "with",
    "can",
    "will",
    "be",
    "has",
    "have",
    "help",
    "would",
    "like",
    "here",
    "check",
    "out",
    "we'd",
    "we're",
    "let's",
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
        grounding_mode: str = GROUNDING_MODE,
        min_grounding_similarity: float = MIN_GROUNDING_SIMILARITY,
    ):
        self.max_chars = max_chars
        self.min_grounding_overlap = min_grounding_overlap
        self.rule_matcher = RuleMatcher()
        self.verify_links = verify_links
        self.link_check_timeout = link_check_timeout
        if grounding_mode not in ("embedding", "lexical"):
            raise ValueError(f"grounding_mode must be 'embedding' or 'lexical', got {grounding_mode!r}")
        self.grounding_mode = grounding_mode
        self.min_grounding_similarity = min_grounding_similarity

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

    def validate_urls(self, text: str) -> tuple[bool, list[str]]:
        """Returns True if all included URLs belong to whitelisted Apple domains."""
        urls = URL_EXTRACTOR.findall(text)
        invalid_urls = []
        for url in urls:
            normalized = url if url.startswith("http") else f"https://{url}"
            if not WHITELISTED_URL_PATTERN.match(normalized):
                invalid_urls.append(url)
        return len(invalid_urls) == 0, invalid_urls

    def check_pii_echo(self, draft: str, source_customer_text: str | None) -> tuple[bool, list[str]]:
        """Returns (ok, echoed_values) -- False if the draft repeats PII that
        appeared in the customer's own message.

        Cards and SSNs are checked as well as emails and phone numbers. They were
        not, which meant the check advertised as "defence in depth" would happily
        let a reply re-publish a full card number the customer had tweeted -- the
        single worst thing this guardrail could fail to catch. Found by a red-team
        pass.
        """
        if not source_customer_text:
            return True, []
        echoed = []
        for pattern in (EMAIL_REGEX, PHONE_REGEX, CREDIT_CARD_REGEX, SSN_REGEX):
            for match in pattern.findall(source_customer_text):
                value = match if isinstance(match, str) else next((m for m in match if m), "")
                if not value:
                    continue
                if value.lower() in draft.lower() and value not in echoed:
                    echoed.append(value)
        return len(echoed) == 0, echoed

    def check_unsafe_advice(self, text: str) -> tuple[bool, list[str]]:
        is_unsafe, rules = self.rule_matcher.detect_unsafe_advice(text)
        return not is_unsafe, rules

    def check_grounding_lexical(self, draft: str, retrieved_snippets: list[str] | None) -> tuple[bool, float]:
        """Bag-of-content-words overlap. Returns (ok, overlap_ratio).

        Kept as the no-model fallback. Its limits are real and documented in this
        class's docstring: vocabulary recycling passes, paraphrase fails.
        """
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

    def check_grounding_embedding(self, draft: str, retrieved_snippets: list[str] | None) -> tuple[bool, float]:
        """Max cosine similarity between the draft and any retrieved snippet.

        Uses src/embeddings.py's cached get_encoder(), so this shares the single
        already-loaded all-MiniLM-L6-v2 instance rather than loading a second
        copy -- the duplicate-model-load problem that contributed to an OOM on
        the 512MB deploy target.

        Falls back to the lexical check (rather than failing the draft, or
        crashing) when no encoder can be constructed, which is the state of CI
        and the offline test suite.
        """
        if not retrieved_snippets:
            return True, 1.0
        if draft.strip() in [s.strip() for s in retrieved_snippets]:
            return True, 1.0
        try:
            import numpy as np

            from src.config import EMBEDDING_MODEL_NAME
            from src.embeddings import get_encoder

            encoder = get_encoder(EMBEDDING_MODEL_NAME)
            vectors = encoder.encode(
                [draft, *list(retrieved_snippets)],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            draft_vec, snippet_vecs = vectors[0], vectors[1:]
            similarity = float(np.max(snippet_vecs @ draft_vec))
        except Exception as e:
            # No model available (CI, offline, restricted network). Degrading to
            # the lexical check is the honest behaviour: refusing every draft
            # would turn a missing optional dependency into a total outage, and
            # passing every draft would silently disable a safety check.
            logger.warning(f"Embedding grounding unavailable ({e}); falling back to lexical overlap")
            return self.check_grounding_lexical(draft, retrieved_snippets)
        return similarity >= self.min_grounding_similarity, similarity

    def check_grounding(self, draft: str, retrieved_snippets: list[str] | None) -> tuple[bool, float]:
        """Dispatches to the configured grounding implementation.

        Signature and return shape are unchanged, so existing callers and tests
        are unaffected by the embedding check being added behind it.
        """
        if self.grounding_mode == "embedding":
            return self.check_grounding_embedding(draft, retrieved_snippets)
        return self.check_grounding_lexical(draft, retrieved_snippets)

    def check_link_reachability(self, text: str) -> tuple[bool, list[str]]:
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
            LINK_CHECK_OUTCOMES[result.status] = LINK_CHECK_OUTCOMES.get(result.status, 0) + 1
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
        source_customer_text: str | None = None,
        retrieved_snippets: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
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

        grounding_ok, grounding_score = self.check_grounding(text, retrieved_snippets)
        if not grounding_ok:
            _floor = self.min_grounding_similarity if self.grounding_mode == "embedding" else self.min_grounding_overlap
            _measure = "embedding similarity" if self.grounding_mode == "embedding" else "lexical overlap"
            violations.append(
                f"UNGROUNDED: Draft has only {grounding_score:.0%} {_measure} with retrieved "
                f"historical resolutions (min {_floor:.0%})"
            )

        link_ok, link_problems = self.check_link_reachability(text)
        if not link_ok:
            violations.append(f"UNVERIFIED_LINK: Draft cites a link that appears dead or fabricated: {link_problems}")

        return len(violations) == 0, violations
