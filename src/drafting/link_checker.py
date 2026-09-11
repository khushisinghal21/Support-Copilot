"""Live URL verification for drafted replies.

`OutputGuardrail.validate_urls` already restricts drafted links to a
whitelist of official Apple domains, but a domain-shaped regex only proves
the DOMAIN is right -- it can't tell a real page from a fabricated one on
that same domain. That gap is not hypothetical: manually verifying a live
dashboard reply found the LLM had cited "apple.co/directmessage", which
passes the whitelist regex (right domain, plausible slug) but actually
302-redirects to Apple's generic homepage, because no such page exists.

This module closes that gap by actually fetching the URL and checking
whether it resolves to real, specific content rather than a dead link or a
bounce back to the bare domain root.

Deliberately opt-in and OFF by default (see ENABLE_LIVE_LINK_CHECK in
src/config.py): src/eval/runner.py's bulk benchmark run shares the same
GroundedReplyGenerator/OutputGuardrail construction path used for live
queries, so if this defaulted on, a fresh clone running
`python -m src.eval.runner` over 150-250 golden-set rows would fire that
many live HTTP calls per run, breaking the "runs offline in well under 15
minutes" guarantee (docs/DECISION_LOG.md #2) and making the harness flaky
in any network-restricted environment -- this project has already hit a
hard outbound-network block once, in the sandboxed session documented in
docs/AUDIT_AND_FIX_PLAN.md (huggingface.co was unreachable there). Turn it
on (ENABLE_LIVE_LINK_CHECK=true in .env) for live, single-query use
(dashboard / CLI / REST API), where checking the one or two links in a
single real customer-facing reply is cheap and directly prevents shipping
a dead or fabricated URL -- just don't leave it on for a full eval run.

Network failures (no connectivity, DNS failure, timeout) are reported as
"inconclusive", not "broken" -- a sandboxed or offline environment
shouldn't fail-closed on every single reply just because it currently has
no route to the internet. That would turn a nice-to-have verification step
into an outage. The caller (OutputGuardrail.check_link_reachability) only
treats "broken" and "suspicious_redirect" as guardrail violations and
merely logs a warning for "inconclusive" results.
"""

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Matches full http(s) URLs, and the bare "apple.co/..." / "support.apple.com/..."
# style links this project's prompts and historical corpus actually use.
_URL_PATTERN = re.compile(
    r'https?://[^\s)>\]"\']+'
    r'|(?<![\w.])(?:apple\.co|support\.apple\.com|iforgot\.apple\.com|reportaproblem\.apple\.com)/[^\s)>\]"\']+',
    re.IGNORECASE,
)

_USER_AGENT = "Mozilla/5.0 (compatible; HiverSupportAgent/1.0; link-verification)"


class LinkCheckResult(NamedTuple):
    url: str
    status: str  # "verified" | "broken" | "suspicious_redirect" | "inconclusive"
    detail: str


def extract_urls(text: str) -> list[str]:
    """Finds candidate URLs (full or bare apple.co-style) in a drafted reply."""
    return _URL_PATTERN.findall(text)


def _normalize(url: str) -> str:
    return url if url.startswith("http") else f"https://{url}"


def verify_url(url: str, timeout: float = 3.0) -> LinkCheckResult:
    """Fetches a URL and classifies whether it's real. Never raises."""
    full_url = _normalize(url)
    try:
        parsed_original = urllib.parse.urlparse(full_url)
        had_specific_path = len(parsed_original.path.strip("/")) > 0

        req = urllib.request.Request(full_url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 405:
                # Some servers (Apple's included, for some paths) reject HEAD.
                req = urllib.request.Request(full_url, method="GET", headers={"User-Agent": _USER_AGENT})
                resp = urllib.request.urlopen(req, timeout=timeout)
            else:
                raise

        with resp:
            status_code = resp.status
            final_url = resp.geturl()

        if not (200 <= status_code < 400):
            return LinkCheckResult(url, "broken", f"HTTP {status_code}")

        parsed_final = urllib.parse.urlparse(final_url)
        landed_on_root = len(parsed_final.path.strip("/")) == 0
        if had_specific_path and landed_on_root:
            # Asked for a specific page, got bounced to the bare domain root.
            # This is exactly the apple.co/directmessage pattern: a 2xx/3xx
            # status that LOOKS fine, but the specific resource never existed.
            return LinkCheckResult(
                url,
                "suspicious_redirect",
                f"redirected to the domain root ({final_url}) instead of the requested page",
            )

        return LinkCheckResult(url, "verified", f"HTTP {status_code}, resolves to {final_url}")

    except urllib.error.HTTPError as e:
        return LinkCheckResult(url, "broken", f"HTTP {e.code}")
    except Exception as e:
        logger.warning(f"Link check for {full_url} could not complete (network/DNS/timeout): {e}")
        return LinkCheckResult(url, "inconclusive", str(e))


def verify_all_urls(text: str, timeout: float = 3.0) -> list[LinkCheckResult]:
    """Extracts and verifies every URL found in the given text."""
    return [verify_url(u, timeout=timeout) for u in extract_urls(text)]
