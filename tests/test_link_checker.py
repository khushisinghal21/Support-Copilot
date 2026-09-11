"""Tests for src/drafting/link_checker.py and OutputGuardrail.check_link_reachability.

Real motivating case: a live dashboard reply cited "apple.co/directmessage",
which passes the domain-whitelist regex (src/drafting/guardrails.py's
validate_urls) but manually verifying it with a real HTTP fetch showed it
302-redirects to Apple's generic homepage -- the specific page never
existed. These tests mock urllib.request.urlopen so they run offline and
deterministically, covering: a real/specific page (verified), a dead link
(broken), a bounce to the domain root (suspicious_redirect), and a network
failure (inconclusive, not treated as broken).
"""

import urllib.error
from unittest.mock import MagicMock, patch

from src.drafting.guardrails import OutputGuardrail
from src.drafting.link_checker import extract_urls, verify_url


def _fake_response(status=200, geturl="https://support.apple.com/kb/HT201264"):
    resp = MagicMock()
    resp.status = status
    resp.geturl.return_value = geturl
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_extract_urls_finds_bare_apple_co_and_full_urls():
    text = "Check out apple.co/battery-tips or https://support.apple.com/kb/HT201264 for help."
    urls = extract_urls(text)
    assert "apple.co/battery-tips" in urls
    assert "https://support.apple.com/kb/HT201264" in urls


def test_verify_url_marks_real_specific_page_as_verified():
    with patch("urllib.request.urlopen", return_value=_fake_response(
        status=200, geturl="https://support.apple.com/kb/HT201264"
    )):
        result = verify_url("support.apple.com/kb/HT201264")
    assert result.status == "verified"


def test_verify_url_marks_dead_link_as_broken():
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
        "https://apple.co/nonexistent", 404, "Not Found", None, None
    )):
        result = verify_url("apple.co/nonexistent")
    assert result.status == "broken"
    assert "404" in result.detail


def test_verify_url_marks_bounce_to_domain_root_as_suspicious():
    """The real apple.co/directmessage case: a 2xx/3xx that looks fine, but
    the specific path never existed and the server bounces to the bare
    domain root instead of a 404."""
    with patch("urllib.request.urlopen", return_value=_fake_response(
        status=200, geturl="https://www.apple.com/"
    )):
        result = verify_url("apple.co/directmessage")
    assert result.status == "suspicious_redirect"


def test_verify_url_network_failure_is_inconclusive_not_broken():
    """A sandboxed/offline environment shouldn't fail-closed on every reply
    just because it currently has no route to the internet."""
    with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
        result = verify_url("apple.co/battery-tips")
    assert result.status == "inconclusive"


def test_guardrail_link_check_is_off_by_default():
    """verify_links defaults to False -- must not make a live network call
    unless explicitly opted in (see src/config.py's ENABLE_LIVE_LINK_CHECK)."""
    guardrail = OutputGuardrail()
    with patch("urllib.request.urlopen") as mock_urlopen:
        ok, problems = guardrail.check_link_reachability("Visit apple.co/support for help.")
    mock_urlopen.assert_not_called()
    assert ok is True
    assert problems == []


def test_guardrail_link_check_flags_broken_link_when_enabled():
    guardrail = OutputGuardrail(verify_links=True)
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
        "https://apple.co/fake", 404, "Not Found", None, None
    )):
        ok, problems = guardrail.check_link_reachability("Visit apple.co/fake for help.")
    assert ok is False
    assert len(problems) == 1
    assert "apple.co/fake" in problems[0]


def test_guardrail_evaluate_escalates_on_unverified_link_when_enabled():
    guardrail = OutputGuardrail(verify_links=True)
    with patch("urllib.request.urlopen", return_value=_fake_response(
        status=200, geturl="https://www.apple.com/"
    )):
        passed, violations = guardrail.evaluate("We'd like to help -- see apple.co/directmessage for next steps.")
    assert passed is False
    assert any("UNVERIFIED_LINK" in v for v in violations)


def test_guardrail_evaluate_passes_verified_link_when_enabled():
    guardrail = OutputGuardrail(verify_links=True)
    with patch("urllib.request.urlopen", return_value=_fake_response(
        status=200, geturl="https://support.apple.com/kb/HT201264"
    )):
        passed, violations = guardrail.evaluate(
            "Check out some battery maximizing tips here: support.apple.com/kb/HT201264"
        )
    assert passed is True
    assert violations == []
