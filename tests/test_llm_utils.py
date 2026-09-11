"""Tests for src/llm_utils.py's build_thinking_config.

Real regression #1: `ThinkingConfig(thinking_budget=0)` worked to disable
thinking on gemini-2.5-flash but returned "400 INVALID_ARGUMENT" against
gemini-3.6-flash in production. Gemini 3.x uses an enum `thinking_level`
field instead of the numeric `thinking_budget` field Gemini 2.x uses, and
sending the wrong one for a given model generation is rejected outright.

Real regression #2: on the user's actual Python-3.9 venv, the installed
google-genai release predates thinking_level entirely (Python 3.9 can only
install pre-2.0 google-genai, and thinking_level shipped in 2.0.0), so
`ThinkingConfig(thinking_level="minimal")` raised a pydantic
"extra_forbidden" validation error instead of the 400. build_thinking_config
now checks the installed SDK's actual schema and returns None rather than
crashing when neither field is available for a (model, SDK) combination.
"""

from google.genai import types

import src.llm_utils as llm_utils
from src.llm_utils import build_thinking_config


def test_gemini_2_generation_uses_thinking_budget():
    cfg = build_thinking_config("gemini-2.5-flash")
    assert cfg.thinking_budget == 0
    assert cfg.thinking_level is None


def test_gemini_3_generation_uses_thinking_level():
    cfg = build_thinking_config("gemini-3.6-flash")
    assert cfg.thinking_level == types.ThinkingLevel.MINIMAL
    assert cfg.thinking_budget is None


def test_other_gemini_3_variants_also_use_thinking_level():
    for model in ["gemini-3.8-flash", "gemini-3-pro-preview", "gemini-3.1-flash-lite"]:
        cfg = build_thinking_config(model)
        assert cfg.thinking_level == types.ThinkingLevel.MINIMAL


def test_degrades_to_none_when_sdk_lacks_thinking_level(monkeypatch):
    """Simulates the user's actual Python-3.9 environment: an old
    google-genai release whose ThinkingConfig schema has thinking_budget
    but not thinking_level. Against a Gemini 3 model this must return None
    (skip thinking_config) rather than constructing an invalid field and
    crashing on every generation call."""
    monkeypatch.setattr(llm_utils, "_THINKING_CONFIG_FIELDS", {"thinking_budget", "include_thoughts"})
    assert build_thinking_config("gemini-3.6-flash") is None
    # Gemini 2.x is unaffected since thinking_budget is still present.
    cfg = build_thinking_config("gemini-2.5-flash")
    assert cfg.thinking_budget == 0


def test_degrades_to_none_when_sdk_lacks_thinking_budget():
    """Symmetry check: a hypothetical SDK/model combo where thinking_budget
    itself is unavailable for a Gemini 2.x model should also degrade to
    None rather than crash."""
    import src.llm_utils as module

    original = module._THINKING_CONFIG_FIELDS
    try:
        module._THINKING_CONFIG_FIELDS = {"thinking_level", "include_thoughts"}
        assert module.build_thinking_config("gemini-2.5-flash") is None
    finally:
        module._THINKING_CONFIG_FIELDS = original
