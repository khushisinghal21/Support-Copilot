"""Shared helpers for calling Gemini via the google-genai SDK.

Currently just the thinking-config builder, factored out of
src/drafting/generator.py and src/eval/judge.py so both callers stay in sync
as Gemini's thinking-control API keeps shifting (see build_thinking_config's
docstring for the real regressions this fixes).
"""

from typing import Optional

from google.genai import types

# google-genai has required Python >=3.10 since its 2.0.0 release (the same
# release that added Gemini 3 support, including thinking_level). This
# project's target venv (per every real terminal log seen from it) is
# Python 3.9.6, which is past end-of-life and can only install pre-2.0
# google-genai releases -- so `pip install google-genai>=1.0.0` silently
# resolves to the newest 1.x release rather than erroring, and that
# release's ThinkingConfig schema simply predates thinking_level entirely.
# Upgrading the project's Python to 3.10+ would be the clean fix (it would
# also clear the separate google-auth/google-api-core EOL warnings already
# showing up in every server log), but that's a bigger, more invasive
# change to make unilaterally -- see build_thinking_config's
# graceful-degradation fallback below for what happens without it.
_THINKING_CONFIG_FIELDS = set(getattr(types.ThinkingConfig, "model_fields", {}).keys())


def build_thinking_config(model_name: str) -> Optional["types.ThinkingConfig"]:
    """Returns a ThinkingConfig that minimizes/disables thinking, using
    whichever field the installed SDK version and given model generation
    actually accept -- or None if neither is available, so a call site can
    omit thinking_config entirely rather than crash.

    Real regression #1: `ThinkingConfig(thinking_budget=0)` worked to
    disable thinking on gemini-2.5-flash, but the same call against
    gemini-3.6-flash came back "400 INVALID_ARGUMENT" with no further
    detail. This isn't a bug in this code -- it's a real, documented Gemini
    API change: the Gemini 3.x generation replaced the numeric
    `thinking_budget` field with an enum `thinking_level`
    ("minimal"/"low"/"medium"/"high"), and the two fields are mutually
    exclusive -- sending the wrong one for a given model generation is
    rejected outright rather than ignored. Gemini 2.x models are the
    reverse: they only understand thinking_budget.

    Real regression #2: fixing that by branching on model_name still broke
    in a *different* way against an older, Python-3.9-compatible
    google-genai release (see the module-level note above) --
    `ThinkingConfig(thinking_level="minimal")` raised a pydantic
    "1 validation error ... Extra inputs are not permitted" instead of a
    "400 INVALID_ARGUMENT", because that release's ThinkingConfig model
    simply has no thinking_level field at all to reject a bad *value* for.
    Checking `types.ThinkingConfig.model_fields` at import time (rather
    than assuming a fixed schema) makes this degrade gracefully instead of
    crashing on every single generation call: an old SDK talking to a
    Gemini 3 model gets no thinking_config and falls back to the model's
    own default (dynamic) thinking -- the finish_reason/short-reply checks
    in generator.py and judge.py remain as the real safety net if that
    still eats too much of the token budget.
    """
    wants_budget = model_name.startswith("gemini-2")
    if wants_budget and "thinking_budget" in _THINKING_CONFIG_FIELDS:
        return types.ThinkingConfig(thinking_budget=0)
    if not wants_budget and "thinking_level" in _THINKING_CONFIG_FIELDS:
        # Pass the enum member when this SDK exposes it, rather than relying on
        # pydantic to coerce the string. Verified that the string form DOES still
        # coerce correctly ("minimal" -> ThinkingLevel.MINIMAL), so this is not a
        # bug fix -- it is removing a dependency on coercion that mypy could not
        # see through and a future SDK could tighten. getattr, because an older
        # release can have the field without exporting the enum.
        level_enum = getattr(types, "ThinkingLevel", None)
        level = getattr(level_enum, "MINIMAL", "minimal") if level_enum else "minimal"
        return types.ThinkingConfig(thinking_level=level)
    return None
