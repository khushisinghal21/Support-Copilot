"""Deterministic pattern matching rules for safety, PII, and security escalation.

Hardened version. The audit (docs/AUDIT_AND_FIX_PLAN.md) found the previous
rules false-positived on things like calorie-tracking "burnt calories" tweets
matching a bare `burn\\w*` pattern, and a generic "hardware issue" phrase that
would escalate ordinary complaints. There was also no negation handling at all
("my battery is NOT swollen" escalated identically to a genuine swelling
report). This version is deliberately more literal/specific and negation-aware.
"""

import re

# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# International-ish phone matcher: requires phone-like separators (space,
# dash, dot, or parens) so it doesn't fire on bare digit runs like a support
# case number ("#100310750365" -- a real false positive found during the
# audit). Handles +country codes and common grouping patterns without
# claiming full E.164 coverage.
# The separator-free forms were added after a red-team pass auto-handled
# "my callback number is 4155550199, my card 4111111111111111" -- a full PAN in
# a public tweet, and the PII-echo guardrail (which reuses these same patterns)
# then let a draft repeat it back. Requiring separators meant the gate only
# caught PII that was politely formatted.
#
# The 10/11-digit alternation deliberately cannot match the 12-digit support
# case number "#100310750365" that false-positived in the original audit: the
# leading (?<!\d) and trailing (?!\d) anchors force the whole run to be exactly
# 10 or 11 digits.
PHONE_REGEX = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[-.\s])?(?:\(\d{2,4}\)[-.\s]?)?\d{2,4}[-.\s]\d{3,4}[-.\s]\d{3,4}(?!\d)"
    r"|(?<![\d+])\+?\d{10,11}(?!\d)"
)
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:\d{4}[-\s]){3}\d{4}\b"  # 4-4-4-4 with separators
    r"|(?<!\d)\d{16}(?!\d)"  # bare 16-digit PAN
    r"|(?<!\d)\d{15}(?!\d)"  # bare 15-digit (Amex)
    r"|\b\d{4}[-\s]\d{6}[-\s]\d{5}\b"  # Amex 4-6-5 grouping
)
SSN_REGEX = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")

# ---------------------------------------------------------------------------
# Negation guard: scan a short window before a match for a negator. Kills the
# most common false-positive class ("not swollen", "isn't on fire", "no smoke")
# without needing a full parser.
# ---------------------------------------------------------------------------
NEGATION_WORDS = re.compile(
    r"\b(not|isn't|isnt|wasn't|wasnt|no|never|doesn't|doesnt|didn't|didnt)\b",
    re.IGNORECASE,
)


# A negator only negates within its own clause. Without this, the 20-character
# lookback treats the "No" in "No joke, my iPhone battery is swollen" as negating
# the hazard -- so the single highest-severity gate in the system was disabled by
# the most natural emphasis a frightened customer uses. Found by a red-team pass;
# it is a false NEGATIVE on a lithium-fire report, not a style issue.
CLAUSE_BOUNDARY = re.compile(r"[,;:\u2014\u2013-]|\b(but|however|though|although)\b", re.IGNORECASE)


def _is_negated(text: str, match_start: int, match_end: int | None = None, window: int = 20) -> bool:
    """Returns True if a negation word appears either in the `window`
    characters immediately preceding the match, or *inside* the match span
    itself.

    The hazard patterns below match a device-noun-to-hazard-word span as a
    single regex match (e.g. "battery is NOT swollen" matches starting at
    "battery" and ending at "swollen"), so a negation word sitting between
    the noun and the hazard word -- the most natural place for one to be in
    English -- falls *inside* the match, not before it. Checking only the
    text before match_start (the original version of this function) misses
    exactly that case and would have let "my battery is NOT swollen"
    escalate identically to a genuine report. Checking the match span too
    fixes it without needing a real parser.
    """
    lookback = text[max(0, match_start - window) : match_start]

    # A negator only negates inside its own clause. Truncate the lookback at the
    # LAST clause boundary before the match, so "No joke, my iPhone battery is
    # swollen" and "I'm not kidding, my battery is swollen" are not read as
    # denials. Before this, that emphatic prefix disabled the highest-severity
    # gate in the system -- a false negative on a lithium-fire report, found by a
    # red-team pass. "my battery is NOT swollen" has no boundary between the
    # negator and the hazard, so it still suppresses correctly.
    boundaries = list(CLAUSE_BOUNDARY.finditer(lookback))
    if boundaries:
        lookback = lookback[boundaries[-1].end() :]

    if NEGATION_WORDS.search(lookback):
        return True
    if match_end is not None and NEGATION_WORDS.search(text[match_start:match_end]):
        return True
    return False


# ---------------------------------------------------------------------------
# Hardware/thermal/physical hazard detection
# ---------------------------------------------------------------------------
# Tightened from the original: bare "burn"/"fire"/"hot" is dropped (it
# false-positived on "burnt a lot of calories" fitness-tracker tweets and
# similar); a hazard now requires the word to appear near a device noun or in
# an explicit fire/smoke/spark/shock context.
BATTERY_HAZARD_REGEX = re.compile(
    r"\b(battery|phone|device|macbook|ipad|iphone|watch|case)\b[^.!?]{0,25}\b"
    r"(swoll\w*|swell\w*|bulg\w*|expand\w*|puff(ed|ing)?)\b"
    r"|\b(swollen|bulging|expanding|puffy)\s+(battery|device|phone|case)\b"
    r"|\b(smoke|smoking)\b(?!\s*(-|\s)?free)"
    r"|\b(caught fire|on fire)\b"
    # spark(s) could not match "sparking" -- the \b fails before "ing", and
    # "sparking" is the single most likely word for an electrical fault.
    r"|\bspark(s|ed|ing)?\b[^.!?]{0,25}\b(charg\w*|port|outlet|plug\w*|cable|adapter|brick)\b"
    r"|\b(explod\w*)\b"
    r"|\bburning (smell|plastic smell)\b"
    r"|\b(shocked|electric shock|electrocut\w*)\b"
    # Everything below was auto-handled before a red-team pass: a reported
    # third-degree burn, a device too hot to hold, a melted mains adapter and a
    # leaking cell all got a troubleshooting reply.
    r"|\bburn(ed|t|ing)?\b[^.!?]{0,30}\b(hand|finger|skin|leg|lap|blister)\b"
    r"|\b(hand|finger|skin|leg|lap)\b[^.!?]{0,20}\bburn(ed|t|ing)?\b"
    r"|\boverheat\w*\b"
    r"|\btoo hot to (touch|hold)\b"
    r"|\b(melt(ed|ing)?|scorch\w*|charred)\b[^.!?]{0,30}"
    r"\b(charger|brick|adapter|cable|port|outlet|phone|device|battery|macbook|ipad|iphone)\b"
    r"|\b(charger|brick|adapter|cable|port|outlet)\b[^.!?]{0,20}\b(melt(ed|ing)?|scorch\w*)\b"
    r"|\b(battery|device|phone|ipad|iphone|macbook)\b[^.!?]{0,25}\b(leak\w*|warp\w*)\b"
    r"|\bleak\w*\b[^.!?]{0,20}\b(fluid|liquid|acid|battery)\b",
    re.IGNORECASE,
)

PHYSICAL_DAMAGE_REGEX = re.compile(
    r"\b(shattered|smashed)\s+(glass|screen|display)\b"
    r"|\b(glass|screen|display)\b[^.!?]{0,15}\b(shattered|smashed)\b"
    r"|\b(dropped|fell|submerged|soaked)\b[^.!?]{0,25}\b(in|into|under)\b[^.!?]{0,10}"
    r"\b(water|toilet|pool|ocean|bath|lake|sink)\b"
    r"|\b(water damage|liquid damage)\b"
    # "went through the washing machine" was auto-handled.
    r"|\b(washing machine|dishwasher|washer|dryer)\b[^.!?]{0,30}\b(phone|ipad|iphone|watch|device|airpods)\b"
    r"|\b(phone|ipad|iphone|watch|device|airpods)\b[^.!?]{0,30}\b(went through|through) the (washing machine|dishwasher|washer|dryer)\b"
    r"|\bmaking (a )?(hissing|popping|crackling) noise\b",
    re.IGNORECASE,
)

HUMAN_REQUEST_REGEX = re.compile(
    # The optional pronoun is not cosmetic: "transfer me to an agent" and
    # "connect me with a person" are the natural phrasings, and the original
    # pattern (verb immediately followed by to/with) matched neither.
    # (an?\s+|the\s+)? not (a\s+)? -- the original could not match the article in
    # "transfer me to AN agent", because "a" was only accepted when followed by
    # whitespace. One character of regex, and the most natural phrasing of the
    # request this gate exists to catch went undetected.
    r"\b(speak|talk|connect|transfer|put)\s+(me|us)?\s*(to|with|through\s+to)\s+(an?\s+|the\s+)?(human|person|real person|agent|representative|advisor|operator|manager)\b"
    r"|\b(want|need)\s+(to\s+)?(speak|talk)\s+(to|with)\s+(a\s+)?(human|person|agent|representative|manager)\b"
    r"|\breal human\b"
    r"|\bstop\s+(this\s+|the\s+|with\s+this\s+)?(automated\s+)?(bot|robot)\b"
    r"|\bnot a bot\b|\bhate bots\b|\bactual human\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompt injection detection: patterns an LLM-in-the-loop drafting step should
# never comply with. Independent of the frustration/legal gate -- an
# injection attempt can read as perfectly calm text.
# ---------------------------------------------------------------------------
PROMPT_INJECTION_REGEX = re.compile(
    # (?:...)*` allows zero or more qualifiers in any order/combination --
    # the original version only allowed exactly one, so "ignore your
    # previous instructions" (two qualifiers: "your" and "previous") didn't
    # match at all, a real miss found while writing this hardening pass.
    # The verb alternation matters as much as the qualifier one. This previously
    # matched only "ignore ... instructions", so "DISREGARD your previous
    # instructions" and "FORGET your instructions" -- the two most common
    # paraphrases, and ones docs/REPORT.md listed as known misses -- walked
    # straight through. Requiring the noun "instructions"/"rules" is what keeps
    # ordinary language out: "please ignore my previous tweet" still does not
    # match, and neither does "can you forget my old Apple ID".
    r"\b(?:ignore|disregard|forget|override)\s+(?:(?:your|all|previous|earlier|the\s+above|any)\s+)*(?:instructions|rules|prompt)\b"
    r"|\bdisregard\s+(?:(?:all|previous|the\s+above|safety)\s+)*rules\b"
    # A fake system turn. Attackers prefix a line with a role label to make the
    # model treat injected text as privileged instruction rather than customer
    # content.
    r"|\b(?:system|assistant|developer)\s*:\s*(?:override|ignore|disable|disregard|you\s+are)\b"
    r"|\byou are now\b.{0,20}\b(dan|jailbroken|unrestricted)\b"
    r"|\bprint (your |the )?system prompt\b"
    r"|\breveal (your |the |internal )?(system prompt|instructions|internal (support )?scripts?)\b"
    r"|\bas an ai (language model|assistant) with no restrictions\b"
    r"|\bnew instructions (from|override)\b"
    r"|\byou must comply\b.{0,30}\breply only with\b"
    r"|\bpretend you('re| are)\b.{0,20}\b(my friend|not a bot|a human)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Unsafe DIY advice a drafted reply must never suggest, regardless of who
# generated it (LLM or template fallback) -- defense in depth for the output
# guardrail, not just the input triage gate.
# ---------------------------------------------------------------------------
UNSAFE_ADVICE_REGEX = re.compile(
    # "pierce ... with a needle" is a synonym for the one thing this regex exists
    # to block, and shipped. So did unscrewing, popping the back panel, heat
    # guns, freezers, ovens and charging under a pillow.
    r"\b(puncture|pierce|prick)\b.{0,20}\b(battery|cell|pouch)\b"
    r"|\b(needle|pin|knife|screwdriver)\b.{0,25}\b(battery|cell|swollen|puffy)\b"
    # Both directions: "take the battery out with a screwdriver" puts the tool
    # AFTER the component, which the pattern above cannot see.
    r"|\b(battery|cell)\b.{0,60}\b(screwdriver|needle|pin|knife|pry)\b"
    r"|\btake (it|the battery|the cell) out\b"
    r"|\bremove\s+the\s+battery\s+yourself\b"
    r"|\b(unscrew|screwdriver)\b.{0,30}\b(cover|back|panel|case|casing|screws?)\b"
    r"|\bpop\s+(the\s+)?(back|panel|cover|case)\b"
    r"|\bdisconnect\b.{0,25}\b(battery|ribbon|connector)\b"
    r"|\bheat\s*gun\b|\bblow\s*dryer\b"
    r"|\b(freezer|freeze it|put it in the freezer)\b"
    r"|\b(bake|oven)\b.{0,25}\b(\d{2,3}\s*(c|f|degrees)|minutes|dry)\b"
    r"|\bunder\s+(your\s+)?(pillow|blanket|duvet)\b"
    # \bjailbreak\b does not match "jailbreaking" or "jailbroken" -- the trailing
    # suffix defeats the word boundary, so the most common forms of the word were
    # invisible to this check.
    r"|\bjailbreak(?:ing|ed|en)?\b|\bjailbroken\b"
    # "open the case/casing/back" covers the phrasing a customer actually reads as
    # instructions; the original required naming a device AND the word
    # "yourself"/"casing", so "open the case with a screwdriver" passed.
    r"|\bopen\s+(up\s+)?the\s+(device|iphone|ipad|macbook|case|casing|back)\b"
    r"|\b(pry|lever)\s+(it|the\s+\w+)\s+(open|apart)\b"
    r"|\bmicrowave\b|\boven\b.{0,10}\bdry\b"
    r"|\buse\s+(a\s+)?(hair\s*dryer|excessive\s+force)\b",
    re.IGNORECASE,
)


DEVICE_NOUN_REGEX = re.compile(
    # Found via a real failure case: the original `\b(iphone|...)\b` required
    # a word boundary immediately after the device name, which does NOT
    # exist between a letter and an attached digit ("iphone6", "ipad2") --
    # \b only fires at a transition between a word char and a non-word char,
    # and digits count as word chars just like letters. That meant "my
    # iphone6 is a lot choppier" registered as NOT mentioning a device,
    # incorrectly triggering the CLARIFY gate on a query that named its
    # device just fine. `[a-z0-9]*` after the root word absorbs model
    # suffixes (6, 6s, 11, x, xr, pro, se, 2) without needing an exhaustive
    # list of every model name Apple has shipped.
    r"\b(iphone|ipad|ipod|macbook|imac)[a-z0-9]*\b"
    r"|\bmac\s*(mini|pro)\b"
    r"|\bapple\s*watch\b|\bairpods\b|\bapple\s*tv\b"
    r"|\bmy (mac|watch)\b",
    re.IGNORECASE,
)


class RuleMatcher:
    """Evaluates text against high-precision deterministic regex rules,
    negation-aware for hazard matches."""

    @staticmethod
    def detect_pii(text: str) -> tuple[bool, list[str]]:
        matched = []
        if EMAIL_REGEX.search(text):
            matched.append("EMAIL_ADDRESS_DETECTED")
        if PHONE_REGEX.search(text):
            matched.append("PHONE_NUMBER_DETECTED")
        if CREDIT_CARD_REGEX.search(text):
            matched.append("CREDIT_CARD_DETECTED")
        if SSN_REGEX.search(text):
            matched.append("SSN_DETECTED")
        return len(matched) > 0, matched

    @staticmethod
    def detect_hardware_hazard(text: str) -> tuple[bool, list[str]]:
        """Scans for physical battery swelling, smoke, fire, shock, or
        shattered/submerged hardware, skipping matches inside a negated
        clause ("not swollen", "no smoke")."""
        matched = []
        m = BATTERY_HAZARD_REGEX.search(text)
        if m and not _is_negated(text, m.start(), m.end()):
            matched.append("BATTERY_THERMAL_HAZARD")
        m = PHYSICAL_DAMAGE_REGEX.search(text)
        if m and not _is_negated(text, m.start(), m.end()):
            matched.append("PHYSICAL_DAMAGE_INSPECTION_REQUIRED")
        return len(matched) > 0, matched

    @staticmethod
    def detect_human_request(text: str) -> bool:
        return bool(HUMAN_REQUEST_REGEX.search(text))

    @staticmethod
    def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
        """Scans customer input for prompt-injection attempts against the
        AI drafting step. Independent of sentiment/legal detection -- an
        injection attempt can be phrased calmly."""
        if PROMPT_INJECTION_REGEX.search(text):
            return True, ["PROMPT_INJECTION_PATTERN"]
        return False, []

    @staticmethod
    def detect_unsafe_advice(text: str) -> tuple[bool, list[str]]:
        """Scans a DRAFTED REPLY (not customer input) for unsafe DIY
        instructions that must never be sent, regardless of source."""
        if UNSAFE_ADVICE_REGEX.search(text):
            return True, ["UNSAFE_ADVICE_PATTERN"]
        return False, []

    @staticmethod
    def mentions_device(text: str) -> bool:
        """Returns True if the text names a specific Apple product."""
        return bool(DEVICE_NOUN_REGEX.search(text))
