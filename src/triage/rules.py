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
# The domain part is bounded per label rather than written as one open
# `[A-Za-z0-9.-]+\.`, which was quadratic: the character class includes the dot,
# so the engine retried every split of a long dotted run. Measured on the old
# pattern with a payload of "a." repeated -- 16KB took 0.245s and 64KB took
# 3.60s, and unlike URL_EXTRACTOR this one runs on RAW CUSTOMER TEXT in triage
# gate 3 and again in check_pii_echo. The HTTP layer caps bodies at 4000 chars,
# but `src/cli.py` and the eval harness construct TweetInput directly and bypass
# that cap entirely, so the bound has to be in the pattern. Found by adversarial
# review round 3.
#
# Also dropped the stray "|" inside `[A-Z|a-z]`, which made the literal pipe a
# valid TLD character.
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,8}[A-Za-z]{2,24}\b")

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
    # "+919876543210" -- a country code plus a 10-digit national number is 12-13
    # digits, so the 10/11-digit branch above could not match it and round 3
    # auto-handled a customer's phone number, which the draft then read back. The
    # explicit leading "+" is what keeps this from colliding with the 12-digit
    # support case number "#100310750365" that the original audit fixed: that one
    # has no plus sign.
    r"|(?<!\d)\+\d{1,3}\d{9,11}(?!\d)"
)
# Separator-free branches constrained to real issuer prefixes, so an IMEI is not
# reported as a credit card (round 2) -- but with the issuer ranges actually
# right, which the first version of this was not.
#
# WHAT ROUND 3 FOUND WRONG WITH THE FIRST VERSION
# -----------------------------------------------
# The comment claimed "Covered: ... JCB (35), Diners (36/38)" and DECISION_LOG 32
# repeated it. Diners Club / Carte Blanche PANs are FOURTEEN digits; the branch
# was `3[68]\d{14}`, i.e. sixteen. So the pattern matched no real Diners card and
# missed every one -- while the disclosure said the only gap was issuers outside
# the list. A customer tweeting a live Diners PAN was auto-handled, and
# check_pii_echo (which reuses these patterns) then let the draft republish it.
# Also missed: 13- and 19-digit Visa, the 644-649 Discover range, and any
# separator that is not ASCII space or hyphen -- iOS and macOS substitute an en
# dash for a typed hyphen by default, so "4111-1111-1111-1111" typed on a Mac
# arrives as U+2013 and matched nothing.
#
# REMAINING LIMIT, stated: the 16-digit `35` JCB range overlaps IMEI-SV (an IMEI
# plus a 2-digit software version, on the same `35` TAC prefixes), so issuer
# prefixes cannot separate those two at that length. JCB is narrowed to its real
# 3528-3589 range, which excludes most IMEI TACs but not all; a 16-digit number
# in that range is reported as a card, and that reason code can be wrong. Keeping
# the detection and disclosing the mislabel is the right way round here -- the
# alternative drops a real card class.
_CARD_SEP = r"[-\s‐-―]"
CREDIT_CARD_REGEX = re.compile(
    rf"\b(?:\d{{4}}{_CARD_SEP}){{3}}\d{{4}}\b"  # 4-4-4-4 with separators
    rf"|\b\d{{4}}{_CARD_SEP}\d{{6}}{_CARD_SEP}\d{{5}}\b"  # Amex 4-6-5
    rf"|\b\d{{4}}{_CARD_SEP}\d{{6}}{_CARD_SEP}\d{{4}}\b"  # Diners 4-6-4
    # Separator-free, by issuer range:
    r"|(?<!\d)4\d{12}(?:\d{3})?(?:\d{3})?(?!\d)"  # Visa 13/16/19
    r"|(?<!\d)5[1-5]\d{14}(?!\d)"  # Mastercard
    r"|(?<!\d)2(?:22[1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)\d{12}(?!\d)"  # Mastercard 2-series
    r"|(?<!\d)3[47]\d{13}(?!\d)"  # Amex (15)
    r"|(?<!\d)6(?:011|5\d\d|4[4-9]\d)\d{12}(?!\d)"  # Discover
    r"|(?<!\d)35(?:2[89]|[3-8]\d)\d{12}(?!\d)"  # JCB 3528-3589
    r"|(?<!\d)3(?:0[0-5]|[68]\d)\d{11}(?!\d)"  # Diners / Carte Blanche (14)
)
# Dots added after round 3 auto-handled "my ssn is 123.45.6789". The separator
# is cosmetic; the disclosure is not.
SSN_REGEX = re.compile(r"\b\d{3}[-.\s\u2010-\u2015]\d{2}[-.\s\u2010-\u2015]\d{4}\b")

# ---------------------------------------------------------------------------
# Negation guard: scan a short window before a match for a negator. Kills the
# most common false-positive class ("not swollen", "isn't on fire", "no smoke")
# without needing a full parser.
# ---------------------------------------------------------------------------
NEGATION_WORDS = re.compile(
    r"\b(not|isn't|isnt|wasn't|wasnt|no|never|doesn't|doesnt|didn't|didnt)\b",
    re.IGNORECASE,
)


# NEGATION, THIRD AND FINAL SHAPE
# -------------------------------
# The first three versions of this logic were window heuristics, and each was
# broken by ordinary English in the next review round:
#
#   v1  a 20-char lookback. "No joke, my iPhone battery is swollen" read as a
#       denial -- the highest-severity gate in the system disabled by the most
#       natural emphasis a frightened customer uses.
#   v2  truncate the lookback at the nearest clause boundary. Round 2 deleted the
#       comma and the identical bypass returned.
#   v3  blank a closed list of emphatic idioms. Round 3 typed "No joking" instead
#       of "No joke" and the identical bypass returned again. Measured end to end:
#       "No joking my iPhone battery is swollen" -> AUTO_HANDLE, answered with a
#       troubleshooting reply.
#
# Three rounds, one defect, because all three asked a question with no correct
# answer: "is there a negation word somewhere near the hazard?" Every idiom list
# has a next inflection. Round 3 also produced
#   "My iPhone battery is not just warm, it's swollen"
#   "My iPad battery is no longer flat, it's swollen"
#   "My MacBook battery has never been this swollen"
#   "My iPhone has no case and the battery is swollen"
# all CLEAN, none of which any idiom list reaches.
#
# The question that does have a correct answer is structural: does the negator
# attach to the HAZARD WORD? In English a negator negates the predicate it
# immediately precedes. "is not swollen" is a denial; in "not just warm, it's
# swollen" the "not" attaches to "warm". So: take the last negator at or before
# the hazard, and treat it as a denial only if nothing but copulas and
# intensifying adverbs separate the two.
#
# This subsumes every case the idiom list existed for, without an idiom list: in
# "No joke my battery is swollen" the words "joke my battery is" sit between "No"
# and "swollen", so "No" cannot attach to it. Same for "No joking", "I shit you
# not", and phrasings nobody has thought of yet -- which is the property all
# three previous versions lacked.
# What may stand between a negator and the hazard word without breaking the
# attachment. Two groups, for two different reasons.
#
# COPULAS AND INTENSIFIERS: "is not swollen", "is not really swollen",
# "was never visibly swollen" are all denials.
#
# HAZARD VOCABULARY: the matched span ENDS with the hazard phrase, and those
# phrases are not all one word -- "on fire", "caught fire", "too hot to touch",
# "electric shock". Those tokens are listed here so the attachment test can see
# past them, rather than trying to guess how many trailing words to ignore.
# Guessing was tried first: stripping up to four trailing tokens also ate the
# real content in "my battery no joke is swollen", turning a hazard report back
# into a denial. Naming the vocabulary cannot make that mistake.
#
# Deliberately ABSENT, because these are the words that distinguish a denial from
# an emphasis: "just" ("not just warm, it's swollen"), "this" ("never been this
# swollen"), "longer" ("no longer flat, it's swollen"), every possessive, and all
# punctuation. That is what makes the rule hold without an idiom list.
NEGATION_FILLER = re.compile(
    r"^(?:\s|\b(?:"
    # copulas, auxiliaries, intensifiers
    r"is|are|am|was|were|be|been|being|get|gets|got|getting|look|looks|looking|"
    r"seem|seems|feel|feels|smell|smells|really|actually|very|that|so|too|even|yet|quite|"
    r"particularly|noticeably|visibly|currently|still|at|all|a|an|the|"
    # hazard-phrase vocabulary, so multi-word hazards do not read as content
    r"on|catch|catches|caught|fire|smoke|smoking|smoky|swoll\w*|swell\w*|bulg\w*|expand\w*|"
    r"puff\w*|burn\w*|melt\w*|scorch\w*|charred|explod\w*|shock\w*|electrocut\w*|hot|touch|"
    r"hold|to|leak\w*|warp\w*|spark\w*|overheat\w*|hiss\w*|pop\w*|crackl\w*|shatter\w*|"
    r"smash\w*|damage|damaged|wet|submerged|soaked"
    r")\b)*$",
    re.IGNORECASE,
)

# Retained for the PII/legal matchers below, which match short single spans where
# clause-level truncation is the right rule.
CLAUSE_BOUNDARY = re.compile(r"[,;:\u2014\u2013-]|\b(but|however|though|although)\b", re.IGNORECASE)


def _is_negated(text: str, match_start: int, match_end: int | None = None, window: int = 20) -> bool:
    """True when a negator structurally attaches to the matched hazard.

    `window` is how far before the match to look for a negator, kept for
    call-site compatibility. What changed is the test applied once one is found
    (see the comment above): a negator counts only when everything between it and
    the hazard word is copula or intensifier, so "battery is not swollen" is a
    denial while "battery is not just warm, it's swollen" is not.
    """
    end = match_end if match_end is not None else match_start
    region_start = max(0, match_start - window)
    region = text[region_start:end]

    last = None
    for m in NEGATION_WORDS.finditer(region):
        last = m
    if last is None:
        return False

    # Everything between the negator and the END of the match -- hazard word
    # included, since NEGATION_FILLER knows the hazard vocabulary. If all of it is
    # copula, intensifier or hazard word, the negator attaches to the hazard and
    # this is a denial. One substantive word in between and it does not.
    between = region[last.end() :]
    return bool(NEGATION_FILLER.match(between))


# ---------------------------------------------------------------------------
# Hardware/thermal/physical hazard detection
# ---------------------------------------------------------------------------
# Tightened from the original: bare "burn"/"fire"/"hot" is dropped (it
# false-positived on "burnt a lot of calories" fitness-tracker tweets and
# similar); a hazard now requires the word to appear near a device noun or in
# an explicit fire/smoke/spark/shock context.
BATTERY_HAZARD_REGEX = re.compile(
    r"\b(battery|phone|device|macbook|ipad|iphone|watch|case)\b[^.!?]{0,60}\b"
    r"(swoll\w*|swell\w*|bulg\w*|expand\w*|puff(ed|ing)?)\b"
    r"|\b(swollen|bulging|expanding|puffy)\s+(battery|device|phone|case)\b"
    r"|\b(smoke|smoking)\b(?!\s*(-|\s)?free)"
    r"|\b(caught fire|on fire)\b"
    # spark(s) could not match "sparking" -- the \b fails before "ing", and
    # "sparking" is the single most likely word for an electrical fault.
    r"|\bspark(s|ed|ing)?\b[^.!?]{0,25}\b(charg\w*|port|outlet|plug\w*|cable|adapter|brick)\b"
    r"|\b(explod\w*)\b"
    r"|\bburning (smell|plastic smell)\b"
    # The reverse ordering, which is how people actually say it. Round 3:
    # "my iphone smells like burning plastic" -> CLEAN, answered with a
    # troubleshooting reply. A burning-plastic smell from a sealed lithium device
    # is a thermal-runaway signal; it is not a wording edge case.
    r"|\bsmell(s|ing|ed|t)?\b[^.!?]{0,25}\b(burn\w*|smoke|smoky|acrid|chemical)\b"
    r"|\bsmells? burnt\b"
    r"|\b(shocked|electric shock|electrocut\w*)\b"
    # Everything below was auto-handled before a red-team pass: a reported
    # third-degree burn, a device too hot to hold, a melted mains adapter and a
    # leaking cell all got a troubleshooting reply.
    r"|\bburn(ed|t|ing)?\b[^.!?]{0,30}\b(hand|finger|skin|leg|lap|blister)\b"
    r"|\b(hand|finger|skin|leg|lap)\b[^.!?]{0,20}\bburn(ed|t|ing)?\b"
    # "overheat" ALONE was a hazard here, which made "My MacBook overheats when I
    # run Final Cut Pro" -- the single most common Mac complaint there is -- an
    # ESCALATE at risk_score 1.0 under reason code HARDWARE_PHYSICAL_DAMAGE. That
    # is both a false positive and a false *label*: a thermal-throttling
    # complaint is not physical damage. Confirmed by adversarial review round 2.
    #
    # It now needs one corroborating severity signal in the same clause. This is a
    # deliberate recall-for-precision trade, recorded in docs/DECISION_LOG.md: a
    # bare "it overheats" no longer escalates here, and instead takes the normal
    # path where the frustration gate and the output guardrails still apply. The
    # corroboration list is kept wide on purpose -- anything suggesting heat that
    # has left the realm of slow-fan-noise still fires.
    # The corroboration list is the round-2 precision trade (DECISION_LOG 31).
    # Round 3 showed the first version of the list was far narrower than decision
    # 31 claimed: it carried `swoll\w*` but not bulging, warping, expanding,
    # leaking or the one-word spelling "shutdown", so
    #   "My iPhone is overheating and there is a bulge in the back"
    #   "phone overheating, it just shutdown by itself"
    #   "My iPhone overheats, the back panel has warped"
    #   "iphone overheating so bad I cannot hold it"
    # were all CLEAN -- and "overheating AND bulging" is the textbook
    # pre-failure presentation of a swelling lithium cell, exactly the case the
    # trade was not supposed to cost. The window is 80 rather than 40 because a
    # normal English clause between the two ("overheats so badly that the
    # aluminium chassis has started to melt") is longer than 40 characters.
    r"|\boverheat\w*\b[^.!?]{0,80}\b(burn\w*|hot to (the )?(touch|hold)|cannot hold|can'?t hold|"
    r"smell\w*|smok\w*|melt\w*|scorch\w*|charred|blister\w*|bulg\w*|swoll\w*|swell\w*|expand\w*|"
    r"puff\w*|warp\w*|leak\w*|hiss\w*|pop(ping|ped)?|crackl\w*|red mark|painful|"
    r"shut ?(s|ting)? ?(itself )?(down|off)|shutdown|won'?t turn on|unsafe|danger\w*)\b"
    r"|\b(burn\w*|hot to (the )?(touch|hold)|smell\w*|smok\w*|melt\w*|scorch\w*|charred|blister\w*|"
    r"bulg\w*|swoll\w*|warp\w*|unsafe|danger\w*)\b[^.!?]{0,80}\boverheat\w*\b"
    r"|\btoo hot to (touch|hold)\b"
    r"|\b(melt(ed|ing)?|scorch\w*|charred)\b[^.!?]{0,30}"
    r"\b(charger|brick|adapter|cable|port|outlet|phone|device|battery|macbook|ipad|iphone)\b"
    r"|\b(charger|brick|adapter|cable|port|outlet)\b[^.!?]{0,20}\b(melt(ed|ing)?|scorch\w*)\b"
    r"|\b(battery|device|phone|ipad|iphone|macbook)\b[^.!?]{0,60}\b(leak\w*|warp\w*)\b"
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
    r"|\bnot a bot\b|\bhate bots\b|\bactual human\b"
    # Every branch above requires a speak/talk/connect/transfer/put verb, so
    # round 3 got "Get me a human, now.", "I want a human to look at this." and
    # "Can you escalate this to a supervisor?" all auto-handled. "supervisor" and
    # "someone (who actually works) at Apple" were absent from the noun list
    # entirely. A customer asking for a person is the least ambiguous escalation
    # signal this system receives; it should not depend on which verb they chose.
    r"|\b(get|give|find)\s+(me|us)\s+(an?\s+|the\s+)?(human|person|real person|agent|representative|"
    r"advisor|operator|manager|supervisor)\b"
    r"|\b(want|need)\s+(an?\s+|the\s+)?(human|person|real person|agent|representative|advisor|"
    r"operator|manager|supervisor)\b"
    r"|\bescalate\s+(this|it|my\s+\w+)?\s*(to\s+)?(an?\s+|the\s+)?(human|person|agent|representative|"
    r"advisor|manager|supervisor|someone)\b"
    r"|\b(speak|talk)\s+(to|with)\s+(someone|somebody)\b"
    r"|\bsomeone\s+who\s+(actually\s+)?works\s+at\b",
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
    # Round 3 walked five phrasings straight through this, all of which DO carry
    # the required noun (so they are not the documented no-noun xfail):
    #   "Ignore all PRIOR instructions ..."        -- "prior" was not a qualifier
    #   "Ignore THE INSTRUCTIONS ABOVE ..."        -- "above" was only accepted
    #                                                 before the noun, never after
    #   "Forget THE RULES ..."                     -- bare "the" was not a qualifier
    #   "Ignore previous INSTRUCTION ..."          -- the noun had to be plural
    #   "Your new INSTRUCTIONS ARE TO ..."         -- no imperative verb at all
    # "ignore all prior instructions" is plausibly the single most common
    # phrasing in the wild. The qualifier set is widened, the noun is allowed in
    # the singular, a trailing "above"/"in this thread" is accepted, and the
    # verbless possessive form gets its own branch.
    r"\b(?:ignore|disregard|forget|override|bypass)\s+"
    r"(?:(?:your|all|any|previous|prior|earlier|preceding|former|original|the|those|these|above)\s+)*"
    r"(?:instruction|instructions|rule|rules|prompt|prompts|guideline|guidelines|directive|directives)"
    r"(?:\s+(?:above|before|so\s+far|in\s+this\s+thread))?\b"
    r"|\b(?:your|the)\s+new\s+(?:instruction|instructions|rule|rules|directive|directives)\s+(?:are|is)\b"
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
