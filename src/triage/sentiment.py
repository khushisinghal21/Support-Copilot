"""Sentiment, customer frustration, and churn/fraud threat analyzer.

Hardened version. The audit (docs/AUDIT_AND_FIX_PLAN.md) found real
false-positive escalations from the original single-keyword design: tweets
where a customer was *reporting a phishing scam to Apple* (not personally
victimized) tripped on the words "scam"/"fraud", and "IM SUING" over a slow
charge -- hyperbole Apple's own historical agents handled as a routine ticket
-- tripped on "sue". The fix is to split triggers into three tiers instead of
one flat keyword list:

  1. STRONG signals (genuine account compromise, or legal/regulatory language
     rarely used in jest) -- escalate on a single mention.
  2. SOFT signals ("sue"/"suing", "fraud"/"scam"/"scammer", generic churn
     lines) -- these are the words that produced real false positives, so
     they now require corroboration: another soft signal, a strong signal,
     an anger marker, or an explicit first-person victim phrase ("my
     account", "charged me", "stole from me").
  3. Anger/profanity and punctuation/caps intensity -- unchanged in spirit,
     but the generic-complaint words ("trash", "garbage", "useless",
     "horrible", "disaster") were removed from the anger list since real
     customers use those about ordinary bugs constantly; keeping them made
     the anger gate fire on completely mundane complaints.
"""

import re

from src.config import FRUSTRATION_THRESHOLD

# Tier 1: genuine account compromise -- escalate-worthy on a single mention,
# because in this domain "hacked"/"compromised" is almost never said about
# someone else's account.
ACCOUNT_COMPROMISE_KEYWORDS = [
    "hacked",
    "compromised",
    "unauthorized charge",
    "unauthorized access",
    "unauthorized transaction",
    "unauthorized purchase",
    "unauthorized login",
    # Bare "unauthorized" was here and is deliberately gone. A single tier-1
    # keyword scores exactly at the 0.60 threshold, so it escalated on its own --
    # and "unauthorized" is a word APPLE'S OWN SOFTWARE PRINTS. Round 3 measured
    # all three of these routed to ESCALATE / HIGH_FRUSTRATION_CHURN_RISK with
    # marker ACCOUNT_COMPROMISE:
    #   "My iPhone keeps saying this accessory is unauthorized, how do I fix it?"
    #   "iTunes says unauthorized computer when I try to sync, any help?"
    #   "my macbook says the software is unauthorized after the update"
    # Wrong action and a false stated reason, in the module whose own docstring
    # says generic words were removed precisely because customers use them about
    # ordinary bugs constantly. The four specific phrases above cover every real
    # compromise report the bare word was catching.
    "identity theft",
    # Added after a red-team pass. Gate 5 detected the VOCABULARY of compromise
    # but not the EVENT: "Someone logged into my Apple ID from another country,
    # changed my password and is buying gift cards with my card" scored 0.00 and
    # was auto-answered with a link, because it contains none of the words above.
    # These describe what a takeover actually looks like when a customer reports
    # it in their own words.
    "someone logged into",
    "someone else logged in",
    "logged in from another country",
    # NOT a bare "changed my password" -- that fires on the benign "I changed my
    # password and now sync works fine". A takeover report names someone else
    # doing it, or says it happened TO them.
    "someone changed my password",
    "they changed my password",
    # The passive forms stay in this tier. Round 3 found them escalating
    # "my password was changed by the iOS update and now mail wont sync", and the
    # first attempt at a fix moved them to the soft tier -- which lost a round-1
    # detection ("my password was changed and I cannot get in", a plausible
    # takeover report) and was therefore a safety regression traded for a
    # precision gain. They are instead filtered by BENIGN_PASSWORD_CHANGE below,
    # which suppresses only the passive form when the customer NAMES a non-human
    # agent for the change. Keeping the detection and narrowing the exception is
    # the right direction when one error is a wasted human review and the other is
    # an unattended account takeover.
    "password was changed",
    "password has been changed",
    "changed my password without",
    "locked me out of my",
    "took over my",
    "has full control of my",
    "ported my number",
    "sim swap",
    "signed in from a device i don",
    "i didn't authorize",
    "i didnt authorize",
    "i did not authorize",
    "not my purchase",
    "didn't make this purchase",
    "didnt make this purchase",
]

# Tier 1: legal/regulatory language that's rarely used in jest.
STRONG_LEGAL_KEYWORDS = [
    "lawyer",
    "attorney",
    "lawsuit",
    "police",
    "bbb",
    "better business bureau",
    "consumer protection",
]

# Tier 2: words that DID produce real false positives in the audited data
# (customers reporting a scam, or hyperbolic "suing" over a minor issue) --
# require corroboration before they count.
SOFT_CHURN_KEYWORDS = [
    "sue",
    "suing",
    "fraud",
    "scam",
    "scammer",
    "scammed",
    "stole",
    "stolen",
    "switching to android",
    "switching to samsung",
    "never buying apple again",
    "canceling everything",
    "cancel this now",
]

# Explicit first-person victim framing -- corroborates a soft signal.
VICTIM_PHRASE_REGEX = re.compile(
    r"\bmy (account|money|card|bank)\b|\bcharged me\b|\bfrom my account\b|"
    r"\bstole (my|from me)\b|\bdrained\b|\bdamaged (my|cartilage)\b",
    re.IGNORECASE,
)

# Anger/profanity markers -- trimmed of generic complaint words ("trash",
# "garbage", "useless", "horrible", "disaster", "pathetic") that real
# customers use constantly about ordinary bugs, which made the original list
# false-positive on completely mundane complaints.
ANGER_KEYWORDS = [
    "fucking",
    "fuck",
    "shit",
    "bullshit",
    "idiots",
    "unacceptable",
    "furious",
    "outraged",
    "ripoff",
    "rip off",
]

ACCOUNT_COMPROMISE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ACCOUNT_COMPROMISE_KEYWORDS) + r")\b", re.IGNORECASE
)
STRONG_LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in STRONG_LEGAL_KEYWORDS) + r")\b", re.IGNORECASE
)
SOFT_CHURN_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in SOFT_CHURN_KEYWORDS) + r")\b", re.IGNORECASE)
# The passive "my password was changed" is a takeover report UNLESS the customer
# names what changed it. Round 3: "my password was changed by the iOS update and
# now mail wont sync" escalated as ACCOUNT_COMPROMISE -- wrong action, false
# stated reason. This suppresses only that case, and only when the agent named is
# a thing rather than a person: "changed by someone" is still a compromise.
BENIGN_PASSWORD_CHANGE = re.compile(
    r"\bpassword (?:was|has been) changed\b\s*(?:by|during|after|when|because of|due to)\s+"
    r"(?:the\s+|an?\s+|my\s+)?"
    r"(?:i|me|myself|update|updates|upgrade|ios|ipados|macos|system|reset|restore|backup|migration|"
    r"setup|wizard|it|apple\s+support)\b",
    re.IGNORECASE,
)
_PASSIVE_PASSWORD_FORMS = ("password was changed", "password has been changed")

ANGER_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in ANGER_KEYWORDS) + r")\b", re.IGNORECASE)
EXCLAMATION_PATTERN = re.compile(r"!{2,}|\?{2,}")


class SentimentAnalyzer:
    """Analyzes text for emotional distress, high customer frustration, and legal/churn risk."""

    def __init__(self, threshold: float = FRUSTRATION_THRESHOLD):
        self.threshold = threshold

    def compute_frustration(self, text: str) -> tuple[float, list[str]]:
        """Calculates a normalized customer frustration score [0.0, 1.0] and triggered markers."""
        score = 0.0
        markers = []

        compromise_matches = list({m.lower() for m in ACCOUNT_COMPROMISE_PATTERN.findall(text)})
        # Drop the passive password forms when the customer named a non-human
        # agent for the change. Only those forms are dropped -- any other
        # compromise signal in the same message still counts.
        if BENIGN_PASSWORD_CHANGE.search(text):
            compromise_matches = [m for m in compromise_matches if m not in _PASSIVE_PASSWORD_FORMS]
        strong_matches = list({m.lower() for m in STRONG_LEGAL_PATTERN.findall(text)})
        soft_matches = list({m.lower() for m in SOFT_CHURN_PATTERN.findall(text)})
        anger_matches = list({m.lower() for m in ANGER_PATTERN.findall(text)})
        has_victim_phrase = bool(VICTIM_PHRASE_REGEX.search(text))

        if compromise_matches:
            score += 0.60 + min(0.15, (len(compromise_matches) - 1) * 0.10)
            markers.append(f"ACCOUNT_COMPROMISE: {compromise_matches}")

        if strong_matches:
            score += 0.60 + min(0.20, (len(strong_matches) - 1) * 0.10)
            markers.append(f"STRONG_LEGAL_THREAT: {strong_matches}")

        if soft_matches:
            corroborated = (
                bool(compromise_matches)
                or bool(strong_matches)
                or len(soft_matches) >= 2
                or bool(anger_matches)
                or has_victim_phrase
            )
            if corroborated:
                # The per-extra-signal increment is 0.15, not 0.10, so that the
                # "corroborated by a second soft signal" case actually crosses the
                # threshold. At 0.10 it scored 0.55 against a 0.60 threshold, so
                # this branch emitted a marker literally named
                # CORROBORATED_CHURN_THREAT and then returned False -- "they
                # scammed me and stole my money" and "this is a scam, I am suing"
                # were both auto-handled. The module docstring has claimed since it
                # was written that "another soft signal" corroborates; it did not.
                # The other corroborators (strong, compromise, anger, victim
                # phrase) each add their own weight and always crossed, which is
                # why three review rounds did not catch this one.
                score += 0.45 + min(0.20, (len(soft_matches) - 1) * 0.15)
                markers.append(f"CORROBORATED_CHURN_THREAT: {soft_matches}")
            else:
                # Uncorroborated soft signal (e.g. a lone "sue"/"scam" mention,
                # the exact pattern that false-positived in the audit) -- a
                # small nudge, not enough alone to cross the default threshold.
                score += 0.15
                markers.append(f"UNCORROBORATED_SOFT_SIGNAL_LOW_WEIGHT: {soft_matches}")

        if anger_matches:
            score += 0.35
            markers.append(f"ANGER_KEYWORDS: {anger_matches}")

        if EXCLAMATION_PATTERN.search(text):
            score += 0.12
            markers.append("EXCESSIVE_PUNCTUATION")

        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 15:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.40:
                score += 0.18
                markers.append(f"UPPERCASE_SHOUTING: {round(upper_ratio * 100)}%")

        normalized_score = min(1.0, round(score, 2))
        return normalized_score, markers

    def is_severe_frustration(self, text: str) -> tuple[bool, float, list[str]]:
        """Returns True if frustration exceeds the threshold."""
        score, markers = self.compute_frustration(text)
        return score >= self.threshold, score, markers
