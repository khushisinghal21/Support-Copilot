# Decision Log: 15 Non-Obvious Engineering Decisions

**Deliverable 5.** The non-obvious decisions made while building the `@AppleSupport` AI triage/reply agent, and why. Each one follows the same pattern: what was decided, and the concrete reason or evidence behind it -- not just "what I built" but "what I chose over what, and why that choice wins." See `docs/REPORT.md` for the full report and `docs/AUDIT_AND_FIX_PLAN.md` for the detailed audit history behind several of these.

1. **Picked @AppleSupport, not a retail brand.** Consumer electronics support has strict diagnostic procedures, real safety stakes (lithium battery hazards), and clearly defined escalation rules -- a domain where "get it right or escalate" is a meaningful test, not just a UI wrapper around generic replies.

2. **Embedded ChromaDB instead of a hosted vector database.** Keeps the entire benchmark running fully offline, in one command, in well under 15 minutes -- with zero external services to configure, pay for, or have rate-limit an interviewer trying to reproduce the results.

3. **A 9-gate deterministic cascade instead of one LLM safety score.** Prompt injection, hazards, PII, human requests, sentiment, confidence, clarify, similarity, and generation guardrails each run as an explicit, ordered rule -- not a single model call an LLM could hallucinate its way past on a real physical hazard.

4. **Calibrated intent confidence with temperature scaling (T=0.08).** Raw cosine similarity isn't a real probability. Scaling it produces a bounded, calibrated confidence score the triage gate can actually threshold against, instead of an arbitrary similarity number with no defined meaning.

5. **Never let the agent request PII -- and block it from echoing PII the customer already posted.** Most designs only guard the agent's own questions. This one also scans the customer's tweet for PII-shaped text (like a phone number) and blocks a reply that echoes it back -- closing a leak most systems miss entirely.

6. **Retrieval is filtered by classified intent, not open across the whole corpus.** Without this, a battery-drain question could retrieve an iPad-display fix on wording similarity alone. Filtering by intent first keeps every retrieved historical reply topically relevant.

7. **Any unhandled crash fails closed to ESCALATE, never to a silent auto-reply.** A global try/except wraps the entire pipeline. If any component breaks, the ticket routes to a human with reason code `SYSTEM_EXCEPTION_FAIL_CLOSED` -- the customer never receives a reply built from a pipeline that crashed midway.

8. **Rewrote the hazard/PII regex to understand negation, because the naive version had real false positives.** "Battery didn't catch fire" matched a bare `fire` keyword pattern. Added a negation-window check and tighter device-noun proximity rules, verified against real Kaggle tweets the old rule would have wrongly escalated.

9. **Escalation on legal/churn language now requires corroboration, not one keyword.** Manual review of real cases found genuine false positives: "reported a scam **to** Apple" (the customer wasn't the victim), and a support case number that happened to match a phone-number regex. Split triggers into an auto-escalate tier (unambiguous: "lawyer," "police") and a soft tier ("sue," "scam") that needs a second corroborating signal before escalating.

10. **Added a third action, CLARIFY, instead of forcing every ambiguous ticket into AUTO_HANDLE or ESCALATE.** Real @AppleSupport agents sometimes just ask "which device is this?" A confidence-band gate now does the same for device-dependent intents where the customer never named a device -- instead of over-escalating every case that's merely unclear.

11. **Rebuilt the golden set so the system can't be graded against its own logic.** The original labels came from the same logic family the system under test would later be scored against, with a hardcoded agreement floor that meant the eval could never fail. Rebuilt using an independent keyword heuristic for labelling, then manually reviewed every real escalation case -- and disclosed that this heuristic disagreed with the original classifier on **~41% of examples**, an honest signal of how noisy this domain really is.

12. **Removed the hardcoded kappa floors and reported the real, weak agreement number.** The code used to force `max(kappa, 0.72)` regardless of the actual computed value. Removed both floors; the real human-judge kappa is reported honestly even where it's weak (0.07, "Slight agreement") -- an eval harness that can't report a bad result isn't measuring anything.

13. **Whitelisted link domains to `apple.co` and `support.apple.com`.** Any third-party domain in a drafted reply is stripped or flagged before it ships, closing an obvious phishing-shaped failure mode independent of the live link checker below it.

14. **Added a lexical-overlap grounding check because a real embedding model wasn't reliably available in every environment this needs to run.** `check_grounding()` checks that a draft's content words actually overlap with what was retrieved -- a cheap, dependency-free proxy for "did the model just invent this," usable even where a heavier model can't load.

15. **Populated the RAG corpus with real historical @AppleSupport replies, not a hand-written seed set -- with leakage explicitly excluded.** Every `source_tweet_id` present in the golden set is excluded from the corpus, so the system can never retrieve its own answer key during evaluation. Falls back to a small hand-written seed corpus only if the real data file is unavailable.

---

## Addendum (2026-09-10): Root cause of the "apple.co/directmessage" fabrication, found and fixed

The live link checker (`src/drafting/link_checker.py`) correctly caught drafted replies citing `apple.co/directmessage` as broken -- that URL 302-redirects to Apple's generic homepage. Tracing it back, the root cause wasn't the LLM hallucinating on its own: `src/drafting/prompts.py`'s system prompt literally instructed it to "direct them to DM (apple.co/directmessage)." That link was never real -- Twitter/X direct messages require the customer to already be logged in and initiate the DM themselves, so there's no fixed public "click here to DM us" URL to hand out.

**Fix**: rewrote the prompt to ask for a DM in plain text (no URL at all), and restricted "Links" to only ever cite a specific, real support article already present in the retrieved historical context. The link checker stays in place as a safety net for any other fabricated URL, but this removes the one bug that was causing it to fire on nearly every PII/account-escalation case in the first place.

This is the same principle behind several of the 15 decisions above: when a guardrail catches something, trace it to the actual cause -- here, a bad prompt instruction -- rather than treating the catch itself as the fix.

---

## Explicitly Rejected / Out of Scope (2026-09-10)

Decisions made *against* building certain things, kept here for the same reason the 15 decisions above are: so the reasoning isn't lost.

**Rejected by design** (added complexity and cost, with no clear safety win over the current deterministic cascade): a live per-ticket **debate agent** for triage routing, LLM self-reported confidence scores, a fine-tuned intent classifier, a learned meta-model combining multiple signals.

**Tested and rejected, with evidence:** an agentic reformulation loop that retries intent classification on low confidence made accuracy *worse* -- **48.9-50.5% vs. the 52.2% single-pass baseline**. A 3-role **debate agent** as the LLM judge cost **3x the API calls with no measurable accuracy gain** over a single-call judge, so it was simplified back down.

**Safety-driven scope boundaries** (deliberate, not oversights): no tool-calling or real actions -- the agent only drafts and recommends, it never executes a refund, account change, or anything else. No persistent memory across messages -- stateless by design, so one compromised turn can't poison later replies. No data-poisoning defense for the retrieval corpus -- out of scope because the corpus is currently static, not a live-updating vector store, so that attack surface doesn't exist yet.

**Known limitations, deferred, not hidden:** a semantic-similarity fallback for grounding synonyms (a real gap; GloVe embeddings are available, but building this properly was too large to rush into this submission). Full multilingual retrieval (built language-aware BM25 routing; real multilingual embeddings are still missing). Real sentence-transformer embeddings for retrieval (blocked by sandbox network restrictions during development -- a scope/time item, not abandoned). Temperature-calibrated softmax recalibration on intent confidence (would require rebuilding the classifier architecture; the actual calibration bug turned out to be data leakage in the golden set, not softmax temperature, so fixing the real bug made this unnecessary). A higher worker count in the parallel eval runner (more threads than available API keys just queue behind the same rate limit -- no speedup).
