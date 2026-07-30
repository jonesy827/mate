"""Safety rails: what Mate refuses to approve without explicit confirmation."""

import re

DESTRUCTIVE = re.compile(
    r"force-push|--force\b|\bTRUNCATE\b|DROP\s+TABLE|"
    r"terraform\s+(apply|destroy)|\bprod(uction)?\b|\brm\s+-rf?\b|"
    r"git\s+reset\s+--hard",
    re.IGNORECASE)


def is_destructive(pane_text: str) -> bool:
    """True if the pending on-screen action looks dangerous enough to require
    reading it back to the user verbatim before sending approval."""
    return bool(DESTRUCTIVE.search(pane_text))


# Deterministic yes/no check on the user's spoken reply before a staged
# message is delivered to an agent. Veto always outranks affirmative:
# "no, don't send it" contains "send" and MUST block. No affirmative at
# all blocks too -- the only failure mode is one extra round trip.
AFFIRM_WORDS = {"yes", "yeah", "yep", "send", "confirm", "correct"}
AFFIRM_PHRASES = (("go", "ahead"), ("do", "it"))
VETO_WORDS = {"no", "dont", "stop", "wait", "cancel", "hold", "change", "not"}


def approves_send(transcript: str) -> bool:
    """True only if the utterance clearly approves sending: at least one
    affirmative and zero veto words (word-boundary match, apostrophes
    normalized so don't == dont)."""
    normalized = transcript.lower().replace("’", "'").replace("'", "")
    words = re.findall(r"[a-z]+", normalized)
    if set(words) & VETO_WORDS:
        return False
    if set(words) & AFFIRM_WORDS:
        return True
    return any(words[i:i + len(p)] == list(p)
               for p in AFFIRM_PHRASES for i in range(len(words)))
