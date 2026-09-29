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
# "okay" is the most common spoken yes on the phone; it is safe to accept
# only because a veto anywhere in the utterance still wins ("no okay wait"
# blocks). That is also why VETO_PHRASES exists: the hesitations people
# say on a phone ride along with okay/sure and contain no veto WORD at all
# ("okay, hang on", "sure, one second"). Without them, accepting okay/sure
# would turn a pause into a delivery.
AFFIRM_WORDS = {"yes", "yeah", "yep", "send", "confirm", "correct",
                "okay", "ok", "sure"}
AFFIRM_PHRASES = (("go", "ahead"), ("do", "it"))
VETO_WORDS = {"no", "dont", "stop", "wait", "cancel", "hold", "change",
              "not", "hang", "never", "nevermind", "scrap"}
VETO_PHRASES = (("one", "sec"), ("one", "second"), ("one", "minute"),
                ("give", "me"))


def _has_phrase(words: list[str], phrases: tuple) -> bool:
    return any(words[i:i + len(p)] == list(p)
               for p in phrases for i in range(len(words)))


def approves_send(transcript: str) -> bool:
    """True only if the utterance clearly approves sending: at least one
    affirmative and no veto word or phrase (word-boundary match,
    apostrophes normalized so don't == dont)."""
    normalized = transcript.lower().replace("’", "'").replace("'", "")
    words = re.findall(r"[a-z]+", normalized)
    if set(words) & VETO_WORDS or _has_phrase(words, VETO_PHRASES):
        return False
    if set(words) & AFFIRM_WORDS:
        return True
    return _has_phrase(words, AFFIRM_PHRASES)
