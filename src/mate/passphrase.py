"""Spoken-passphrase gate for phone callers.

Caller ID can be spoofed, so the allowlist alone is weak proof of who is
on the line. With MATE_REQUIRE_PASSPHRASE on (the default), a SIP caller
must speak the four-word MATE_PASSPHRASE before Mate will do anything;
the check runs in code on the raw STT transcript — the turn never reaches
the LLM while locked. Matching squashes both sides to lowercase letters
and digits, so "Correct Horse battery-staple" survives casing, hyphens
and STT spacing quirks, but all four words must come back, in order.
"""

import os
import re
import sys

ENABLE_VAR = "MATE_REQUIRE_PASSPHRASE"
PHRASE_VAR = "MATE_PASSPHRASE"
WORD_COUNT = 4

_OFF = {"0", "false", "no", "off"}


def passphrase_required(env=None) -> bool:
    """The gate is on unless MATE_REQUIRE_PASSPHRASE is explicitly off."""
    raw = (env if env is not None else os.environ).get(ENABLE_VAR, "")
    return raw.strip().lower() not in _OFF


def configured_phrase(env=None) -> str:
    return ((env if env is not None else os.environ)
            .get(PHRASE_VAR, "")).strip()


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def phrase_heard(transcript: str, phrase: str) -> bool:
    """True only if the whole passphrase occurs in the transcript: every
    word, in order, contiguous once punctuation/spacing is squashed away.
    An empty phrase never unlocks."""
    want = _squash(phrase)
    return bool(want) and want in _squash(transcript)


def ensure_launch_phrase(argv: list[str], env=None) -> None:
    """Refuse to start a phone-facing worker (dev/start) without a valid
    four-word passphrase, unless the gate is switched off. On a terminal,
    prompts for one instead of dying; the phrase then holds for this run
    (dev-mode job processes inherit it) — add it to .env to make it stick.
    console mode has no SIP path and is exempt."""
    target = env if env is not None else os.environ
    if not {"dev", "start"} & set(argv[1:]):
        return
    if not passphrase_required(target):
        return
    raw = (target.get(PHRASE_VAR) or "").strip()
    prompted = False
    while len(raw.split()) != WORD_COUNT:
        problem = (f"{PHRASE_VAR} must be exactly {WORD_COUNT} words "
                   f"(got {len(raw.split())})." if raw
                   else f"{PHRASE_VAR} is not set.")
        if not sys.stdin.isatty():
            raise SystemExit(
                f"{problem} Callers must speak this phrase before Mate "
                f"will act. Put {WORD_COUNT} words in .env, e.g.\n"
                f'  {PHRASE_VAR}="correct horse battery staple"\n'
                f"or set {ENABLE_VAR}=0 to run without the gate.")
        try:
            raw = input(f"{problem} Enter {WORD_COUNT} words callers must "
                        "speak: ").strip()
        except EOFError:
            raise SystemExit(problem) from None
        prompted = True
    target[PHRASE_VAR] = raw
    if prompted:
        print(f"Using that phrase for this run. Add it to .env to keep it:\n"
              f'  {PHRASE_VAR}="{raw}"')
