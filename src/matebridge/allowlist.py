"""Caller allowlist — the code-enforced gate on who may phone Mate.

Caller ID is asserted by the originating carrier and can be spoofed, so
this stops strangers who find the DID, not a targeted impersonator (see
the README threat model). It is still the only boundary that matters
against a hostile caller: the confirmation rail protects against lossy
transcription, not against an attacker, because the attacker is the one
saying "yes".

Fail-closed everywhere: no allowlist means no callers, and a number that
cannot be normalized never matches anything.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

ENV_VAR = "MATE_ALLOWED_NUMBERS"

# LiveKit SIP sets this attribute on the inbound participant; its absence
# means the participant is not a phone caller (console mode, playground).
SIP_ATTR = "sip.phoneNumber"

_FORMATTING = re.compile(r"[\s().\-]")


def normalize_number(raw: str) -> str:
    """Fold a dialable number to a canonical +digits form, or "".

    Formatting characters (spaces, dashes, dots, parens) are stripped.
    A bare 11-digit number starting with 1 and a bare 10-digit number are
    assumed to be US (+1...) — write full E.164 in the allowlist to avoid
    relying on that. Anything non-numeric normalizes to "" and can never
    match.
    """
    digits = _FORMATTING.sub("", raw.strip())
    had_plus = digits.startswith("+")
    digits = digits.lstrip("+")
    if not digits.isdigit():
        return ""
    if had_plus:
        return "+" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    return "+" + digits


def allowed_callers(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Parse MATE_ALLOWED_NUMBERS (comma-separated) into normalized form."""
    raw = (os.environ if env is None else env).get(ENV_VAR, "")
    return frozenset(
        n for part in raw.split(",")
        if part.strip() and (n := normalize_number(part))
    )


def is_allowed(caller: str | None, allowed: frozenset[str]) -> bool:
    """True only for a normalizable caller present in a non-empty allowlist."""
    if not caller:
        return False
    n = normalize_number(caller)
    return bool(n) and n in allowed


def sip_caller(attributes: Mapping[str, str] | None) -> str | None:
    """The phone number of a SIP participant, or None for non-SIP ones."""
    return (attributes or {}).get(SIP_ATTR) or None
