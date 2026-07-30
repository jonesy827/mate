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
