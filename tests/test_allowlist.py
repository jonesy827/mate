"""Caller allowlist: normalization, env parsing, fail-closed matching,
and the worker's refuse-to-start guard."""

import pytest

from matebridge.agent import _require_allowlist
from matebridge.allowlist import (
    ENV_VAR,
    allowed_callers,
    is_allowed,
    normalize_number,
    sip_caller,
)

# --- normalization -----------------------------------------------------

@pytest.mark.parametrize("raw", [
    "+14052790756",
    "+1 (405) 279-0756",
    "14052790756",
    "4052790756",
    " +1 405.279.0756 ",
])
def test_normalize_us_variants(raw):
    assert normalize_number(raw) == "+14052790756"


def test_normalize_international_kept_verbatim():
    assert normalize_number("+61 2 9374 4000") == "+61293744000"


@pytest.mark.parametrize("raw", ["", "  ", "anonymous", "+1405CALLME", "+"])
def test_normalize_garbage_is_empty(raw):
    assert normalize_number(raw) == ""


# --- env parsing -------------------------------------------------------

def test_allowed_callers_parses_and_normalizes():
    env = {ENV_VAR: "+1 405 279 0756, 4055551234,, "}
    assert allowed_callers(env) == {"+14052790756", "+14055551234"}


def test_allowed_callers_empty_or_missing():
    assert allowed_callers({}) == frozenset()
    assert allowed_callers({ENV_VAR: ""}) == frozenset()
    assert allowed_callers({ENV_VAR: " , "}) == frozenset()


# --- matching (fail-closed) --------------------------------------------

ALLOWED = allowed_callers({ENV_VAR: "+14052790756"})


def test_allowed_number_matches_across_formats():
    assert is_allowed("+14052790756", ALLOWED)
    assert is_allowed("14052790756", ALLOWED)
    assert is_allowed("(405) 279-0756", ALLOWED)


def test_unknown_number_blocked():
    assert not is_allowed("+15550001111", ALLOWED)


def test_missing_or_garbage_caller_blocked():
    assert not is_allowed(None, ALLOWED)
    assert not is_allowed("", ALLOWED)
    assert not is_allowed("anonymous", ALLOWED)


def test_empty_allowlist_blocks_everyone():
    assert not is_allowed("+14052790756", frozenset())


def test_garbage_never_matches_garbage():
    # both normalize to "" -- must NOT be treated as equal
    env = {ENV_VAR: "anonymous"}
    assert not is_allowed("anonymous", allowed_callers(env))


# --- SIP attribute extraction ------------------------------------------

def test_sip_caller_reads_attribute():
    assert sip_caller({"sip.phoneNumber": "+14052790756"}) == "+14052790756"


def test_sip_caller_none_for_non_sip():
    assert sip_caller({}) is None
    assert sip_caller(None) is None
    assert sip_caller({"sip.phoneNumber": ""}) is None


# --- worker startup guard ----------------------------------------------

def test_worker_refuses_to_start_without_allowlist():
    with pytest.raises(SystemExit, match=ENV_VAR):
        _require_allowlist(["agent.py", "dev"], env={})
    with pytest.raises(SystemExit, match=ENV_VAR):
        _require_allowlist(["agent.py", "start"], env={ENV_VAR: " "})


def test_worker_starts_with_allowlist():
    _require_allowlist(["agent.py", "dev"], env={ENV_VAR: "+14052790756"})


def test_console_mode_exempt():
    _require_allowlist(["agent.py", "console"], env={})
