"""CallScreen: the caller allowlist and the spoken-passphrase gate as the
entrypoint wires them — who gets dropped, when the lock is armed, and what
the failed-call streak does. Fakes stand in for the room, the LLM session
and the worker kill, so every decision is checked without LiveKit."""

import asyncio

import pytest

from mate.passphrase import (
    ENABLE_VAR,
    MAX_FAILED_CALLS,
    PHRASE_VAR,
    FailedCalls,
)
from mate.screening import CallScreen

pytestmark = pytest.mark.asyncio

PHRASE = "correct horse battery staple"
ALLOWED = frozenset({"+14055551234"})


class FakeParticipant:
    """Duck-types a remote participant: SIP callers carry sip.phoneNumber,
    console/playground participants carry nothing."""

    def __init__(self, number: str | None = None):
        self.attributes = {"sip.phoneNumber": number} if number else {}


class FakeMate:
    MAX_PASSPHRASE_ATTEMPTS = 3

    def __init__(self, locked=False, passed=False):
        self.locked = locked
        self.passphrase_passed = passed
        self.phrase = None
        self.hangup = None
        self.on_unlock = None

    def lock(self, phrase, hangup=None, on_unlock=None):
        self.locked = True
        self.phrase = phrase
        self.hangup = hangup
        self.on_unlock = on_unlock


class Recorder:
    def __init__(self):
        self.deleted = 0
        self.said: list[str] = []
        self.killed = 0
        self.watcher_starts = 0


def make_screen(tmp_path, mate=None, allowed=ALLOWED, delete_raises=False):
    rec = Recorder()

    async def delete_room():
        rec.deleted += 1
        if delete_raises:
            raise RuntimeError("room already gone")

    async def say(text):
        rec.said.append(text)

    def kill():
        rec.killed += 1

    def start_watcher():
        rec.watcher_starts += 1

    failed = FailedCalls(tmp_path / "failed_calls.json")
    screen = CallScreen(mate or FakeMate(), allowed, failed,
                        delete_room=delete_room, say=say, kill=kill,
                        start_watcher=start_watcher)
    return screen, rec, failed


async def drain(screen):
    """Await the fire-and-forget tasks the sync handlers start."""
    while screen._tasks:
        await asyncio.gather(*list(screen._tasks))


@pytest.fixture(autouse=True)
def _gate_env(monkeypatch):
    monkeypatch.setenv(PHRASE_VAR, PHRASE)
    monkeypatch.delenv(ENABLE_VAR, raising=False)


# --- allowlist -------------------------------------------------------------

async def test_allowlisted_caller_is_let_through(tmp_path):
    screen, rec, _ = make_screen(tmp_path)
    ok = await screen.screen_participants([FakeParticipant("+1 405 555 1234")])
    assert ok is True
    assert rec.deleted == 0


async def test_blocked_caller_is_dropped_before_the_session(tmp_path):
    screen, rec, _ = make_screen(tmp_path)
    ok = await screen.screen_participants([FakeParticipant("+15125559999")])
    assert ok is False
    assert rec.deleted == 1


async def test_empty_allowlist_rejects_every_phone_caller(tmp_path):
    # fail closed: no allowlist means no callers
    screen, rec, _ = make_screen(tmp_path, allowed=frozenset())
    assert await screen.screen_participants(
        [FakeParticipant("+14055551234")]) is False
    assert rec.deleted == 1


async def test_non_sip_participants_are_not_screened(tmp_path):
    # console/playground participants already authenticated with a token
    screen, rec, _ = make_screen(tmp_path, allowed=frozenset())
    assert await screen.screen_participants([FakeParticipant()]) is True
    assert rec.deleted == 0


async def test_late_joiner_is_screened_too(tmp_path):
    screen, rec, _ = make_screen(tmp_path)
    screen.screen_late_joiner(FakeParticipant("+15125559999"))
    await drain(screen)
    assert rec.deleted == 1
    screen.screen_late_joiner(FakeParticipant("+14055551234"))
    screen.screen_late_joiner(FakeParticipant())
    await drain(screen)
    assert rec.deleted == 1


async def test_delete_failure_does_not_raise_into_the_job(tmp_path):
    screen, rec, _ = make_screen(tmp_path, delete_raises=True)
    assert await screen.screen_participants(
        [FakeParticipant("+15125559999")]) is False
    assert rec.deleted == 1


# --- arming the passphrase gate -------------------------------------------

async def test_gate_arms_for_a_sip_caller_already_in_the_room(tmp_path):
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    assert await screen.arm_initial_callers(
        [FakeParticipant("+14055551234")]) is True
    assert mate.locked is True
    assert mate.phrase == PHRASE
    assert mate.hangup == screen.hangup_failed_caller
    assert mate.on_unlock == screen.unlocked
    assert rec.deleted == 0


async def test_non_sip_session_is_never_gated(tmp_path):
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    assert await screen.arm_initial_callers([FakeParticipant()]) is True
    assert mate.locked is False
    assert rec.deleted == 0


async def test_gate_switched_off_leaves_the_call_open(tmp_path, monkeypatch):
    monkeypatch.setenv(ENABLE_VAR, "0")
    mate = FakeMate()
    screen, _rec, _ = make_screen(tmp_path, mate)
    assert await screen.arm_initial_callers(
        [FakeParticipant("+14055551234")]) is True
    assert mate.locked is False


async def test_no_configured_phrase_rejects_the_call(tmp_path, monkeypatch):
    # fail closed: the gate is on but there is nothing to check against
    monkeypatch.delenv(PHRASE_VAR, raising=False)
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    assert await screen.arm_initial_callers(
        [FakeParticipant("+14055551234")]) is False
    assert mate.locked is False
    assert rec.deleted == 1
    assert rec.killed == 0


async def test_tripped_streak_refuses_the_call_and_kills_the_worker(tmp_path):
    mate = FakeMate()
    screen, rec, failed = make_screen(tmp_path, mate)
    for _ in range(MAX_FAILED_CALLS):
        failed.record_failure()
    assert await screen.arm_initial_callers(
        [FakeParticipant("+14055551234")]) is False
    assert mate.locked is False
    assert rec.deleted == 1
    assert rec.killed == 1


# --- late SIP joiner ------------------------------------------------------

async def test_late_sip_joiner_is_gated_and_prompted_after_the_greeting(
        tmp_path):
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    screen.mark_greeted()
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert mate.locked is True
    assert rec.said == ["G'day. What's the passphrase?"]


async def test_late_joiner_before_the_greeting_arms_but_stays_quiet(tmp_path):
    # the greeting itself is about to ask for the passphrase
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert mate.locked is True
    assert rec.said == []


async def test_late_joiner_leaves_verified_or_locked_sessions_alone(tmp_path):
    passed = FakeMate(passed=True)
    screen, rec, _ = make_screen(tmp_path, passed)
    screen.mark_greeted()
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert rec.said == [] and passed.locked is False

    mid = FakeMate(locked=True)
    screen, rec, _ = make_screen(tmp_path, mid)
    screen.mark_greeted()
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert rec.said == []  # already mid-prompt, don't re-ask

    non_sip = FakeMate()
    screen, rec, _ = make_screen(tmp_path, non_sip)
    screen.gate_late_sip_joiner(FakeParticipant())
    await drain(screen)
    assert non_sip.locked is False


async def test_late_joiner_on_a_tripped_streak_is_refused(tmp_path):
    mate = FakeMate()
    screen, rec, failed = make_screen(tmp_path, mate)
    for _ in range(MAX_FAILED_CALLS):
        failed.record_failure()
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert mate.locked is False
    assert rec.deleted == 1 and rec.killed == 1


async def test_late_joiner_without_a_phrase_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(PHRASE_VAR, raising=False)
    mate = FakeMate()
    screen, rec, _ = make_screen(tmp_path, mate)
    screen.gate_late_sip_joiner(FakeParticipant("+14055551234"))
    await drain(screen)
    assert mate.locked is False
    assert rec.deleted == 1 and rec.killed == 0


# --- streak semantics -----------------------------------------------------

async def test_unlock_resets_the_streak_and_starts_the_watcher(tmp_path):
    screen, rec, failed = make_screen(tmp_path)
    failed.record_failure()
    failed.record_failure()
    screen.unlocked()  # what Mate calls the moment the phrase is accepted
    assert failed.count() == 0  # an authenticated call ends the streak
    assert rec.watcher_starts == 1


async def test_hangup_records_a_failure_without_killing_early(tmp_path):
    screen, rec, failed = make_screen(tmp_path)
    await screen.hangup_failed_caller()
    assert rec.deleted == 1
    assert failed.count() == 1
    assert rec.killed == 0


async def test_hangup_kills_the_worker_on_the_last_failure(tmp_path):
    screen, rec, failed = make_screen(tmp_path)
    for _ in range(MAX_FAILED_CALLS - 1):
        failed.record_failure()
    await screen.hangup_failed_caller()
    assert failed.count() == MAX_FAILED_CALLS
    assert rec.killed == 1


async def test_hangup_kills_even_if_the_room_delete_fails(tmp_path):
    # the streak matters more than a tidy hangup
    screen, rec, failed = make_screen(tmp_path, delete_raises=True)
    for _ in range(MAX_FAILED_CALLS - 1):
        failed.record_failure()
    await screen.hangup_failed_caller()
    assert rec.killed == 1
