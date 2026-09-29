"""Who is allowed to talk to Mate: the caller allowlist and the spoken
passphrase gate, plus the failed-call lockout that shuts the worker down.

Split out of agent.py's entrypoint() so every decision here is reachable
from unit tests. Nothing in this module imports LiveKit: the room actions
arrive as injected callables — delete_room() to drop the call, say(text)
to speak, kill() to bring the worker down, start_watcher() to let the
fleet watcher run once a caller is verified.
"""

import asyncio
import logging

from .allowlist import ENV_VAR, is_allowed, sip_caller
from .passphrase import (
    MAX_FAILED_CALLS,
    PHRASE_VAR,
    configured_phrase,
    passphrase_required,
)

logger = logging.getLogger("mate")


class CallScreen:
    """Caller gate. Only SIP participants (they carry sip.phoneNumber) are
    screened — console/playground participants already authenticated with a
    LiveKit token. Fail-closed: an empty allowlist rejects every phone
    caller. This is the only boundary against a hostile caller; the
    confirmation rail guards against transcription error, not attackers.

    The spoken-passphrase gate is the second factor on top of the allowlist,
    since caller ID can be spoofed. Only SIP callers are gated — console and
    playground participants already authenticated with a LiveKit token.
    Verification runs in code on the raw transcript
    (Mate.on_user_turn_completed); the LLM never sees a locked turn. While
    locked the fleet watcher does not run either — no fleet announcements to
    an unverified caller.
    """

    def __init__(self, mate, allowed, failed_calls, *, delete_room, say,
                 kill, start_watcher):
        self.mate = mate
        self.allowed = allowed
        self.failed_calls = failed_calls
        self._delete_room = delete_room
        self._say = say
        self._kill = kill
        self._start_watcher = start_watcher
        # flipped once the deterministic greeting has been spoken: before
        # that, a joining caller hears the greeting itself and must not be
        # prompted on top of it
        self.greeted = False
        # strong refs to fire-and-forget rejects/prompts started from the
        # (synchronous) participant_connected handlers
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def mark_greeted(self) -> None:
        self.greeted = True

    # ---- allowlist ------------------------------------------------------

    async def reject_call(self, caller: str) -> None:
        logger.warning("blocked call from %s: not in %s", caller, ENV_VAR)
        try:
            await self._delete_room()
        except Exception:
            logger.exception("could not delete room for blocked caller")

    async def screen_participants(self, participants) -> bool:
        """Screen the participants already in the room. False means the call
        was rejected and the job must stop.

        The SIP caller is normally already in the room when the job starts
        (the dispatch rule created the room for them) — reject before the
        session ever opens its mouth. screen_late_joiner covers stragglers.
        """
        for p in list(participants):
            caller = sip_caller(p.attributes)
            if caller is not None and not is_allowed(caller, self.allowed):
                await self.reject_call(caller)
                return False
        return True

    def screen_late_joiner(self, participant) -> None:
        caller = sip_caller(participant.attributes)
        if caller is not None and not is_allowed(caller, self.allowed):
            self._spawn(self.reject_call(caller))

    # ---- passphrase gate ------------------------------------------------

    def unlocked(self) -> None:
        self.failed_calls.reset()  # an authenticated call ends the streak
        self._start_watcher()

    async def hangup_failed_caller(self) -> None:
        logger.warning("caller failed the passphrase %d times — "
                       "hanging up", self.mate.MAX_PASSPHRASE_ATTEMPTS)
        try:
            await self._delete_room()
        except Exception:
            logger.exception("could not hang up failed-passphrase caller")
        streak = self.failed_calls.record_failure()
        if streak >= MAX_FAILED_CALLS:
            logger.critical("%d calls in a row failed the passphrase — "
                            "someone is guessing", streak)
            self._kill()

    async def reject_no_phrase(self) -> None:
        # unreachable when launched via __main__ (ensure_launch_phrase),
        # but fail closed if the gate is on with nothing to check against
        logger.error("%s required but not set — rejecting call", PHRASE_VAR)
        try:
            await self._delete_room()
        except Exception:
            logger.exception("could not delete room (no passphrase set)")

    async def reject_tripped(self) -> None:
        # the streak tripped but this worker is somehow still taking calls
        # (shutdown not landed yet, or the signal failed) — refuse the call
        # and pull the plug again
        logger.critical("%d calls in a row failed the passphrase — "
                        "refusing call and shutting down",
                        self.failed_calls.count())
        try:
            await self._delete_room()
        except Exception:
            logger.exception("could not delete room (failure lockout)")
        self._kill()

    def arm_lock(self) -> bool:
        """Arm the gate. False means no phrase is configured — the caller
        must be rejected (fail closed)."""
        phrase = configured_phrase()
        if not phrase:
            return False
        self.mate.lock(phrase, self.hangup_failed_caller,
                       on_unlock=self.unlocked)
        return True

    async def arm_initial_callers(self, participants) -> bool:
        """Arm the gate for a SIP caller already in the room. False means the
        call was rejected and the job must stop."""
        sip_present = any(sip_caller(p.attributes) is not None
                          for p in participants)
        if sip_present and passphrase_required():
            if self.failed_calls.count() >= MAX_FAILED_CALLS:
                await self.reject_tripped()
                return False
            if not self.arm_lock():
                await self.reject_no_phrase()
                return False
        return True

    def gate_late_sip_joiner(self, participant) -> None:
        # the allowlist's participant_connected handler covers stragglers;
        # this one gives the same stragglers the passphrase gate. Already
        # unlocked-by-phrase (or mid-prompt) sessions are left alone.
        if sip_caller(participant.attributes) is None:
            return
        if (not passphrase_required() or self.mate.passphrase_passed
                or self.mate.locked):
            return
        if self.failed_calls.count() >= MAX_FAILED_CALLS:
            self._spawn(self.reject_tripped())
            return
        if not self.arm_lock():
            self._spawn(self.reject_no_phrase())
            return
        logger.info("SIP caller joined mid-session — arming passphrase gate")
        if self.greeted:
            self._spawn(self._say("G'day. What's the passphrase?"))
