"""Fleet watching: which panes were delegated work this call, and the
proactive spoken interjections when one of them blocks or finishes.

Split out of agent.py's entrypoint() so the announce/catch-up/resubscribe
decisions are reachable from unit tests: FleetWatcher takes an injected
`say` coroutine (production hands it session.say, tests a recorder), so
nothing here imports LiveKit.
"""

import asyncio
import logging
import time

from .herdr_client import HerdrError

logger = logging.getLogger("mate")

# pane.agent_status_changed subscriptions are per-pane, so the fleet watcher
# resubscribes on this interval to pick up panes created since (spawn_task
# and friends). Also bounds the catch-up latency for events missed while
# between subscriptions.
WATCH_RESUBSCRIBE_SECS = 30.0


class Delegations:
    """Panes handed work this call, with enough state to tell a real finish
    from the delivery race: right after prompt_agent returns, herdr is still
    typing the prompt into the pane (herdr 0.8.0 delays its own Enter past
    Claude Code's paste guard, #1878), so the pane sits idle for a couple of
    seconds — idle+delegated alone must NOT count as finished, or the watcher
    announces a stale reply and permanently eats the real notification.

    A pane is finish_ready once it has been seen working. If it is never seen
    working within GRACE_SECS, the task was quick enough that every
    observation missed the working state — idle counts as finished."""

    GRACE_SECS = 20.0

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._panes: dict[str, dict] = {}

    def add(self, pane_id: str) -> None:
        self._panes[pane_id] = {"at": self._clock(), "started": False}

    def discard(self, pane_id: str) -> None:
        self._panes.pop(pane_id, None)

    def mark_started(self, pane_id: str) -> None:
        entry = self._panes.get(pane_id)
        if entry:
            entry["started"] = True

    def finish_ready(self, pane_id: str) -> bool:
        entry = self._panes.get(pane_id)
        if entry is None:
            return False
        if entry["started"]:
            return True
        return self._clock() - entry["at"] >= self.GRACE_SECS

    def __contains__(self, pane_id: str) -> bool:
        return pane_id in self._panes

    def __iter__(self):
        return iter(self._panes)


class FleetWatcher:
    """Proactive interjection when an agent blocks, or when a pane Mate
    delegated to finishes (kills the "let me check again" loop).

    `say` is an async callable(text): production passes a wrapper over
    session.say, tests pass a recorder. It is only ever called with
    code-composed sentences — the LLM is not in this loop.
    """

    def __init__(self, herdr, mate, say):
        self.herdr = herdr
        self.mate = mate
        self._say_out = say

    async def say(self, text: str) -> None:
        logger.info("watch_fleet: announcing: %s", text)
        await self._say_out(text)
        logger.info("watch_fleet: announcement spoken")

    async def announce(self, status, pane_id, data) -> None:
        mate = self.mate
        if status == "working":
            if pane_id in mate.delegated:
                mate.delegated.mark_started(pane_id)
                logger.info("watch_fleet: delegated pane %s started "
                            "working", pane_id)
            return
        if status == "blocked":
            label = await mate._pane_label(pane_id)
            await self.say(f"Hey mate, the {label} agent is waiting on your "
                           "approval. Want me to read its question?")
        elif (status in ("idle", "done")
              and pane_id in mate.delegated):
            if not mate.delegated.finish_ready(pane_id):
                # delivery race: the pane was never seen working after
                # delivery. Early on, herdr is still typing the prompt;
                # announcing now would read a STALE reply and discard the
                # pane, eating the real notification later.
                logger.info("watch_fleet: pane %s idle but not "
                            "finish-ready yet, skipping", pane_id)
                return
            mate.delegated.discard(pane_id)  # one announcement per task
            # one snapshot for both lookups: the reply and the label are
            # read from the same session.snapshot instead of two round
            # trips a moment apart. On a failed fetch each falls back to
            # its own (which reports the failure in its own way).
            try:
                snap = await self.herdr.snapshot()
            except HerdrError:
                snap = None
            # Fetch the reply here: asking qwen to call agent_report from
            # an injected instruction doesn't work reliably.
            reply = await mate._agent_replies(pane_id, snap=snap)
            label = await mate._pane_label(pane_id, snap=snap)
            # imported here, not at module scope: agent.py imports this
            # module for Delegations/FleetWatcher, so a top-level import
            # back into it would be a cycle. Shaping speech stays a speech
            # concern and lives in agent.py.
            from .agent import tts_summary
            if reply.startswith("ERROR"):
                logger.warning("watch_fleet: finish on %s but reply "
                               "unreadable: %s", pane_id, reply)
                await self.say(f"Hey mate, the {label} agent just finished, "
                               "but I couldn't read its reply. Want me to "
                               "read its screen instead?")
            else:
                await self.say(f"Hey mate, the {label} agent just finished. "
                               f"{tts_summary(reply)} Want the full report?")

    async def run(self):
        # herdr's pane.agent_status_changed subscription is PER-PANE: a bare
        # {"type": ...} is rejected with invalid_request (missing pane_id).
        # So each cycle: snapshot -> subscribe to every current pane ->
        # resubscribe every WATCH_RESUBSCRIBE_SECS to pick up new panes.
        # The snapshot doubles as a catch-up pass for delegated panes that
        # finished while we weren't subscribed.
        while True:
            try:
                snap = await self.herdr.snapshot()
                # catch-up: delegated panes that changed state between
                # subscriptions (idle/done only -- re-announcing "blocked"
                # every cycle would nag; "working" just marks started)
                for a in snap.get("agents", []):
                    if (a.get("pane_id") in self.mate.delegated
                            and a.get("agent_status")
                            in ("idle", "done", "working")):
                        await self.announce(a["agent_status"], a["pane_id"], a)
                pane_ids = [p["pane_id"] for p in snap.get("panes", [])
                            if p.get("pane_id")]
                logger.info("watch_fleet: cycle: %d panes, delegated=%s",
                            len(pane_ids), list(self.mate.delegated))
                if not pane_ids:
                    await asyncio.sleep(WATCH_RESUBSCRIBE_SECS)
                    continue
                events = self.herdr.events(
                    [{"type": "pane.agent_status_changed", "pane_id": p}
                     for p in pane_ids])
                loop = asyncio.get_running_loop()
                deadline = loop.time() + WATCH_RESUBSCRIBE_SECS
                try:
                    while (remaining := deadline - loop.time()) > 0:
                        # only the WAIT is under a timeout -- announce() can
                        # hold the floor for 10+ s of TTS and must never be
                        # cancelled mid-speech by the resubscribe deadline
                        try:
                            msg = await asyncio.wait_for(
                                anext(events), remaining)
                        except (TimeoutError, StopAsyncIteration):
                            break
                        data = msg.get("data", {})
                        logger.info("watch_fleet: event pane=%s status=%s",
                                    data.get("pane_id"),
                                    data.get("agent_status"))
                        await self.announce(data.get("agent_status"),
                                            data.get("pane_id"), data)
                finally:
                    await events.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("watch_fleet: watcher error, retrying in 5s")
                await asyncio.sleep(5)
