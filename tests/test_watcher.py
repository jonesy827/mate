"""FleetWatcher: what gets spoken when a pane blocks or a delegated pane
goes idle, the delivery-race skip, and one full run() cycle (snapshot
catch-up -> per-pane subscribe -> event announce)."""

import asyncio

import pytest

from mate import watcher as watcher_mod
from mate.herdr_client import HerdrError
from mate.watcher import WATCH_RESUBSCRIBE_SECS, Delegations, FleetWatcher

pytestmark = pytest.mark.asyncio


class FakeMate:
    """Only what the watcher touches: the delegated set and the two async
    lookups it calls to build a sentence."""

    def __init__(self, clock=None, reply="All done. I fixed the parser."):
        self.delegated = (Delegations(clock) if clock is not None
                          else Delegations())
        self.reply = reply
        self.labels: dict[str, str] = {}
        # snapshots the watcher handed to the two lookups (one fetch, both
        # calls get the same object)
        self.snaps: list = []

    async def _pane_label(self, pane_id, snap=None):
        self.snaps.append(snap)
        return self.labels.get(pane_id, "coding")

    async def _agent_replies(self, pane_id, messages: int = 1, snap=None):
        self.snaps.append(snap)
        return self.reply


class FakeHerdr:
    def __init__(self, snapshots=None, events_msgs=()):
        self.snapshots = list(snapshots or [])
        self._scripted = bool(self.snapshots)
        self.events_msgs = list(events_msgs)
        self.subscriptions: list[list[dict]] = []
        self.closed = 0
        self.snapshot_calls = 0

    async def snapshot(self):
        self.snapshot_calls += 1
        if self.snapshots:
            return self.snapshots.pop(0)
        if self._scripted:
            # the watcher loop only stops when cancelled; a run() test drives
            # it for exactly as many cycles as it has snapshots for
            raise asyncio.CancelledError
        return {"agents": [], "panes": []}

    def events(self, subscriptions):
        self.subscriptions.append(list(subscriptions))
        return self._stream()

    async def _stream(self):
        try:
            for msg in self.events_msgs:
                yield msg
        finally:
            self.closed += 1


def make_watcher(mate=None, herdr=None):
    said: list[str] = []

    async def say(text):
        said.append(text)

    mate = mate or FakeMate()
    herdr = herdr or FakeHerdr()
    return FleetWatcher(herdr, mate, say), said, mate, herdr


# --- announce -------------------------------------------------------------

async def test_blocked_pane_prompts_the_user():
    mate = FakeMate()
    mate.labels["w1:p1"] = "mate"
    watcher, said, _, _ = make_watcher(mate)
    await watcher.announce("blocked", "w1:p1", {})
    assert said == ["Hey mate, the mate agent is waiting on your approval. "
                    "Want me to read its question?"]


async def test_blocked_announces_even_for_undelegated_panes():
    # a pane nobody delegated to can still be stuck on an approval prompt
    watcher, said, mate, _ = make_watcher()
    assert "w9:p9" not in mate.delegated
    await watcher.announce("blocked", "w9:p9", {})
    assert len(said) == 1


async def test_delegated_finish_is_announced_with_a_summary():
    mate = FakeMate(reply="Fixed the parser. Tests pass. Third sentence.")
    mate.labels["w1:p1"] = "herdr"
    watcher, said, _, _ = make_watcher(mate)
    mate.delegated.add("w1:p1")
    mate.delegated.mark_started("w1:p1")
    await watcher.announce("idle", "w1:p1", {})
    assert said == ["Hey mate, the herdr agent just finished. Fixed the "
                    "parser. Tests pass. Want the full report?"]
    # one announcement per task: the pane leaves the delegated set
    assert "w1:p1" not in mate.delegated


async def test_finish_announcement_takes_one_snapshot_for_both_lookups():
    mate = FakeMate()
    watcher, said, _, herdr = make_watcher(mate)
    mate.delegated.add("w1:p1")
    mate.delegated.mark_started("w1:p1")
    await watcher.announce("idle", "w1:p1", {})
    assert herdr.snapshot_calls == 1
    # the reply and the label were read from the very same snapshot
    assert mate.snaps[0] is not None
    assert mate.snaps[0] is mate.snaps[1]
    assert "just finished" in said[0]


async def test_finish_announcement_survives_a_failed_snapshot():
    class NoSnapshotHerdr(FakeHerdr):
        async def snapshot(self):
            raise HerdrError("session.snapshot", "io", "socket gone")

    mate = FakeMate()
    watcher, said, _, _ = make_watcher(mate, NoSnapshotHerdr())
    mate.delegated.add("w1:p1")
    mate.delegated.mark_started("w1:p1")
    await watcher.announce("idle", "w1:p1", {})
    # no shared snapshot: each lookup falls back to fetching its own
    assert mate.snaps == [None, None]
    assert "just finished" in said[0]


async def test_unreadable_reply_offers_the_screen_instead():
    mate = FakeMate(reply="ERROR: no transcript adapter for codex agents")
    watcher, said, _, _ = make_watcher(mate)
    mate.delegated.add("w1:p1")
    mate.delegated.mark_started("w1:p1")
    await watcher.announce("done", "w1:p1", {})
    assert "couldn't read its reply" in said[0]
    assert "w1:p1" not in mate.delegated


async def test_idle_before_finish_ready_is_skipped():
    # the delivery race: herdr is still typing the prompt, so this idle is
    # the pre-start state, not a finish
    mate = FakeMate()
    watcher, said, _, _ = make_watcher(mate)
    mate.delegated.add("w1:p1")
    await watcher.announce("idle", "w1:p1", {})
    assert said == []
    assert "w1:p1" in mate.delegated  # still owed a real announcement


async def test_grace_expiry_counts_a_never_started_pane_as_finished():
    # herdr 0.8.0 owns prompt submission (#1878), so a pane that never shows
    # working within the grace window is a task quick enough that every
    # observation missed it — announce it, no Enter nudge in between
    clock = [100.0]
    mate = FakeMate(clock=lambda: clock[0])
    watcher, said, _, _ = make_watcher(mate)
    mate.delegated.add("w1:p1")
    clock[0] += Delegations.GRACE_SECS + 1
    await watcher.announce("idle", "w1:p1", {})
    assert "just finished" in said[0]
    assert "w1:p1" not in mate.delegated


async def test_working_marks_started_and_says_nothing():
    watcher, said, mate, _ = make_watcher()
    mate.delegated.add("w1:p1")
    await watcher.announce("working", "w1:p1", {})
    assert said == []
    assert mate.delegated.finish_ready("w1:p1")


async def test_idle_pane_that_was_never_delegated_is_ignored():
    watcher, said, _, _ = make_watcher()
    await watcher.announce("idle", "w1:p1", {})
    assert said == []


# --- run(): catch-up, subscribe, event announce ---------------------------

async def test_run_catches_up_from_the_snapshot_then_reads_events():
    mate = FakeMate()
    snap = {
        "agents": [
            # delegated and finished while we were between subscriptions
            {"pane_id": "w1:p1", "agent_status": "idle"},
            # not delegated: no catch-up announcement
            {"pane_id": "w2:p1", "agent_status": "idle"},
        ],
        "panes": [{"pane_id": "w1:p1"}, {"pane_id": "w2:p1"}, {}],
    }
    event = {"event": "pane.agent_status_changed",
             "data": {"pane_id": "w2:p1", "agent_status": "blocked"}}
    # two snapshots: the cycle's own, then the one announce() fetches once
    # and shares between the reply and the label lookups
    herdr = FakeHerdr(snapshots=[snap, snap], events_msgs=[event])
    watcher, said, _, _ = make_watcher(mate, herdr)
    mate.delegated.add("w1:p1")
    mate.delegated.mark_started("w1:p1")

    with pytest.raises(asyncio.CancelledError):
        await watcher.run()

    assert "just finished" in said[0]          # catch-up pass
    assert "waiting on your approval" in said[1]  # live event
    # subscriptions are PER-PANE (a bare type is rejected by herdr), and
    # panes without a pane_id are skipped
    assert herdr.subscriptions == [[
        {"type": "pane.agent_status_changed", "pane_id": "w1:p1"},
        {"type": "pane.agent_status_changed", "pane_id": "w2:p1"},
    ]]
    assert herdr.closed == 1


async def test_run_resubscribes_to_pick_up_panes_spawned_mid_call(monkeypatch):
    # the reason WATCH_RESUBSCRIBE_SECS exists: subscriptions are PER-PANE,
    # so a pane spawned after the subscribe is invisible until the deadline
    # expires and the next cycle subscribes to the fresh pane list. The
    # stream here never ends on its own, so the cycle can only be ended by
    # that deadline -- the path a live herdr always takes.
    class LiveStreamHerdr(FakeHerdr):
        async def _stream(self):
            try:
                for msg in self.events_msgs:
                    yield msg
                await asyncio.Event().wait()  # a real stream just stays open
            finally:
                self.closed += 1

    monkeypatch.setattr(watcher_mod, "WATCH_RESUBSCRIBE_SECS", 0.01)
    first = {"agents": [], "panes": [{"pane_id": "w1:p1"}]}
    second = {"agents": [], "panes": [{"pane_id": "w1:p1"},
                                      {"pane_id": "w3:p1"}]}
    herdr = LiveStreamHerdr(snapshots=[first, second])
    watcher, _said, _, _ = make_watcher(None, herdr)

    with pytest.raises(asyncio.CancelledError):
        # the outer wait_for is the regression net: a watcher that stops
        # resubscribing would sit on the live stream forever, and this
        # fails it in seconds instead of hanging the suite
        await asyncio.wait_for(watcher.run(), 5)

    assert herdr.subscriptions == [
        [{"type": "pane.agent_status_changed", "pane_id": "w1:p1"}],
        [{"type": "pane.agent_status_changed", "pane_id": "w1:p1"},
         {"type": "pane.agent_status_changed", "pane_id": "w3:p1"}],
    ]
    assert herdr.closed == 2  # each stream closed before resubscribing


async def test_run_sleeps_when_the_fleet_has_no_panes(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(secs):
        slept.append(secs)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    herdr = FakeHerdr(snapshots=[{"agents": [], "panes": []}])
    watcher, _said, _, _ = make_watcher(None, herdr)
    with pytest.raises(asyncio.CancelledError):
        await watcher.run()
    assert slept == [WATCH_RESUBSCRIBE_SECS]
    assert herdr.subscriptions == []


async def test_run_retries_after_a_herdr_error(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(secs):
        slept.append(secs)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    class FlakyHerdr(FakeHerdr):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def snapshot(self):
            self.calls += 1
            if self.calls == 1:
                raise HerdrError("session.snapshot", "io", "socket gone")
            raise asyncio.CancelledError

    watcher, _, _, _ = make_watcher(None, FlakyHerdr())
    with pytest.raises(asyncio.CancelledError):
        await watcher.run()
    assert slept == [5]
