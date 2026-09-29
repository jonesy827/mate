"""Watcher-announcement plumbing: TTS sanitizing/summarizing, the guardrails
voice toggle, the delivery-race guard in Delegations, and the guardrails-off
immediate-delivery path."""

import asyncio

import pytest

from mate.agent import (
    Mate,
    detect_rail_toggle,
    tts_sanitize,
    tts_summary,
)
from mate.watcher import Delegations, FleetWatcher

pytestmark = pytest.mark.asyncio


# --- tts_sanitize / tts_summary -------------------------------------------

def test_sanitize_strips_markdown_and_code():
    text = ("## Done\n\n- Fixed the **bug** in `agent.py`\n"
            "```python\nprint('hi')\n```\n"
            "See [the docs](https://example.com) for more. 🎉")
    out = tts_sanitize(text)
    assert "```" not in out and "print" not in out
    assert "**" not in out and "`" not in out and "#" not in out
    assert "https://" not in out and "the docs" in out
    assert "🎉" not in out
    assert "Fixed the bug in agent.py" in out


def test_sanitize_transliterates_accents():
    # the ascii filter used to eat the accented letter itself ("naive" came
    # out as "nave"); NFKD + dropping combining marks keeps the letter
    out = tts_sanitize("The naïve café résumé from Zoë — done.")
    assert "naive cafe resume" in out
    assert "Zoe" in out


def test_summary_limits_to_two_sentences():
    text = "First thing. Second thing. Third thing. Fourth thing."
    assert tts_summary(text) == "First thing. Second thing."


def test_summary_truncates_and_terminates():
    out = tts_summary("word " * 200)
    assert len(out) <= 321
    assert out[-1] in ".!?"


def test_summary_of_unreadable_reply():
    assert "read out" in tts_summary("```\n\n```")


# --- detect_rail_toggle ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("guardrails off", False),
    ("Guard rails off.", False),
    ("guard-rails, off", False),
    ("turn the guardrails off please", False),
    ("guardrails on", True),
    ("okay put the guard rails on again", True),
    ("guardrails off... no wait, guardrails on", True),
    ("guardrails on, actually guard rails off", False),
    ("how are the agents doing", None),
    ("tell it to guard the rails of the staircase", None),
    ("", None),
])
def test_detect_rail_toggle(text, expected):
    assert detect_rail_toggle(text) is expected


# --- Delegations: the delivery-race guard ---------------------------------

def test_delegated_pane_not_finish_ready_at_delivery():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    assert "w1:p1" in d
    # right after delivery herdr is still typing -- idle must not count
    assert not d.finish_ready("w1:p1")


def test_seen_working_makes_finish_ready():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    d.mark_started("w1:p1")
    assert d.finish_ready("w1:p1")


def test_never_started_pane_finishes_after_one_grace_window():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    clock[0] += Delegations.GRACE_SECS - 1
    assert not d.finish_ready("w1:p1")
    clock[0] += 2
    # grace expired with no working state: the task was quick enough that
    # every observation missed it -- announce it (herdr 0.8.0 owns prompt
    # submission, so there is no stuck-Enter case to nudge first)
    assert d.finish_ready("w1:p1")


def test_started_pane_stays_finish_ready_past_grace():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    d.mark_started("w1:p1")
    clock[0] += Delegations.GRACE_SECS * 3
    assert d.finish_ready("w1:p1")


def test_discard_and_unknown_panes():
    d = Delegations()
    d.add("w1:p1")
    d.discard("w1:p1")
    assert "w1:p1" not in d
    assert not d.finish_ready("w1:p1")
    d.mark_started("nope")  # no-op, no raise
    d.discard("nope")


def test_iterates_pane_ids():
    d = Delegations()
    d.add("w1:p1")
    d.add("w2:p1")
    assert sorted(d) == ["w1:p1", "w2:p1"]


# --- guardrails toggle intercept + immediate delivery ---------------------

class FakeMessage:
    def __init__(self, text):
        self.content = [text]

    @property
    def text_content(self):
        return "\n".join(c for c in self.content if isinstance(c, str))


class RecordingHerdr:
    def __init__(self):
        self.prompts = []
        self.spawns = []
        self.deliveries = []

    async def prompt_agent(self, target, text):
        self.prompts.append((target, text))
        return {}

    async def spawn(self, repo_path, branch, agent="claude"):
        self.spawns.append((repo_path, branch, agent))
        return {"pane_id": "w9:p1", "workspace_id": "w9",
                "agent_name": branch}

    async def deliver_task(self, pane_id, task):
        self.deliveries.append((pane_id, task))


def make_mate(herdr):
    """One fake plays both roles Mate talks to: the protocol client
    (prompt_agent) and the delivery quirk layer (spawn, deliver_task)."""
    return Mate(herdr, delivery=herdr)


async def test_toggle_off_flips_flag_and_injects_note():
    mate = make_mate(RecordingHerdr())
    msg = FakeMessage("guardrails off")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is False
    assert len(msg.content) == 2
    assert "OFF" in msg.content[1] and "code intercept" in msg.content[1]


async def test_toggle_back_on():
    mate = make_mate(RecordingHerdr())
    mate.rail_enabled = False
    msg = FakeMessage("alright guard rails on")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert "ON" in msg.content[1]


async def test_redundant_toggle_still_acknowledged():
    mate = make_mate(RecordingHerdr())
    msg = FakeMessage("guardrails on")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert "already" in msg.content[1]


async def test_normal_speech_injects_nothing():
    mate = make_mate(RecordingHerdr())
    msg = FakeMessage("how's the refactor going")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert len(msg.content) == 1


async def test_rails_off_tell_agent_delivers_immediately():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    mate.rail_enabled = False
    out = await mate.tell_agent(None, pane_id="w1:p1", text="run the tests")
    assert "guardrails off" in out and "delivered" in out
    assert herdr.prompts == [("w1:p1", "run the tests")]
    assert "w1:p1" in mate.delegated
    assert not mate.delegated.finish_ready("w1:p1")  # race guard still applies


async def test_rails_off_spawn_starts_immediately():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    mate.rail_enabled = False
    out = await mate.spawn_task(None, repo_path="/repo", branch="main",
                                task="fix the tests")
    assert "started immediately" in out
    assert herdr.spawns == [("/repo", "main", "claude")]
    # task lands via the background deliverer; delegated joins on delivery
    while mate._bg:
        await asyncio.gather(*list(mate._bg))
    assert herdr.deliveries == [("w9:p1", "fix the tests")]
    assert "w9:p1" in mate.delegated


async def test_rails_on_still_stages():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    out = await mate.tell_agent(None, pane_id="w1:p1", text="hello")
    assert "NOT SENT YET" in out
    assert herdr.prompts == []


# --- the snapshot the watcher shares between both finish lookups ----------

SNAP = {
    "workspaces": [{"workspace_id": "w9", "label": "songhaus"}],
    "panes": [{"pane_id": "w9:p1", "workspace_id": "w9",
               "terminal_title_stripped": "claude"}],
    # codex: a kind with no transcript adapter, so the reply path can be
    # driven off the snapshot alone without a transcript file on disk
    "agents": [{"pane_id": "w9:p1", "agent": "codex", "cwd": "/src/songhaus",
                "agent_session": {"kind": "id", "value": "s1"}}],
}


class CountingHerdr(RecordingHerdr):
    """Counts session.snapshot fetches, so a snapshot handed in by the
    caller can be proved to have replaced them."""

    def __init__(self, snap=None):
        super().__init__()
        self.snap = SNAP if snap is None else snap
        self.snapshot_calls = 0

    async def snapshot(self):
        self.snapshot_calls += 1
        return self.snap


async def test_pane_label_uses_the_snapshot_it_is_given():
    herdr = CountingHerdr()
    mate = make_mate(herdr)
    assert await mate._pane_label("w9:p1", snap=SNAP) == "songhaus"
    assert herdr.snapshot_calls == 0
    # ...and still fetches its own when the caller has none
    assert await mate._pane_label("w9:p1") == "songhaus"
    assert herdr.snapshot_calls == 1


async def test_pane_label_of_a_pane_missing_from_the_given_snapshot():
    # a shared snapshot is a moment old by the time the label is read: an
    # unknown pane takes the generic word, it does not refetch
    herdr = CountingHerdr()
    mate = make_mate(herdr)
    assert await mate._pane_label("w1:p1", snap=SNAP) == "coding"
    assert herdr.snapshot_calls == 0


async def test_agent_replies_dispatches_off_the_snapshot_it_is_given():
    herdr = CountingHerdr()
    mate = make_mate(herdr)
    out = await mate._agent_replies("w9:p1", snap=SNAP)
    assert "transcript is unavailable" in out
    assert herdr.snapshot_calls == 0


async def test_agent_replies_when_the_given_snapshot_has_no_such_agent():
    herdr = CountingHerdr()
    mate = make_mate(herdr)
    out = await mate._agent_replies("w1:p1", snap=SNAP)
    assert out.startswith("ERROR: no coding agent is registered")
    assert herdr.snapshot_calls == 0


async def test_watcher_finish_lookups_work_against_the_real_mate():
    # test_watcher covers the announce logic against FakeMate; this pins the
    # other half of that contract — the real _agent_replies/_pane_label
    # taking the single snapshot the watcher fetches for both of them.
    said: list[str] = []

    async def say(text):
        said.append(text)

    herdr = CountingHerdr()
    mate = make_mate(herdr)
    mate.delegated.add("w9:p1")
    mate.delegated.mark_started("w9:p1")
    await FleetWatcher(herdr, mate, say).announce("idle", "w9:p1", {})
    assert herdr.snapshot_calls == 1  # one fetch, both lookups
    assert "the songhaus agent just finished" in said[0]
    assert "couldn't read its reply" in said[0]  # codex has no adapter
