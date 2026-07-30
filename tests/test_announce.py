"""Watcher-announcement plumbing: TTS sanitizing/summarizing, the guardrails
voice toggle, the delivery-race guard in Delegations, and the guardrails-off
immediate-delivery path."""

import asyncio

import pytest

from matebridge.agent import (
    Delegations,
    Mate,
    detect_rail_toggle,
    tts_sanitize,
    tts_summary,
)

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


def test_never_started_pane_asks_for_nudge_not_finish():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    clock[0] += Delegations.GRACE_SECS - 1
    assert not d.finish_ready("w1:p1")
    assert not d.needs_nudge("w1:p1")
    clock[0] += 2
    # grace expired but never seen working: the Enter was probably swallowed
    # -- nudge instead of announcing a stale reply as "finished"
    assert not d.finish_ready("w1:p1")
    assert d.needs_nudge("w1:p1")


def test_nudge_then_second_grace_counts_as_finished():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    clock[0] += Delegations.GRACE_SECS + 1
    d.mark_nudged("w1:p1")
    assert not d.needs_nudge("w1:p1")  # one nudge only
    assert not d.finish_ready("w1:p1")  # fresh grace window after the nudge
    clock[0] += Delegations.GRACE_SECS + 1
    assert d.finish_ready("w1:p1")  # truly-fast task, announce it
    assert not d.needs_nudge("w1:p1")


def test_nudged_pane_that_starts_working_is_normal_again():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    clock[0] += Delegations.GRACE_SECS + 1
    d.mark_nudged("w1:p1")
    d.mark_started("w1:p1")  # the nudge submitted the prompt
    assert d.finish_ready("w1:p1")
    assert not d.needs_nudge("w1:p1")


def test_started_pane_never_needs_nudge():
    clock = [100.0]
    d = Delegations(clock=lambda: clock[0])
    d.add("w1:p1")
    d.mark_started("w1:p1")
    clock[0] += Delegations.GRACE_SECS * 3
    assert not d.needs_nudge("w1:p1")
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

    async def nudge_enter(self, pane_id):
        self.nudges = [*getattr(self, "nudges", []), pane_id]


async def test_toggle_off_flips_flag_and_injects_note():
    mate = Mate(RecordingHerdr())
    msg = FakeMessage("guardrails off")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is False
    assert len(msg.content) == 2
    assert "OFF" in msg.content[1] and "code intercept" in msg.content[1]


async def test_toggle_back_on():
    mate = Mate(RecordingHerdr())
    mate.rail_enabled = False
    msg = FakeMessage("alright guard rails on")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert "ON" in msg.content[1]


async def test_redundant_toggle_still_acknowledged():
    mate = Mate(RecordingHerdr())
    msg = FakeMessage("guardrails on")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert "already" in msg.content[1]


async def test_normal_speech_injects_nothing():
    mate = Mate(RecordingHerdr())
    msg = FakeMessage("how's the refactor going")
    await mate.on_user_turn_completed(None, msg)
    assert mate.rail_enabled is True
    assert len(msg.content) == 1


async def test_rails_off_tell_agent_delivers_immediately():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
    mate.rail_enabled = False
    out = await mate.tell_agent(None, pane_id="w1:p1", text="run the tests")
    assert "guardrails off" in out and "delivered" in out
    assert herdr.prompts == [("w1:p1", "run the tests")]
    assert "w1:p1" in mate.delegated
    assert not mate.delegated.finish_ready("w1:p1")  # race guard still applies
    # paste-guard self-heal: a bare Enter follows the delivery
    while mate._bg:
        await asyncio.gather(*list(mate._bg))
    assert herdr.nudges == ["w1:p1"]


async def test_rails_off_spawn_starts_immediately():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
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
    mate = Mate(herdr)
    out = await mate.tell_agent(None, pane_id="w1:p1", text="hello")
    assert "NOT SENT YET" in out
    assert herdr.prompts == []
