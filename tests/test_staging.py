"""Stage-and-confirm rail: tell_agent/spawn_task stage, send_staged delivers
only after a new user turn whose raw transcript passes approves_send."""

import asyncio
import json

import pytest

from mate.agent import Mate
from mate.herdr_client import HerdrError

pytestmark = pytest.mark.asyncio


class FakeCtx:
    """Duck-types RunContext.session.history.items for user_transcripts()."""

    class _Item:
        def __init__(self, role, text):
            self.role = role
            self.text_content = text

    def __init__(self, user_texts):
        items = [self._Item("user", t) for t in user_texts]
        items.append(self._Item("assistant", "ok"))  # must be ignored
        history = type("H", (), {"items": items})()
        self.session = type("S", (), {"history": history})()


class RecordingHerdr:
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []
        self.spawns: list[tuple[str, str]] = []
        self.deliveries: list[tuple[str, str]] = []

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


async def test_tell_agent_stages_without_sending():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    out = await mate.tell_agent(FakeCtx(["tell it to rerun the tests"]),
                                pane_id="w1:p1", text="rerun the tests")
    assert "NOT SENT YET" in out
    assert "rerun the tests" in out
    assert "word for word" in out
    assert herdr.prompts == []
    assert "w1:p1" not in mate.delegated


async def test_send_staged_with_nothing_staged():
    mate = make_mate(RecordingHerdr())
    out = await mate.send_staged(FakeCtx(["yes"]))
    assert out.startswith("ERROR")
    assert "nothing is staged" in out


async def test_turn_gate_blocks_without_new_user_turn():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    ctx = FakeCtx(["send hello to the agent"])
    await mate.tell_agent(ctx, pane_id="w1:p1", text="hello")
    out = await mate.send_staged(ctx)  # same history: no reply yet
    assert out.startswith("NOT SENT")
    assert "not replied" in out
    assert herdr.prompts == []


async def test_veto_word_blocks_even_with_send_in_transcript():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.send_staged(FakeCtx(["first", "no, don't send it"]))
    assert out.startswith("NOT SENT")
    assert "clear yes" in out
    assert herdr.prompts == []
    # stays staged so the model can re-ask and retry
    out2 = await mate.send_staged(FakeCtx(["first", "no, don't send it",
                                           "yes send it"]))
    assert out2.startswith("delivered")


async def test_unclear_reply_blocks():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.send_staged(FakeCtx(["first", "hmm what time is it"]))
    assert out.startswith("NOT SENT")
    assert herdr.prompts == []


async def test_clear_yes_delivers_staged_text_verbatim():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1",
                          text="check out the cleanup branch")
    out = await mate.send_staged(FakeCtx(["first", "yep"]))
    assert out.startswith("delivered")
    assert herdr.prompts == [("w1:p1", "check out the cleanup branch")]
    assert "w1:p1" in mate.delegated
    # one delivery per confirmation: the stage is consumed
    out2 = await mate.send_staged(FakeCtx(["first", "yep", "yes"]))
    assert out2.startswith("ERROR")
    assert "nothing is staged" in out2


async def test_restaging_overwrites_previous_stage():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1",
                          text="check out the clean branch")
    await mate.tell_agent(FakeCtx(["first", "no, the cleanup branch"]),
                          pane_id="w1:p1", text="check out the cleanup branch")
    out = await mate.send_staged(
        FakeCtx(["first", "no, the cleanup branch", "yes send it"]))
    assert out.startswith("delivered")
    assert herdr.prompts == [("w1:p1", "check out the cleanup branch")]


async def test_stage_survives_a_few_intervening_turns():
    # the user can hesitate or ask one thing in between; the read-back is
    # still fresh enough at the limit
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.send_staged(
        FakeCtx(["first", "hang on", "what was that", "yes send it"]))
    assert out.startswith("delivered")
    assert herdr.prompts == [("w1:p1", "hello")]


async def test_stale_stage_expires_and_is_dropped():
    # _staged is one slot: without this, a "yes" four turns later would
    # deliver whatever was staged before the conversation moved on
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    history = ["first", "hang on", "what was that", "never mind", "yes"]
    out = await mate.send_staged(FakeCtx(history))
    assert out.startswith("NOT SENT")
    assert "expired" in out
    assert herdr.prompts == []
    # dropped, not just refused: a fresh yes cannot resurrect it
    out2 = await mate.send_staged(FakeCtx([*history, "yes send it"]))
    assert out2.startswith("ERROR")
    assert "nothing is staged" in out2
    assert herdr.prompts == []


async def test_discard_staged():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.discard_staged(None)
    assert "discarded" in out
    out2 = await mate.send_staged(FakeCtx(["first", "yes"]))
    assert "nothing is staged" in out2
    assert await mate.discard_staged(None) == "nothing was staged."


async def test_spawn_task_rail_parallels_tell_agent():
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    out = await mate.spawn_task(FakeCtx(["first"]),
                                repo_path="/repo", branch="main",
                                task="fix the tests")
    assert "NOT STARTED YET" in out
    assert herdr.spawns == []
    blocked = await mate.send_staged(FakeCtx(["first", "wait"]))
    assert blocked.startswith("NOT SENT")
    sent = await mate.send_staged(FakeCtx(["first", "wait", "go ahead"]))
    assert json.loads(sent)["pane_id"] == "w9:p1"
    assert herdr.spawns == [("/repo", "main", "claude")]
    while mate._bg:
        await asyncio.gather(*list(mate._bg))
    assert herdr.deliveries == [("w9:p1", "fix the tests")]
    assert "w9:p1" in mate.delegated


async def test_spawn_task_harness_passthrough():
    # herdr hosts many agent kinds; the tool passes the (normalized) kind
    # through and the spoken staging line names non-default harnesses
    herdr = RecordingHerdr()
    mate = make_mate(herdr)
    out = await mate.spawn_task(FakeCtx(["first"]), repo_path="/repo",
                                branch="main", task="t", agent="Codex")
    assert "a codex agent" in out
    sent = await mate.send_staged(FakeCtx(["first", "yes"]))
    assert json.loads(sent)["pane_id"] == "w9:p1"
    assert herdr.spawns == [("/repo", "main", "codex")]
    while mate._bg:
        await asyncio.gather(*list(mate._bg))


async def test_tell_agent_queues_message_while_agent_boots():
    # a just-spawned agent answers agent_not_ready for its whole boot (over
    # a minute in a big folder) — tell_agent must queue the message for
    # background delivery, not bounce "isn't ready" back at the user
    class BootingHerdr(RecordingHerdr):
        async def prompt_agent(self, target, text):
            raise HerdrError("agent.prompt", "agent_not_ready", "pending")

        async def snapshot(self):
            return {"panes": [], "workspaces": []}

    herdr = BootingHerdr()
    mate = make_mate(herdr)
    mate.rail_enabled = False
    out = await mate.tell_agent(None, pane_id="wA:p1", text="start the task")
    assert "queued" in out and "still starting up" in out
    assert "wA:p1" not in mate.delegated
    while mate._bg:
        await asyncio.gather(*list(mate._bg))
    assert herdr.deliveries == [("wA:p1", "start the task")]
    assert "wA:p1" in mate.delegated


async def test_agent_not_found_surfaces_from_send_staged():
    class NoAgentHerdr(RecordingHerdr):
        async def prompt_agent(self, target, text):
            raise HerdrError("agent.prompt", "agent_not_found",
                             f"agent target {target} not found")

    mate = make_mate(NoAgentHerdr())
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello?")
    out = await mate.send_staged(FakeCtx(["first", "yes"]))
    assert out.startswith("ERROR")
    assert "no coding agent" in out
    assert "w1:p1" not in mate.delegated
