"""Stage-and-confirm rail: tell_agent/spawn_task stage, send_staged delivers
only after a new user turn whose raw transcript passes approves_send."""

import json

import pytest

from matebridge.agent import Mate
from matebridge.herdr_client import HerdrError

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
        self.spawns: list[tuple[str, str, str]] = []

    async def prompt_agent(self, target, text):
        self.prompts.append((target, text))
        return {}

    async def spawn(self, repo_path, branch, task):
        self.spawns.append((repo_path, branch, task))
        return {"pane_id": "w9:p1", "workspace_id": "w9"}


async def test_tell_agent_stages_without_sending():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
    out = await mate.tell_agent(FakeCtx(["tell it to rerun the tests"]),
                                pane_id="w1:p1", text="rerun the tests")
    assert "NOT SENT YET" in out
    assert "rerun the tests" in out
    assert "word for word" in out
    assert herdr.prompts == []
    assert "w1:p1" not in mate.delegated


async def test_send_staged_with_nothing_staged():
    mate = Mate(RecordingHerdr())
    out = await mate.send_staged(FakeCtx(["yes"]))
    assert out.startswith("ERROR")
    assert "nothing is staged" in out


async def test_turn_gate_blocks_without_new_user_turn():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
    ctx = FakeCtx(["send hello to the agent"])
    await mate.tell_agent(ctx, pane_id="w1:p1", text="hello")
    out = await mate.send_staged(ctx)  # same history: no reply yet
    assert out.startswith("NOT SENT")
    assert "not replied" in out
    assert herdr.prompts == []


async def test_veto_word_blocks_even_with_send_in_transcript():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
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
    mate = Mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.send_staged(FakeCtx(["first", "hmm what time is it"]))
    assert out.startswith("NOT SENT")
    assert herdr.prompts == []


async def test_clear_yes_delivers_staged_text_verbatim():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
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
    mate = Mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1",
                          text="check out the clean branch")
    await mate.tell_agent(FakeCtx(["first", "no, the cleanup branch"]),
                          pane_id="w1:p1", text="check out the cleanup branch")
    out = await mate.send_staged(
        FakeCtx(["first", "no, the cleanup branch", "yes send it"]))
    assert out.startswith("delivered")
    assert herdr.prompts == [("w1:p1", "check out the cleanup branch")]


async def test_discard_staged():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello")
    out = await mate.discard_staged(None)
    assert "discarded" in out
    out2 = await mate.send_staged(FakeCtx(["first", "yes"]))
    assert "nothing is staged" in out2
    assert await mate.discard_staged(None) == "nothing was staged."


async def test_spawn_task_rail_parallels_tell_agent():
    herdr = RecordingHerdr()
    mate = Mate(herdr)
    out = await mate.spawn_task(FakeCtx(["first"]),
                                repo_path="/repo", branch="main",
                                task="fix the tests")
    assert "NOT STARTED YET" in out
    assert herdr.spawns == []
    blocked = await mate.send_staged(FakeCtx(["first", "wait"]))
    assert blocked.startswith("NOT SENT")
    sent = await mate.send_staged(FakeCtx(["first", "wait", "go ahead"]))
    assert json.loads(sent)["pane_id"] == "w9:p1"
    assert herdr.spawns == [("/repo", "main", "fix the tests")]
    assert "w9:p1" in mate.delegated


async def test_agent_not_found_surfaces_from_send_staged():
    class NoAgentHerdr(RecordingHerdr):
        async def prompt_agent(self, target, text):
            raise HerdrError("agent.prompt", "agent_not_found",
                             f"agent target {target} not found")

    mate = Mate(NoAgentHerdr())
    await mate.tell_agent(FakeCtx(["first"]), pane_id="w1:p1", text="hello?")
    out = await mate.send_staged(FakeCtx(["first", "yes"]))
    assert out.startswith("ERROR")
    assert "no coding agent" in out
    assert "w1:p1" not in mate.delegated
