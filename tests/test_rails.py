"""Safety-rail behavior of Mate's send_answer tool: destructive-looking
pane prompts stage through the same code-enforced rail as tell_agent —
there is no confirmed-bypass tool, and "guardrails off" does not apply."""

import pytest

from mate.agent import LOCKED_MSG, Mate
from mate.herdr_client import HerdrError
from test_staging import FakeCtx

pytestmark = pytest.mark.asyncio

DESTRUCTIVE_PANE = "About to run: git push --force origin main. Proceed?"
OTHER_DESTRUCTIVE_PANE = "About to run: rm -rf build/. Proceed?"
MOVED_ON_PANE = "All tests passed. Anything else?"


class StubHerdr:
    """Duck-typed HerdrClient: canned pane text, records sent keys."""

    def __init__(self, pane_text: str):
        self.pane_text = pane_text
        self.sent: list[tuple[str, list[str]]] = []

    async def read_pane(self, pane_id, lines=80, source="recent"):
        return self.pane_text

    async def send_keys(self, pane_id, keys):
        self.sent.append((pane_id, keys))


class ScriptedHerdr(StubHerdr):
    """Pane text that changes between reads: each read takes the next
    scripted screen, the last one repeats. Reads go read_pane tool ->
    send_answer's own read -> send_staged's re-read."""

    def __init__(self, texts: list[str]):
        super().__init__(texts[0])
        self.texts = list(texts)

    async def read_pane(self, pane_id, lines=80, source="recent"):
        self.pane_text = self.texts[0]
        if len(self.texts) > 1:
            self.texts.pop(0)
        return self.pane_text


async def test_send_answer_requires_prior_read():
    herdr = StubHerdr("Proceed? [y/n]")
    mate = Mate(herdr)
    out = await mate.send_answer(None, pane_id="w1:p1", keys=["y", "Enter"])
    assert out.startswith("REFUSED")
    assert herdr.sent == []


async def test_send_answer_after_read_is_sent():
    herdr = StubHerdr("Run npm test? [y/n]")
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    out = await mate.send_answer(None, pane_id="w1:p1", keys=["y", "Enter"])
    assert out == "sent"
    assert herdr.sent == [("w1:p1", ["y", "Enter"])]


async def test_send_answer_stages_destructive_without_sending():
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    out = await mate.send_answer(FakeCtx(["approve it"]),
                                 pane_id="w1:p1", keys=["y", "Enter"])
    assert out.startswith("NOT SENT")
    assert "destructive" in out and "verbatim" in out
    assert herdr.sent == []


async def test_staged_keys_blocked_without_new_user_turn():
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    ctx = FakeCtx(["approve it"])
    await mate.send_answer(ctx, pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(ctx)  # same history: no reply yet
    assert out.startswith("NOT SENT")
    assert herdr.sent == []


async def test_staged_keys_delivered_verbatim_on_clear_yes():
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    await mate.send_answer(FakeCtx(["approve it"]),
                           pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(FakeCtx(["approve it", "yes go ahead"]))
    assert out == "sent"
    assert herdr.sent == [("w1:p1", ["y", "Enter"])]


async def test_staged_keys_blocked_by_veto():
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    await mate.send_answer(FakeCtx(["approve it"]),
                           pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(FakeCtx(["approve it", "no, cancel that"]))
    assert out.startswith("NOT SENT")
    assert herdr.sent == []


async def test_destructive_stages_even_with_guardrails_off():
    # rail_enabled only relaxes messaging/spawns; destructive approvals
    # always need the code-verified spoken yes
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    mate.rail_enabled = False
    await mate.read_pane(None, pane_id="w1:p1")
    out = await mate.send_answer(FakeCtx(["approve it"]),
                                 pane_id="w1:p1", keys=["y", "Enter"])
    assert out.startswith("NOT SENT")
    assert herdr.sent == []
    sent = await mate.send_staged(FakeCtx(["approve it", "yes"]))
    assert sent == "sent"
    assert herdr.sent == [("w1:p1", ["y", "Enter"])]


async def test_staged_keys_dropped_when_the_screen_moved_on():
    # the yes approved a SCREEN: if the destructive prompt is gone by the
    # time it arrives, the moment has passed and the keys must not fire
    # into whatever is up now
    herdr = ScriptedHerdr([DESTRUCTIVE_PANE, DESTRUCTIVE_PANE,
                           MOVED_ON_PANE])
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    await mate.send_answer(FakeCtx(["approve it"]),
                           pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(FakeCtx(["approve it", "yes go ahead"]))
    assert out.startswith("NOT SENT")
    assert "read_pane again" in out
    assert herdr.sent == []


async def test_staged_keys_still_fire_on_a_different_destructive_screen():
    # the limit of the re-read, pinned so it is not mistaken for more than
    # it is: the bar is "the regex still matches", NOT "the same screen the
    # user heard". Nothing is stored about the approved text, so a swap to
    # another destructive prompt inside the stage window still receives the
    # keys. Tighten _deliver (store and compare the read-back text) and this
    # test changes with it — as does the README.
    herdr = ScriptedHerdr([DESTRUCTIVE_PANE, DESTRUCTIVE_PANE,
                           OTHER_DESTRUCTIVE_PANE])
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    await mate.send_answer(FakeCtx(["approve it"]),
                           pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(FakeCtx(["approve it", "yes go ahead"]))
    assert out == "sent"
    assert herdr.sent == [("w1:p1", ["y", "Enter"])]


async def test_staged_keys_dropped_when_the_recheck_read_fails():
    class FlakyHerdr(ScriptedHerdr):
        async def read_pane(self, pane_id, lines=80, source="recent"):
            if len(self.texts) == 1:  # the send_staged re-read
                raise HerdrError("pane.read", "pane_not_found", "gone")
            return await super().read_pane(pane_id, lines, source)

    herdr = FlakyHerdr([DESTRUCTIVE_PANE, DESTRUCTIVE_PANE, DESTRUCTIVE_PANE])
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    await mate.send_answer(FakeCtx(["approve it"]),
                           pane_id="w1:p1", keys=["y", "Enter"])
    out = await mate.send_staged(FakeCtx(["approve it", "yes"]))
    assert out.startswith("NOT SENT")
    assert "pane_not_found" in out
    assert herdr.sent == []


async def test_every_tool_refuses_while_locked():
    # defense in depth: a locked turn never reaches the LLM, so no tool
    # should ever be called — but if one is, it must refuse. StubHerdr has
    # no snapshot/prompt_agent, so any tool that acted would raise here.
    herdr = StubHerdr(DESTRUCTIVE_PANE)
    mate = Mate(herdr)
    mate.lock("correct horse battery staple")
    assert await mate.fleet_status(None) == LOCKED_MSG
    assert await mate.read_pane(None, pane_id="w1:p1") == LOCKED_MSG
    assert await mate.agent_report(None, pane_id="w1:p1") == LOCKED_MSG
    assert await mate.wait_for_agent(None, pane_id="w1:p1") == LOCKED_MSG
    assert await mate.list_known_agents(None) == LOCKED_MSG
    assert await mate.forget_agent(None, name="anything") == LOCKED_MSG
    assert await mate.discard_staged(None) == LOCKED_MSG
    assert await mate.send_staged(None) == LOCKED_MSG
    assert await mate.tell_agent(None, pane_id="w1:p1",
                                 text="hi") == LOCKED_MSG
    assert await mate.send_answer(None, pane_id="w1:p1",
                                  keys=["y"]) == LOCKED_MSG
    assert await mate.spawn_task(None, repo_path="/repo", branch="b",
                                 task="t") == LOCKED_MSG
    assert await mate.spawn_in_folder(None, folder_name="mate",
                                      task="t") == LOCKED_MSG
    assert herdr.sent == []
    # a refused read must not satisfy the read-before-approve rail either
    assert mate._read_panes == set()


async def test_no_confirmed_bypass_tool():
    assert not hasattr(Mate, "send_answer_confirmed")


async def test_herdr_error_becomes_tool_message():
    class BrokenHerdr(StubHerdr):
        async def read_pane(self, pane_id, lines=80, source="recent"):
            raise HerdrError("pane.read", "pane_not_found", "no such pane")

    mate = Mate(BrokenHerdr(""))
    out = await mate.read_pane(None, pane_id="w9:p9")
    assert out == "ERROR: pane_not_found: no such pane (pane w9:p9)"
