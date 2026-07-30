"""Safety-rail behavior of Mate's send_answer tool: destructive-looking
pane prompts stage through the same code-enforced rail as tell_agent —
there is no confirmed-bypass tool, and "guardrails off" does not apply."""

import pytest

from mate.agent import Mate
from mate.herdr_client import HerdrError
from test_staging import FakeCtx

pytestmark = pytest.mark.asyncio

DESTRUCTIVE_PANE = "About to run: git push --force origin main. Proceed?"


class StubHerdr:
    """Duck-typed HerdrClient: canned pane text, records sent keys."""

    def __init__(self, pane_text: str):
        self.pane_text = pane_text
        self.sent: list[tuple[str, list[str]]] = []

    async def read_pane(self, pane_id, lines=80, source="recent"):
        return self.pane_text

    async def send_keys(self, pane_id, keys):
        self.sent.append((pane_id, keys))


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


async def test_no_confirmed_bypass_tool():
    assert not hasattr(Mate, "send_answer_confirmed")


async def test_herdr_error_becomes_tool_message():
    class BrokenHerdr(StubHerdr):
        async def read_pane(self, pane_id, lines=80, source="recent"):
            raise HerdrError("pane.read", "pane_not_found", "no such pane")

    mate = Mate(BrokenHerdr(""))
    out = await mate.read_pane(None, pane_id="w9:p9")
    assert out == "ERROR: pane_not_found: no such pane (pane w9:p9)"
