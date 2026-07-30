"""Safety-rail behavior of Mate's send_answer tools."""

import pytest

from matebridge.agent import Mate

pytestmark = pytest.mark.asyncio


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


async def test_send_answer_blocks_destructive():
    herdr = StubHerdr("About to run: git push --force origin main. Proceed?")
    mate = Mate(herdr)
    await mate.read_pane(None, pane_id="w1:p1")
    out = await mate.send_answer(None, pane_id="w1:p1", keys=["y", "Enter"])
    assert out.startswith("REFUSED")
    assert "destructive" in out
    assert herdr.sent == []


async def test_send_answer_confirmed_bypasses_check():
    herdr = StubHerdr("About to run: git push --force origin main. Proceed?")
    mate = Mate(herdr)
    out = await mate.send_answer_confirmed(
        None, pane_id="w1:p1", keys=["y", "Enter"])
    assert out == "sent"
    assert herdr.sent == [("w1:p1", ["y", "Enter"])]
