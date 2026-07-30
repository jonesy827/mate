"""spawn_in_folder: STT-tolerant folder resolution, tiered code-composed
confirmations, and the known-agents memory."""

import asyncio
import json

import pytest

from matebridge.agent import Mate
from matebridge.folders import (
    KnownAgents,
    resolve_folder,
    speakable_path,
    squash,
)

pytestmark = pytest.mark.asyncio


class FakeCtx:
    class _Item:
        def __init__(self, role, text):
            self.role = role
            self.text_content = text

    def __init__(self, user_texts):
        items = [self._Item("user", t) for t in user_texts]
        history = type("H", (), {"items": items})()
        self.session = type("S", (), {"history": history})()


class RecordingHerdr:
    def __init__(self):
        self.folder_spawns = []
        self.deliveries = []
        self.deliver_error = None

    async def spawn_in_folder(self, path, label, agent="claude"):
        self.folder_spawns.append((path, label, agent))
        return {"workspace": {}, "pane_id": "w5:p1", "agent_name": label}

    async def deliver_task(self, pane_id, task):
        if self.deliver_error is not None:
            raise self.deliver_error
        self.deliveries.append((pane_id, task))


def make_roots(tmp_path, *names):
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    for n in names:
        (root / n).mkdir()
    return [root]


def make_mate(tmp_path, *dirnames):
    herdr = RecordingHerdr()
    known = KnownAgents(tmp_path / "known.json")
    mate = Mate(herdr, known=known, roots=make_roots(tmp_path, *dirnames))
    return mate, herdr


# --- resolver --------------------------------------------------------------

def test_resolver_exact_squashed_match(tmp_path):
    roots = make_roots(tmp_path, "matebridge", "herdr")
    assert [p.name for p in resolve_folder("mate bridge", roots)] == ["matebridge"]
    assert [p.name for p in resolve_folder("Mate-Bridge", roots)] == ["matebridge"]


def test_resolver_close_match_catches_stt_spellings(tmp_path):
    roots = make_roots(tmp_path, "matebridge", "herdr")
    assert [p.name for p in resolve_folder("herder", roots)] == ["herdr"]


def test_resolver_no_match_and_hidden_dirs_skipped(tmp_path):
    roots = make_roots(tmp_path, "matebridge")
    (roots[0] / ".git").mkdir()
    assert resolve_folder("frobnicator", roots) == []
    assert resolve_folder("git", roots) == []
    assert resolve_folder("", roots) == []


def test_speakable_path():
    assert speakable_path("/home/jonesy/src/matebridge") == \
        "home, jonesy, src, matebridge"


# --- KnownAgents memory ----------------------------------------------------

def test_memory_remember_get_forget_persists(tmp_path):
    f = tmp_path / "known.json"
    k = KnownAgents(f)
    k.remember("matebridge", "/home/jonesy/src/matebridge")
    # squash-matched lookup: spoken spelling need not match stored spelling
    assert KnownAgents(f).get("Mate Bridge") == "/home/jonesy/src/matebridge"
    k2 = KnownAgents(f)
    assert k2.forget("mate-bridge") is True
    assert KnownAgents(f).get("matebridge") is None
    assert k2.forget("matebridge") is False


def test_memory_names_ordering_and_dedup(tmp_path):
    k = KnownAgents(tmp_path / "known.json")
    k.remember("herdr", "/a")
    k.remember("matebridge", "/b")
    k.remember("Herdr", "/c")  # same squashed name replaces the old entry
    names = k.names()
    assert names[0] == ("Herdr", "/c")
    assert len(names) == 2


def test_memory_survives_corrupt_file(tmp_path):
    f = tmp_path / "known.json"
    f.write_text("{not json")
    assert KnownAgents(f).names() == []


# --- spawn_in_folder tool --------------------------------------------------

async def test_unknown_folder_gets_full_path_confirmation(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    out = await mate.spawn_in_folder(FakeCtx(["spawn one in matebridge"]),
                                     folder_name="mate bridge",
                                     task="fix the tests")
    # no live session in tests -> falls back to read-exactly, sentence intact
    assert "Read this to the user EXACTLY" in out
    assert "full path" in out and "src, matebridge" in out
    assert "fix the tests" in out
    assert herdr.folder_spawns == []


async def drain_bg(mate):
    """Let the background task-delivery coroutines run to completion."""
    while mate._bg:
        await asyncio.gather(*list(mate._bg))


async def test_confirmed_spawn_delivers_and_remembers(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    await mate.spawn_in_folder(FakeCtx(["first"]), folder_name="matebridge",
                               task="fix the tests")
    sent = await mate.send_staged(FakeCtx(["first", "yes go ahead"]))
    result = json.loads(sent)
    assert result["pane_id"] == "w5:p1"
    assert len(herdr.folder_spawns) == 1
    path, label, agent = herdr.folder_spawns[0]
    assert path.endswith("src/matebridge") and label == "matebridge"
    assert agent == "claude"
    assert mate.known.get("matebridge") == path
    # the task is handed over in the background once claude boots; the pane
    # joins delegated only after it actually lands
    assert "w5:p1" not in mate.delegated
    await drain_bg(mate)
    assert herdr.deliveries == [("w5:p1", "fix the tests")]
    assert "w5:p1" in mate.delegated


async def test_spawn_without_task_skips_delivery(tmp_path):
    # "open up a new agent in songhaus" with no task: herdr rejects empty
    # prompts (empty_agent_prompt), so nothing must be delivered — the agent
    # just opens ready. No delegated entry either: nothing to finish.
    mate, herdr = make_mate(tmp_path, "matebridge")
    out = await mate.spawn_in_folder(FakeCtx(["x"]), folder_name="matebridge",
                                     task="")
    assert "no task yet" in out
    sent = await mate.send_staged(FakeCtx(["x", "yes"]))
    result = json.loads(sent)
    assert result["task_delivery"].startswith("none")
    await drain_bg(mate)
    assert herdr.deliveries == []
    assert "w5:p1" not in mate.delegated


async def test_failed_background_delivery_keeps_workspace_out_of_delegated(
        tmp_path):
    from matebridge.herdr_client import HerdrError
    mate, herdr = make_mate(tmp_path, "matebridge")
    herdr.deliver_error = HerdrError("agent.prompt", "agent_not_ready", "x")
    await mate.spawn_in_folder(FakeCtx(["x"]), folder_name="matebridge",
                               task="t")
    await mate.send_staged(FakeCtx(["x", "yes"]))
    await drain_bg(mate)
    assert herdr.deliveries == []
    assert "w5:p1" not in mate.delegated  # never "finished" announcements


async def test_known_folder_gets_short_confirmation(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    real = str((tmp_path / "src" / "matebridge"))
    mate.known.remember("matebridge", real)
    out = await mate.spawn_in_folder(FakeCtx(["again"]),
                                     folder_name="matebridge",
                                     task="more tests")
    assert "go ahead?" in out
    assert "full path" not in out  # short form, no path readback
    sent = await mate.send_staged(FakeCtx(["again", "yep"]))
    assert json.loads(sent)["pane_id"] == "w5:p1"
    assert herdr.folder_spawns[0][0] == real
    await drain_bg(mate)


async def test_stale_memory_falls_back_to_full_confirmation(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    mate.known.remember("matebridge", str(tmp_path / "gone"))
    out = await mate.spawn_in_folder(FakeCtx(["x"]), folder_name="matebridge",
                                     task="t")
    assert "no longer exists" in out
    assert "full path" in out


async def test_no_match_errors_and_ambiguity_asks(tmp_path):
    mate, _ = make_mate(tmp_path, "matebridge", "mate-bridge-two")
    out = await mate.spawn_in_folder(FakeCtx(["x"]), folder_name="zzz",
                                     task="t")
    assert out.startswith("ERROR")
    # both dirs squash-match closely on "matebridge two"-ish input? exact
    # match still wins for the exact name
    exact = await mate.spawn_in_folder(FakeCtx(["x"]),
                                       folder_name="matebridge", task="t")
    assert "EXACTLY" in exact


async def test_rails_off_spawns_immediately_and_remembers(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    mate.rail_enabled = False
    out = await mate.spawn_in_folder(FakeCtx(["x"]),
                                     folder_name="mate bridge", task="t")
    assert "guardrails off" in out and "started immediately" in out
    assert len(herdr.folder_spawns) == 1
    assert mate.known.get("matebridge") == herdr.folder_spawns[0][0]
    await drain_bg(mate)
    assert herdr.deliveries == [("w5:p1", "t")]
    assert "w5:p1" in mate.delegated


async def test_non_default_agent_kind_passes_through(tmp_path):
    mate, herdr = make_mate(tmp_path, "matebridge")
    out = await mate.spawn_in_folder(FakeCtx(["x"]), folder_name="matebridge",
                                     task="t", agent="Pi")
    assert "a pi agent" in out  # spoken confirmation names the harness
    sent = await mate.send_staged(FakeCtx(["x", "yes"]))
    assert json.loads(sent)["pane_id"] == "w5:p1"
    assert herdr.folder_spawns[0][2] == "pi"
    await drain_bg(mate)


async def test_list_and_forget_tools(tmp_path):
    mate, _ = make_mate(tmp_path)
    assert "no known agents" in await mate.list_known_agents(None)
    mate.known.remember("matebridge", "/x")
    listed = json.loads(await mate.list_known_agents(None))
    assert listed == [{"name": "matebridge", "path": "/x"}]
    assert "forgotten" in await mate.forget_agent(None, name="mate bridge")
    assert "ERROR" in await mate.forget_agent(None, name="matebridge")
