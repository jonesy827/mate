"""agent_report: reading claude replies from the session transcript."""

import json

import pytest

import matebridge.agent as agent_mod
from matebridge.agent import Mate, claude_transcript_path, read_transcript_replies

pytestmark = pytest.mark.asyncio

SESSION = "11111111-2222-3333-4444-555555555555"
CWD = "/home/jonesy/src/japan-translate"


def write_transcript(root, texts):
    d = root / "-home-jonesy-src-japan-translate"
    d.mkdir(parents=True)
    path = d / f"{SESSION}.jsonl"
    lines = [json.dumps({"type": "user", "message": {"content": "hi"}})]
    for t in texts:
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": t},
            ]},
        }))
    lines.append("not json at all")
    path.write_text("\n".join(lines) + "\n")
    return path


class SnapshotHerdr:
    def __init__(self, agents):
        self._agents = agents

    async def snapshot(self):
        return {"agents": self._agents}


def claude_agent(pane_id="w1:p1"):
    return {"pane_id": pane_id, "agent": "claude", "cwd": CWD,
            "agent_session": {"kind": "id", "value": SESSION}}


def test_transcript_path_munges_cwd():
    p = claude_transcript_path(CWD, SESSION)
    assert p.name == f"{SESSION}.jsonl"
    assert p.parent.name == "-home-jonesy-src-japan-translate"


def test_read_transcript_replies_skips_junk(tmp_path):
    path = write_transcript(tmp_path, ["first answer", "second answer"])
    assert read_transcript_replies(path, 1) == ["second answer"]
    assert read_transcript_replies(path, 5) == ["first answer", "second answer"]


async def test_agent_report_returns_last_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "CLAUDE_PROJECTS", tmp_path)
    write_transcript(tmp_path, ["the project is japan-translate"])
    mate = Mate(SnapshotHerdr([claude_agent()]))
    out = await mate.agent_report(None, pane_id="w1:p1")
    assert out == "the project is japan-translate"


async def test_agent_report_no_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "CLAUDE_PROJECTS", tmp_path)
    mate = Mate(SnapshotHerdr([]))
    out = await mate.agent_report(None, pane_id="w1:p1")
    assert out.startswith("ERROR")
    assert "read_pane" in out


async def test_agent_report_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "CLAUDE_PROJECTS", tmp_path)
    mate = Mate(SnapshotHerdr([claude_agent()]))
    out = await mate.agent_report(None, pane_id="w1:p1")
    assert out.startswith("ERROR")
    assert "read_pane" in out


class StatusHerdr(SnapshotHerdr):
    """Agent statuses served in sequence; sticks on the last one."""

    def __init__(self, statuses):
        self._statuses = list(statuses)
        super().__init__([])

    async def snapshot(self):
        status = (self._statuses.pop(0) if len(self._statuses) > 1
                  else self._statuses[0])
        agent = dict(claude_agent(), agent_status=status)
        return {"agents": [agent]}


async def test_wait_for_agent_returns_reply_when_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "CLAUDE_PROJECTS", tmp_path)
    write_transcript(tmp_path, ["all done boss"])
    mate = Mate(StatusHerdr(["working", "idle"]))
    mate.delegated.add("w1:p1")
    out = await mate.wait_for_agent(None, pane_id="w1:p1", seconds=5)
    assert "all done boss" in out
    assert "w1:p1" not in mate.delegated


async def test_wait_for_agent_times_out_without_looping(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "CLAUDE_PROJECTS", tmp_path)
    mate = Mate(StatusHerdr(["working"]))
    out = await mate.wait_for_agent(None, pane_id="w1:p1", seconds=1)
    assert "STOP checking" in out


# delegation marking now happens at send_staged delivery time -- covered in
# tests/test_staging.py (test_clear_yes_delivers_staged_text_verbatim).
