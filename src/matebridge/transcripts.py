"""Per-harness transcript adapters for agent_report.

herdr can host many agent kinds (claude, codex, gemini, ...). Spawning,
messaging, and status-watching work for all of them through herdr itself,
but reading an agent's *replies* back means knowing where that harness
writes its session transcript and how to parse it — which is per-harness
knowledge. ADAPTERS maps herdr's agent kind to a reader; a kind without an
entry still runs fine, Mate just falls back to read_pane for its output.

An adapter is `(cwd, session_id, count) -> list[str]`: the last `count`
assistant replies, newest last. herdr's integration hook reports the
session id + cwd for detected agents, which is what adapters key on. May
raise OSError when the transcript file is missing/unreadable.

Claude Code: one JSONL transcript per session under
~/.claude/projects/<munged-cwd>/<session-id>.jsonl.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

CLAUDE_PROJECTS = Path(os.path.expanduser(
    os.environ.get("MATE_CLAUDE_PROJECTS", "~/.claude/projects")))


def claude_transcript_path(cwd: str, session_id: str) -> Path:
    munged = re.sub(r"[^A-Za-z0-9-]", "-", cwd)
    return CLAUDE_PROJECTS / munged / f"{session_id}.jsonl"


def read_transcript_replies(path: Path, count: int) -> list[str]:
    """Last `count` assistant text messages from a Claude Code transcript."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") != "assistant":
                continue
            for block in entry.get("message", {}).get("content", []):
                if (isinstance(block, dict) and block.get("type") == "text"
                        and block.get("text", "").strip()):
                    texts.append(block["text"])
    return texts[-count:]


def _claude_replies(cwd: str, session_id: str, count: int) -> list[str]:
    return read_transcript_replies(
        claude_transcript_path(cwd, session_id), count)


Adapter = Callable[[str, str, int], list[str]]

ADAPTERS: dict[str, Adapter] = {
    "claude": _claude_replies,
}


def adapter_for(kind: str | None) -> Adapter | None:
    return ADAPTERS.get(kind or "")


def supported_kinds() -> str:
    return ", ".join(sorted(ADAPTERS))


async def agent_last_reply(herdr, pane_id: str, count: int = 1) -> str | None:
    """Last reply text of the agent in pane_id, or None.

    Quiet variant of Mate._agent_replies for callers that want an excerpt
    or nothing (the voice agent keeps its own version with LLM-facing
    ERROR strings). None for harnesses without a transcript adapter.
    """
    try:
        snap = await herdr.snapshot()
    except Exception:
        return None
    agent = next((a for a in snap.get("agents", [])
                  if a.get("pane_id") == pane_id), None)
    if agent is None:
        return None
    reader = adapter_for(agent.get("agent"))
    session = agent.get("agent_session") or {}
    if reader is None or session.get("kind") != "id":
        return None
    try:
        replies = reader(agent.get("cwd", ""), session["value"], max(1, count))
    except OSError:
        return None
    return "\n\n".join(replies) if replies else None
