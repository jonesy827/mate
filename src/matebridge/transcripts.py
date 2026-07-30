"""Reading Claude Code session transcripts (shared by agent.py and the
Telegram bridge).

Claude Code writes one JSONL transcript per session under
~/.claude/projects/<munged-cwd>/<session-id>.jsonl. herdr's integration hook
reports the session id + cwd, which is enough to find it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CLAUDE_PROJECTS = Path(os.path.expanduser("~/.claude/projects"))


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


async def agent_last_reply(herdr, pane_id: str, count: int = 1) -> str | None:
    """Last reply text of the claude agent in pane_id, or None.

    Quiet variant of Mate._agent_replies for callers that want an excerpt
    or nothing (the voice agent keeps its own version with LLM-facing
    ERROR strings).
    """
    try:
        snap = await herdr.snapshot()
    except Exception:  # noqa: BLE001 - any herdr trouble means "no excerpt"
        return None
    agent = next((a for a in snap.get("agents", [])
                  if a.get("pane_id") == pane_id), None)
    if agent is None or agent.get("agent") != "claude":
        return None
    session = agent.get("agent_session") or {}
    if session.get("kind") != "id":
        return None
    path = claude_transcript_path(agent.get("cwd", ""), session["value"])
    try:
        replies = read_transcript_replies(path, max(1, count))
    except OSError:
        return None
    return "\n\n".join(replies) if replies else None
