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
Codex: exact native session ID -> read-only state index -> rollout JSONL.
Only final assistant message records are exposed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import deque
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


CODEX_STATE_DIR = Path(os.environ.get("MATE_CODEX_HOME", os.environ.get("CODEX_HOME", "~/.codex"))).expanduser()


def codex_transcript_path(session_id: str) -> Path:
    """Resolve an exact native session ID; never guess from cwd or recency.

    Prefer Codex's read-only index: resumed paginated sessions can have a new
    rollout filename. Fall back to a unique file for installations without it.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
        raise OSError("Invalid Codex session ID")
    db = CODEX_STATE_DIR / "state_5.sqlite"
    if db.is_file():
        try:
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=1) as conn:
                row = conn.execute("SELECT rollout_path FROM threads WHERE id = ?",
                                   (session_id,)).fetchone()
            if row:
                path = Path(row[0]).resolve()
                if not path.is_relative_to(CODEX_STATE_DIR.resolve()):
                    raise OSError("Codex rollout is outside its state directory")
                return path
            raise OSError("Codex session is absent from its index")
        except sqlite3.Error:
            pass  # optional index absent, busy, or changed schema
    paths = list((CODEX_STATE_DIR / "sessions").glob(f"**/rollout-*-{session_id}.jsonl"))
    if len(paths) != 1:
        raise OSError("No unique Codex transcript for session")
    return paths[0]


def read_codex_replies(path: Path, session_id: str, count: int) -> list[str]:
    """Read final user-facing answers, excluding commentary, tools, and reasoning.

    Read only the indexed rollout segment. If a resumed segment has no final
    answer yet, callers fall back to the screen instead of guessing old history.
    """
    texts = deque(maxlen=max(1, min(count, 10)))
    seen = set()
    with path.open(encoding="utf-8") as fh:
        try:
            first = json.loads(next(fh))
            meta = first.get("payload", {})
            if first.get("type") != "session_meta" or meta.get("id", meta.get("session_id")) != session_id:
                raise OSError("Codex transcript identity mismatch")
        except (ValueError, StopIteration, AttributeError) as exc:
            raise OSError("Invalid Codex transcript metadata") from exc
        for line in fh:
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a concurrent write may leave a partial final line
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if (entry.get("type") != "response_item" or not isinstance(payload, dict)
                    or payload.get("type") != "message" or payload.get("role") != "assistant"
                    or payload.get("phase") not in (None, "final", "final_answer")):
                continue
            mid = payload.get("id")
            if mid and mid in seen:
                continue
            text = "\n".join(block["text"] for block in payload.get("content", [])
                             if isinstance(block, dict) and block.get("type") == "output_text"
                             and isinstance(block.get("text"), str)).strip()
            if text:
                texts.append(text)
                if mid:
                    seen.add(mid)
    return list(texts)


def _codex_replies(cwd: str, session_id: str, count: int) -> list[str]:
    return read_codex_replies(codex_transcript_path(session_id), session_id, count)


async def resolve_agent_session(herdr, agent: dict) -> dict:
    """Detailed pane metadata may contain identity absent from snapshots."""
    if (agent.get("agent_session") or {}).get("kind") == "id":
        return agent
    try:
        pane = (await herdr.call("pane.get", pane_id=agent["pane_id"])).get("pane", {})
    except (OSError, RuntimeError, AttributeError, KeyError):
        return agent
    if pane.get("agent") == agent.get("agent") and pane.get("agent_session"):
        return dict(agent, agent_session=pane["agent_session"])
    return agent


Adapter = Callable[[str, str, int], list[str]]

ADAPTERS: dict[str, Adapter] = {
    "claude": _claude_replies,
    "codex": _codex_replies,
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
    agent = await resolve_agent_session(herdr, agent)
    reader = adapter_for(agent.get("agent"))
    session = agent.get("agent_session") or {}
    if reader is None or session.get("kind") != "id":
        return None
    try:
        replies = reader(agent.get("cwd", ""), session["value"], max(1, count))
    except OSError:
        return None
    return "\n\n".join(replies) if replies else None
