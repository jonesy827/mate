"""Async client for the herdr socket API (herdr 0.7.5, protocol 17).

Wire format: newline-delimited JSON over a Unix socket.
Requests:  {"id": "<string>", "method": "pane.read", "params": {...}}

Ground-truth protocol rules (verified; see matebridge-groundtruth/REPORT.md):
- ONE request per connection: the server sends one response line, then
  closes. Pipelining or reusing a connection gets ECONNRESET/EPIPE.
- events.subscribe is the exception: that connection stays open and streams
  frames shaped {"event": "pane.agent_status_changed", "data": {...}}
  (no "id"; event names are dotted, same as the subscription types).
- "id" must be a string; "id", "method" and "params" are all required.
- Unknown methods produce NO reply (the connection just closes).
- Errors: {"id": ..., "error": {"code": "...", "message": "..."}}.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

DEFAULT_SOCK = Path(os.environ.get(
    "HERDR_SOCKET", Path.home() / ".config/herdr/herdr.sock"))

# The protocol this client's workarounds were verified against. A different
# herdr is warned about at startup, never refused: the workarounds degrade
# to no-ops on a herdr that behaves, and real breakage surfaces loudly as
# HerdrErrors.
TESTED_PROTOCOL = 17


def protocol_note(pong: dict) -> str | None:
    """None when herdr speaks the tested protocol, else a warning line for
    the startup log. herdr's ping response carries version + protocol."""
    proto = pong.get("protocol")
    if proto == TESTED_PROTOCOL:
        return None
    return (f"herdr {pong.get('version', '?')} speaks protocol {proto!r}; "
            f"the workarounds in herdr_client.py are tuned for protocol "
            f"{TESTED_PROTOCOL} and may misbehave")


class HerdrError(RuntimeError):
    def __init__(self, method: str, code: str, message: str):
        super().__init__(f"{method}: [{code}] {message}")
        self.method = method
        self.code = code
        self.message = message


class HerdrClient:
    """Connection-per-request client; safe for concurrent use."""

    def __init__(self, sock_path: Path | str = DEFAULT_SOCK,
                 timeout: float = 15.0):
        self.sock_path = str(sock_path)
        self.timeout = timeout
        self._ids = itertools.count(1)

    async def call(self, method: str, **params: Any) -> dict:
        reader, writer = await asyncio.open_unix_connection(self.sock_path)
        try:
            rid = f"mb-{next(self._ids)}"
            writer.write(json.dumps(
                {"id": rid, "method": method, "params": params}).encode() + b"\n")
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), self.timeout)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if not line:
            raise HerdrError(method, "no_reply",
                             "server closed without replying "
                             "(unknown method or protocol error)")
        msg = json.loads(line)
        if err := msg.get("error"):
            raise HerdrError(method, err.get("code", "?"), err.get("message", ""))
        return msg.get("result", {})

    async def events(
            self, subscriptions: Iterable[dict | str]) -> AsyncIterator[dict]:
        """Subscribe and yield event frames {"event": ..., "data": ...} forever.

        Each subscription is a dict passed through verbatim, e.g.
        {"type": "pane.agent_status_changed", "pane_id": "w2:p1"}; a bare
        string t becomes {"type": t}. Pane-scoped types
        (pane.agent_status_changed, pane.scroll_changed, pane.output_matched)
        REQUIRE a pane_id — the server rejects a bare {"type": ...} with
        invalid_request. Event frames use the same dotted names as the
        subscription types, with a flat data payload (verified live:
        {"event": "pane.agent_status_changed", "data": {"pane_id": ...,
        "agent_status": ..., ...}}).
        """
        reader, writer = await asyncio.open_unix_connection(self.sock_path)
        try:
            writer.write(json.dumps({
                "id": f"mb-ev-{next(self._ids)}",
                "method": "events.subscribe",
                "params": {"subscriptions": [
                    {"type": s} if isinstance(s, str) else s
                    for s in subscriptions]},
            }).encode() + b"\n")
            await writer.drain()
            ack = json.loads(await asyncio.wait_for(
                reader.readline(), self.timeout))
            if err := ack.get("error"):
                raise HerdrError("events.subscribe",
                                 err.get("code", "?"), err.get("message", ""))
            while True:
                line = await reader.readline()
                if not line:
                    return  # server went away
                msg = json.loads(line)
                if "event" in msg:
                    yield msg
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    # ---- convenience wrappers used by the voice agent's tools ----

    async def snapshot(self) -> dict:
        result = await self.call("session.snapshot")
        return result.get("snapshot", result)

    async def agents(self) -> dict:
        return await self.call("agent.list")

    async def read_pane(self, pane_id: str, lines: int = 80,
                        source: str = "recent") -> str:
        result = await self.call("pane.read", pane_id=pane_id, source=source,
                                 lines=lines, strip_ansi=True)
        return result.get("read", {}).get("text", "")

    async def send_keys(self, pane_id: str, keys: list[str]):
        return await self.call("pane.send_keys", pane_id=pane_id, keys=keys)

    async def send_input(self, pane_id: str, text: str, submit: bool = True):
        params: dict[str, Any] = {"pane_id": pane_id, "text": text}
        if submit:
            params["keys"] = ["Enter"]
        return await self.call("pane.send_input", **params)

    async def prompt_agent(self, target: str, text: str,
                           wait: dict | None = None):
        params: dict[str, Any] = {"target": target, "text": text}
        if wait:
            params["wait"] = wait
        return await self.call("agent.prompt", **params)

    # A freshly created pane's shell needs a moment before agent.start
    # succeeds (herdr: agent_pane_busy "not an available shell"). Verified
    # live 2026-07-30: immediate agent.start after workspace.create fails.
    START_RETRIES = 12
    START_RETRY_DELAY = 0.4
    # herdr rejects agent.prompt (agent_not_ready) until it *detects* the
    # launched agent idle — claude's first boot in a folder (trust dialog,
    # big repo) can take well over a minute, so task delivery must be
    # patient and must never be treated as a launch failure. timeout_ms
    # stretches herdr's own managed-launch deadline (default 30s) to match.
    LAUNCH_TIMEOUT_MS = 120_000
    PROMPT_RETRIES = 240
    PROMPT_RETRY_DELAY = 0.5
    # herdr 0.7.5 bug seen live 2026-07-30: a managed launch can stay
    # launch_pending forever even though detection reports the agent idle
    # ("not an active named agent" on every prompt). After this many prompt
    # attempts, if the snapshot proves the pane hosts a settled agent, type
    # the message straight into the pane instead of using the gated prompt.
    FALLBACK_AFTER = 40  # attempts (~20s), rechecked every 10 after that
    # Claude Code's paste guard sometimes swallows the Enter herdr sends
    # 300ms after the text, leaving the message stuck in the input box. A
    # bare Enter shortly after delivery is a no-op if the message went
    # through and submits it if it didn't.
    NUDGE_DELAY = 2.0

    async def start_agent(self, pane_id: str, name: str,
                          kind: str = "claude") -> str:
        """agent.start with shell-settle retries. The requested name is
        first folded to herdr's naming rules (seen live 2026-07-30: label
        "RackCoach" → instant invalid_agent_name). Agent names are unique
        fleet-wide in herdr, so a taken name gets a numeric suffix. Returns
        the agent name actually used; raises HerdrError on failure."""
        base = name = _sanitize_agent_name(name)
        for attempt in range(self.START_RETRIES):
            try:
                await self.call("agent.start", pane_id=pane_id,
                                name=name, kind=kind,
                                timeout_ms=self.LAUNCH_TIMEOUT_MS)
                return name
            except HerdrError as e:
                if e.code == "duplicate_agent_name":
                    suffix = f"-{attempt + 2}"
                    name = base[:_AGENT_NAME_MAX - len(suffix)] + suffix
                    continue
                if (e.code == "agent_pane_busy"
                        and attempt < self.START_RETRIES - 1):
                    await asyncio.sleep(self.START_RETRY_DELAY)
                    continue
                raise
        return name

    async def _pane_hosts_settled_agent(self, pane_id: str) -> bool:
        """True when herdr's detection shows a coding agent sitting idle or
        blocked in this pane — the precondition for typing a message into
        the pane directly (never type into a bare shell)."""
        try:
            listed = await self.agents()
        except HerdrError:
            return False
        for a in listed.get("agents", []):
            if (a.get("pane_id") == pane_id and a.get("agent")
                    and a.get("agent_status") in ("idle", "blocked")):
                return True
        return False

    async def nudge_enter(self, pane_id: str) -> None:
        """One bare Enter, a beat after a delivery — submits a message the
        paste guard left stuck in the input box, no-op otherwise."""
        await asyncio.sleep(self.NUDGE_DELAY)
        try:
            await self.send_keys(pane_id, ["Enter"])
        except HerdrError:
            pass  # the nudge is best-effort; the message itself was sent

    async def deliver_task(self, pane_id: str, task: str) -> None:
        """Hand a prompt to an agent, retrying through the whole boot window
        (agent_not_ready until herdr detects the agent idle). If herdr's
        launch tracking is stuck (launch_pending forever while the agent is
        visibly idle — seen live on 0.7.5), falls back to typing into the
        pane once the snapshot proves an agent is settled there. Raises
        HerdrError if the agent never becomes promptable."""
        for attempt in range(self.PROMPT_RETRIES):
            try:
                await self.prompt_agent(pane_id, task)
                await self.nudge_enter(pane_id)
                return
            except HerdrError as e:
                if (e.code not in ("agent_not_ready", "agent_not_found")
                        or attempt >= self.PROMPT_RETRIES - 1):
                    raise
                if (attempt >= self.FALLBACK_AFTER
                        and (attempt - self.FALLBACK_AFTER) % 10 == 0
                        and await self._pane_hosts_settled_agent(pane_id)):
                    await self.send_input(pane_id, task)
                    await self.nudge_enter(pane_id)
                    return
                await asyncio.sleep(self.PROMPT_RETRY_DELAY)

    async def spawn_in_folder(self, path: str, label: str,
                              agent: str = "claude") -> dict:
        """Open a workspace directly in an existing folder (no worktree) and
        start a coding agent (claude by default). The agent edits the real
        working tree. The task is NOT delivered here — the caller hands it
        over with deliver_task() once the agent finishes booting. Only when
        agent.start itself fails is the workspace closed again; once the
        agent is running the workspace is never torn down (a slow boot is
        not a failed launch)."""
        ws = await self.call("workspace.create", cwd=path, label=label,
                             focus=False)
        pane_id = _find_pane_id(ws)
        if not pane_id:
            return {"workspace": ws, "pane_id": None}
        try:
            name = await self.start_agent(pane_id, label, kind=agent)
        except HerdrError:
            ws_id = _find_workspace_id(ws)
            if ws_id:
                try:
                    await self.call("workspace.close", workspace_id=ws_id,
                                    force=True)
                except HerdrError:
                    pass  # surface the original launch error, not this one
            raise
        return {"workspace": ws, "pane_id": pane_id, "agent_name": name}

    async def spawn(self, repo_path: str, branch: str) -> dict:
        """Create a worktree workspace in repo_path and start Claude. The
        task is delivered by the caller via deliver_task() once ready."""
        wt = await self.call("worktree.create", cwd=repo_path, branch=branch,
                             focus=False)
        # worktree.create response shape is untested ground (needs a git
        # workspace); find the new workspace's pane defensively.
        pane_id = _find_pane_id(wt)
        name = None
        if pane_id:
            name = await self.start_agent(pane_id, branch)
        return {"worktree": wt, "pane_id": pane_id, "agent_name": name}


_AGENT_NAME_MAX = 32


def _sanitize_agent_name(label: str) -> str:
    """Fold an arbitrary label (folder or branch name) into a valid herdr
    agent name: ^[a-z][a-z0-9_-]{0,31}$. herdr rejects anything else with
    invalid_agent_name before even touching the pane."""
    name = re.sub(r"[^a-z0-9_-]+", "-", label.lower())
    name = re.sub(r"-+", "-", name).lstrip("0123456789-_").rstrip("-_")
    if not name:
        return "agent"
    return name[:_AGENT_NAME_MAX].rstrip("-_")


def _find_workspace_id(obj: Any) -> str | None:
    """Depth-first hunt for a workspace_id in a response of unknown shape."""
    if isinstance(obj, dict):
        if isinstance(obj.get("workspace_id"), str):
            return obj["workspace_id"]
        for v in obj.values():
            if (found := _find_workspace_id(v)) is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            if (found := _find_workspace_id(v)) is not None:
                return found
    return None


def _find_pane_id(obj: Any) -> str | None:
    """Depth-first hunt for a pane_id in a response of unknown shape."""
    if isinstance(obj, dict):
        if isinstance(obj.get("pane_id"), str):
            return obj["pane_id"]
        for v in obj.values():
            if (found := _find_pane_id(v)) is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            if (found := _find_pane_id(v)) is not None:
                return found
    return None
