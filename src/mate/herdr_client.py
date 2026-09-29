"""Async client for the herdr socket API (herdr 0.8.0, protocol 19).

Wire format: newline-delimited JSON over a Unix socket.
Requests:  {"id": "<string>", "method": "pane.read", "params": {...}}

Ground-truth protocol rules (verified; see mate-groundtruth/REPORT.md):
- ONE request per connection: the server sends one response line, then
  closes. Pipelining or reusing a connection gets ECONNRESET/EPIPE.
- events.subscribe is the exception: that connection stays open and streams
  frames shaped {"event": "pane.agent_status_changed", "data": {...}}
  (no "id"; event names are dotted, same as the subscription types).
- "id" must be a string; "id", "method" and "params" are all required.
- Unknown methods produce NO reply (the connection just closes).
- Errors: {"id": ..., "error": {"code": "...", "message": "..."}}.

Pure protocol only: call(), the event stream, and 1:1 method wrappers.
Every workaround for a herdr *bug* lives in delivery.py so an upgrade has
one file to audit.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

DEFAULT_SOCK = Path(os.path.expanduser(os.environ.get(
    "HERDR_SOCKET", str(Path.home() / ".config/herdr/herdr.sock"))))

# The protocol the workarounds in delivery.py were verified against. A
# different herdr is warned about at startup, never refused: the
# workarounds degrade to no-ops on a herdr that behaves, and real breakage
# surfaces loudly as HerdrErrors.
# Verified live against 0.8.0 (protocol 19) on 2026-08-18: every method,
# param shape, response field, and subscription mate uses is unchanged from
# protocol 17; the 17->19 bump is purely additive (workspace.move_block,
# workspace.reordered). The 0.8.0 prompt-Enter fix (#1878) let delivery.py
# drop its guarded Enter nudge entirely.
TESTED_PROTOCOL = 19


def protocol_note(pong: dict) -> str | None:
    """None when herdr speaks the tested protocol, else a warning line for
    the startup log. herdr's ping response carries version + protocol."""
    proto = pong.get("protocol")
    if proto == TESTED_PROTOCOL:
        return None
    return (f"herdr {pong.get('version', '?')} speaks protocol {proto!r}; "
            f"the workarounds in delivery.py are tuned for protocol "
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
