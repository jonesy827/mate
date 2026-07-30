"""Telegram bridge daemon: pushes herdr fleet events to the user's phone and
routes typed replies back to agents. Runs OUTSIDE any voice session (the
voice agent's watcher only exists while a LiveKit job is live).

Run by hand (never at boot):  python -m matebridge.telegram_bridge

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment or .env
(see README "Telegram notifications" for the BotFather checklist). Exits
cleanly with a pointer if unconfigured.

Two-way grammar (typed text is deterministic input — no confirmation rail):
  - reply to one of the bridge's notifications  -> routes to that agent
  - "songhaus: run the tests"                   -> workspace-prefix routing
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from dotenv import load_dotenv

from . import notify
from .herdr_client import HerdrClient, HerdrError
from .transcripts import agent_last_reply

logger = logging.getLogger("matebridge.telegram")

# Same per-pane subscription model as agent.py's watch_fleet: herdr rejects a
# bare {"type": "pane.agent_status_changed"}, so resubscribe on an interval
# to pick up new panes. (Kept local: importing agent.py would drag livekit
# into this daemon.)
RESUBSCRIBE_SECS = 30.0
REPLY_EXCERPT_CHARS = 500
MAX_UPDATE_AGE_SECS = 300  # unacked updates replay for 24h; skip stale ones
SENT_MAP_MAX = 50

USAGE_HINT = ('to message an agent, reply to one of my notifications or use '
              '"<workspace>: <instruction>", e.g. "songhaus: run the tests"')


def watch_transition(last: dict[str, str], pane_id: str,
                     status: str | None) -> str | None:
    """Update last-seen state; return "finished"/"blocked" when this change
    deserves a push, else None.

    First sighting of a pane only baselines it (no announcing history), and
    a repeated status never re-pushes — one finish, one push.
    """
    if not pane_id or not status:
        return None
    prev = last.get(pane_id)
    last[pane_id] = status
    if prev is None or status == prev:
        return None
    if status in ("idle", "done") and prev == "working":
        return "finished"
    if status == "blocked":
        return "blocked"
    return None


def resolve_route(text: str, reply_to: int | None,
                  sent_map: dict[int, str], snap: dict) -> dict:
    """Where should an incoming Telegram message go?

    Returns {"pane_id", "label", "text"} on success, {"error": hint} not.
    Pure so it can be unit-tested without a socket.
    """
    text = (text or "").strip()
    if not text:
        return {"error": USAGE_HINT}

    agents = snap.get("agents", [])
    panes = {p.get("pane_id"): p for p in snap.get("panes", [])}
    labels = {w.get("workspace_id"): (w.get("label") or w.get("workspace_id"))
              for w in snap.get("workspaces", [])}

    def label_for(pane_id: str) -> str:
        ws = (panes.get(pane_id) or {}).get("workspace_id")
        return labels.get(ws, pane_id)

    # 1) reply to one of our notifications -> that pane, text as-is
    if reply_to is not None and reply_to in sent_map:
        pane_id = sent_map[reply_to]
        if any(a.get("pane_id") == pane_id for a in agents):
            return {"pane_id": pane_id, "label": label_for(pane_id),
                    "text": text}
        return {"error": f"the agent that notification came from ({pane_id}) "
                         "is gone. " + USAGE_HINT}

    # 2) "<workspace-prefix>: <instruction>"
    if ":" not in text:
        return {"error": USAGE_HINT}
    prefix, _, body = text.partition(":")
    prefix, body = prefix.strip().lower(), body.strip()
    if not prefix or not body:
        return {"error": USAGE_HINT}
    matches = [ws_id for ws_id, label in labels.items()
               if str(label).lower().startswith(prefix)]
    if not matches:
        return {"error": f'no workspace matches "{prefix}". ' + USAGE_HINT}
    if len(matches) > 1:
        names = ", ".join(str(labels[m]) for m in matches)
        return {"error": f'"{prefix}" is ambiguous ({names}) — '
                         "use more of the name"}
    ws_id = matches[0]
    pane_id = next((a.get("pane_id") for a in agents
                    if (panes.get(a.get("pane_id")) or {})
                    .get("workspace_id") == ws_id), None)
    if pane_id is None:
        return {"error": f'no coding agent is running in "{labels[ws_id]}"'}
    return {"pane_id": pane_id, "label": str(labels[ws_id]), "text": body}


class Bridge:
    def __init__(self, herdr: HerdrClient):
        self.herdr = herdr
        self.last_status: dict[str, str] = {}
        self.sent_map: dict[int, str] = {}  # our message_id -> pane_id

    def remember(self, message: dict | None, pane_id: str) -> None:
        if message and isinstance(message.get("message_id"), int):
            self.sent_map[message["message_id"]] = pane_id
            while len(self.sent_map) > SENT_MAP_MAX:
                self.sent_map.pop(next(iter(self.sent_map)))

    async def push(self, kind: str, pane_id: str, label: str) -> None:
        if kind == "blocked":
            text = f"{label}: blocked — an agent is waiting for input"
        else:
            text = f"{label}: agent finished"
            reply = await agent_last_reply(self.herdr, pane_id)
            if reply:
                text += "\n\n" + reply[:REPLY_EXCERPT_CHARS]
        self.remember(await notify.send_message(text), pane_id)

    async def handle_status(self, snap_labels: dict, pane_ws: dict,
                            pane_id: str, status: str | None,
                            workspace_id: str | None = None) -> None:
        kind = watch_transition(self.last_status, pane_id, status)
        if kind is None:
            return
        ws = workspace_id or pane_ws.get(pane_id)
        await self.push(kind, pane_id, str(snap_labels.get(ws, pane_id)))

    async def watch(self) -> None:
        """Fleet watcher: same shape as agent.py's watch_fleet — snapshot,
        per-pane subscribe, resubscribe every RESUBSCRIBE_SECS, and an outer
        loop that can never die silently."""
        while True:
            try:
                snap = await self.herdr.snapshot()
                labels = {w.get("workspace_id"):
                          (w.get("label") or w.get("workspace_id"))
                          for w in snap.get("workspaces", [])}
                pane_ws = {p.get("pane_id"): p.get("workspace_id")
                           for p in snap.get("panes", [])}
                # catch-up: transitions that happened between subscriptions
                for a in snap.get("agents", []):
                    await self.handle_status(labels, pane_ws,
                                             a.get("pane_id"),
                                             a.get("agent_status"))
                pane_ids = [p for p in pane_ws if p]
                if not pane_ids:
                    await asyncio.sleep(RESUBSCRIBE_SECS)
                    continue
                events = self.herdr.events(
                    [{"type": "pane.agent_status_changed", "pane_id": p}
                     for p in pane_ids])
                loop = asyncio.get_running_loop()
                deadline = loop.time() + RESUBSCRIBE_SECS
                try:
                    while (remaining := deadline - loop.time()) > 0:
                        try:
                            msg = await asyncio.wait_for(
                                anext(events), remaining)
                        except (TimeoutError, StopAsyncIteration):
                            break
                        data = msg.get("data", {})
                        await self.handle_status(
                            labels, pane_ws, data.get("pane_id"),
                            data.get("agent_status"),
                            data.get("workspace_id"))
                finally:
                    await events.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("watcher error, retrying in 5s")
                await asyncio.sleep(5)

    async def handle_message(self, msg: dict) -> None:
        route = resolve_route(
            msg.get("text", ""),
            (msg.get("reply_to_message") or {}).get("message_id"),
            self.sent_map,
            await self.herdr.snapshot())
        if "error" in route:
            await notify.send_message(route["error"])
            return
        try:
            await self.herdr.prompt_agent(route["pane_id"], route["text"])
        except HerdrError as e:
            await notify.send_message(
                f'could not deliver to {route["label"]}: {e.message}')
            return
        self.remember(
            await notify.send_message(f'delivered to {route["label"]}'),
            route["pane_id"])

    async def poll(self) -> None:
        """Long-poll getUpdates. Only THIS daemon may poll (a second poller
        on the same token gets HTTP 409)."""
        cfg = notify.config()
        assert cfg is not None  # main() gates on this
        offset = 0
        while True:
            try:
                updates = await notify.api_call(
                    "getUpdates",
                    {"offset": offset, "timeout": 50,
                     "allowed_updates": ["message"]},
                    timeout=httpx.Timeout(60.0))  # > long-poll timeout
                for u in updates or []:
                    offset = max(offset, u.get("update_id", 0) + 1)
                    msg = u.get("message") or {}
                    if msg.get("chat", {}).get("id") != cfg[1]:
                        continue  # not the user: drop silently, never reply
                    if time.time() - msg.get("date", 0) > MAX_UPDATE_AGE_SECS:
                        continue  # stale replayed update after a restart
                    await self.handle_message(msg)
                if updates is None:
                    await asyncio.sleep(5)  # api_call already logged it
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("poll error, retrying in 5s")
                await asyncio.sleep(5)

    async def run(self) -> None:
        await notify.send_telegram("mate bridge online")
        await asyncio.gather(self.watch(), self.poll())


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    if notify.config() is None:
        print("telegram bridge not configured: set TELEGRAM_BOT_TOKEN and "
              "TELEGRAM_CHAT_ID in .env — see README.md "
              '"Telegram notifications"')
        return 0
    try:
        asyncio.run(Bridge(HerdrClient()).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
