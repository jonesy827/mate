"""Fire-and-forget Telegram pushes. Safe to call from anywhere: never
raises, and is a silent no-op when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
are unset.

Plain text only (no parse_mode): MarkdownV2 needs 18 characters escaped and
agent output would trip it constantly; a formatting 400 is a lost
notification.
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("matebridge.telegram")

API = "https://api.telegram.org/bot{token}/{method}"
MAX_TEXT = 4096  # sendMessage hard limit

# Lazily created inside the running loop; tests inject a MockTransport via
# _transport and reset _client.
_client: httpx.AsyncClient | None = None
_transport: httpx.AsyncBaseTransport | None = None


def config() -> tuple[str, int] | None:
    """(token, chat_id) from the environment, or None if not configured."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return None
    try:
        return token, int(chat)
    except ValueError:
        log.warning("TELEGRAM_CHAT_ID is not an integer: %r", chat)
        return None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(10.0),
                                    transport=_transport)
    return _client


async def api_call(method: str, payload: dict,
                   timeout: httpx.Timeout | None = None) -> dict | None:
    """POST one Bot API method. Returns the decoded result, None on any
    failure. Never raises."""
    cfg = config()
    if cfg is None:
        return None
    try:
        r = await client().post(API.format(token=cfg[0], method=method),
                                json=payload, timeout=timeout)
        if r.status_code != 200:
            log.warning("telegram %s -> %s: %s",
                        method, r.status_code, r.text[:200])
            return None
        body = r.json()
        return body.get("result") if body.get("ok") else None
    except Exception:
        log.exception("telegram %s failed", method)
        return None


async def send_message(text: str, reply_to: int | None = None) -> dict | None:
    """Send a plain-text message; returns the Telegram message object
    (contains message_id) or None."""
    cfg = config()
    if cfg is None:
        return None
    payload: dict = {"chat_id": cfg[1], "text": text[:MAX_TEXT],
                     "disable_web_page_preview": True}
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to
    return await api_call("sendMessage", payload)


async def send_telegram(text: str) -> bool:
    """Push a plain-text message. Never raises — False on any failure or
    when unconfigured."""
    return await send_message(text) is not None
