"""Telegram bridge: routing, dedupe, and the notify no-op/truncation rules.
No network: notify gets an httpx.MockTransport."""

import json

import httpx
import pytest

import matebridge.notify as notify
from matebridge.telegram_bridge import resolve_route, watch_transition

SNAP = {
    "workspaces": [
        {"workspace_id": "w1", "label": "songhaus"},
        {"workspace_id": "w2", "label": "songhaus-mobile"},
        {"workspace_id": "w3", "label": "trainsongz"},
        {"workspace_id": "w4", "label": "empty-ws"},
    ],
    "panes": [
        {"pane_id": "w1:p1", "workspace_id": "w1"},
        {"pane_id": "w2:p1", "workspace_id": "w2"},
        {"pane_id": "w3:p1", "workspace_id": "w3"},
        {"pane_id": "w4:p1", "workspace_id": "w4"},
    ],
    "agents": [
        {"pane_id": "w1:p1", "agent": "claude"},
        {"pane_id": "w2:p1", "agent": "claude"},
        {"pane_id": "w3:p1", "agent": "claude"},
    ],
}


# ---- resolve_route ----

def test_route_reply_to_notification():
    r = resolve_route("run the tests", 42, {42: "w3:p1"}, SNAP)
    assert r == {"pane_id": "w3:p1", "label": "trainsongz",
                 "text": "run the tests"}


def test_route_reply_to_vanished_agent():
    snap = dict(SNAP, agents=[])
    r = resolve_route("hi", 42, {42: "w3:p1"}, snap)
    assert "gone" in r["error"]


def test_route_prefix_match():
    r = resolve_route("train: fix the flaky test", None, {}, SNAP)
    assert r == {"pane_id": "w3:p1", "label": "trainsongz",
                 "text": "fix the flaky test"}


def test_route_prefix_ambiguous():
    r = resolve_route("songhaus: do it", None, {}, SNAP)
    assert "ambiguous" in r["error"]
    assert "songhaus-mobile" in r["error"]


def test_route_longer_prefix_disambiguates():
    r = resolve_route("songhaus-m: do it", None, {}, SNAP)
    assert r["pane_id"] == "w2:p1"


def test_route_unknown_prefix():
    r = resolve_route("nonesuch: hello", None, {}, SNAP)
    assert "no workspace matches" in r["error"]


def test_route_no_colon_no_reply_is_usage_hint():
    r = resolve_route("just some text", None, {}, SNAP)
    assert "reply to one of my notifications" in r["error"]


def test_route_workspace_without_agent():
    r = resolve_route("empty: hello", None, {}, SNAP)
    assert "no coding agent" in r["error"]


def test_route_empty_text():
    assert "error" in resolve_route("   ", None, {}, SNAP)


# ---- watch_transition dedupe ----

def test_transition_first_sighting_baselines_only():
    last = {}
    assert watch_transition(last, "p", "idle") is None
    assert watch_transition(last, "p", "idle") is None  # still nothing


def test_transition_working_to_idle_pushes_once():
    last = {"p": "working"}
    assert watch_transition(last, "p", "idle") == "finished"
    assert watch_transition(last, "p", "idle") is None  # dedupe
    assert watch_transition(last, "p", "working") is None
    assert watch_transition(last, "p", "done") == "finished"


def test_transition_blocked_pushes_until_unblocked():
    last = {"p": "working"}
    assert watch_transition(last, "p", "blocked") == "blocked"
    assert watch_transition(last, "p", "blocked") is None
    assert watch_transition(last, "p", "working") is None
    assert watch_transition(last, "p", "blocked") == "blocked"  # re-blocked


def test_transition_idle_from_blocked_is_quiet():
    # only working -> idle/done announces a finish
    last = {"p": "blocked"}
    assert watch_transition(last, "p", "idle") is None


# ---- notify ----

@pytest.fixture
def telegram_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:AAtest")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987")
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "ok": True, "result": {"message_id": 7}})

    monkeypatch.setattr(notify, "_client", None)
    monkeypatch.setattr(notify, "_transport", httpx.MockTransport(handler))
    yield sent
    notify._client = None


@pytest.mark.asyncio
async def test_notify_unconfigured_is_noop(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notify, "_client", None)
    assert await notify.send_telegram("hello") is False
    assert notify._client is None  # never even built a client


@pytest.mark.asyncio
async def test_notify_bad_chat_id_is_noop(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:AAtest")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "not-a-number")
    assert notify.config() is None


@pytest.mark.asyncio
async def test_notify_sends_and_truncates(telegram_env):
    ok = await notify.send_telegram("x" * 5000)
    assert ok is True
    (payload,) = telegram_env
    assert payload["chat_id"] == 987
    assert len(payload["text"]) == notify.MAX_TEXT
    assert payload["disable_web_page_preview"] is True
    assert "parse_mode" not in payload


@pytest.mark.asyncio
async def test_notify_returns_message_for_sent_map(telegram_env):
    msg = await notify.send_message("ping", reply_to=3)
    assert msg["message_id"] == 7
    assert telegram_env[0]["reply_to_message_id"] == 3


@pytest.mark.asyncio
async def test_notify_http_error_is_false(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:AAtest")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987")
    monkeypatch.setattr(notify, "_client", None)
    monkeypatch.setattr(notify, "_transport", httpx.MockTransport(
        lambda req: httpx.Response(429, json={"ok": False})))
    try:
        assert await notify.send_telegram("hi") is False
    finally:
        notify._client = None
