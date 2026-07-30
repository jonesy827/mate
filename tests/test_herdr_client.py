"""HerdrClient tests against a fake server that enforces the REAL protocol
semantics established in matebridge-groundtruth/REPORT.md:

- one request per connection (respond once, then close)
- unknown method: close without any reply
- events.subscribe: ack, then stream frames, connection stays open
"""

import asyncio
import json

import pytest

from matebridge.herdr_client import HerdrClient, HerdrError, _find_pane_id

pytestmark = pytest.mark.asyncio


class FakeHerdr:
    def __init__(self):
        self.requests: list[dict] = []
        self.server = None
        self.path = None

    async def start(self, tmp_path):
        self.path = str(tmp_path / "herdr.sock")
        self.server = await asyncio.start_unix_server(self._handle, self.path)
        return self.path

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, reader, writer):
        line = await reader.readline()
        if not line:
            writer.close()
            return
        req = json.loads(line)
        self.requests.append(req)
        method = req["method"]
        rid = req["id"]

        if method == "events.subscribe":
            # Real server: pane-scoped subscription types REQUIRE pane_id;
            # a bare {"type": "pane.agent_status_changed"} is rejected.
            pane_scoped = ("pane.agent_status_changed", "pane.scroll_changed",
                           "pane.output_matched")
            for sub in req["params"]["subscriptions"]:
                if sub["type"] in pane_scoped and "pane_id" not in sub:
                    writer.write(json.dumps({"id": "", "error": {
                        "code": "invalid_request",
                        "message": "invalid request: missing field `pane_id`",
                    }}).encode() + b"\n")
                    await writer.drain()
                    writer.close()
                    return
            writer.write(json.dumps(
                {"id": rid, "result": {"type": "subscription_started"}},
            ).encode() + b"\n")
            await writer.drain()
            for i in range(3):
                # frames use dotted names + flat data, like the real server
                writer.write(json.dumps({
                    "event": "pane.agent_status_changed",
                    "data": {"pane_id": f"w1:p{i}", "agent_status": "blocked",
                             "workspace_id": "w1"},
                }).encode() + b"\n")
                await writer.drain()
            writer.close()
            return

        if method == "totally.bogus":
            writer.close()  # real server: no reply at all
            return

        if method == "agent.get":
            resp = {"id": rid, "error": {
                "code": "agent_not_found", "message": "nope"}}
        elif method == "pane.read":
            resp = {"id": rid, "result": {"type": "pane_read", "read": {
                "pane_id": req["params"]["pane_id"], "source": "recent",
                "format": "text", "text": "hello from pane\n",
                "revision": 3, "truncated": False}}}
        elif method == "session.snapshot":
            resp = {"id": rid, "result": {"type": "session_snapshot",
                                          "snapshot": {"panes": [], "workspaces": []}}}
        else:
            resp = {"id": rid, "result": {"type": "ok", "echo": method}}
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()
        writer.close()  # ONE request per connection, like the real server


@pytest.fixture
async def fake(tmp_path):
    f = FakeHerdr()
    await f.start(tmp_path)
    yield f
    await f.stop()


async def test_basic_call(fake):
    c = HerdrClient(fake.path)
    result = await c.call("ping")
    assert result["echo"] == "ping"
    assert fake.requests[0]["params"] == {}
    assert isinstance(fake.requests[0]["id"], str)  # ids must be strings


async def test_concurrent_calls(fake):
    # connection-per-request must make parallel calls safe
    c = HerdrClient(fake.path)
    results = await asyncio.gather(*(c.call("ping") for _ in range(10)))
    assert all(r["echo"] == "ping" for r in results)
    assert len(fake.requests) == 10
    assert len({r["id"] for r in fake.requests}) == 10  # unique ids


async def test_error_raises(fake):
    c = HerdrClient(fake.path)
    with pytest.raises(HerdrError) as ei:
        await c.call("agent.get", target="nope")
    assert ei.value.code == "agent_not_found"


async def test_silent_close_raises_no_reply(fake):
    c = HerdrClient(fake.path, timeout=2.0)
    with pytest.raises(HerdrError) as ei:
        await c.call("totally.bogus")
    assert ei.value.code == "no_reply"


async def test_read_pane_extracts_text(fake):
    c = HerdrClient(fake.path)
    text = await c.read_pane("w1:p1", lines=40)
    assert text == "hello from pane\n"
    params = fake.requests[0]["params"]
    assert params["source"] == "recent"
    assert params["strip_ansi"] is True


async def test_send_input_appends_enter(fake):
    c = HerdrClient(fake.path)
    await c.send_input("w1:p1", "echo hi")
    assert fake.requests[0]["params"] == {
        "pane_id": "w1:p1", "text": "echo hi", "keys": ["Enter"]}
    await c.send_input("w1:p1", "partial", submit=False)
    assert "keys" not in fake.requests[1]["params"]


async def test_snapshot_unwraps(fake):
    c = HerdrClient(fake.path)
    snap = await c.snapshot()
    assert snap == {"panes": [], "workspaces": []}


async def test_events_stream(fake):
    c = HerdrClient(fake.path)
    got = []
    async for msg in c.events(
            [{"type": "pane.agent_status_changed", "pane_id": "w1:p1"}]):
        got.append(msg)
    assert len(got) == 3
    assert got[0]["event"] == "pane.agent_status_changed"
    assert got[0]["data"]["agent_status"] == "blocked"
    sub = fake.requests[0]
    assert sub["params"] == {"subscriptions": [
        {"type": "pane.agent_status_changed", "pane_id": "w1:p1"}]}


async def test_events_dict_subscriptions_pass_through(fake):
    # dicts go on the wire verbatim; bare strings become {"type": t}
    c = HerdrClient(fake.path)
    async for _ in c.events([
            {"type": "pane.agent_status_changed", "pane_id": "w2:p1"},
            {"type": "pane.agent_status_changed", "pane_id": "w2:p2"},
            "workspace.created"]):
        break
    assert fake.requests[0]["params"]["subscriptions"] == [
        {"type": "pane.agent_status_changed", "pane_id": "w2:p1"},
        {"type": "pane.agent_status_changed", "pane_id": "w2:p2"},
        {"type": "workspace.created"}]


async def test_events_pane_scoped_without_pane_id_rejected(fake):
    # the exact bug that silently killed watch_fleet: fleet-wide subscribe
    c = HerdrClient(fake.path)
    with pytest.raises(HerdrError) as ei:
        async for _ in c.events(["pane.agent_status_changed"]):
            pass
    assert ei.value.code == "invalid_request"


async def test_find_pane_id():
    assert _find_pane_id({"a": [{"pane": {"pane_id": "w2:p9"}}]}) == "w2:p9"
    assert _find_pane_id({"nothing": [1, "x", None]}) is None
