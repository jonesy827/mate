"""HerdrClient tests against a fake server that enforces the REAL protocol
semantics established in mate-groundtruth/REPORT.md:

- one request per connection (respond once, then close)
- unknown method: close without any reply
- events.subscribe: ack, then stream frames, connection stays open
"""

import asyncio
import json

import pytest

from mate.herdr_client import (
    TESTED_PROTOCOL,
    HerdrClient,
    HerdrError,
    _find_pane_id,
    _sanitize_agent_name,
    protocol_note,
)

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


class ScriptedClient(HerdrClient):
    """Overrides call() with a per-method script of results; HerdrError
    entries are raised. Retry delays zeroed so tests run instantly."""

    START_RETRY_DELAY = 0
    PROMPT_RETRY_DELAY = 0
    NUDGE_DELAY = 0
    PROMPT_VERIFY_WAIT = 0

    def __init__(self, script):
        super().__init__("/nonexistent")
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    async def call(self, method, **params):
        self.calls.append((method, params))
        queue = self.script.get(method, [])
        result = queue.pop(0) if queue else {}
        if isinstance(result, HerdrError):
            raise result
        return result


def _busy():
    return HerdrError("agent.start", "agent_pane_busy",
                      "agent target pane w9:p1 is not an available shell")


async def test_spawn_in_folder_retries_until_shell_ready():
    # the songhaus bug, part 1: agent.start 100ms after workspace.create
    # fails because the fresh pane has no available shell yet
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [_busy(), _busy(), {}],
    })
    result = await c.spawn_in_folder("/src/songhaus", "songhaus")
    assert result["pane_id"] == "w9:p1"
    assert result["agent_name"] == "songhaus"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert len(starts) == 3
    assert starts[0]["kind"] == "claude"
    # herdr's managed-launch deadline stretched past its 30s default so a
    # slow claude boot isn't abandoned server-side either
    assert starts[0]["timeout_ms"] == HerdrClient.LAUNCH_TIMEOUT_MS
    assert not any(m == "workspace.close" for m, _ in c.calls)
    # spawn never prompts: the task is delivered separately once ready
    assert not any(m == "agent.prompt" for m, _ in c.calls)


async def test_spawn_in_folder_unique_name_on_duplicate():
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [HerdrError("agent.start", "duplicate_agent_name",
                                   "taken"), {}],
    })
    result = await c.spawn_in_folder("/src/songhaus", "songhaus")
    assert result["agent_name"] == "songhaus-2"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert [s["name"] for s in starts] == ["songhaus", "songhaus-2"]


def test_sanitize_agent_name():
    # herdr requires ^[a-z][a-z0-9_-]{0,31}$ (invalid_agent_name otherwise)
    assert _sanitize_agent_name("RackCoach") == "rackcoach"
    assert _sanitize_agent_name("My Repo!") == "my-repo"
    assert _sanitize_agent_name("Fix/The Thing") == "fix-the-thing"
    assert _sanitize_agent_name("123abc") == "abc"  # must start with a letter
    assert _sanitize_agent_name("snake_case_ok") == "snake_case_ok"
    assert _sanitize_agent_name("!!!") == "agent"
    assert _sanitize_agent_name("") == "agent"
    assert _sanitize_agent_name("x" * 50) == "x" * 32


async def test_spawn_in_folder_sanitizes_agent_name():
    # the RackCoach bug: the folder label went to agent.start verbatim and
    # herdr rejected the capital letters instantly, three calls in a row
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [{}],
    })
    result = await c.spawn_in_folder("/src/RackCoach", "RackCoach")
    assert result["agent_name"] == "rackcoach"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["name"] == "rackcoach"
    assert not any(m == "workspace.close" for m, _ in c.calls)


async def test_duplicate_suffix_stays_within_name_limit():
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [HerdrError("agent.start", "duplicate_agent_name",
                                   "taken"), {}],
    })
    result = await c.spawn_in_folder("/src/x", "X" * 40)
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["name"] == "x" * 32
    assert starts[1]["name"] == "x" * 30 + "-2"
    assert result["agent_name"] == starts[1]["name"]


async def test_spawn_in_folder_agent_kind_passes_through():
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [{}],
    })
    await c.spawn_in_folder("/src/x", "x", agent="pi")
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["kind"] == "pi"


async def test_spawn_in_folder_closes_workspace_on_start_failure():
    fatal = HerdrError("agent.start", "invalid_agent_name", "bad")
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [fatal],
    })
    with pytest.raises(HerdrError) as ei:
        await c.spawn_in_folder("/src/x", "x")
    assert ei.value.code == "invalid_agent_name"
    closes = [p for m, p in c.calls if m == "workspace.close"]
    assert closes == [{"workspace_id": "w9", "force": True}]


async def test_spawn_in_folder_gives_up_after_persistent_busy():
    c = ScriptedClient({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [_busy() for _ in range(HerdrClient.START_RETRIES)],
    })
    with pytest.raises(HerdrError) as ei:
        await c.spawn_in_folder("/src/x", "x")
    assert ei.value.code == "agent_pane_busy"
    closes = [m for m, _ in c.calls if m == "workspace.close"]
    assert closes == ["workspace.close"]


async def test_deliver_task_retries_through_boot():
    # the songhaus bug, part 2: claude's first boot in a folder keeps
    # agent.prompt at agent_not_ready for a long time — delivery must wait
    # it out instead of declaring the launch failed
    not_ready = HerdrError("agent.prompt", "agent_not_ready", "pending")
    c = ScriptedClient({
        "agent.prompt": [not_ready] * 30 + [{}],
    })
    await c.deliver_task("w9:p1", "do it")
    assert sum(1 for m, _ in c.calls if m == "agent.prompt") == 31
    # paste-guard self-heal: one bare Enter follows every delivery
    keys = [p for m, p in c.calls if m == "pane.send_keys"]
    assert keys == [{"pane_id": "w9:p1", "keys": ["Enter"]}]


async def test_deliver_task_falls_back_to_pane_input_when_launch_stuck():
    # herdr 0.7.5 bug seen live: launch_pending never clears even though
    # the agent is detected idle — agent.prompt refuses forever. Once the
    # snapshot proves an idle agent owns the pane, type into it directly.
    not_ready = HerdrError("agent.prompt", "agent_not_ready",
                           "agent w9:p1 is not an active named agent")
    # two agent.list entries: delivery samples the agent's state once up
    # front (dropped-prompt detection), then the fallback guard reads it
    idle = {"agents": [{"pane_id": "w9:p1", "agent": "claude",
                        "agent_status": "idle"}]}
    c = ScriptedClient({
        "agent.prompt": [not_ready] * (HerdrClient.FALLBACK_AFTER + 1),
        "agent.list": [idle, idle],
    })
    await c.deliver_task("w9:p1", "do it")
    sends = [p for m, p in c.calls if m == "pane.send_input"]
    assert sends == [{"pane_id": "w9:p1", "text": "do it",
                      "keys": ["Enter"]}]
    keys = [m for m, _ in c.calls if m == "pane.send_keys"]
    assert keys == ["pane.send_keys"]


async def test_deliver_task_never_types_into_a_bare_shell():
    # fallback must not fire when no settled agent owns the pane — typing
    # a task into a shell would execute it as a command
    not_ready = HerdrError("agent.prompt", "agent_not_ready", "pending")
    c = ScriptedClient({
        "agent.prompt": [not_ready] * HerdrClient.PROMPT_RETRIES,
        "agent.list": [{"agents": []}
                       for _ in range(HerdrClient.PROMPT_RETRIES)],
    })
    with pytest.raises(HerdrError):
        await c.deliver_task("w9:p1", "rm -rf importantdir")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


def _listed_agent(status="idle", seq=381):
    return {"agents": [{"pane_id": "w9:p1", "agent": "claude",
                        "agent_status": status, "state_change_seq": seq}]}


async def test_deliver_task_types_in_when_prompt_silently_dropped():
    # herdr 0.7.5 bug seen live (worktree-workspace pane): agent.prompt
    # answers agent_prompted but never types anything — the agent's
    # state_change_seq stays frozen. Delivery must notice and type directly.
    c = ScriptedClient({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent() for _ in range(10)],
    })
    await c.deliver_task("w9:p1", "do it")
    sends = [p for m, p in c.calls if m == "pane.send_input"]
    assert sends == [{"pane_id": "w9:p1", "text": "do it",
                      "keys": ["Enter"]}]


async def test_deliver_task_no_fallback_when_prompt_lands():
    # state_change_seq advanced after the accepted prompt -> it landed;
    # typing as well would deliver the task twice
    c = ScriptedClient({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(seq=381), _listed_agent(seq=382)],
    })
    await c.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


async def test_dropped_prompt_fallback_needs_settled_agent():
    # frozen seq but the agent shows working: it is busy on something else,
    # so the settled-agent guard must veto typing into its terminal
    c = ScriptedClient({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(status="working") for _ in range(10)],
    })
    await c.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


async def test_deliver_task_raises_when_agent_never_ready():
    c = ScriptedClient({
        "agent.prompt": [HerdrError("agent.prompt", "agent_not_ready", "x")
                         for _ in range(HerdrClient.PROMPT_RETRIES)],
    })
    with pytest.raises(HerdrError) as ei:
        await c.deliver_task("w9:p1", "t")
    assert ei.value.code == "agent_not_ready"
    # delivery failure must never tear anything down
    assert not any(m == "workspace.close" for m, _ in c.calls)


# --- protocol note (warn-only version check on the preflight pong) -----

async def test_protocol_note_quiet_on_tested_protocol():
    pong = {"type": "pong", "version": "0.7.5",
            "protocol": TESTED_PROTOCOL}
    assert protocol_note(pong) is None


async def test_protocol_note_warns_on_other_protocol():
    note = protocol_note({"type": "pong", "version": "0.9.0",
                          "protocol": TESTED_PROTOCOL + 1})
    assert note is not None
    assert "0.9.0" in note and str(TESTED_PROTOCOL) in note


async def test_protocol_note_warns_when_protocol_missing():
    note = protocol_note({"type": "pong"})
    assert note is not None and str(TESTED_PROTOCOL) in note
