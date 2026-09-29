import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from livekit import rtc
from livekit.agents import StopResponse, llm

from mate.approval import ApprovalConfig
from mate.folders import KnownAgents
from mate.live import LiveBridge, LiveConfig, ResponsesTools
from mate.live_agent import LiveMate


def config():
    return LiveConfig("test-key", "gpt-live-1", "gpt-6-luna", "marin", ApprovalConfig())


def make_mate(tmp_path):
    herdr = SimpleNamespace(snapshot=AsyncMock(return_value={}),
                            prompt_agent=AsyncMock(return_value={}))
    mate = LiveMate(herdr, config(), known=KnownAgents(tmp_path / "known.json"))
    mate.bridge.say = AsyncMock()
    mate.bridge.append = AsyncMock()
    return mate


async def say_user(mate, text):
    if mate.locked:
        await mate.on_user_turn_completed(None, llm.ChatMessage(role="user", content=[text]))
        return
    start = mate.input_end_ms + 1500
    mate.record_live_input(text, start, start + 200)


def test_live_config_requires_separate_key_and_has_backend(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "some-other-provider")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        LiveConfig.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    session = LiveConfig.from_env().session([], "backend rules")
    assert LiveConfig.from_env().approval.api_key == "test-key"
    assert LiveConfig.from_env().approval.model == "gpt-6-luna"
    assert session["delegation"]["responses"]["reasoning"] == {"effort": "none"}
    assert session["model"] == "gpt-live-1"
    assert session["delegation"]["responses"]["parallel_tool_calls"] is False
    assert session["audio"]["format"]["rate"] == 24000


async def test_live_send_needs_readback_but_no_transcript_or_classifier(tmp_path):
    mate = make_mate(tmp_path)
    out = await mate.execute_live_tool("tell_agent", '{"pane_id":"w1:p1","text":"test it"}')
    assert "Staged only" in out
    mate.herdr.prompt_agent.assert_not_awaited()
    assert "test it" in mate.bridge.say.call_args.args[0]
    # GPT-Live/backend owns interpretation of approval, including when transcript
    # deltas arrive late or contain continuous background conversation.
    assert not mate.turns
    assert "delivered" in await mate.execute_live_tool("send_staged", "{}")
    mate.herdr.prompt_agent.assert_awaited_once_with("w1:p1", "test it")
    assert "nothing is staged" in await mate.execute_live_tool("send_staged", "{}")


async def test_send_before_readback_finishes_is_refused(tmp_path):
    mate = make_mate(tmp_path)
    results = []

    async def playback(text):
        results.append(await mate.execute_live_tool("send_staged", "{}"))

    mate.bridge.say.side_effect = playback
    await mate.execute_live_tool("tell_agent", '{"pane_id":"p","text":"hello"}')
    assert "NOT SENT" in results[0]
    mate.herdr.prompt_agent.assert_not_awaited()


@pytest.mark.parametrize("change", ["lock", "disconnect", "discard"])
async def test_live_send_still_checks_session_and_staging(tmp_path, change):
    mate = make_mate(tmp_path)
    await mate.execute_live_tool("tell_agent", '{"pane_id":"p","text":"hello"}')
    if change == "lock":
        mate.lock("one two three four")
    elif change == "disconnect":
        mate.bridge.failed = True
    else:
        await mate.execute_live_tool("discard_staged", "{}")
    result = await mate.execute_live_tool("send_staged", "{}")
    assert result.startswith(("ERROR", "REFUSED"))
    mate.herdr.prompt_agent.assert_not_awaited()


async def test_restage_sends_only_corrected_request(tmp_path):
    mate = make_mate(tmp_path)
    await mate.execute_live_tool("tell_agent", '{"pane_id":"p","text":"make video"}')
    await mate.execute_live_tool("tell_agent", '{"pane_id":"p","text":"make video and host via Cloudflare tunnel"}')
    await mate.execute_live_tool("send_staged", "{}")
    mate.herdr.prompt_agent.assert_awaited_once_with("p", "make video and host via Cloudflare tunnel")
    assert mate.bridge.say.await_count == 2


async def test_failed_delivery_is_consumed_and_never_retried(tmp_path):
    mate = make_mate(tmp_path)
    await mate.execute_live_tool("tell_agent", '{"pane_id":"p","text":"hello"}')
    mate.herdr.prompt_agent.side_effect = TimeoutError()
    assert "outcome unknown" in await mate.execute_live_tool("send_staged", "{}")
    assert "nothing is staged" in await mate.execute_live_tool("send_staged", "{}")
    mate.herdr.prompt_agent.assert_awaited_once()


async def test_locked_caller_never_reaches_cloud_or_tools(tmp_path):
    mate = make_mate(tmp_path)
    mate.lock("one two three four")
    mate.bridge.start = AsyncMock()
    with pytest.raises(StopResponse):
        await say_user(mate, "wrong words")
    assert mate.turns == []
    mate.bridge.start.assert_not_awaited()
    assert "REFUSED" in await mate.execute_live_tool("fleet_status", "{}")
    mate.herdr.snapshot.assert_not_awaited()
    with pytest.raises(StopResponse):
        await say_user(mate, "one two three four")
    mate.bridge.start.assert_awaited_once()
    assert mate.turns == []  # passphrase is not backend context


async def test_tool_argument_validation(tmp_path):
    mate = make_mate(tmp_path)
    for args in ('{}', '{"pane_id":1,"text":"hello"}',
                 '{"pane_id":"p","text":"hello","ctx":"forged"}', '[]', 'bad'):
        assert "ERROR" in await mate.execute_live_tool("tell_agent", args)
    mate.herdr.prompt_agent.assert_not_awaited()


async def test_response_calls_collected_and_deduplicated():
    sent = []
    execute = AsyncMock(return_value="delivered")

    async def send(event):
        sent.append(event)

    loop = ResponsesTools(send, execute)
    call = {"type": "response.output_item.done", "response_id": "r1", "item": {
        "type": "function_call", "call_id": "c1", "name": "tell_agent", "arguments": "{}"}}
    done = {"type": "response.completed", "response": {"id": "r1", "output": []}}
    loop.feed(call)
    loop.feed(call)
    assert loop.queue.empty()
    loop.feed(done)
    loop.feed(done)
    worker = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(loop.queue.join(), 1)
        execute.assert_awaited_once()
        assert [e["type"] for e in sent] == ["response.item.create", "response.create"]
        assert sent[0]["item"]["call_id"] == "c1"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_failed_backend_response_does_not_execute():
    loop = ResponsesTools(AsyncMock(), AsyncMock())
    loop.feed({"type": "response.output_item.done", "response_id": "r", "item": {
        "type": "function_call", "call_id": "c", "name": "send_staged", "arguments": "{}"}})
    loop.feed({"type": "response.failed", "response": {"id": "r"}})
    assert loop.queue.empty()
    assert not loop.pending


async def test_live_envelopes_correlate_idless_items_across_delegations():
    execute = AsyncMock(return_value="report")
    loop = ResponsesTools(AsyncMock(), execute)
    for d, r in (("d1", "r1"), ("d2", "r2")):
        loop.feed({"type": "response.created", "response": {"id": r}}, d)
        loop.feed({"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": d, "name": "agent_report",
            "arguments": json.dumps({"pane_id": d})}}, d)
    loop.feed({"type": "response.failed", "response": {"id": "r2"}}, "d2")
    loop.feed({"type": "response.completed", "response": {"id": "r1"}}, "d1")
    worker = asyncio.create_task(loop.run())
    try:
        await asyncio.wait_for(loop.queue.join(), 1)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    execute.assert_awaited_once_with("agent_report", '{"pane_id": "d1"}')
    # A continuation uses the same delegation but a fresh response ID.
    loop.feed({"type": "response.created", "response": {"id": "r3"}}, "d1")
    loop.feed({"type": "response.output_item.done", "item": {
        "type": "function_call", "call_id": "c3", "name": "fleet_status",
        "arguments": "{}"}}, "d1")
    assert "c3" in loop.pending["r3"]


async def test_websocket_start_audio_and_graceful_close(monkeypatch, tmp_path):
    mate = make_mate(tmp_path)
    events = asyncio.Queue()
    events.put_nowait(json.dumps({"type": "session.started", "session": {"id": "s"}}))
    sent = []

    class Socket:
        async def send(self, raw):
            event = json.loads(raw)
            sent.append(event)
            if event["type"] == "session.close":
                events.put_nowait(json.dumps({"type": "session.closed"}))

        async def recv(self):
            return await events.get()

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.recv()

        async def close(self):
            pass

    connector = AsyncMock(return_value=Socket())
    monkeypatch.setattr("mate.live.connect", connector)
    bridge = LiveBridge(mate, config())
    mate.bridge = bridge
    await bridge.start()
    assert bridge.ready
    assert sent[0]["type"] == "session.start"
    assert sent[0]["session"]["delegation"]["responses"]["tools"]
    frame = rtc.AudioFrame(b"\x01\x00" * 480, 24000, 1, 480)
    bridge.push_audio(frame)
    await asyncio.sleep(0.01)
    audio = [e for e in sent if e["type"] == "session.input_audio.append"]
    assert audio and base64.b64decode(audio[0]["audio"])
    await bridge.aclose()
    assert bridge.closed.is_set()
    assert all(t.done() for t in bridge.tasks)
    assert not bridge.failed


async def test_playback_preserves_pcm_and_drops_audio_while_locked():
    frames = []
    played = asyncio.Event()

    async def capture(frame):
        frames.append(frame)

    async def wait():
        played.set()

    sink = SimpleNamespace(capture_frame=capture, flush=lambda: None, wait_for_playout=wait)
    mate = SimpleNamespace(locked=False, execute_live_tool=AsyncMock(),
                           session=SimpleNamespace(output=SimpleNamespace(audio=sink)))
    bridge = LiveBridge(mate, config())
    data = b"\x01\x02" * 480
    bridge.output.put_nowait(data)
    worker = asyncio.create_task(bridge.play_audio())
    try:
        await asyncio.wait_for(played.wait(), 1)
        assert bytes(frames[0].data) == data
        assert frames[0].sample_rate == 24000
        assert frames[0].num_channels == 1
        mate.locked = True
        bridge.output.put_nowait(data)
        await asyncio.sleep(0)
        assert len(frames) == 1
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_changed_destructive_screen_requires_new_confirmation(tmp_path):
    mate = make_mate(tmp_path)
    mate.herdr.read_pane = AsyncMock(return_value="rm -rf /tmp/old")
    mate.herdr.send_keys = AsyncMock()
    await say_user(mate, "approve the deletion")
    await mate.execute_live_tool("read_pane", '{"pane_id":"p"}')
    await mate.execute_live_tool("send_answer", '{"pane_id":"p","keys":["Enter"]}')
    assert "rm -rf /tmp/old" in mate.bridge.say.call_args.args[0]
    await say_user(mate, "go ahead")
    mate.herdr.read_pane.return_value = "rm -rf /tmp/new"
    assert "screen changed" in await mate.execute_live_tool("send_staged", "{}")
    mate.herdr.send_keys.assert_not_awaited()


async def test_live_preflight_uses_cloud_without_approval_or_old_llm(monkeypatch):
    import httpx

    from mate import agent

    monkeypatch.setenv("MATE_VOICE_MODE", "live")
    monkeypatch.setenv("OPENAI_API_KEY", "test-cloud-key")
    monkeypatch.setenv("MATE_APPROVAL_URL", "http://approval.local/v1")
    monkeypatch.setenv("MATE_APPROVAL_API_KEY", "local")
    seen = []

    def respond(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"data": []})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(agent.httpx, "AsyncClient", lambda **kwargs: client_type(
        **kwargs, transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(agent, "HerdrClient", lambda: SimpleNamespace(call=AsyncMock(return_value={})))
    status = await agent.check_endpoints()
    assert set(status) == {"stt", "tts", "openai", "herdr"}
    assert not any(status.values())
    assert ("https://api.openai.com/v1/models", "Bearer test-cloud-key") in seen
    assert all("approval.local" not in url for url, _ in seen)


async def test_connection_failure_clears_stage_and_stops_session():
    session = SimpleNamespace(shutdown=Mock())
    mate = SimpleNamespace(session=session, locked=False, execute_live_tool=AsyncMock(),
                           _staged={"text": "must not send"})
    bridge = LiveBridge(mate, config())
    bridge.ready = True

    async def broken():
        raise ConnectionError("disconnected")

    task = asyncio.create_task(broken())
    await asyncio.gather(task, return_exceptions=True)
    bridge._task_done(task)
    assert bridge.failed and not bridge.ready
    assert mate._staged is None
    session.shutdown.assert_called_once_with(drain=False)


async def test_rejected_startup_closes_socket_and_session(monkeypatch):
    session = SimpleNamespace(shutdown=Mock())
    mate = SimpleNamespace(session=session, locked=False, execute_live_tool=AsyncMock(),
                           instructions="test", live_tool_schemas=lambda: [])
    socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), recv=AsyncMock(
        return_value=json.dumps({"type": "error", "error": {"message": "not available"}})))
    monkeypatch.setattr("mate.live.connect", AsyncMock(return_value=socket))
    bridge = LiveBridge(mate, config())
    with pytest.raises(RuntimeError, match="rejected"):
        await bridge.start()
    assert bridge.failed and not bridge.ready
    socket.close.assert_awaited_once()
    session.shutdown.assert_called_once_with(drain=False)


async def test_read_only_tool_does_not_wait_for_transcription(tmp_path):
    mate = make_mate(tmp_path)
    mate.transcription_pending = True
    result = await asyncio.wait_for(mate.execute_live_tool("fleet_status", "{}"), 0.2)
    assert "NOT SENT" not in result
    mate.herdr.snapshot.assert_awaited_once()


async def test_postlogin_whisper_text_is_ignored(tmp_path):
    mate = make_mate(tmp_path)
    await mate.on_user_turn_completed(None, llm.ChatMessage(role="user", content=["yes"]))
    assert not mate.turns
    assert not mate.conversation


async def test_audio_tee_stops_whisper_after_unlock_and_resumes_on_relock(tmp_path, monkeypatch):
    from livekit.agents import Agent

    mate = make_mate(tmp_path)
    frames = [rtc.AudioFrame(b"\x00\x00" * 480, 24000, 1, 480) for _ in range(3)]
    local_frames = []
    mate.bridge.push_audio = Mock()

    async def local_stt(agent, audio, settings):
        async for frame in audio:
            local_frames.append(frame)
            yield SimpleNamespace()

    monkeypatch.setattr(Agent.default, "stt_node", local_stt)

    async def microphone():
        for locked, frame in zip((True, False, True), frames, strict=True):
            mate.locked = locked
            yield frame

    async for _ in mate.stt_node(microphone(), None):
        pass
    assert local_frames == [frames[0], frames[2]]
    assert mate.bridge.push_audio.call_count == 3
