import asyncio
import json
import logging
from unittest.mock import AsyncMock

from mate.audit import HideSDKTranscripts, event
from mate.live import ResponsesTools


def test_trace_redacts_credentials_and_passphrase(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("MATE_PASSPHRASE", "one two three four")
    with caplog.at_level(logging.INFO, logger="mate.audit"):
        event("user.turn", text="ONE, TWO three four sk-test-secret", nested={"key": "sk-other-token"})
    assert "one" not in caplog.text.lower()
    assert "sk-" not in caplog.text
    assert "[redacted]" in caplog.text


def test_sdk_transcripts_are_suppressed_before_authentication():
    guard = HideSDKTranscripts()
    for message in ("received user transcript", "conversation_item_added"):
        record = logging.LogRecord("livekit.agents", logging.DEBUG, "", 0, message, (), None)
        assert not guard.filter(record)
    record = logging.LogRecord("livekit.agents", logging.INFO, "", 0, "received job request", (), None)
    assert guard.filter(record)


async def test_tool_trace_correlates_execution_and_cached_result(caplog):
    loop = ResponsesTools(AsyncMock(), AsyncMock(return_value="NOT SENT: pending transcript"))
    with caplog.at_level(logging.INFO, logger="mate.audit"):
        worker = asyncio.create_task(loop.run())
        try:
            for rid in ("r1", "r2"):
                loop.feed({"type": "response.output_item.done", "response_id": rid, "item": {
                    "type": "function_call", "name": "send_staged", "call_id": "same",
                    "arguments": "{}"}})
                loop.feed({"type": "response.completed", "response": {"id": rid}})
                await asyncio.wait_for(loop.queue.join(), 1)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
    entries = [json.loads(r.message) for r in caplog.records if r.name == "mate.audit"]
    done = [e for e in entries if e["event"] == "tool.finished"]
    assert len(done) == 1
    assert done[0]["call_id"] == "same"
    assert done[0]["outcome"] == "NOT SENT"
    assert done[0]["elapsed_ms"] >= 0
    assert any(e["event"] == "tool.cached" for e in entries)
    loop.execute.assert_awaited_once()
