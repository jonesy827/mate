import json

import httpx
import pytest

from mate.approval import ApprovalConfig, ApprovalGate


@pytest.mark.parametrize("content,finish,expected", [
    ('{"decision":"approve"}', "stop", "approve"),
    ('{"decision":"reject"}', "stop", "reject"),
    ('{"decision":"unclear"}', "stop", "unclear"),
    ('{"decision":"approve"}', "length", "unclear"),
    ('{"decision":"approve","action":"changed"}', "stop", "unclear"),
    ('{"decision":true}', "stop", "unclear"),
    ('```json\n{"decision":"approve"}\n```', "stop", "unclear"),
    ('[]', "stop", "unclear"),
    ('not json', "stop", "unclear"),
])
async def test_gate_validates_response_and_sends_context(content, finish, expected):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": finish, "message": {"content": content}}]})

    gate = ApprovalGate(ApprovalConfig(), transport=httpx.MockTransport(respond))
    result = await gate.decide(action={"text": "test it"},
                               confirmation="Send test it?",
                               conversation=[{"role": "user", "content": "tell the agent"}],
                               reply="looks good")
    assert result == expected
    payload = requests[0]
    assert payload["model"] == "gpt-6-luna"
    assert payload["reasoning_effort"] == "none"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "chat_template_kwargs" not in payload
    evidence = json.loads(payload["messages"][1]["content"])
    assert evidence["latest_user_reply"] == "looks good"
    assert evidence["pending_action"] == {"text": "test it"}
    assert evidence["spoken_confirmation"] == "Send test it?"


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "bad_shape"])
async def test_unavailable_gate_never_approves(failure):
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("busy GPU")
        if failure == "unavailable":
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": []})
    gate = ApprovalGate(ApprovalConfig(), transport=httpx.MockTransport(respond))
    assert await gate.decide(action={}, confirmation="Send?", conversation=[], reply="yes") == "unclear"
