#!/usr/bin/env python3
"""One tool-call round trip against the local router LLM. Prints TTFT.

Uses the exact request shape validated in the router test battery:
- chat_template_kwargs {"enable_thinking": false}  (no <think> blocks)
- presence_penalty 0                               (server default 1.5 breaks tools)
Run: .venv/bin/python scripts/smoke_llm.py
"""

import json
import os
import time
import urllib.request

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8003/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-long")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "fleet_status",
        "description": "Full fleet snapshot: every workspace, pane, agent and its state.",
        "parameters": {"type": "object", "properties": {}},
    },
}]

FLEET = {"workspaces": [{"label": "matebridge", "id": "w1", "agent_status": "working"},
                        {"label": "webshop", "id": "w2", "agent_status": "blocked"}],
         "panes": [], "agents": []}


def post(messages, tools=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "presence_penalty": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{LLM_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"], time.time() - t0


def main():
    messages = [
        {"role": "system", "content": "You are Mate, a voice assistant supervising "
         "coding agents. Use tools to check state before answering. Speak briefly."},
        {"role": "user", "content": "how's the webshop project doing"},
    ]
    msg, t1 = post(messages, TOOLS)
    calls = msg.get("tool_calls") or []
    print(f"turn 1: {t1:.2f}s  tool_calls={[c['function']['name'] for c in calls]}")
    assert calls, f"expected a tool call, got: {msg.get('content')!r}"

    messages.append(msg)
    messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                     "content": json.dumps(FLEET)})
    msg, t2 = post(messages, TOOLS)
    print(f"turn 2: {t2:.2f}s  reply: {msg.get('content')!r}")
    assert msg.get("content"), "expected a spoken reply"
    assert "<think>" not in (msg.get("content") or ""), "thinking leaked!"
    print(f"SMOKE OK  (full round trip {t1 + t2:.2f}s)")


if __name__ == "__main__":
    main()
