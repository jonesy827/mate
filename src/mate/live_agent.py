"""Mate tools behind GPT-Live, with model-managed confirmation and one-shot delivery."""

import json
from types import SimpleNamespace

from livekit.agents import Agent, RunContext, StopResponse, function_tool
from livekit.agents.llm.utils import (
    build_legacy_openai_schema,
    function_arguments_to_pydantic_model,
)

from .agent import LOCKED_MSG, Mate
from .audit import event as trace
from .live import LiveBridge


class LiveMate(Mate):
    def __init__(self, herdr, config, **kwargs):
        super().__init__(herdr, **kwargs)
        self.bridge = LiveBridge(self, config)
        self.turns: list[str] = []
        self.conversation: list[dict] = []
        self.revision = 0
        self.input_end_ms = -1
        self.tool_map = {tool.info.name: tool for tool in self.tools}
        self.tool_models = {name: function_arguments_to_pydantic_model(tool)
                            for name, tool in self.tool_map.items()}

    def live_tool_schemas(self) -> list[dict]:
        return [dict(build_legacy_openai_schema(tool, internally_tagged=True), strict=False)
                for tool in self.tool_map.values()]

    def context(self):
        items = [SimpleNamespace(role="user", text_content=text) for text in self.turns]
        return SimpleNamespace(session=SimpleNamespace(history=SimpleNamespace(items=items)))

    async def stt_node(self, audio, model_settings):
        async def tee():
            async for frame in audio:
                self.bridge.push_audio(frame)
                if self.locked:
                    yield frame

        async for event in Agent.default.stt_node(self, tee(), model_settings):
            if self.locked:
                yield event

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        # Late Whisper results after login must never become approval evidence.
        if not self.locked:
            return
        try:
            await super().on_user_turn_completed(turn_ctx, new_message)
        except StopResponse:
            if not self.locked:
                await self.bridge.start()
            raise

    def record_live_input(self, delta: str, start_ms: float, end_ms: float) -> None:
        if self.locked or not delta:
            return
        self.revision += 1
        # Transcript grouping is for context and logging, never a send gate.
        new_turn = not self.turns or start_ms - self.input_end_ms >= 1000
        if new_turn:
            self.turns.append(delta)
        else:
            self.turns[-1] += delta
        self.input_end_ms = max(self.input_end_ms, end_ms)
        self.conversation.append({"role": "user", "content": delta})
        trace("user.transcript", source="gpt-live", text=delta,
              start_ms=start_ms, end_ms=end_ms, revision=self.revision)

    def record_live_speech(self, delta: str) -> None:
        if self.conversation and self.conversation[-1]["role"] == "assistant":
            self.conversation[-1]["content"] = (
                self.conversation[-1]["content"] + delta)[-8000:]
        else:
            self.conversation.append({"role": "assistant", "content": delta})

    async def _speak_confirmation(self, text: str) -> bool:
        staged = self._staged
        # Even remembered folders must name the task being authorized.
        if staged and staged["kind"] == "spawn_folder" and staged.get("task"):
            if staged["task"] not in text:
                text = f"{text.rstrip('?')} Task: {staged['task']}. Should I go ahead?"
        await self.bridge.say(text)
        self.conversation.append({"role": "assistant", "content": text})
        if staged is not None and self._staged is staged:
            staged["_confirmation"] = text
        return True

    @function_tool
    async def send_staged(self, ctx: RunContext):
        """Send the exact staged action ONCE after the user approves its readback.

        You interpret the reply. If unclear, ask; if corrected, restage; if
        declined, discard_staged. Never claim delivery without this tool's result.
        """
        if self.locked:
            return LOCKED_MSG
        if self.bridge.failed or self.bridge.closing:
            return "REFUSED: voice connection is unavailable."
        staged = self._staged
        if staged is None:
            return "ERROR: nothing is staged. Stage the request first."
        if not staged.get("_confirmation"):
            return "NOT SENT: the staged action has not finished its readback."
        self._staged = None  # consume before the first await; never auto-retry
        trace("delivery.attempt", action_kind=staged["kind"], pane_id=staged.get("pane_id"),
              authorization="live_backend")
        result = await self._deliver(staged)
        trace("delivery.result", action_kind=staged["kind"], pane_id=staged.get("pane_id"),
              result=result)
        return result

    async def execute_live_tool(self, name: str, arguments: str) -> str:
        if self.locked:
            return LOCKED_MSG
        if self.bridge.failed or self.bridge.closing:
            return "REFUSED: voice connection is unavailable."
        tool = self.tool_map.get(name)
        if tool is None:
            return "ERROR: unknown tool."
        try:
            values = json.loads(arguments)
            if not isinstance(values, dict) or set(values) - self.tool_models[name].model_fields.keys():
                return "ERROR: unexpected tool arguments."
            values = self.tool_models[name].model_validate(values, strict=True).model_dump()
        except (ValueError, TypeError):
            return "ERROR: invalid tool arguments."
        if self.locked or self.bridge.failed:
            return LOCKED_MSG
        before = self._staged
        try:
            result = await tool(self.context(), **values)
        except Exception as exc:
            trace("tool.exception", name=name, error_type=type(exc).__name__)
            # Do not retry mutations after exceptions: delivery may have
            # happened. Do not expose arbitrary exception payloads.
            return "ERROR: tool execution failed; outcome unknown. Do not retry automatically."
        if name == "fleet_status":
            try:
                snapshot = json.loads(result)
                trace("fleet.coverage", agents=[a.get("pane_id") for a in snapshot.get("agents", [])])
            except (ValueError, TypeError, AttributeError):
                trace("fleet.coverage_unavailable")
        elif name == "agent_report":
            trace("agent.report", pane_id=values.get("pane_id"),
                  messages=values.get("messages"), chars=len(str(result)))
        staged = self._staged
        if staged is not None and staged is not before and not staged.get("_confirmation"):
            if staged["kind"] == "tell":
                label = await self._pane_label(staged["pane_id"])
                text = f"Send to {label}: {staged['text']}. Should I send it?"
            elif staged["kind"] == "spawn":
                text = (f"Start a {staged['agent']} agent in {staged['repo_path']}, "
                        f"branch {staged['branch']}, with task: {staged['task']}. Go ahead?")
            elif staged["kind"] == "keys":
                text = (f"The agent is asking to do this: {staged['screen']}. "
                        f"Send keys {staged['keys']} to approve it?")
            else:
                return "NOT SENT: no confirmation was played. Ask the user to restate the request."
            await self._speak_confirmation(text)
            return "Staged only. Application already spoke the confirmation. Wait for the user's reply."
        return str(result)

    async def on_exit(self) -> None:
        await self.bridge.aclose()
