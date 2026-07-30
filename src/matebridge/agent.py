"""Mate — LiveKit Agents worker bridging voice to a herdr fleet.

Run:  python -m matebridge.agent console   # terminal mic/speaker desk test
      python -m matebridge.agent dev       # worker + Agents Playground
      python -m matebridge.agent start     # production (SIP calls dispatch here)

Local services expected:
  :8003  llama.cpp qwen3.6-35b-a3b (thinking disabled via chat_template_kwargs)
  :8001  speaches (faster-whisper, OpenAI-compatible STT)
  :8880  kokoro-fastapi (OpenAI-compatible TTS)
  ~/.config/herdr/herdr.sock
"""

import asyncio
import json
import logging
import os

import httpx
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero

from .herdr_client import HerdrClient, HerdrError
from .safety import is_destructive

logger = logging.getLogger("matebridge")

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8003/v1")
STT_URL = os.environ.get("STT_URL", "http://localhost:8001/v1")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8880/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-long")
STT_MODEL = os.environ.get("STT_MODEL", "Systran/faster-whisper-medium")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")

INSTRUCTIONS = """You are Mate, a hands-free voice assistant supervising a fleet of
coding agents in herdr. The user is often driving. Speak short, natural sentences:
no markdown, no lists, no code blocks, no emoji. Two to four sentences unless asked.

Always refresh state with tools before reporting status; never answer from memory.
Resolve casual project names to workspace labels via fleet_status. If a reference
is genuinely ambiguous, ask one short question instead of guessing.
Before approving anything an agent is waiting on, read its pane and tell the user
exactly what will run if it looks destructive."""


class Mate(Agent):
    def __init__(self, herdr: HerdrClient):
        super().__init__(instructions=INSTRUCTIONS)
        self.herdr = herdr
        self._read_panes: set[str] = set()  # rails: read before approve

    @function_tool
    async def fleet_status(self, ctx: RunContext):
        """Full fleet snapshot: every workspace, pane, agent and its state."""
        snap = await self.herdr.snapshot()
        # trim to what the router needs; keep the prompt small for prefill speed
        return json.dumps({
            "workspaces": [
                {"label": w.get("label"), "id": w.get("workspace_id"),
                 "agent_status": w.get("agent_status")}
                for w in snap.get("workspaces", [])],
            "panes": [
                {"pane_id": p.get("pane_id"),
                 "workspace_id": p.get("workspace_id"),
                 "agent_status": p.get("agent_status"),
                 "title": p.get("terminal_title_stripped")}
                for p in snap.get("panes", [])],
            "agents": snap.get("agents", []),
        })

    @function_tool
    async def read_pane(self, ctx: RunContext, pane_id: str, lines: int = 60):
        """Read the last lines of a pane's terminal output (what an agent is doing or asking)."""
        try:
            text = await self.herdr.read_pane(pane_id, lines)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        self._read_panes.add(pane_id)
        return text[-4000:]

    @function_tool
    async def send_answer(self, ctx: RunContext, pane_id: str, keys: list[str]):
        """Answer a blocked agent's on-screen prompt with keys, e.g. ["1","Enter"]."""
        if pane_id not in self._read_panes:
            return ("REFUSED: read the pane first so the user knows what "
                    "they are approving.")
        try:
            text = await self.herdr.read_pane(pane_id, 40)
            if is_destructive(text):
                return ("REFUSED: the pending action looks destructive. Read it to "
                        "the user verbatim and get explicit confirmation, then call "
                        "send_answer_confirmed.")
            await self.herdr.send_keys(pane_id, keys)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        return "sent"

    @function_tool
    async def send_answer_confirmed(self, ctx: RunContext, pane_id: str,
                                    keys: list[str]):
        """Send keys after the user explicitly confirmed a destructive action read to them verbatim."""
        try:
            await self.herdr.send_keys(pane_id, keys)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        return "sent"

    @function_tool
    async def tell_agent(self, ctx: RunContext, pane_id: str, text: str):
        """Send a natural-language instruction to a coding agent running in a pane."""
        try:
            await self.herdr.prompt_agent(pane_id, text)
        except HerdrError as e:
            if e.code == "agent_not_found":
                return (f"ERROR: no coding agent is registered in pane {pane_id} — "
                        "it is just a shell. An agent appears only once claude (or "
                        "another integrated agent) is launched inside a herdr pane. "
                        "Tell the user that pane has no agent to talk to.")
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        return "delivered"

    @function_tool
    async def spawn_task(self, ctx: RunContext, repo_path: str, branch: str,
                         task: str):
        """Create a worktree in a repo and start a coding agent on a task."""
        try:
            result = await self.herdr.spawn(repo_path, branch, task)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message}"
        return json.dumps(result)


async def check_endpoints() -> dict[str, str | None]:
    """Return {name: error or None} for each required local service."""
    targets = {
        "llm": f"{LLM_URL}/models",
        "stt": f"{STT_URL}/models",
        "tts": f"{TTS_URL}/models",
    }
    errors: dict[str, str | None] = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in targets.items():
            try:
                r = await client.get(url)
                errors[name] = None if r.status_code < 500 else f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001 - report anything as down
                errors[name] = f"{type(e).__name__}: {e}"
    try:
        await HerdrClient().call("ping")
        errors["herdr"] = None
    except Exception as e:  # noqa: BLE001
        errors["herdr"] = f"{type(e).__name__}: {e}"
    return errors


async def entrypoint(ctx: JobContext):
    status = await check_endpoints()
    down = {k: v for k, v in status.items() if v}
    if down:
        details = "\n".join(f"  {k}: {v}" for k, v in down.items())
        raise RuntimeError(
            f"matebridge cannot start; unreachable services:\n{details}\n"
            f"(llm={LLM_URL} stt={STT_URL} tts={TTS_URL} "
            f"herdr={HerdrClient().sock_path})")

    herdr = HerdrClient()
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(base_url=STT_URL, api_key="local", model=STT_MODEL),
        llm=openai.LLM(
            base_url=LLM_URL, api_key="local", model=LLM_MODEL,
            temperature=0.2,
            extra_body={
                # llama.cpp: never emit <think> blocks in a voice pipeline
                "chat_template_kwargs": {"enable_thinking": False},
                # server default presence_penalty=1.5 breaks tool calling
                "presence_penalty": 0,
            },
        ),
        # "tts-1" (not "kokoro"): the plugin treats unknown model names as
        # OpenAI's SSE-streaming models and parses the response as SSE JSON,
        # but kokoro returns raw audio bytes -> "no audio frames were pushed".
        # kokoro-fastapi aliases tts-1 to kokoro, and tts-1 selects the
        # raw-audio stream path (AUDIO_STREAM_MODELS) in the plugin.
        tts=openai.TTS(base_url=TTS_URL, api_key="local", model="tts-1",
                       voice=TTS_VOICE),
    )

    async def watch_fleet():
        # proactive interjection when an agent blocks mid-call
        async for msg in herdr.events(["pane.agent_status_changed"]):
            data = msg.get("data", {})
            status = (data.get("agent_status")
                      or data.get("pane", {}).get("agent_status"))
            if status == "blocked":
                await session.generate_reply(
                    instructions="Briefly tell the user this agent just got "
                                 "blocked and offer to read its question: "
                                 + json.dumps(data))

    watcher = asyncio.create_task(watch_fleet())

    async def _stop_watcher():
        watcher.cancel()

    ctx.add_shutdown_callback(_stop_watcher)

    await session.start(agent=Mate(herdr), room=ctx.room)
    await session.generate_reply(
        instructions="Greet briefly: say you're here and how many agents "
                     "are running.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
