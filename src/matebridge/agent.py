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
import re
from pathlib import Path

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

# Claude Code writes one JSONL transcript per session under
# ~/.claude/projects/<munged-cwd>/<session-id>.jsonl. herdr's integration hook
# reports the session id + cwd, which is enough to find it.
CLAUDE_PROJECTS = Path(os.path.expanduser("~/.claude/projects"))


def claude_transcript_path(cwd: str, session_id: str) -> Path:
    munged = re.sub(r"[^A-Za-z0-9-]", "-", cwd)
    return CLAUDE_PROJECTS / munged / f"{session_id}.jsonl"


def read_transcript_replies(path: Path, count: int) -> list[str]:
    """Last `count` assistant text messages from a Claude Code transcript."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") != "assistant":
                continue
            for block in entry.get("message", {}).get("content", []):
                if (isinstance(block, dict) and block.get("type") == "text"
                        and block.get("text", "").strip()):
                    texts.append(block["text"])
    return texts[-count:]

INSTRUCTIONS = """You are Mate, a hands-free voice assistant supervising a fleet of
coding agents in herdr. The user is often driving. Speak short, natural sentences:
no markdown, no lists, no code blocks, no emoji. Two to four sentences unless asked.

Always refresh state with tools before reporting status; never answer from memory.
For "what did the agent say/find/conclude", prefer agent_report (full replies from
its transcript). Use read_pane for what is on screen now: pending prompts, errors,
running commands.
After delegating work with tell_agent or spawn_task, NEVER poll or re-check on
your own. Quick question: call wait_for_agent once. Anything longer: tell the
user you'll be notified when it finishes — you will be, automatically.
Resolve casual project names to workspace labels via fleet_status. If a reference
is genuinely ambiguous, ask one short question instead of guessing.
Before approving anything an agent is waiting on, read its pane and tell the user
exactly what will run if it looks destructive."""


class Mate(Agent):
    def __init__(self, herdr: HerdrClient):
        super().__init__(instructions=INSTRUCTIONS)
        self.herdr = herdr
        self._read_panes: set[str] = set()  # rails: read before approve
        # panes given work this call; watch_fleet announces when they finish
        self.delegated: set[str] = set()

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

    async def _agent_replies(self, pane_id: str, messages: int = 1) -> str:
        """Last replies from the pane's claude transcript, or an ERROR string."""
        try:
            snap = await self.herdr.snapshot()
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message}"
        agent = next((a for a in snap.get("agents", [])
                      if a.get("pane_id") == pane_id), None)
        if agent is None:
            return (f"ERROR: no coding agent is registered in pane {pane_id}. "
                    "Use read_pane to see the raw terminal instead.")
        session = agent.get("agent_session") or {}
        if agent.get("agent") != "claude" or session.get("kind") != "id":
            return ("ERROR: transcript reading is only wired up for claude "
                    "agents. Use read_pane instead.")
        path = claude_transcript_path(agent.get("cwd", ""), session["value"])
        try:
            replies = read_transcript_replies(path, max(1, min(messages, 10)))
        except OSError:
            return (f"ERROR: transcript not readable at {path}. "
                    "Use read_pane instead.")
        if not replies:
            return "The agent has not written any replies yet this session."
        return "\n\n---\n\n".join(replies)[-8000:]

    @function_tool
    async def agent_report(self, ctx: RunContext, pane_id: str,
                           messages: int = 1):
        """Read the coding agent's last full reply/replies from its session
        transcript. Better than read_pane for "what did it say/find" — screen
        scrollback loses long answers, the transcript never does."""
        return await self._agent_replies(pane_id, messages)

    @function_tool
    async def wait_for_agent(self, ctx: RunContext, pane_id: str,
                             seconds: int = 15):
        """Wait briefly (max 20s) for a busy agent to finish, then return its
        reply. Use after tell_agent for QUICK questions only. If it is still
        working when time runs out, do NOT call again — the user will be
        notified automatically when the agent finishes."""
        deadline = asyncio.get_event_loop().time() + max(1, min(seconds, 20))
        status = "unknown"
        while asyncio.get_event_loop().time() < deadline:
            try:
                snap = await self.herdr.snapshot()
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message}"
            agent = next((a for a in snap.get("agents", [])
                          if a.get("pane_id") == pane_id), None)
            if agent is None:
                return f"ERROR: no coding agent is registered in pane {pane_id}."
            status = agent.get("agent_status", "unknown")
            if status in ("idle", "done", "blocked"):
                self.delegated.discard(pane_id)
                reply = await self._agent_replies(pane_id)
                return f"agent finished (status: {status}). Its reply:\n{reply}"
            await asyncio.sleep(1.0)
        return (f"agent is still {status}. STOP checking — the user will be "
                "notified automatically the moment it finishes. Tell them "
                "that and move on.")

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
        self.delegated.add(pane_id)
        return ("delivered. For a quick question, call wait_for_agent once. "
                "For anything longer, tell the user they'll be notified when "
                "it finishes — do not check again on your own.")

    @function_tool
    async def spawn_task(self, ctx: RunContext, repo_path: str, branch: str,
                         task: str):
        """Create a worktree in a repo and start a coding agent on a task."""
        try:
            result = await self.herdr.spawn(repo_path, branch, task)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message}"
        if result.get("pane_id"):
            self.delegated.add(result["pane_id"])
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
        turn_handling={
            # Batch whisper takes ~4-5s to return a transcript, so the default
            # 2s false-interruption window always expires first: Mate resumes
            # speaking, then cuts off again when the transcript lands. Give
            # STT time to confirm the interruption was real.
            "interruption": {"false_interruption_timeout": 6.0},
        },
    )

    async def watch_fleet():
        # proactive interjection when an agent blocks, or when a pane Mate
        # delegated to this call finishes (kills the "let me check again" loop)
        async for msg in herdr.events(["pane.agent_status_changed"]):
            data = msg.get("data", {})
            status = (data.get("agent_status")
                      or data.get("pane", {}).get("agent_status"))
            pane_id = (data.get("pane_id")
                       or data.get("pane", {}).get("pane_id"))
            if status == "blocked":
                await session.generate_reply(
                    instructions="Briefly tell the user this agent just got "
                                 "blocked and offer to read its question: "
                                 + json.dumps(data))
            elif (status in ("idle", "done")
                  and pane_id in mate.delegated):
                mate.delegated.discard(pane_id)  # one announcement per task
                await session.generate_reply(
                    instructions="The agent you delegated work to just "
                                 "finished. Use agent_report on pane "
                                 f"{pane_id}, then give the user a one- or "
                                 "two-sentence summary of what it did.")

    mate = Mate(herdr)
    watcher = asyncio.create_task(watch_fleet())

    async def _stop_watcher():
        watcher.cancel()

    ctx.add_shutdown_callback(_stop_watcher)

    await session.start(agent=mate, room=ctx.room)
    await session.generate_reply(
        instructions="Greet briefly: say you're here and how many agents "
                     "are running.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
