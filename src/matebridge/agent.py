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
import sys
import time

import httpx
from livekit import api as lk_api
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

from .allowlist import ENV_VAR, allowed_callers, is_allowed, sip_caller
from .folders import KnownAgents, resolve_folder, speakable_path
from .herdr_client import HerdrClient, HerdrError, protocol_note
from .safety import approves_send, is_destructive
from .transcripts import adapter_for, supported_kinds

logger = logging.getLogger("matebridge")

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8003/v1")
STT_URL = os.environ.get("STT_URL", "http://localhost:8001/v1")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8880/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b-long")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "local")
STT_MODEL = os.environ.get("STT_MODEL", "Systran/faster-whisper-medium")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")


def llm_options() -> dict:
    """kwargs for the voice-brain openai.LLM.

    The sampling block is local-only tuning: a real LLM_API_KEY means a
    hosted provider, and those reject params they don't allow (OpenAI 400s
    on top_k, and gpt-5-era models on any non-default temperature), so
    hosted runs on provider defaults.
    """
    opts: dict = {"base_url": LLM_URL, "api_key": LLM_API_KEY,
                  "model": LLM_MODEL}
    if LLM_API_KEY == "local":
        opts.update(
            # Official qwen rec for non-thinking mode: temp 0.7, top_p 0.8,
            # top_k 20. At temp 0.2 Mate repeated identical sentences
            # verbatim in live testing.
            temperature=0.7,
            extra_body={
                # llama.cpp: never emit <think> blocks in a voice pipeline
                "chat_template_kwargs": {"enable_thinking": False},
                "top_p": 0.8,
                "top_k": 20,
                # server default presence_penalty=1.5 breaks tool calling
                "presence_penalty": 0,
            },
        )
    return opts

# pane.agent_status_changed subscriptions are per-pane, so the fleet watcher
# resubscribes on this interval to pick up panes created since (spawn_task
# and friends). Also bounds the catch-up latency for events missed while
# between subscriptions.
WATCH_RESUBSCRIBE_SECS = 30.0

INSTRUCTIONS = """You are Mate, a hands-free voice assistant supervising a fleet of
coding agents in herdr. The user is often driving and cannot look at a screen.
Call the user "mate" — greet them with it and drop it in naturally, though not
in every single sentence.

Voice output: plain text only — no markdown, lists, code blocks, or emoji. One to
three short sentences unless asked for more. Vary your acknowledgments; never
repeat a sentence you already said. When speaking, refer to agents by workspace
or project name, not pane ids.

Tools are the only source of truth. Refresh with them before reporting status;
never answer from memory. Never claim an action you did not perform with a tool
this turn. You cannot cancel, undo, or stop anything already sent — if asked to,
say so and offer to send the agent a correcting message. If a tool returns
ERROR, tell the user plainly what failed.

When asked for status or "how is it going" on an agent, never answer from the
last thing you heard: call fleet_status, then agent_report with messages=2 or
more for that agent. If those replies still leave it unclear what the project
or task actually is, call agent_report again with more messages (up to 5)
before answering. Lead with what it is working on, then where it stands.

Messaging an agent is a two-step rail while guardrails are ON (the default).
tell_agent and spawn_task only STAGE: read the staged message back to the user
word for word, ask whether to send it, and stop. Only after the user replies,
call send_staged. If it returns NOT SENT that is normal, not an error — ask
explicitly "should I send it?" and call send_staged again after they answer. If
the user declines, call discard_staged. After a message is delivered, NEVER
poll or re-check on your own. Quick question: call wait_for_agent once.
Anything longer: tell the user you'll be notified when it finishes — you will
be, automatically.

Spawning a new agent: default to spawn_in_folder with the user's spoken
folder name — code resolves the real path and speaks the confirmation for
you, so after calling it say nothing until the user answers, then call
send_staged or discard_staged. Folders the user confirmed before get a short
confirmation automatically. Use spawn_task only when the user explicitly
asks for a worktree or a new branch. list_known_agents shows saved folders;
forget_agent removes one.

Guardrails toggle: the user can say "guardrails off" or "guardrails on" at any
time. The toggle is detected and enforced in code, not by you — never claim to
have switched it yourself, and a bracketed note is injected into the
conversation whenever the state changes. While guardrails are OFF, tell_agent
and spawn_task deliver immediately with no read-back or confirmation; after
delivering, briefly tell the user what was sent and to whom.

For "what did the agent say/find/conclude", prefer agent_report (full replies
from its transcript). Use read_pane for what is on screen now: pending prompts,
errors, running commands.
Resolve casual project names to workspace labels via fleet_status. If a reference
is genuinely ambiguous, ask one short question instead of guessing.
Before approving anything an agent is waiting on, read its pane first. If the
pending action looks destructive, send_answer stages the keys instead of
sending: read the on-screen action to the user verbatim, ask whether to
approve it, and call send_staged only after they reply. "Guardrails off" does
not bypass this — destructive approvals always need the spoken yes."""


# ---------------------------------------------------------------------------
# speech + rail-toggle helpers (pure functions, unit tested)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def tts_sanitize(text: str) -> str:
    """Agent replies are markdown; kokoro speaks punctuation literally. Strip
    code blocks, links, bullets, emphasis and non-ascii so the result reads
    as plain sentences."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", "", text, flags=re.M)
    text = re.sub(r"^#{1,6}[ \t]+", "", text, flags=re.M)
    text = re.sub(r"[*_#>|~]", "", text)
    text = "".join(ch for ch in text
                   if ch.isascii() and (ch.isprintable() or ch in "\n\t"))
    return re.sub(r"\s+", " ", text).strip()


def tts_summary(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    """First sentence or two of a sanitized reply — the spoken heads-up when
    an agent finishes. The full reply stays available via agent_report."""
    clean = tts_sanitize(text)
    if not clean:
        return "It didn't say anything I can read out."
    sentences: list[str] = []
    for s in _SENTENCE_END.split(clean):
        if sentences and (len(sentences) >= max_sentences
                          or len(" ".join(sentences)) + len(s) > max_chars):
            break
        sentences.append(s)
    summary = " ".join(sentences)[:max_chars].strip()
    if summary[-1] not in ".!?":
        summary += "."
    return summary


_RAIL_ON = ("guardrailson", "guardrailon")
_RAIL_OFF = ("guardrailsoff", "guardrailoff")


def detect_rail_toggle(transcript: str) -> bool | None:
    """True = guardrails on, False = off, None = no toggle phrase. Matched on
    the raw STT transcript with everything but letters removed, so "guard
    rails off", "guardrails off" and "guard-rails, off" all count. If both
    phrases appear, the last one spoken wins."""
    squashed = re.sub(r"[^a-z]", "", transcript.lower())
    on = max((squashed.rfind(p) for p in _RAIL_ON), default=-1)
    off = max((squashed.rfind(p) for p in _RAIL_OFF), default=-1)
    if on == off == -1:
        return None
    return on > off


class Delegations:
    """Panes handed work this call, with enough state to tell a real finish
    from the delivery race: right after prompt_agent returns, herdr is still
    typing the prompt into the pane, so the pane sits idle for a couple of
    seconds — idle+delegated alone must NOT count as finished, or the watcher
    announces a stale reply and permanently eats the real notification.

    A pane is finish_ready once it has been seen working. If it is never seen
    working within GRACE_SECS, that usually means herdr's delayed Enter got
    swallowed by the agent's paste guard and the prompt is sitting typed but
    unsubmitted — the watcher then sends a one-shot Enter nudge (needs_nudge/
    mark_nudged). Only after the nudge plus a second grace window with still
    no working state does idle count as finished (a task so quick that every
    observation missed the working state)."""

    GRACE_SECS = 20.0

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._panes: dict[str, dict] = {}

    def add(self, pane_id: str) -> None:
        self._panes[pane_id] = {"at": self._clock(), "started": False,
                                "nudged": False}

    def discard(self, pane_id: str) -> None:
        self._panes.pop(pane_id, None)

    def mark_started(self, pane_id: str) -> None:
        entry = self._panes.get(pane_id)
        if entry:
            entry["started"] = True

    def needs_nudge(self, pane_id: str) -> bool:
        entry = self._panes.get(pane_id)
        if entry is None:
            return False
        return (not entry["started"] and not entry["nudged"]
                and self._clock() - entry["at"] >= self.GRACE_SECS)

    def mark_nudged(self, pane_id: str) -> None:
        entry = self._panes.get(pane_id)
        if entry:
            entry["nudged"] = True
            entry["at"] = self._clock()  # fresh grace window after the nudge

    def finish_ready(self, pane_id: str) -> bool:
        entry = self._panes.get(pane_id)
        if entry is None:
            return False
        if entry["started"]:
            return True
        return (entry["nudged"]
                and self._clock() - entry["at"] >= self.GRACE_SECS)

    def __contains__(self, pane_id: str) -> bool:
        return pane_id in self._panes

    def __iter__(self):
        return iter(self._panes)


def user_transcripts(ctx) -> list[str]:
    """STT transcripts of the user's turns so far, oldest first, straight from
    the live session history — not the model's paraphrase of them. Returns []
    when no session is attached (unit tests pass ctx=None)."""
    history = getattr(getattr(ctx, "session", None), "history", None)
    if history is None:
        return []
    return [item.text_content or ""
            for item in getattr(history, "items", [])
            if getattr(item, "role", None) == "user"]


class Mate(Agent):
    def __init__(self, herdr: HerdrClient, known: KnownAgents | None = None,
                 roots: list | None = None):
        super().__init__(instructions=INSTRUCTIONS)
        self.herdr = herdr
        # name -> path memory of spawn targets the user has confirmed;
        # roots override is for tests (default: MATE_SRC_ROOTS / ~/src)
        self.known = known if known is not None else KnownAgents()
        self._roots = roots
        self._read_panes: set[str] = set()  # rails: read before approve
        # panes given work this call; watch_fleet announces when they finish
        self.delegated = Delegations()
        # background task-delivery coroutines (a spawned claude can take
        # minutes to boot; the call must not block on it). Strong refs so
        # they aren't garbage-collected mid-flight.
        self._bg: set[asyncio.Task] = set()
        # stage-and-confirm rail master switch, voice-toggled via
        # on_user_turn_completed ("guardrails on/off") — enforced in code,
        # never trusted to the model
        self.rail_enabled = True
        # stage-and-confirm rail: tell_agent/spawn_task park the exact
        # payload here; send_staged delivers it only after (a) at least one
        # new user turn since staging and (b) that turn's raw STT transcript
        # passes approves_send. The model never re-supplies the text, so what
        # was read back is byte-for-byte what gets delivered.
        self._staged: dict | None = None

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Code-level intercept on every finalized user transcript: detect the
        guardrail toggle phrase, flip the flag here (the model is never asked
        to), and inject the state change into this message so the LLM knows
        from this turn onward — the note persists in chat history."""
        toggle = detect_rail_toggle(new_message.text_content or "")
        if toggle is None:
            return
        already = toggle == self.rail_enabled
        self.rail_enabled = toggle
        state = ("ON: tell_agent and spawn_task stage only, and delivery "
                 "requires a spoken confirmation via send_staged"
                 if toggle else
                 "OFF: tell_agent and spawn_task deliver immediately with no "
                 "read-back or confirmation")
        logger.info("guardrails voice toggle: now %s%s",
                    "ON" if toggle else "OFF",
                    " (was already)" if already else "")
        new_message.content.append(
            "[code intercept: guardrail toggle phrase heard. Guardrails are "
            + ("already " if already else "now ") + state
            + ". Briefly confirm this to the user.]")

    async def _pane_label(self, pane_id: str) -> str:
        """Speakable name for a pane — workspace label, else pane title, else
        a generic fallback. Pane ids read as gibberish over TTS."""
        try:
            snap = await self.herdr.snapshot()
        except HerdrError:
            return "coding"
        pane = next((p for p in snap.get("panes", [])
                     if p.get("pane_id") == pane_id), None)
        if pane:
            ws = next((w for w in snap.get("workspaces", [])
                       if w.get("workspace_id") == pane.get("workspace_id")),
                      None)
            if ws and ws.get("label"):
                return str(ws["label"])
            if pane.get("terminal_title_stripped"):
                return str(pane["terminal_title_stripped"])
        return "coding"

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
        """Answer a blocked agent's on-screen prompt with keys, e.g.
        ["1","Enter"]. If the pending action looks destructive nothing is
        sent: the keys are staged — read the on-screen action to the user
        verbatim, ask whether to approve it, and call send_staged after they
        reply."""
        if pane_id not in self._read_panes:
            return ("REFUSED: read the pane first so the user knows what "
                    "they are approving.")
        try:
            text = await self.herdr.read_pane(pane_id, 40)
            if is_destructive(text):
                # destructive approvals ALWAYS take the rail — rail_enabled
                # (the "guardrails off" convenience toggle) covers messaging
                # and spawns, never this
                self._staged = {"kind": "keys", "pane_id": pane_id,
                                "keys": keys,
                                "turns": len(user_transcripts(ctx))}
                return (f"NOT SENT: the pending action looks destructive, so "
                        f"the keys {keys} are staged for pane {pane_id}. Read "
                        "the pending on-screen action to the user verbatim, "
                        "ask whether to approve it, and stop. Call "
                        "send_staged only after they reply; if they decline, "
                        "call discard_staged.")
            await self.herdr.send_keys(pane_id, keys)
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        return "sent"

    async def _agent_replies(self, pane_id: str, messages: int = 1) -> str:
        """Last replies from the pane's agent transcript, or an ERROR string.
        Dispatches on the harness via the transcripts adapter registry."""
        try:
            snap = await self.herdr.snapshot()
        except HerdrError as e:
            return f"ERROR: {e.code}: {e.message}"
        agent = next((a for a in snap.get("agents", [])
                      if a.get("pane_id") == pane_id), None)
        if agent is None:
            return (f"ERROR: no coding agent is registered in pane {pane_id}. "
                    "Use read_pane to see the raw terminal instead.")
        kind = agent.get("agent")
        reader = adapter_for(kind)
        session = agent.get("agent_session") or {}
        if reader is None or session.get("kind") != "id":
            return (f"ERROR: no transcript adapter for {kind} agents "
                    f"(adapters exist for: {supported_kinds()}). "
                    "Use read_pane instead.")
        try:
            replies = reader(agent.get("cwd", ""), session["value"],
                             max(1, min(messages, 10)))
        except OSError:
            return (f"ERROR: the {kind} transcript is not readable. "
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
            if status == "working":
                self.delegated.mark_started(pane_id)
            if status in ("idle", "done", "blocked"):
                if (pane_id in self.delegated
                        and not self.delegated.finish_ready(pane_id)):
                    # just delivered: herdr is still typing the prompt, so
                    # "idle" here is the pre-start state, not a finish
                    await asyncio.sleep(1.0)
                    continue
                self.delegated.discard(pane_id)
                reply = await self._agent_replies(pane_id)
                return f"agent finished (status: {status}). Its reply:\n{reply}"
            await asyncio.sleep(1.0)
        return (f"agent is still {status}. STOP checking — the user will be "
                "notified automatically the moment it finishes. Tell them "
                "that and move on.")

    async def _speak_confirmation(self, text: str) -> bool:
        """Speak a code-composed confirmation via TTS, bypassing the LLM —
        what is spoken is byte-for-byte what was staged. Returns False when
        no live session is attached (unit tests, console edge cases); the
        caller then falls back to a read-this-exactly tool result."""
        try:
            session = self.session
        except Exception:
            return False
        if session is None:
            return False
        await session.say(text)
        return True

    def _deliver_task_in_background(self, pane_id: str | None, task: str,
                                    label: str) -> None:
        """Hand the first prompt to a freshly spawned agent without blocking
        the call: claude can take minutes to boot (herdr answers
        agent_not_ready the whole time), and killing or stalling on that
        would be wrong — the songhaus bug. The pane joins `delegated` only
        once the task actually lands, so the watcher's grace/nudge clock
        starts at delivery, not at spawn. An empty task means the user asked
        for an agent with nothing to do yet — herdr rejects empty prompts
        (empty_agent_prompt), so there is nothing to deliver."""
        if not pane_id or not task.strip():
            return
        t = asyncio.create_task(self._deliver_task_bg(pane_id, task, label))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _deliver_task_bg(self, pane_id: str, task: str,
                               label: str) -> None:
        try:
            await self.herdr.deliver_task(pane_id, task)
        except Exception:
            logger.exception("background task delivery to %s (%s) failed",
                             pane_id, label)
            await self._speak_confirmation(
                f"Hey mate, the {label} agent came up but never took the "
                "task. The workspace is still open if you want a look.")
            return
        logger.info("background task delivery to %s (%s) succeeded",
                    pane_id, label)
        self.delegated.add(pane_id)

    async def _nudge_after_tell(self, pane_id: str) -> None:
        try:
            await self.herdr.nudge_enter(pane_id)
        except Exception:
            logger.exception("post-tell enter nudge failed for %s", pane_id)

    async def _deliver(self, staged: dict) -> str:
        """Actually deliver a staged payload (shared by send_staged and the
        guardrails-off immediate path)."""
        if staged["kind"] == "spawn_folder":
            try:
                result = await self.herdr.spawn_in_folder(
                    staged["path"], staged["name"],
                    agent=staged.get("agent", "claude"))
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message}"
            # the user just said yes to this exact path (or has guardrails
            # off): remember it so next time gets the short confirmation
            self.known.remember(staged["name"], staged["path"])
            self._deliver_task_in_background(
                result.get("pane_id"), staged["task"], staged["name"])
            result["task_delivery"] = (
                "queued — the task is handed over automatically as soon as "
                "the agent finishes booting; the user does not need to wait"
                if staged["task"].strip() else
                "none — no task was given; the agent opens ready and waits "
                "for instructions")
            return json.dumps(result)
        if staged["kind"] == "spawn":
            try:
                result = await self.herdr.spawn(
                    staged["repo_path"], staged["branch"],
                    agent=staged.get("agent", "claude"))
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message}"
            self._deliver_task_in_background(
                result.get("pane_id"), staged["task"], staged["branch"])
            return json.dumps(result)
        if staged["kind"] == "keys":
            try:
                await self.herdr.send_keys(staged["pane_id"], staged["keys"])
            except HerdrError as e:
                return (f"ERROR: {e.code}: {e.message} "
                        f"(pane {staged['pane_id']})")
            return "sent"
        pane_id = staged["pane_id"]
        try:
            await self.herdr.prompt_agent(pane_id, staged["text"])
        except HerdrError as e:
            if e.code == "agent_not_ready":
                # the agent is still booting (herdr refuses prompts until it
                # detects it idle — can take minutes on a first launch).
                # Queue the message instead of bouncing it back to the user.
                label = await self._pane_label(pane_id)
                self._deliver_task_in_background(
                    pane_id, staged["text"], label)
                return (f"the {label} agent is still starting up — the "
                        "message is queued and will be handed over the "
                        "moment it is ready. Tell the user that; nothing "
                        "more to do.")
            if e.code == "agent_not_found":
                return (f"ERROR: no coding agent is registered in pane {pane_id} — "
                        "it is just a shell. An agent appears only once claude (or "
                        "another integrated agent) is launched inside a herdr pane. "
                        "Tell the user that pane has no agent to talk to.")
            return f"ERROR: {e.code}: {e.message} (pane {pane_id})"
        # paste-guard self-heal: herdr accepted the prompt, but Claude Code
        # sometimes eats the delayed Enter (seen live: message stuck in the
        # input box, invisible to the watcher when the pane was already
        # working). A trailing Enter is a no-op when delivery worked.
        t = asyncio.create_task(self._nudge_after_tell(pane_id))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        self.delegated.add(pane_id)
        return ("delivered. For a quick question, call wait_for_agent once. "
                "For anything longer, tell the user they'll be notified when "
                "it finishes — do not check again on your own.")

    @function_tool
    async def tell_agent(self, ctx: RunContext, pane_id: str, text: str):
        """Stage a natural-language instruction for a coding agent. With
        guardrails on (default) nothing is sent yet: read the staged text back
        to the user word for word, ask whether to send it, and call
        send_staged after they reply. With guardrails off it is delivered
        immediately."""
        if not self.rail_enabled:
            self._staged = None
            result = await self._deliver(
                {"kind": "tell", "pane_id": pane_id, "text": text})
            if result.startswith("ERROR"):
                return result
            return (f'guardrails off — delivered to pane {pane_id} '
                    f'immediately: "{text}"\n' + result)
        self._staged = {"kind": "tell", "pane_id": pane_id, "text": text,
                        "turns": len(user_transcripts(ctx))}
        return (f'staged for pane {pane_id}: "{text}"\n'
                "NOT SENT YET. Read that back to the user word for word, ask "
                "whether to send it, and stop. Call send_staged only after "
                "they reply.")

    @function_tool
    async def spawn_task(self, ctx: RunContext, repo_path: str, branch: str,
                         task: str, agent: str = "claude"):
        """Stage a new worktree + coding agent on a task. Leave agent as
        "claude" unless the user names a different harness. With guardrails
        on (default) nothing is created yet: read the staged task back to the
        user word for word, ask whether to go ahead, and call send_staged
        after they reply. With guardrails off it starts immediately."""
        agent = agent.strip().lower() or "claude"
        agent_phrase = ("a new agent" if agent == "claude"
                        else f"a {agent} agent")
        if not self.rail_enabled:
            self._staged = None
            result = await self._deliver(
                {"kind": "spawn", "repo_path": repo_path, "branch": branch,
                 "task": task, "agent": agent})
            if result.startswith("ERROR"):
                return result
            return ("guardrails off — task started immediately.\n" + result)
        self._staged = {"kind": "spawn", "repo_path": repo_path,
                        "branch": branch, "task": task, "agent": agent,
                        "turns": len(user_transcripts(ctx))}
        return (f'staged: {agent_phrase} in {repo_path} (branch {branch}) '
                f'with task "{task}"\n'
                "NOT STARTED YET. Read that back to the user word for word, "
                "ask whether to go ahead, and stop. Call send_staged only "
                "after they reply.")

    @function_tool
    async def spawn_in_folder(self, ctx: RunContext, folder_name: str,
                              task: str, agent: str = "claude"):
        """Spawn a new coding agent directly in an existing source folder (it
        edits the real checkout — use spawn_task only if the user explicitly
        asks for a worktree or branch). Pass the user's SPOKEN folder name;
        the path is resolved and confirmed in code. Leave agent as "claude"
        unless the user names a different harness. If the user gave no task,
        pass task as an empty string — the agent opens ready and waits; do
        not invent a task. After calling this, do not read anything back —
        wait for the user's answer, then call send_staged (yes) or
        discard_staged (no)."""
        agent = agent.strip().lower() or "claude"
        # spoken confirmations name the harness only when it isn't the default
        agent_phrase = "a new agent" if agent == "claude" else f"a {agent} agent"
        known_path = self.known.get(folder_name)
        stale = known_path is not None and not os.path.isdir(known_path)
        if known_path and not stale:
            # tier 2: previously confirmed target -> short confirmation
            name, path = folder_name, known_path
            confirmation = (f"Spawning in {name} — go ahead?"
                            if agent == "claude" else
                            f"Spawning {agent_phrase} in {name} — go ahead?")
        else:
            candidates = resolve_folder(folder_name, self._roots)
            if not candidates:
                return (f'ERROR: no folder matching "{folder_name}" under '
                        "the source roots. Ask the user for the folder name "
                        "again.")
            if len(candidates) > 1:
                options = ", ".join(c.name for c in candidates)
                return (f'AMBIGUOUS: several folders match "{folder_name}": '
                        f"{options}. Ask the user which one they mean.")
            path = str(candidates[0])
            name = candidates[0].name
            # tier 1: unknown target -> full path, stated from the exact
            # bytes that will be used
            task_phrase = (f"task: {task}" if task.strip()
                           else "no task yet, it will open ready and wait")
            confirmation = (
                f"About to spawn {agent_phrase} in {name} — full path "
                f"{speakable_path(path)} — {task_phrase}. Should I go ahead?")
            if stale:
                confirmation = (f"Heads up: the remembered folder for {name} "
                                "no longer exists, so I re-resolved it. "
                                + confirmation)
        if not self.rail_enabled:
            self._staged = None
            result = await self._deliver(
                {"kind": "spawn_folder", "path": path, "name": name,
                 "task": task, "agent": agent})
            if result.startswith("ERROR"):
                return result
            return (f"guardrails off — agent started immediately in {path}.\n"
                    + result)
        self._staged = {"kind": "spawn_folder", "path": path, "name": name,
                        "task": task, "agent": agent,
                        "turns": len(user_transcripts(ctx))}
        if await self._speak_confirmation(confirmation):
            return ("staged; the confirmation question was already SPOKEN to "
                    "the user by code. Do not repeat it or read anything "
                    "back — reply with nothing. When the user answers, call "
                    "send_staged (yes) or discard_staged (no).")
        return ('staged. Read this to the user EXACTLY, then stop: "'
                + confirmation + '"')

    @function_tool
    async def list_known_agents(self, ctx: RunContext):
        """Saved spawn targets (name and folder) the user has previously
        confirmed. Use when the user asks what agents/folders are known."""
        pairs = self.known.names()
        if not pairs:
            return "no known agents saved yet."
        return json.dumps([{"name": n, "path": p} for n, p in pairs])

    @function_tool
    async def forget_agent(self, ctx: RunContext, name: str):
        """Remove a saved spawn target from memory, so its next spawn needs
        the full path confirmation again."""
        if self.known.forget(name):
            return f"forgotten: {name}. Its next spawn needs full confirmation."
        return f'ERROR: no known agent matching "{name}".'

    @function_tool
    async def send_staged(self, ctx: RunContext):
        """Deliver the currently staged message or task, exactly as staged.
        Only call after the user has replied to the read-back."""
        staged = self._staged
        if staged is None:
            return ("ERROR: nothing is staged. Use tell_agent or spawn_task "
                    "first.")
        turns = user_transcripts(ctx)
        if len(turns) <= staged["turns"]:
            return ("NOT SENT: the user has not replied to the read-back yet. "
                    "Read the staged message back, ask whether to send it, "
                    "and call send_staged after they answer.")
        if not approves_send(turns[-1]):
            return ("NOT SENT: could not verify a clear yes in the user's "
                    f'last reply ("{turns[-1]}"). Ask again explicitly — '
                    '"should I send it?" — wait for the answer, then call '
                    "send_staged again. If they do not want it sent, call "
                    "discard_staged.")
        self._staged = None  # one delivery attempt per confirmation
        return await self._deliver(staged)

    @function_tool
    async def discard_staged(self, ctx: RunContext):
        """Drop the staged message without sending it (user said no)."""
        if self._staged is None:
            return "nothing was staged."
        self._staged = None
        return "discarded — nothing was sent."


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
            # local servers ignore the header; against a hosted LLM this
            # makes the probe a real auth check
            headers = ({"Authorization": f"Bearer {LLM_API_KEY}"}
                       if name == "llm" else None)
            try:
                r = await client.get(url, headers=headers)
                bad_auth = name == "llm" and r.status_code in (401, 403)
                errors[name] = (None if r.status_code < 500 and not bad_auth
                                else f"HTTP {r.status_code}")
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"
    try:
        pong = await HerdrClient().call("ping")
        errors["herdr"] = None
        note = protocol_note(pong)
        if note:
            logger.warning(note)
        else:
            logger.info("herdr %s (protocol %s)",
                        pong.get("version"), pong.get("protocol"))
    except Exception as e:
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

    await ctx.connect()

    # Caller gate. Only SIP participants (they carry sip.phoneNumber) are
    # screened — console/playground participants already authenticated with
    # a LiveKit token. Fail-closed: an empty allowlist rejects every phone
    # caller. This is the only boundary against a hostile caller; the
    # confirmation rail guards against transcription error, not attackers.
    allowed = allowed_callers()

    async def _reject_call(caller: str) -> None:
        logger.warning("blocked call from %s: not in %s", caller, ENV_VAR)
        try:
            await ctx.api.room.delete_room(
                lk_api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            logger.exception("could not delete room for blocked caller")

    reject_tasks: set[asyncio.Task] = set()

    def _screen_late_joiner(participant) -> None:
        caller = sip_caller(participant.attributes)
        if caller is not None and not is_allowed(caller, allowed):
            t = asyncio.create_task(_reject_call(caller))
            reject_tasks.add(t)
            t.add_done_callback(reject_tasks.discard)

    # The SIP caller is normally already in the room when the job starts
    # (the dispatch rule created the room for them) — reject before the
    # session ever opens its mouth. The event handler covers stragglers.
    for p in list(ctx.room.remote_participants.values()):
        caller = sip_caller(p.attributes)
        if caller is not None and not is_allowed(caller, allowed):
            await _reject_call(caller)
            return
    ctx.room.on("participant_connected", _screen_late_joiner)

    herdr = HerdrClient()
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(base_url=STT_URL, api_key="local", model=STT_MODEL),
        llm=openai.LLM(**llm_options()),
        # "tts-1" (not "kokoro"): the plugin treats unknown model names as
        # OpenAI's SSE-streaming models and parses the response as SSE JSON,
        # but kokoro returns raw audio bytes -> "no audio frames were pushed".
        # kokoro-fastapi aliases tts-1 to kokoro, and tts-1 selects the
        # raw-audio stream path (AUDIO_STREAM_MODELS) in the plugin.
        tts=openai.TTS(base_url=TTS_URL, api_key="local", model="tts-1",
                       voice=TTS_VOICE),
        turn_handling={
            # 3s (down from 6s, which was sized for 4-5s CPU whisper): GPU
            # whisper returns in ~0.2s + ~0.8s transcript_delay, so 3s still
            # comfortably covers a real interruption being confirmed while
            # cutting the awkward resume-then-stop window in half.
            "interruption": {"false_interruption_timeout": 3.0},
            # A mid-sentence "um" pause let VAD commit the turn before the
            # rest of the utterance's transcript arrived (livekit warned:
            # "transcript arrives after turn has been committed, consider
            # raising min_delay"), splitting one request into two turns.
            "endpointing": {"min_delay": 1.0},
        },
    )

    async def watch_fleet():
        # Proactive interjection when an agent blocks, or when a pane Mate
        # delegated to finishes (kills the "let me check again" loop).
        #
        # herdr's pane.agent_status_changed subscription is PER-PANE: a bare
        # {"type": ...} is rejected with invalid_request (missing pane_id).
        # So each cycle: snapshot -> subscribe to every current pane ->
        # resubscribe every WATCH_RESUBSCRIBE_SECS to pick up new panes.
        # The snapshot doubles as a catch-up pass for delegated panes that
        # finished while we weren't subscribed.
        async def say(text):
            # session.say = deterministic TTS, no LLM in the loop. qwen has
            # twice mangled generate_reply(instructions=...) at exactly this
            # moment (once refusing to call a tool, once repeating its
            # previous sentence verbatim instead of announcing), and the
            # notification moment is too important to gamble on it.
            # add_to_chat_ctx defaults True, so the model still knows what
            # was said.
            logger.info("watch_fleet: announcing: %s", text)
            await session.say(text)
            logger.info("watch_fleet: announcement spoken")

        async def announce(status, pane_id, data):
            if status == "working":
                if pane_id in mate.delegated:
                    mate.delegated.mark_started(pane_id)
                    logger.info("watch_fleet: delegated pane %s started "
                                "working", pane_id)
                return
            if status == "blocked":
                label = await mate._pane_label(pane_id)
                await say(f"Hey mate, the {label} agent is waiting on your "
                          "approval. Want me to read its question?")
            elif (status in ("idle", "done")
                  and pane_id in mate.delegated):
                if not mate.delegated.finish_ready(pane_id):
                    # delivery race: the pane was never seen working after
                    # delivery. Early on, herdr is still typing the prompt;
                    # announcing now would read a STALE reply and discard the
                    # pane, eating the real notification later.
                    if mate.delegated.needs_nudge(pane_id):
                        # grace expired with no working state: herdr's delayed
                        # Enter was probably swallowed by the agent's paste
                        # guard, leaving the prompt typed but unsubmitted.
                        # One bare Enter submits it (and is a no-op on an
                        # empty input box if it did go through).
                        logger.warning("watch_fleet: pane %s never started "
                                       "after delivery — nudging with Enter "
                                       "(submit likely lost)", pane_id)
                        try:
                            await herdr.send_keys(pane_id, ["Enter"])
                        except HerdrError as e:
                            logger.warning("watch_fleet: nudge failed: %s", e)
                        mate.delegated.mark_nudged(pane_id)
                    else:
                        logger.info("watch_fleet: pane %s idle but not "
                                    "finish-ready yet, skipping", pane_id)
                    return
                mate.delegated.discard(pane_id)  # one announcement per task
                # Fetch the reply here: asking qwen to call agent_report from
                # an injected instruction doesn't work reliably.
                reply = await mate._agent_replies(pane_id)
                label = await mate._pane_label(pane_id)
                if reply.startswith("ERROR"):
                    logger.warning("watch_fleet: finish on %s but reply "
                                   "unreadable: %s", pane_id, reply)
                    await say(f"Hey mate, the {label} agent just finished, "
                              "but I couldn't read its reply. Want me to "
                              "read its screen instead?")
                else:
                    await say(f"Hey mate, the {label} agent just finished. "
                              f"{tts_summary(reply)} Want the full report?")

        while True:
            try:
                snap = await herdr.snapshot()
                # catch-up: delegated panes that changed state between
                # subscriptions (idle/done only -- re-announcing "blocked"
                # every cycle would nag; "working" just marks started)
                for a in snap.get("agents", []):
                    if (a.get("pane_id") in mate.delegated
                            and a.get("agent_status")
                            in ("idle", "done", "working")):
                        await announce(a["agent_status"], a["pane_id"], a)
                pane_ids = [p["pane_id"] for p in snap.get("panes", [])
                            if p.get("pane_id")]
                logger.info("watch_fleet: cycle: %d panes, delegated=%s",
                            len(pane_ids), list(mate.delegated))
                if not pane_ids:
                    await asyncio.sleep(WATCH_RESUBSCRIBE_SECS)
                    continue
                events = herdr.events(
                    [{"type": "pane.agent_status_changed", "pane_id": p}
                     for p in pane_ids])
                loop = asyncio.get_running_loop()
                deadline = loop.time() + WATCH_RESUBSCRIBE_SECS
                try:
                    while (remaining := deadline - loop.time()) > 0:
                        # only the WAIT is under a timeout -- announce() can
                        # hold the floor for 10+ s of TTS and must never be
                        # cancelled mid-speech by the resubscribe deadline
                        try:
                            msg = await asyncio.wait_for(
                                anext(events), remaining)
                        except (TimeoutError, StopAsyncIteration):
                            break
                        data = msg.get("data", {})
                        logger.info("watch_fleet: event pane=%s status=%s",
                                    data.get("pane_id"),
                                    data.get("agent_status"))
                        await announce(data.get("agent_status"),
                                       data.get("pane_id"), data)
                finally:
                    await events.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("watch_fleet: watcher error, retrying in 5s")
                await asyncio.sleep(5)

    mate = Mate(herdr)
    watcher = asyncio.create_task(watch_fleet())

    async def _stop_watcher():
        watcher.cancel()

    ctx.add_shutdown_callback(_stop_watcher)

    await session.start(agent=mate, room=ctx.room)
    # deterministic greeting -- no LLM roll on the very first thing heard
    try:
        snap = await herdr.snapshot()
        n = len(snap.get("agents", []))
        fleet = f"{n} agent{'s' if n != 1 else ''} running"
    except Exception:
        fleet = "your fleet is up"
    await session.say(f"G'day mate. {fleet}. What do you need?")


def _require_allowlist(argv: list[str], env=None) -> None:
    """Refuse to run a phone-facing worker (dev/start) without a caller
    allowlist. console mode has no SIP path and is exempt."""
    if not {"dev", "start"} & set(argv[1:]):
        return
    if not allowed_callers(env):
        raise SystemExit(
            f"{ENV_VAR} is not set (or empty). Refusing to take phone calls "
            "without a caller allowlist. Put a comma-separated E.164 list in "
            ".env, e.g.  MATE_ALLOWED_NUMBERS=+14055551234")


if __name__ == "__main__":
    _require_allowlist(sys.argv)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
