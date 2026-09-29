"""Mate — LiveKit Agents worker bridging voice to a herdr fleet.

Run:  python -m mate.agent console   # terminal mic/speaker desk test
      python -m mate.agent dev       # worker + Agents Playground
      python -m mate.agent start     # production (SIP calls dispatch here)

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
import signal
import sys
import unicodedata

import httpx
from livekit import api as lk_api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    StopResponse,
    WorkerOptions,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import openai, silero

from .allowlist import ENV_VAR, allowed_callers
from .delivery import Delivery
from .folders import KnownAgents, resolve_folder, speakable_path
from .herdr_client import HerdrClient, HerdrError, protocol_note
from .passphrase import (
    FailedCalls,
    ensure_launch_phrase,
    phrase_heard,
)
from .safety import approves_send, is_destructive
from .screening import CallScreen
from .transcripts import adapter_for, resolve_agent_session, supported_kinds
from .watcher import Delegations, FleetWatcher

logger = logging.getLogger("mate")

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
    # transliterate before the ascii filter, or accented words lose letters
    # instead of keeping them: "naive" not "nave", "cafe" not "caf".
    # NFKD splits a letter from its accent; dropping the combining marks
    # leaves the plain ascii letter behind.
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))
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


async def fleet_greeting(herdr) -> str:
    """Spoken fleet-size line for greetings: "3 agents running"."""
    try:
        snap = await herdr.snapshot()
        n = len(snap.get("agents", []))
        return f"{n} agent{'s' if n != 1 else ''} running"
    except Exception:
        return "your fleet is up"


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


LOCKED_MSG = "REFUSED: the caller has not said the passphrase."


def kill_worker() -> None:
    """Bring down the whole worker, not just the current call: jobs run in
    child processes of the worker, so SIGTERM the shared process group. A
    shell or systemd parent sits in a different group and is untouched."""
    logger.critical("shutting the worker down")
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        logger.exception("could not signal the worker process group; "
                         "exiting this job only")
        os._exit(1)


def speech_seconds(msg) -> float | None:
    """VAD-measured speaking time of a user turn, from the message metrics
    the framework attaches to every user ChatMessage. None if unavailable
    (unit tests, text-only input)."""
    metrics = getattr(msg, "metrics", None) or {}
    try:
        return metrics["stopped_speaking_at"] - metrics["started_speaking_at"]
    except (KeyError, TypeError):
        return None


class Mate(Agent):
    MAX_PASSPHRASE_ATTEMPTS = 3
    # a passphrase attempt longer than this is a miss no matter what it
    # contains — one turn can't be stuffed with candidate phrases
    MAX_PASSPHRASE_SPEECH_SECS = 15.0
    # _staged is a single slot, so a late "yes" would otherwise deliver
    # whatever was staged however many turns ago. After this many user
    # turns the stage expires: the read-back is no longer what the user has
    # in mind, and the model must stage and read back again.
    MAX_STAGED_TURNS = 3

    def __init__(self, herdr: HerdrClient, known: KnownAgents | None = None,
                 roots: list | None = None, delivery: Delivery | None = None):
        super().__init__(instructions=INSTRUCTIONS)
        self.herdr = herdr
        # spawning and task delivery go through the quirk layer (herdr 0.7.5
        # workarounds); plain reads/keys go straight to the protocol client.
        # Injectable so tests can record deliveries without a real socket.
        self.delivery = delivery if delivery is not None else Delivery(herdr)
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
        # spoken-passphrase gate (lock()): while locked, every user turn is
        # intercepted in code and never reaches the LLM, so no tool can fire
        self.locked = False
        self.passphrase_passed = False
        self._passphrase = ""
        self._attempts = 0
        self._hangup = None
        self._on_unlock = None
        # The Live path supplies a contextual classifier. The chained
        # pipeline retains its existing deterministic confirmation check.
        self._approval_gate = None

    def lock(self, phrase: str, hangup=None, on_unlock=None) -> None:
        """Arm the spoken-passphrase gate. Until the caller says `phrase`,
        on_user_turn_completed answers every turn itself (retry prompts and
        all) and raises StopResponse so the LLM never runs. `hangup` is
        awaited after MAX_PASSPHRASE_ATTEMPTS failures; `on_unlock` is
        called (sync) the moment the phrase is accepted."""
        self.locked = True
        self._passphrase = phrase
        self._hangup = hangup
        self._on_unlock = on_unlock
        self._attempts = 0

    async def _check_passphrase(self, transcript: str,
                                speech_secs: float | None = None) -> None:
        too_long = (speech_secs is not None
                    and speech_secs > self.MAX_PASSPHRASE_SPEECH_SECS)
        if not too_long and phrase_heard(transcript, self._passphrase):
            self.locked = False
            self.passphrase_passed = True
            logger.info("passphrase accepted")
            if self._on_unlock is not None:
                self._on_unlock()
            await self._speak_confirmation(
                f"That's it, mate. {await fleet_greeting(self.herdr)}. "
                "What do you need?")
            return
        # count only — logging the transcript would leak near-misses
        self._attempts += 1
        logger.warning("passphrase attempt %d/%d failed%s",
                       self._attempts, self.MAX_PASSPHRASE_ATTEMPTS,
                       " (spoke too long)" if too_long else "")
        if self._attempts >= self.MAX_PASSPHRASE_ATTEMPTS:
            await self._speak_confirmation(
                "Sorry mate, that's not it. Goodbye.")
            if self._hangup is not None:
                await self._hangup()
            return
        await self._speak_confirmation(
            "Too long, mate. Just the passphrase, please." if too_long
            else "That's not it. What's the passphrase?")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Code-level intercept on every finalized user transcript. While the
        passphrase gate is locked the turn stops here: verification, retry
        prompts and the hangup are all code, and StopResponse keeps the LLM
        out of the loop entirely. Once unlocked, the only intercept is the
        guardrail toggle phrase: detect it, flip the flag here (the model is
        never asked to), and inject the state change into this message so
        the LLM knows from this turn onward — the note persists in chat
        history."""
        if self.locked:
            await self._check_passphrase(new_message.text_content or "",
                                         speech_seconds(new_message))
            raise StopResponse()
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

    async def _pane_label(self, pane_id: str, snap: dict | None = None) -> str:
        """Speakable name for a pane — workspace label, else pane title, else
        a generic fallback. Pane ids read as gibberish over TTS.

        `snap` is an already-fetched session snapshot: the watcher's finish
        announcement needs the label and the reply together, and one
        snapshot serves both. None fetches a fresh one."""
        if snap is None:
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
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
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
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
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
        if self.locked:
            return LOCKED_MSG
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
                                "screen": text,
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

    async def _agent_replies(self, pane_id: str, messages: int = 1,
                             snap: dict | None = None) -> str:
        """Last replies from the pane's agent transcript, or an ERROR string.
        Dispatches on the harness via the transcripts adapter registry.
        `snap` reuses a caller's session snapshot (see _pane_label); None
        fetches a fresh one."""
        if snap is None:
            try:
                snap = await self.herdr.snapshot()
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message}"
        agent = next((a for a in snap.get("agents", [])
                      if a.get("pane_id") == pane_id), None)
        if agent is None:
            return (f"ERROR: no coding agent is registered in pane {pane_id}. "
                    "Use read_pane to see the raw terminal instead.")
        agent = await resolve_agent_session(self.herdr, agent)
        kind = agent.get("agent")
        reader = adapter_for(kind)
        session = agent.get("agent_session") or {}
        reason = None
        replies = []
        if reader is None:
            reason = f"no transcript adapter for {kind} (supported: {supported_kinds()})"
        elif session.get("kind") != "id":
            reason = "native session identity is missing"
        else:
            try:
                replies = await asyncio.to_thread(reader, agent.get("cwd", ""), session["value"],
                                                 max(1, min(messages, 10)))
            except (OSError, KeyError):
                reason = "transcript is unavailable"
        if not replies:
            from .audit import event as trace
            reason = reason or "no final reply in the current transcript segment"
            trace("agent.report_fallback", pane_id=pane_id, agent=kind, reason=reason)
            try:
                screen = await self.herdr.read_pane(pane_id, 100)
            except (HerdrError, OSError, AttributeError):
                return f"ERROR: {reason}; could not read_pane either."
            if not screen.strip():
                return f"ERROR: {reason}; the agent screen is empty."
            return ("From the agent's screen (partial view, not a verified full reply):\n"
                    + screen[-8000:])
        return "\n\n---\n\n".join(replies)[-8000:]

    @function_tool
    async def agent_report(self, ctx: RunContext, pane_id: str,
                           messages: int = 1):
        """Read the coding agent's last full reply/replies from its session
        transcript. Better than read_pane for "what did it say/find" — screen
        scrollback loses long answers, the transcript never does."""
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
        return await self._agent_replies(pane_id, messages)

    @function_tool
    async def wait_for_agent(self, ctx: RunContext, pane_id: str,
                             seconds: int = 15):
        """Wait briefly (max 20s) for a busy agent to finish, then return its
        reply. Use after tell_agent for QUICK questions only. If it is still
        working when time runs out, do NOT call again — the user will be
        notified automatically when the agent finishes."""
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
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
        would be wrong — the songhaus bug. An empty task means the user asked
        for an agent with nothing to do yet — herdr rejects empty prompts
        (empty_agent_prompt), so there is nothing to deliver. The pane joins
        `delegated` only once the task actually lands, so the watcher's grace
        clock starts at delivery, not at spawn."""
        if not pane_id or not task.strip():
            return
        t = asyncio.create_task(self._deliver_task_bg(pane_id, task, label))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _deliver_task_bg(self, pane_id: str, task: str,
                               label: str) -> None:
        try:
            await self.delivery.deliver_task(pane_id, task)
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

    async def _deliver(self, staged: dict) -> str:
        """Actually deliver a staged payload (shared by send_staged and the
        guardrails-off immediate path)."""
        if staged["kind"] == "spawn_folder":
            try:
                result = await self.delivery.spawn_in_folder(
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
                result = await self.delivery.spawn(
                    staged["repo_path"], staged["branch"],
                    agent=staged.get("agent", "claude"))
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message}"
            self._deliver_task_in_background(
                result.get("pane_id"), staged["task"], staged["branch"])
            return json.dumps(result)
        if staged["kind"] == "keys":
            keys_pane = staged["pane_id"]
            # what the user approved is a SCREEN, not a pane. Between the
            # read-back and their yes the agent can have moved on, and these
            # keys would then answer whatever dialog is up now. So re-read
            # the pane and require the same destructive match before firing.
            # Deliberately asymmetric: a destructive prompt that DISAPPEARED
            # means the moment has passed, so a now-clean screen is a
            # refusal, not a green light — keys must never go into an
            # unknown screen. Regex-still-matches is the whole bar.
            try:
                current = await self.herdr.read_pane(keys_pane, 40)
            except HerdrError as e:
                return (f"NOT SENT: pane {keys_pane} could not be re-read "
                        f"before approving ({e.code}: {e.message}), so the "
                        "keys were dropped. Call read_pane again and restart "
                        "the approval.")
            if not is_destructive(current):
                return (f"NOT SENT: pane {keys_pane} is no longer showing the "
                        "action that was read back, so the keys were dropped "
                        "rather than sent into an unknown screen. Call "
                        "read_pane again, tell the user what is on screen "
                        "now, and restart the approval if they still want it.")
            if staged.get("_confirmation") and current != staged.get("screen"):
                return "NOT SENT: the approval screen changed. Read it again and stage a new confirmation."
            try:
                await self.herdr.send_keys(keys_pane, staged["keys"])
            except HerdrError as e:
                return f"ERROR: {e.code}: {e.message} (pane {keys_pane})"
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
        # herdr owns prompt submission now: 0.8.0 delays its own Enter past
        # Claude Code's paste guard (#1878), so no trailing nudge is sent.
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
        if self.locked:
            return LOCKED_MSG
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
        if self.locked:
            return LOCKED_MSG
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
        if self.locked:
            return LOCKED_MSG
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
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
        pairs = self.known.names()
        if not pairs:
            return "no known agents saved yet."
        return json.dumps([{"name": n, "path": p} for n, p in pairs])

    @function_tool
    async def forget_agent(self, ctx: RunContext, name: str):
        """Remove a saved spawn target from memory, so its next spawn needs
        the full path confirmation again."""
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
        if self.known.forget(name):
            return f"forgotten: {name}. Its next spawn needs full confirmation."
        return f'ERROR: no known agent matching "{name}".'

    @function_tool
    async def send_staged(self, ctx: RunContext):
        """Deliver the currently staged message or task, exactly as staged.
        Only call after the user has replied to the read-back."""
        if self.locked:
            return LOCKED_MSG
        staged = self._staged
        if staged is None:
            return ("ERROR: nothing is staged. Use tell_agent or spawn_task "
                    "first.")
        turns = user_transcripts(ctx)
        if len(turns) <= staged["turns"]:
            return ("NOT SENT: the user has not replied to the read-back yet. "
                    "Read the staged message back, ask whether to send it, "
                    "and call send_staged after they answer.")
        if len(turns) - staged["turns"] > self.MAX_STAGED_TURNS:
            # the conversation moved on; a "yes" this late is answering
            # something else, not the read-back
            self._staged = None
            return ("NOT SENT: the staged item expired — the user has spoken "
                    f"{len(turns) - staged['turns']} times since it was "
                    "staged and it has been dropped. Stage it again, read the "
                    "new staging back to the user word for word, and call "
                    "send_staged only after they reply to that.")
        approved = (await self._approval_gate(staged, ctx)
                    if self._approval_gate is not None else approves_send(turns[-1]))
        if self.locked or self._staged is not staged:
            return "NOT SENT: authorization or the pending action changed during confirmation."
        if not approved:
            if self._approval_gate is not None:
                return ("NOT SENT: could not verify approval of this exact action. "
                        "The reply may be unclear or the approval check unavailable. "
                        "Ask whether to proceed and wait for a new reply; if declined, discard it.")
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
        # defense in depth behind the on_user_turn_completed intercept
        if self.locked:
            return LOCKED_MSG
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
    live_config = None
    mode = os.environ.get("MATE_VOICE_MODE", "local")
    if mode not in ("local", "live"):
        return {"config": "MATE_VOICE_MODE must be local or live"}
    if mode == "live":
        from .live import LiveConfig
        try:
            live_config = LiveConfig.from_env()
        except ValueError as e:
            return {"config": str(e)}
        targets.pop("llm")
        targets["openai"] = "https://api.openai.com/v1/models"
    errors: dict[str, str | None] = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in targets.items():
            # local servers ignore the header; against a hosted LLM this
            # makes the probe a real auth check
            headers = ({"Authorization": f"Bearer {LLM_API_KEY}"}
                       if name == "llm" else None)
            if name == "openai":
                headers = {"Authorization": f"Bearer {live_config.api_key}"}
            try:
                r = await client.get(url, headers=headers)
                bad_auth = name in ("llm", "openai") and r.status_code in (401, 403)
                if name == "openai":
                    r.raise_for_status()
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
    from .audit import configure
    configure(ctx.job.id)
    status = await check_endpoints()
    down = {k: v for k, v in status.items() if v}
    if down:
        details = "\n".join(f"  {k}: {v}" for k, v in down.items())
        raise RuntimeError(
            f"mate cannot start; unreachable services:\n{details}\n"
            f"(llm={LLM_URL} stt={STT_URL} tts={TTS_URL} "
            f"herdr={HerdrClient().sock_path})")

    await ctx.connect()

    herdr = HerdrClient()
    live_mode = os.environ.get("MATE_VOICE_MODE", "local") == "live"
    if live_mode:
        from .live import LiveConfig
        from .live_agent import LiveMate
        mate = LiveMate(herdr, LiveConfig.from_env())
    else:
        mate = Mate(herdr)
    # assigned below; `say` and the watcher only ever run once it is live
    session = None
    watcher_holder: list[asyncio.Task] = []

    async def delete_room() -> None:
        await ctx.api.room.delete_room(
            lk_api.DeleteRoomRequest(room=ctx.room.name))

    async def say(text: str) -> None:
        # session.say = deterministic TTS, no LLM in the loop. qwen has
        # twice mangled generate_reply(instructions=...) at exactly this
        # moment (once refusing to call a tool, once repeating its
        # previous sentence verbatim instead of announcing), and the
        # notification moment is too important to gamble on it.
        # add_to_chat_ctx defaults True, so the model still knows what
        # was said.
        if live_mode:
            await mate.bridge.announce(text)
        else:
            await session.say(text)

    def start_watcher() -> None:
        if not watcher_holder:
            watcher_holder.append(asyncio.create_task(watcher.run()))

    screen = CallScreen(mate, allowed_callers(), FailedCalls(),
                        delete_room=delete_room, say=say, kill=kill_worker,
                        start_watcher=start_watcher)

    # Caller allowlist first: a blocked caller must never reach a session.
    if not await screen.screen_participants(
            ctx.room.remote_participants.values()):
        return
    ctx.room.on("participant_connected", screen.screen_late_joiner)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(base_url=STT_URL, api_key="local", model=STT_MODEL),
        llm=None if live_mode else openai.LLM(**llm_options()),
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

    watcher = FleetWatcher(herdr, mate, say)

    # Spoken-passphrase gate: the LLM never sees a locked turn, and while
    # locked the watcher does not run either — no fleet announcements to an
    # unverified caller. The straggler handler is registered before the
    # in-room callers are gated, so a caller arriving mid-arming is still
    # covered.
    ctx.room.on("participant_connected", screen.gate_late_sip_joiner)

    if not await screen.arm_initial_callers(
            ctx.room.remote_participants.values()):
        return

    if not mate.locked:
        start_watcher()

    async def _stop_watcher():
        for w in watcher_holder:
            w.cancel()
        if live_mode:
            await mate.bridge.aclose()

    ctx.add_shutdown_callback(_stop_watcher)

    if live_mode:
        # Live audio has no matching LiveKit text-generation stream.
        # Disable text/audio synchronization for the direct audio writer.
        await session.start(agent=mate, room=ctx.room, room_options=room_io.RoomOptions(
            text_input=False,
            text_output=room_io.TextOutputOptions(sync_transcription=False),
            delete_room_on_close=True,
        ))
    else:
        await session.start(agent=mate, room=ctx.room)
    # deterministic greeting -- no LLM roll on the very first thing heard.
    # `greeted` flips first so a SIP straggler landing mid-greeting still
    # gets a spoken passphrase prompt from gate_late_sip_joiner.
    screen.mark_greeted()
    if mate.locked:
        # no fleet details before the caller proves who they are
        await say("G'day. What's the passphrase?")
    else:
        await say(f"G'day mate. {await fleet_greeting(herdr)}. "
                  "What do you need?")
        if live_mode:
            await mate.bridge.start()


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
    ensure_launch_phrase(sys.argv)
    # a restart is the operator's deliberate reset of the failed-call
    # lockout (job processes never run this — once per service start)
    FailedCalls().reset()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
