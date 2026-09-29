"""The herdr quirk layer: spawning agents and getting a task into one.

Everything here exists because of a herdr bug or a timing wart, so a
future herdr upgrade has ONE file to audit and delete from: re-test each
workaround against the new server and drop the ones it fixed. (The last
one out was the guarded Enter nudge — herdr 0.8.0 delays its own prompt
Enter past Claude Code's paste guard, #1878, so mate no longer sends a
bare Enter after delivery at all.)
herdr_client.py stays a pure protocol client, and the protocol version
these workarounds were verified against is TESTED_PROTOCOL there (a
different herdr is warned about at startup, never refused).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from .herdr_client import HerdrClient, HerdrError

logger = logging.getLogger("mate")


class Delivery:
    """Spawn-and-deliver on top of a HerdrClient, with every herdr 0.7.5
    workaround the live calls needed."""

    # A freshly created pane's shell needs a moment before agent.start
    # succeeds (herdr: agent_pane_busy "not an available shell"). Verified
    # live 2026-07-30: immediate agent.start after workspace.create fails.
    START_RETRIES = 12
    START_RETRY_DELAY = 0.4
    # herdr rejects agent.prompt (agent_not_ready) until it *detects* the
    # launched agent idle — claude's first boot in a folder (trust dialog,
    # big repo) can take well over a minute, so task delivery must be
    # patient and must never be treated as a launch failure. timeout_ms
    # stretches herdr's own managed-launch deadline (default 30s) to match.
    LAUNCH_TIMEOUT_MS = 120_000
    PROMPT_RETRIES = 240
    PROMPT_RETRY_DELAY = 0.5
    # herdr 0.7.5 bug seen live 2026-07-30: a managed launch can stay
    # launch_pending forever even though detection reports the agent idle
    # ("not an active named agent" on every prompt). After this many prompt
    # attempts, if the snapshot proves the pane hosts a settled agent, type
    # the message straight into the pane instead of using the gated prompt.
    FALLBACK_AFTER = 40  # attempts (~20s), rechecked every 10 after that
    # herdr 0.7.5 bug seen live (2026-07-30, worktree workspace): agent.prompt
    # answers agent_prompted but nothing is ever typed into the pane. A landed
    # prompt always advances the agent's state_change_seq within moments (the
    # typed text sends claude working); a dropped one leaves it frozen.
    PROMPT_VERIFY_WAIT = 2.0  # per check
    PROMPT_VERIFY_CHECKS = 2

    def __init__(self, herdr: HerdrClient):
        self.herdr = herdr

    async def start_agent(self, pane_id: str, name: str,
                          kind: str = "claude") -> str:
        """agent.start with shell-settle retries. The requested name is
        first folded to herdr's naming rules (seen live 2026-07-30: label
        "RackCoach" → instant invalid_agent_name). Agent names are unique
        fleet-wide in herdr, so a taken name gets a numeric suffix. Returns
        the agent name actually used; raises HerdrError on failure."""
        base = name = _sanitize_agent_name(name)
        for attempt in range(self.START_RETRIES):
            try:
                await self.herdr.call("agent.start", pane_id=pane_id,
                                      name=name, kind=kind,
                                      timeout_ms=self.LAUNCH_TIMEOUT_MS)
                return name
            except HerdrError as e:
                if e.code == "duplicate_agent_name":
                    suffix = f"-{attempt + 2}"
                    name = base[:_AGENT_NAME_MAX - len(suffix)] + suffix
                    continue
                if (e.code == "agent_pane_busy"
                        and attempt < self.START_RETRIES - 1):
                    await asyncio.sleep(self.START_RETRY_DELAY)
                    continue
                raise
        return name

    async def _pane_hosts_settled_agent(self, pane_id: str) -> bool:
        """True when herdr's detection shows a coding agent sitting idle in
        this pane — the precondition for typing a message into the pane
        directly (never type into a bare shell).

        Idle only, deliberately: a "blocked" agent is sitting on a
        permission or trust dialog, where the Enter that send_input appends
        would ACCEPT the highlighted option, and the typed task would be
        swallowed by the dialog anyway. Waiting for another retry costs
        nothing."""
        try:
            listed = await self.herdr.agents()
        except (HerdrError, OSError):
            # transport too: a slow or restarting herdr raises through
            # call() unwrapped (asyncio.wait_for's TimeoutError is an
            # OSError since 3.10) — treat it as "no settled agent"
            return False
        for a in listed.get("agents", []):
            if (a.get("pane_id") == pane_id and a.get("agent")
                    and a.get("agent_status") == "idle"):
                return True
        return False

    async def _agent_state_marker(self, pane_id: str) -> tuple | None:
        """(state_change_seq, agent_status) for the pane's agent, or None
        when the agent isn't listed or herdr doesn't report the seq (no
        verification possible then — the dropped-prompt fallback must never
        fire on doubt alone, it types text into a terminal)."""
        try:
            listed = await self.herdr.agents()
        except (HerdrError, OSError):
            # transport too, same as _pane_hosts_settled_agent: unreadable
            # state is "no verification possible", never a fallback trigger
            return None
        for a in listed.get("agents", []):
            if a.get("pane_id") == pane_id:
                seq = a.get("state_change_seq")
                return None if seq is None else (seq, a.get("agent_status"))
        return None

    async def _prompt_landed(self, pane_id: str, before: tuple | None) -> bool:
        """False only when the agent's state provably never moved after an
        accepted prompt. Unknown/unreadable state counts as landed."""
        if before is None:
            return True
        for _ in range(self.PROMPT_VERIFY_CHECKS):
            await asyncio.sleep(self.PROMPT_VERIFY_WAIT)
            now = await self._agent_state_marker(pane_id)
            if now is None or now != before:
                return True
        return False

    async def deliver_task(self, pane_id: str, task: str) -> None:
        """Hand a prompt to an agent, retrying through the whole boot window
        (agent_not_ready until herdr detects the agent idle). Two herdr
        0.7.5 failure modes are worked around, both seen live:

        - stuck launch tracking (launch_pending forever while the agent is
          visibly idle — agent.prompt refuses every attempt): after ~20s of
          refusals, type into the pane once the snapshot proves a settled
          agent owns it.
        - silently dropped prompt (worktree-workspace panes: agent.prompt
          answers agent_prompted but never types anything): if the agent's
          state_change_seq is still frozen after an accepted prompt, type
          into the pane — same settled-agent guard, never a bare shell.

        Raises HerdrError if the agent never becomes promptable."""
        marker = await self._agent_state_marker(pane_id)
        for attempt in range(self.PROMPT_RETRIES):
            try:
                await self.herdr.prompt_agent(pane_id, task)
                if not await self._prompt_landed(pane_id, marker):
                    if await self._pane_hosts_settled_agent(pane_id):
                        await self.herdr.send_input(pane_id, task)
                return
            except HerdrError as e:
                if (e.code not in ("agent_not_ready", "agent_not_found")
                        or attempt >= self.PROMPT_RETRIES - 1):
                    raise
                if (attempt >= self.FALLBACK_AFTER
                        and (attempt - self.FALLBACK_AFTER) % 10 == 0
                        and await self._pane_hosts_settled_agent(pane_id)):
                    await self.herdr.send_input(pane_id, task)
                    return
                await asyncio.sleep(self.PROMPT_RETRY_DELAY)

    async def spawn_in_folder(self, path: str, label: str,
                              agent: str = "claude") -> dict:
        """Open a workspace directly in an existing folder (no worktree) and
        start a coding agent (claude by default). The agent edits the real
        working tree. The task is NOT delivered here — the caller hands it
        over with deliver_task() once the agent finishes booting. Only when
        agent.start itself fails is the workspace closed again; once the
        agent is running the workspace is never torn down (a slow boot is
        not a failed launch)."""
        ws = await self.herdr.call("workspace.create", cwd=path, label=label,
                                   focus=False)
        pane_id = _find_pane_id(ws)
        if not pane_id:
            return {"workspace": ws, "pane_id": None}
        try:
            name = await self.start_agent(pane_id, label, kind=agent)
        except HerdrError:
            ws_id = _find_workspace_id(ws)
            if ws_id:
                try:
                    await self.herdr.call("workspace.close",
                                          workspace_id=ws_id, force=True)
                except HerdrError:
                    pass  # surface the original launch error, not this one
            raise
        return {"workspace": ws, "pane_id": pane_id, "agent_name": name}

    async def spawn(self, repo_path: str, branch: str,
                    agent: str = "claude") -> dict:
        """Create a worktree workspace in repo_path and start a coding agent
        (claude by default). The task is delivered by the caller via
        deliver_task() once ready."""
        wt = await self.herdr.call("worktree.create", cwd=repo_path,
                                   branch=branch, focus=False)
        # worktree.create response shape is untested ground (needs a git
        # workspace); find the new workspace's pane defensively.
        pane_id = _find_pane_id(wt)
        name = None
        if pane_id:
            name = await self.start_agent(pane_id, branch, kind=agent)
        return {"worktree": wt, "pane_id": pane_id, "agent_name": name}


_AGENT_NAME_MAX = 32


def _sanitize_agent_name(label: str) -> str:
    """Fold an arbitrary label (folder or branch name) into a valid herdr
    agent name: ^[a-z][a-z0-9_-]{0,31}$. herdr rejects anything else with
    invalid_agent_name before even touching the pane."""
    name = re.sub(r"[^a-z0-9_-]+", "-", label.lower())
    name = re.sub(r"-+", "-", name).lstrip("0123456789-_").rstrip("-_")
    if not name:
        return "agent"
    return name[:_AGENT_NAME_MAX].rstrip("-_")


def _find_workspace_id(obj: Any) -> str | None:
    """Depth-first hunt for a workspace_id in a response of unknown shape."""
    if isinstance(obj, dict):
        if isinstance(obj.get("workspace_id"), str):
            return obj["workspace_id"]
        for v in obj.values():
            if (found := _find_workspace_id(v)) is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            if (found := _find_workspace_id(v)) is not None:
                return found
    return None


def _find_pane_id(obj: Any) -> str | None:
    """Depth-first hunt for a pane_id in a response of unknown shape."""
    if isinstance(obj, dict):
        if isinstance(obj.get("pane_id"), str):
            return obj["pane_id"]
        for v in obj.values():
            if (found := _find_pane_id(v)) is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            if (found := _find_pane_id(v)) is not None:
                return found
    return None
