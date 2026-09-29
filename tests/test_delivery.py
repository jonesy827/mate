"""The herdr quirk layer: spawn retries, task delivery through a slow boot,
and the two dropped-prompt fallbacks.

Every scenario here is a bug or timing wart seen against a live herdr; if an
upgrade fixes one, its test and its workaround go together. (The guarded
Enter nudge was removed with herdr 0.8.0's #1878.)
"""

import pytest

from mate.delivery import (
    Delivery,
    _find_pane_id,
    _sanitize_agent_name,
)
from mate.herdr_client import HerdrClient, HerdrError

pytestmark = pytest.mark.asyncio


class ScriptedClient(HerdrClient):
    """Overrides call() with a per-method script of results; exception
    entries are raised (HerdrError for a protocol error, TimeoutError or
    OSError for the transport failures real call() does not wrap)."""

    def __init__(self, script):
        super().__init__("/nonexistent")
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    async def call(self, method, **params):
        self.calls.append((method, params))
        queue = self.script.get(method, [])
        result = queue.pop(0) if queue else {}
        if isinstance(result, Exception):
            raise result
        return result


class FastDelivery(Delivery):
    """Retry and verification delays zeroed so tests run instantly."""

    START_RETRY_DELAY = 0
    PROMPT_RETRY_DELAY = 0
    PROMPT_VERIFY_WAIT = 0


def scripted(script):
    """(delivery, client) over a scripted herdr; assertions read the
    client's recorded calls."""
    client = ScriptedClient(script)
    return FastDelivery(client), client


def _busy():
    return HerdrError("agent.start", "agent_pane_busy",
                      "agent target pane w9:p1 is not an available shell")


async def test_spawn_in_folder_retries_until_shell_ready():
    # the songhaus bug, part 1: agent.start 100ms after workspace.create
    # fails because the fresh pane has no available shell yet
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [_busy(), _busy(), {}],
    })
    result = await d.spawn_in_folder("/src/songhaus", "songhaus")
    assert result["pane_id"] == "w9:p1"
    assert result["agent_name"] == "songhaus"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert len(starts) == 3
    assert starts[0]["kind"] == "claude"
    # herdr's managed-launch deadline stretched past its 30s default so a
    # slow claude boot isn't abandoned server-side either
    assert starts[0]["timeout_ms"] == Delivery.LAUNCH_TIMEOUT_MS
    assert not any(m == "workspace.close" for m, _ in c.calls)
    # spawn never prompts: the task is delivered separately once ready
    assert not any(m == "agent.prompt" for m, _ in c.calls)


async def test_spawn_in_folder_unique_name_on_duplicate():
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [HerdrError("agent.start", "duplicate_agent_name",
                                   "taken"), {}],
    })
    result = await d.spawn_in_folder("/src/songhaus", "songhaus")
    assert result["agent_name"] == "songhaus-2"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert [s["name"] for s in starts] == ["songhaus", "songhaus-2"]


def test_sanitize_agent_name():
    # herdr requires ^[a-z][a-z0-9_-]{0,31}$ (invalid_agent_name otherwise)
    assert _sanitize_agent_name("RackCoach") == "rackcoach"
    assert _sanitize_agent_name("My Repo!") == "my-repo"
    assert _sanitize_agent_name("Fix/The Thing") == "fix-the-thing"
    assert _sanitize_agent_name("123abc") == "abc"  # must start with a letter
    assert _sanitize_agent_name("snake_case_ok") == "snake_case_ok"
    assert _sanitize_agent_name("!!!") == "agent"
    assert _sanitize_agent_name("") == "agent"
    assert _sanitize_agent_name("x" * 50) == "x" * 32


async def test_find_pane_id():
    assert _find_pane_id({"a": [{"pane": {"pane_id": "w2:p9"}}]}) == "w2:p9"
    assert _find_pane_id({"nothing": [1, "x", None]}) is None


async def test_spawn_in_folder_sanitizes_agent_name():
    # the RackCoach bug: the folder label went to agent.start verbatim and
    # herdr rejected the capital letters instantly, three calls in a row
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [{}],
    })
    result = await d.spawn_in_folder("/src/RackCoach", "RackCoach")
    assert result["agent_name"] == "rackcoach"
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["name"] == "rackcoach"
    assert not any(m == "workspace.close" for m, _ in c.calls)


async def test_duplicate_suffix_stays_within_name_limit():
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [HerdrError("agent.start", "duplicate_agent_name",
                                   "taken"), {}],
    })
    result = await d.spawn_in_folder("/src/x", "X" * 40)
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["name"] == "x" * 32
    assert starts[1]["name"] == "x" * 30 + "-2"
    assert result["agent_name"] == starts[1]["name"]


async def test_spawn_in_folder_agent_kind_passes_through():
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [{}],
    })
    await d.spawn_in_folder("/src/x", "x", agent="pi")
    starts = [p for m, p in c.calls if m == "agent.start"]
    assert starts[0]["kind"] == "pi"


async def test_spawn_in_folder_closes_workspace_on_start_failure():
    fatal = HerdrError("agent.start", "invalid_agent_name", "bad")
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [fatal],
    })
    with pytest.raises(HerdrError) as ei:
        await d.spawn_in_folder("/src/x", "x")
    assert ei.value.code == "invalid_agent_name"
    closes = [p for m, p in c.calls if m == "workspace.close"]
    assert closes == [{"workspace_id": "w9", "force": True}]


async def test_spawn_in_folder_gives_up_after_persistent_busy():
    d, c = scripted({
        "workspace.create": [{"workspace_id": "w9", "pane_id": "w9:p1"}],
        "agent.start": [_busy() for _ in range(Delivery.START_RETRIES)],
    })
    with pytest.raises(HerdrError) as ei:
        await d.spawn_in_folder("/src/x", "x")
    assert ei.value.code == "agent_pane_busy"
    closes = [m for m, _ in c.calls if m == "workspace.close"]
    assert closes == ["workspace.close"]


async def test_deliver_task_retries_through_boot():
    # the songhaus bug, part 2: claude's first boot in a folder keeps
    # agent.prompt at agent_not_ready for a long time — delivery must wait
    # it out instead of declaring the launch failed
    not_ready = HerdrError("agent.prompt", "agent_not_ready", "pending")
    d, c = scripted({
        "agent.prompt": [not_ready] * 30 + [{}],
    })
    await d.deliver_task("w9:p1", "do it")
    assert sum(1 for m, _ in c.calls if m == "agent.prompt") == 31
    # herdr owns prompt submission (0.8.0, #1878): mate sends no Enter
    assert not any(m == "pane.send_keys" for m, _ in c.calls)


async def test_deliver_task_falls_back_to_pane_input_when_launch_stuck():
    # herdr 0.7.5 bug seen live: launch_pending never clears even though
    # the agent is detected idle — agent.prompt refuses forever. Once the
    # snapshot proves an idle agent owns the pane, type into it directly.
    not_ready = HerdrError("agent.prompt", "agent_not_ready",
                           "agent w9:p1 is not an active named agent")
    # agent.list entries: delivery samples the agent's state once up front
    # (dropped-prompt detection), then the fallback guard reads it
    idle = {"agents": [{"pane_id": "w9:p1", "agent": "claude",
                        "agent_status": "idle"}]}
    d, c = scripted({
        "agent.prompt": [not_ready] * (Delivery.FALLBACK_AFTER + 1),
        "agent.list": [idle, idle],
    })
    await d.deliver_task("w9:p1", "do it")
    sends = [p for m, p in c.calls if m == "pane.send_input"]
    assert sends == [{"pane_id": "w9:p1", "text": "do it",
                      "keys": ["Enter"]}]
    assert not any(m == "pane.send_keys" for m, _ in c.calls)


async def test_deliver_task_never_types_into_a_bare_shell():
    # fallback must not fire when no settled agent owns the pane — typing
    # a task into a shell would execute it as a command
    not_ready = HerdrError("agent.prompt", "agent_not_ready", "pending")
    d, c = scripted({
        "agent.prompt": [not_ready] * Delivery.PROMPT_RETRIES,
        "agent.list": [{"agents": []}
                       for _ in range(Delivery.PROMPT_RETRIES)],
    })
    with pytest.raises(HerdrError):
        await d.deliver_task("w9:p1", "rm -rf importantdir")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


def _listed_agent(status="idle", seq=381):
    return {"agents": [{"pane_id": "w9:p1", "agent": "claude",
                        "agent_status": status, "state_change_seq": seq}]}


async def test_deliver_task_types_in_when_prompt_silently_dropped():
    # herdr 0.7.5 bug seen live (worktree-workspace pane): agent.prompt
    # answers agent_prompted but never types anything — the agent's
    # state_change_seq stays frozen. Delivery must notice and type directly.
    d, c = scripted({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent() for _ in range(10)],
    })
    await d.deliver_task("w9:p1", "do it")
    sends = [p for m, p in c.calls if m == "pane.send_input"]
    assert sends == [{"pane_id": "w9:p1", "text": "do it",
                      "keys": ["Enter"]}]


async def test_deliver_task_no_fallback_when_prompt_lands():
    # state_change_seq advanced after the accepted prompt -> it landed;
    # typing as well would deliver the task twice. The verification reads
    # sit between the two samples.
    d, c = scripted({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(seq=381), _listed_agent(seq=381),
                       _listed_agent(seq=382)],
    })
    await d.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


async def test_dropped_prompt_fallback_needs_settled_agent():
    # frozen seq but the agent shows working: it is busy on something else,
    # so the settled-agent guard must veto typing into its terminal
    d, c = scripted({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(status="working") for _ in range(10)],
    })
    await d.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


async def test_dropped_prompt_fallback_skips_a_blocked_agent():
    # blocked = sitting on a permission/trust dialog. pane.send_input ends
    # with Enter, which would ACCEPT the highlighted option, and the typed
    # task would be swallowed by the dialog anyway.
    d, c = scripted({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(status="blocked") for _ in range(10)],
    })
    await d.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)
    assert not any(m == "pane.send_keys" for m, _ in c.calls)


async def test_stuck_launch_fallback_skips_a_blocked_agent():
    # same guard on the other fallback: a first-boot trust dialog keeps
    # agent.prompt at agent_not_ready for minutes while herdr reports the
    # agent blocked — typing into it must wait for idle
    not_ready = HerdrError("agent.prompt", "agent_not_ready", "pending")
    d, c = scripted({
        "agent.prompt": [not_ready] * Delivery.PROMPT_RETRIES,
        "agent.list": [_listed_agent(status="blocked")
                       for _ in range(Delivery.PROMPT_RETRIES)],
    })
    with pytest.raises(HerdrError):
        await d.deliver_task("w9:p1", "do it")
    assert not any(m == "pane.send_input" for m, _ in c.calls)


async def test_deliver_task_survives_a_transport_failure_in_verification():
    # herdr slow or restarting mid-delivery: call() lets asyncio.wait_for's
    # TimeoutError through unwrapped, so it is NOT a HerdrError. The prompt
    # already landed, and letting that escape would make the caller speak
    # "the agent never took the task" and drop the pane from the watcher.
    # Unreadable verification state counts as landed — the fallback types
    # into a terminal, so it must never fire on doubt alone.
    d, c = scripted({
        "agent.prompt": [{}],
        "agent.list": [_listed_agent(),
                       TimeoutError("herdr did not answer in 15s")],
    })
    await d.deliver_task("w9:p1", "do it")  # no raise
    assert not any(m == "pane.send_input" for m, _ in c.calls)
    assert not any(m == "pane.send_keys" for m, _ in c.calls)


async def test_deliver_task_raises_when_agent_never_ready():
    d, c = scripted({
        "agent.prompt": [HerdrError("agent.prompt", "agent_not_ready", "x")
                         for _ in range(Delivery.PROMPT_RETRIES)],
    })
    with pytest.raises(HerdrError) as ei:
        await d.deliver_task("w9:p1", "t")
    assert ei.value.code == "agent_not_ready"
    # delivery failure must never tear anything down
    assert not any(m == "workspace.close" for m, _ in c.calls)
