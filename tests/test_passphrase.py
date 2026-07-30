"""Spoken-passphrase gate: matching, the launch requirement, and the
code-level lock on the Mate agent (locked turns never reach the LLM)."""

import pytest
from livekit.agents import StopResponse

from mate.agent import LOCKED_MSG, Mate
from mate.folders import KnownAgents
from mate.passphrase import (
    ENABLE_VAR,
    PHRASE_VAR,
    ensure_launch_phrase,
    passphrase_required,
    phrase_heard,
)

PHRASE = "correct horse battery staple"


# --- matching --------------------------------------------------------------

def test_phrase_heard_exact_and_stt_variants():
    assert phrase_heard(PHRASE, PHRASE)
    assert phrase_heard("Correct Horse, battery-staple!", PHRASE)
    assert phrase_heard("um, correct horse battery staple please", PHRASE)
    assert phrase_heard("correcthorsebatterystaple", PHRASE)


def test_phrase_heard_rejects_partial_reordered_and_empty():
    assert not phrase_heard("correct horse battery", PHRASE)
    assert not phrase_heard("staple battery horse correct", PHRASE)
    assert not phrase_heard("correct horse staple battery", PHRASE)
    assert not phrase_heard("", PHRASE)
    # an unset phrase must never unlock, whatever was said
    assert not phrase_heard("anything at all", "")


# --- flag + launch gate ----------------------------------------------------

def test_required_defaults_on_and_flag_disables():
    assert passphrase_required({}) is True
    assert passphrase_required({ENABLE_VAR: "1"}) is True
    for off in ("0", "false", "no", "OFF"):
        assert passphrase_required({ENABLE_VAR: off}) is False


def test_launch_gate_passes_console_disabled_and_valid(monkeypatch):
    ensure_launch_phrase(["prog", "console"], {})  # no SIP path -> exempt
    ensure_launch_phrase(["prog", "dev"], {ENABLE_VAR: "0"})
    env = {PHRASE_VAR: PHRASE}
    ensure_launch_phrase(["prog", "dev"], env)
    assert env[PHRASE_VAR] == PHRASE


def test_launch_gate_refuses_without_phrase_non_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="not set"):
        ensure_launch_phrase(["prog", "dev"], {})
    with pytest.raises(SystemExit, match="exactly 4 words"):
        ensure_launch_phrase(["prog", "start"], {PHRASE_VAR: "two words"})


def test_launch_gate_prompts_on_tty_until_four_words(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["too short", PHRASE])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    env = {}
    ensure_launch_phrase(["prog", "dev"], env)
    assert env[PHRASE_VAR] == PHRASE


# --- the lock on Mate ------------------------------------------------------

class Msg:
    def __init__(self, text):
        self.text_content = text
        self.content = [text]


def make_locked_mate(tmp_path):
    mate = Mate(herdr=None, known=KnownAgents(tmp_path / "known.json"),
                roots=[])
    spoken = []

    async def speak(text):
        spoken.append(text)
        return True

    hung_up = []

    async def hangup():
        hung_up.append(True)

    mate._speak_confirmation = speak
    mate.lock(PHRASE, hangup)
    return mate, spoken, hung_up


@pytest.mark.asyncio
async def test_correct_phrase_unlocks_and_greets(tmp_path):
    mate, spoken, hung_up = make_locked_mate(tmp_path)
    with pytest.raises(StopResponse):
        await mate.on_user_turn_completed(None, Msg("Correct horse, battery staple"))
    assert mate.locked is False
    assert hung_up == []
    assert "What do you need?" in spoken[0]


@pytest.mark.asyncio
async def test_wrong_phrase_retries_then_hangs_up(tmp_path):
    mate, spoken, hung_up = make_locked_mate(tmp_path)
    for _ in range(2):
        with pytest.raises(StopResponse):
            await mate.on_user_turn_completed(None, Msg("open the pod bay doors"))
        assert mate.locked is True
        assert hung_up == []
    with pytest.raises(StopResponse):
        await mate.on_user_turn_completed(None, Msg("guardrails off"))
    assert hung_up == [True]
    assert mate.locked is True
    # the toggle phrase in a locked turn must NOT reach the rail toggle
    assert mate.rail_enabled is True
    assert "Goodbye" in spoken[-1]


@pytest.mark.asyncio
async def test_locked_tools_refuse(tmp_path):
    mate, _, _ = make_locked_mate(tmp_path)
    assert await mate.tell_agent(None, pane_id="w1:p1", text="hi") == LOCKED_MSG
    assert await mate.spawn_task(None, repo_path="/x", branch="b",
                                 task="t") == LOCKED_MSG
    assert await mate.spawn_in_folder(None, folder_name="mate",
                                      task="t") == LOCKED_MSG
    assert await mate.send_answer(None, pane_id="w1:p1",
                                  keys=["Enter"]) == LOCKED_MSG
    assert await mate.send_staged(None) == LOCKED_MSG
    assert mate._staged is None


@pytest.mark.asyncio
async def test_unlocked_turns_pass_through_to_toggle(tmp_path):
    mate, _, _ = make_locked_mate(tmp_path)
    with pytest.raises(StopResponse):
        await mate.on_user_turn_completed(None, Msg(PHRASE))
    msg = Msg("guardrails off")
    await mate.on_user_turn_completed(None, msg)  # no StopResponse now
    assert mate.rail_enabled is False
    assert len(msg.content) == 2  # toggle note injected
