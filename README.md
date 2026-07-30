# mate

Call a phone number, talk to "Mate", and it drives a fleet of coding agents
running in [herdr](https://herdr.dev) — spawn agents, hand them tasks, hear
their results. I built it because I wanted to check on and update my Claude
Code sessions from the car. It's a working proof of concept with basic
security, MIT-licensed, published as-is.

```
your phone
   │  PSTN
   ▼
Telnyx DID (+1 405 279 0756)
   │  SIP trunk
   ▼
LiveKit Cloud SIP  (inbound trunk → dispatch rule → room mate-call-<caller>-<rand>)
   │  WebRTC
   ▼
mate worker (this repo, runs on the workstation)
   │  STT ⇄ LLM ⇄ TTS, local by default (LLM can also be the OpenAI API):
   │    :8001 faster-whisper (STT)   :8003 llama.cpp qwen (LLM)   :8880 kokoro (TTS)
   ▼
herdr unix socket (~/.config/herdr/herdr.sock)
   └─ workspaces/panes hosting claude-code agents
```

Media/infra services live in `../matebridge-infra` — see its README for
bring-up. Nothing in this project starts at boot.

## Confirmation rail

Voice transcription is lossy, so nothing outward happens on one utterance.
`tell_agent` / `spawn_task` / `spawn_in_folder` stage the action and read it
back; delivery needs a spoken yes in a **new** turn, verified in code — not
by the LLM. Destructive-looking TUI approvals take the same rail, with no
bypass tool. "Guardrails off" (voice toggle, also detected in code) makes
messaging and spawns immediate but never skips destructive approvals.

## Security

Basic, deliberately. Know what you're deploying:

- Two barriers against hostile callers: the caller allowlist
  (`MATE_ALLOWED_NUMBERS`, enforced fail-closed in the worker and again on
  the LiveKit trunk) and a spoken four-word passphrase (`MATE_PASSPHRASE`,
  on by default — caller ID can be spoofed, the passphrase covers that).
  The passphrase check runs in code on the raw transcript; until it
  passes, the LLM never runs, every acting tool refuses, and no fleet
  status is spoken or announced. Three misses hangs up; an attempt spoken
  for longer than 15 seconds counts as a miss, so one turn can't be
  stuffed with candidate phrases. There is no default phrase — you pick
  your own at setup. Three calls in a row ending in a failed-passphrase
  hangup shuts the whole worker down (the streak survives across calls in
  a state file; restart the worker to take calls again) — so redialling
  buys an attacker 9 guesses total, not unlimited.
- The rail catches bad transcription, not attackers; an attacker says yes
  to their own staged action.
- An allowed caller drives agents with your full user permissions. The
  destructive-prompt regex (`safety.py`) is a heuristic, not a boundary.
- A hosted voice brain (`LLM_API_KEY` set) sends call transcripts and agent
  output to the provider; the default stack is all local.

## Running

Prereqs: infra services up (see `../matebridge-infra`), llama.cpp started
(`systemctl --user start llama-qwen-long`), herdr running
(`tmux new-session -d -s herdr-host herdr`).

```sh
.venv/bin/python -m mate.agent console   # desk test: terminal mic/speaker
set -a && source .env && set +a
.venv/bin/python -m mate.agent dev       # real worker: registers with LiveKit
```

The worker preflights llm/stt/tts/herdr and refuses to start if any are
unreachable. Two rules learned the hard way:

- Never restart the worker during a call — `lk room list | grep mate-call-`
  must be empty first.
- Dev-mode re-imports code per call, so edits usually go live on the next
  call; restart between calls when you need certainty.

## Configuration (`.env`, gitignored, chmod 600)

Copy `.env.example` to `.env` — it documents every variable.

| var | purpose |
|---|---|
| `MATE_ALLOWED_NUMBERS` | **required**: comma-separated E.164 numbers allowed to call in. Anyone else is hung up on before Mate says a word. |
| `MATE_PASSPHRASE` | **required** (unless disabled): four words every phone caller must speak before Mate acts. No default — the worker prompts for one at startup if unset. |
| `MATE_REQUIRE_PASSPHRASE` | default `1`. Set `0` to run without the passphrase gate. |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit Cloud project |
| `LLM_URL` `STT_URL` `TTS_URL` | override local endpoints (defaults `:8003` `:8001` `:8880`) |
| `LLM_MODEL` `STT_MODEL` `TTS_VOICE` | model/voice overrides (default voice `af_heart`) |
| `LLM_API_KEY` | default `local`. Set a real key + `LLM_URL=https://api.openai.com/v1` + `LLM_MODEL` to use the OpenAI API (usage-billed key; a ChatGPT subscription has no API access) |
| `HERDR_SOCKET` | herdr control socket (default `~/.config/herdr/herdr.sock`) |
| `MATE_SRC_ROOTS` | colon-separated roots `spawn_in_folder` searches (default `~/src`) |
| `MATE_CLAUDE_PROJECTS` | Claude Code transcript dir (default `~/.claude/projects`) |

## Phone/SIP setup (one-time, as deployed)

1. **Telnyx**: buy a DID, create a SIP trunk pointed at your LiveKit Cloud
   project's SIP URI, assign the DID.
2. **LiveKit Cloud** (`lk` configured for the project):

   ```sh
   lk sip inbound create trunk.json     # numbers: ["+14052790756"]
   lk sip dispatch create dispatch.json # individual/caller → mate-call-_<caller>_<random>
   ```
3. Set `AllowedNumbers` on the trunk to match `MATE_ALLOWED_NUMBERS`, then
   run the worker.

(Self-hosted LiveKit was abandoned — it hard-crashed this host twice; see
the infra README.)

## Tools

`src/mate/agent.py` defines the `Mate` agent:

- **Fleet**: `fleet_status`, `read_pane`, `agent_report` (reads real replies
  from the session transcript), `wait_for_agent`, `list_known_agents` /
  `forget_agent`.
- **Acting**: `tell_agent`, `spawn_task` (new worktree + branch),
  `spawn_in_folder`, `send_answer` (TUI prompts).
- **Rail**: `send_staged` / `discard_staged`.

Spawns return immediately; a background deliverer hands the task over once
the agent boots (up to 120 s). `watch_fleet` announces deliveries and
finishes during the call, speaking a sanitized two-sentence summary. A
`Delegations` clock stops "finished" announcements for panes that were never
seen working (herdr may still be typing); those get one Enter nudge instead.

## herdr notes (`herdr_client.py`)

Tested against herdr 0.7.5 (protocol 17); mate never patches herdr — these
are client-side workarounds, each pinned by a test. A protocol mismatch is
logged as a warning, not a refusal to start.

- **Names sanitized**: agent names folded to `^[a-z][a-z0-9_-]{0,31}$`.
- **Slow boot ≠ failed launch**: prompt delivery retries `agent_not_ready`
  for up to 120 s; `timeout_ms` stretches herdr's 30 s launch deadline.
- **Stuck-launch fallback** (0.7.5 bug): a launch can stay `launch_pending`
  forever while the agent idles. After ~20 s of refusals, if `agent.list`
  proves a settled agent owns the pane, the message is typed in directly —
  never into a bare shell.
- **Dropped-prompt fallback** (0.7.5 bug): on worktree panes `agent.prompt`
  claims success but types nothing. A landed prompt advances
  `state_change_seq`; if it stays frozen, type-in fallback, same guard.
- **Enter nudge**: Claude Code's paste guard sometimes eats herdr's Enter;
  every delivery is followed ~2 s later by one bare Enter.

### Other harnesses

herdr hosts many agent kinds (`claude`, `codex`, `gemini`, …). Spawning,
messaging, TUI answers, and status watching work for any kind (`spawn_task`
/ `spawn_in_folder` take an `agent` kind). Transcript readback
(`agent_report`, spoken summaries) needs a per-harness adapter in
`transcripts.py` `ADAPTERS` — currently claude only; other kinds get "use
read_pane instead". Adding one is a single function: given cwd + session id,
return the last assistant replies. Note `safety.py`'s destructive-prompt
regex is tuned to Claude Code's approval wording.

## Development

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
```

CI runs both on every push. Every live-discovered bug gets a pinning test;
the suite needs no network, herdr, or GPU.

| module | role |
|---|---|
| `agent.py` | the Mate voice agent: tools, rail, watcher, entrypoint |
| `herdr_client.py` | async herdr socket client + spawn/delivery logic |
| `folders.py` | folder-name → path resolution for `spawn_in_folder` |
| `transcripts.py` | per-harness transcript adapters (claude today) |
| `safety.py` | approval / veto detection for the rail |
| `allowlist.py` | caller allowlist: normalization + fail-closed matching |
| `passphrase.py` | spoken-passphrase gate: matching + launch requirement |
| `scripts/smoke_llm.py` | quick local-LLM sanity check |

## License

MIT — see `LICENSE`.
