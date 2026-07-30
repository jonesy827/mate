# mate

Call a phone number and talk to "Mate". Mate drives a fleet of coding
agents that run in [herdr](https://herdr.dev). You can spawn agents, give
them tasks, and hear their results. I built this tool because I wanted to
monitor and update my Claude Code sessions from the car. It is a working
proof of concept with basic security. It has an MIT license and comes
as-is.

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

The media and infrastructure services are in `../matebridge-infra`. Its
README gives the start procedure. No part of this project starts at boot.

## Confirmation rail

Voice transcription is not accurate. Because of this, one utterance never
causes an outward action. The tools `tell_agent`, `spawn_task`, and
`spawn_in_folder` stage the action and read it back. Delivery occurs only
after a spoken yes in a **new** turn. The code does this check, not the
LLM. TUI approvals that look destructive use the same rail. There is no
bypass tool. "Guardrails off" is a voice toggle that the code also
detects. It makes messages and spawns immediate. It never skips
destructive approvals.

## Security

The security is basic by design. This list gives the limits:

- Two barriers stop hostile callers. The first barrier is the caller
  allowlist (`MATE_ALLOWED_NUMBERS`). The worker enforces it fail-closed,
  and the LiveKit trunk enforces it again. The second barrier is a spoken
  four-word passphrase (`MATE_PASSPHRASE`), on by default. Callers can
  spoof caller ID. The passphrase covers that risk.
- The passphrase check runs in code on the raw transcript. Until the
  check passes, the LLM does not run and each acting tool refuses. Mate
  also speaks no fleet status before the check passes. After three failed
  attempts, Mate ends the call. An attempt longer than 15 seconds counts
  as a failed attempt. Thus one turn cannot contain many candidate
  phrases. There is no default passphrase. You select your own passphrase
  at setup. If three calls in a row end in a failed-passphrase hangup,
  the worker shuts down. A state file keeps the streak count, so the
  count survives across calls. The worker takes calls again after a
  restart. Thus an attacker who redials gets a maximum of 9 guesses.
- The rail catches bad transcription, not attackers. An attacker says yes
  to their own staged action.
- A caller on the allowlist drives agents with your full user
  permissions. The destructive-prompt regex (`safety.py`) is a heuristic,
  not a boundary.
- If you set `LLM_API_KEY`, the hosted LLM receives call transcripts and
  agent output. The default stack is fully local.

## Running

Before you start the worker:

1. Start the infrastructure services (see `../matebridge-infra`).
2. Start llama.cpp: `systemctl --user start llama-qwen-long`.
3. Start herdr: `tmux new-session -d -s herdr-host herdr`.

```sh
.venv/bin/python -m mate.agent console   # desk test: terminal mic/speaker
set -a && source .env && set +a
.venv/bin/python -m mate.agent dev       # real worker: registers with LiveKit
```

The worker does a preflight check of the LLM, STT, TTS, and herdr
services. If one of these services is unreachable, the worker refuses to
start. Obey these two rules:

- Do not restart the worker during a call. First, make sure that the
  output of `lk room list | grep mate-call-` is empty.
- Dev mode imports the code again for each call. Thus edits usually apply
  on the next call. If you must be sure, restart the worker between
  calls.

## Configuration (`.env`, gitignored, chmod 600)

Copy `.env.example` to `.env`. The example file documents each variable.

| var | purpose |
|---|---|
| `MATE_ALLOWED_NUMBERS` | **Required.** Comma-separated E.164 numbers that can call in. Mate ends other calls before it speaks. |
| `MATE_PASSPHRASE` | **Required** unless disabled. Four words that each phone caller must speak before Mate acts. There is no default. If the variable is not set, the worker prompts for one at startup. |
| `MATE_REQUIRE_PASSPHRASE` | Default `1`. Set `0` to run without the passphrase gate. |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit Cloud project credentials. |
| `LLM_URL` `STT_URL` `TTS_URL` | Overrides for the local endpoints (defaults `:8003` `:8001` `:8880`). |
| `LLM_MODEL` `STT_MODEL` `TTS_VOICE` | Model and voice overrides (default voice `af_heart`). |
| `LLM_API_KEY` | Default `local`. To use the OpenAI API, set a real key, `LLM_URL=https://api.openai.com/v1`, and `LLM_MODEL`. The key is usage-billed. A ChatGPT subscription has no API access. |
| `HERDR_SOCKET` | The herdr control socket (default `~/.config/herdr/herdr.sock`). |
| `MATE_SRC_ROOTS` | Colon-separated roots that `spawn_in_folder` searches (default `~/src`). |
| `MATE_CLAUDE_PROJECTS` | The Claude Code transcript directory (default `~/.claude/projects`). |

## Phone/SIP setup (one-time, as deployed)

1. **Telnyx**: Buy a DID. Create a SIP trunk that points to the SIP URI
   of your LiveKit Cloud project. Assign the DID to the trunk.
2. **LiveKit Cloud** (`lk` configured for the project):

   ```sh
   lk sip inbound create trunk.json     # numbers: ["+14052790756"]
   lk sip dispatch create dispatch.json # individual/caller → mate-call-_<caller>_<random>
   ```
3. Set `AllowedNumbers` on the trunk to match `MATE_ALLOWED_NUMBERS`.
   Then start the worker.

Note: I stopped the self-hosted LiveKit test because it crashed this host
twice. The infra README gives details.

## Tools

`src/mate/agent.py` defines the `Mate` agent:

- **Fleet**: `fleet_status`, `read_pane`, `agent_report` (reads real
  replies from the session transcript), `wait_for_agent`,
  `list_known_agents` / `forget_agent`.
- **Acting**: `tell_agent`, `spawn_task` (new worktree + branch),
  `spawn_in_folder`, `send_answer` (TUI prompts).
- **Rail**: `send_staged` / `discard_staged`.

A spawn returns immediately. A background deliverer sends the task after
the agent boots (maximum 120 s). `watch_fleet` announces deliveries and
completions during the call. It speaks a sanitized two-sentence summary.
A `Delegations` clock blocks "finished" announcements for panes that
never showed activity, because herdr can still type in them. These
panes get one Enter nudge instead.

## herdr notes (`herdr_client.py`)

This client is tested against herdr 0.7.5 (protocol 17). mate never
patches herdr. These items are client-side workarounds, and a test pins
each one. A protocol mismatch causes a log warning, not a refusal to
start.

- **Sanitized names**: The client folds agent names to
  `^[a-z][a-z0-9_-]{0,31}$`.
- **A slow boot is not a failed launch**: Prompt delivery retries
  `agent_not_ready` for a maximum of 120 s. The `timeout_ms` value
  extends the 30 s launch deadline of herdr.
- **Stuck-launch fallback** (0.7.5 bug): A launch can stay in
  `launch_pending` forever while the agent idles. After approximately
  20 s of refusals, the client reads `agent.list`. If a settled agent
  owns the pane, the client types the message directly into the pane.
  The client never types into a bare shell.
- **Dropped-prompt fallback** (0.7.5 bug): On worktree panes,
  `agent.prompt` reports success but types nothing. A landed prompt
  increases `state_change_seq`. If the value stays frozen, the client
  uses the same type-in fallback with the same guard.
- **Enter nudge**: The paste guard of Claude Code sometimes eats the
  Enter from herdr. Thus the client sends one bare Enter approximately
  2 s after each delivery.

### Other harnesses

herdr hosts many agent kinds (`claude`, `codex`, `gemini`, and more).
Spawns, messages, TUI answers, and status watching work for each kind.
`spawn_task` and `spawn_in_folder` accept an `agent` kind. Transcript
readback (`agent_report` and spoken summaries) needs a per-harness
adapter in the `ADAPTERS` table in `transcripts.py`. Only a claude
adapter exists now. Other kinds get the reply "use read_pane instead". An
adapter is a single function: it receives a cwd and a session ID, and it
returns the last assistant replies. The destructive-prompt regex in
`safety.py` matches the approval wording of Claude Code only.

## Development

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
```

CI runs both commands on each push. Each bug found in live use gets a
pinning test. The test suite needs no network, no herdr, and no GPU.

| module | role |
|---|---|
| `agent.py` | The Mate voice agent: tools, rail, watcher, entrypoint. |
| `herdr_client.py` | Async herdr socket client, plus spawn and delivery logic. |
| `folders.py` | Folder-name to path resolution for `spawn_in_folder`. |
| `transcripts.py` | Per-harness transcript adapters (claude today). |
| `safety.py` | Approval and veto detection for the rail. |
| `allowlist.py` | Caller allowlist: normalization plus fail-closed matching. |
| `passphrase.py` | Spoken-passphrase gate: matching plus launch requirement. |
| `scripts/smoke_llm.py` | Quick sanity test of the local LLM. |

## License

MIT — see `LICENSE`.
