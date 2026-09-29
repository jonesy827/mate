# mate

Call a phone number and talk to "Mate". Mate drives a fleet of coding
agents that run in [herdr](https://herdr.dev). You can spawn agents, give
them tasks, and hear their results. I built this tool because I wanted to
monitor and update my Claude Code sessions from the car. It is a working
proof of concept with basic security. It has an MIT license and comes
as-is.

Phone provisioning status (2026-09-27): the old Telnyx number
`+12025550102` is deleted and must not be used. A replacement,
`+12025550100`, is active in LiveKit directly. The `mate-calls` dispatch
rule matches this number through its `numbers` field. A real inbound call
reached Mate, passed authentication, and delivered speech to transcription.
The CLI number-assignment endpoint returned an error and its assignment display
remains empty; routing was configured through the SIP dispatch-rule API instead.
The diagram below describes the original Telnyx deployment.

```
your phone
   │  PSTN
   ▼
Telnyx DID (+12025550102)
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

The rules below describe the default `local` voice mode. The optional GPT-Live
mode uses a contextual Luna approval classifier, described below.

Voice transcription is not accurate. Because of this, one utterance never
causes an outward action. The tools `tell_agent`, `spawn_task`, and
`spawn_in_folder` stage the action and read it back. Delivery occurs only
after a spoken yes in a **new** turn. The code does this check, not the
LLM. A stage stays valid for three user turns. After that the code drops
it, because a late yes answers a different question. The model must then
stage the action and read it back again. TUI approvals that look
destructive use the same rail. These approvals get one more check: the
code reads the pane again immediately before it sends the keys. If that
read fails, or if the destructive action is no longer on screen, the code
drops the keys. This check is the same regex, not a comparison with the
text that the user heard. Therefore the keys go only into a screen that
still shows a destructive action. If the agent moves to a *different*
destructive prompt inside the three-turn window, the keys go into that
prompt. There is no bypass tool. "Guardrails off" is a voice toggle that
the code also detects. It makes messages and spawns immediate. It never
skips destructive approvals.

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
- The same regex guards the automatic Enter nudge. In a Claude Code
  dialog, Enter accepts the selected option. Therefore mate reads the
  pane first and skips the nudge if the agent is blocked, if the screen
  shows a destructive action, or if the check fails.
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

## GPT-Live voice mode

Set `MATE_VOICE_MODE=live` and `OPENAI_API_KEY` in your environment to use
GPT-Live for conversation with a managed Responses backend selecting Mate's
tools. The default `local` mode remains available. This integration uses the
Live WebSocket protocol directly; the pinned LiveKit OpenAI plugin's Realtime
implementation uses a different protocol.

```text
Phone → LiveKit → GPT-Live (listening and speaking)
                              ↕ delegation
                         Responses backend
                              ↕ function requests/results
                         Mate Python tools → Herdr

User approves readback → GPT-Live delegates → backend calls send_staged
```

`MATE_BACKEND_MODEL` selects the delegated model (default `gpt-6-luna`).
`MATE_BACKEND_REASONING` defaults to `none`, the lowest supported effort.
`MATE_LIVE_MODEL` defaults to `gpt-live-1`; `MATE_LIVE_VOICE` defaults to
`marin`. An API key with access to both models is required. The startup model
endpoint check verifies credentials; the Live session handshake verifies its
configuration. A ChatGPT subscription does not supply API usage.

Keep Whisper and Kokoro running for this mode; Qwen is no longer needed:

- Authentication stays local. No Live connection opens until the caller
  passes the existing passphrase gate. The passphrase is not sent to the
  delegated model.
- Ordinary conversation and fleet announcements use GPT-Live. Authentication
  and exact action readbacks use local Kokoro so code knows the wording and
  when playback finished. These short clips cannot be interrupted.
- Whisper receives audio only while the passphrase gate is locked. After login,
  GPT-Live handles conversation and interprets confirmation with its backend.
  Transcript fragments are logged, but do not gate tools or delivery.
- Sending is two steps: stage and read back the exact request, then call
  `send_staged` after the user approves. Corrections restage the message and
  trigger another readback; refusals discard it. There is no separate Luna
  approval classifier, silence timer, phrase match, or transcript-turn expiry.
  Luna remains the tool-selecting backend. Guardrails stay on in Live mode.
- Code requires a completed readback, an authenticated live connection, and a
  pending action. It consumes that action before attempting delivery, so repeats
  cannot resend it. Interpreting the user's approval is now the voice/backend
  models' responsibility, rather than an independent application check.
- Destructive prompt approvals still re-read the pane and require the same
  screen before sending keys. The local voice mode keeps its existing gate.
- `delivery.attempt` and `delivery.result` record actual execution. A verbal
  claim of success alone is not a delivery receipt. Unknown outcomes must not
  be automatically retried.

The optional approval classifier module and its `MATE_APPROVAL_*` configuration
remain available for standalone experiments; Live calls no longer use or probe it.

After configuring `.env` and starting the existing local services:

```sh
set -a && source .env && set +a
MATE_VOICE_MODE=live .venv/bin/python -m mate.agent console
# Once the desk test passes, register for phone calls:
MATE_VOICE_MODE=live .venv/bin/python -m mate.agent dev
```

The desk test should cover interruptions, an agent status request, staging a
message, “looks good,” “yes, but wait,” changing the task during confirmation,
and a disconnected call. Verify what actually arrived in Herdr. Mocked tests
exercise the transport and staged delivery without GPU inference or API calls;
real voice behavior and interpretation of approval require this live test.

Each call writes a bounded trace to `.logs/<job-id>.log` (gitignored, private
directory). The trace records authenticated user turns, Live speech fragments,
delegation/response IDs, tool calls and timings, result sizes and outcomes,
fleet agent IDs, report coverage, delivery attempts/results, and audio byte totals.
Tool results containing terminal or transcript content are not copied in full.
Configured credentials and the passphrase are redacted. SDK transcript debug
messages are suppressed in Mate; authenticated turns are recorded instead.
The separate Whisper service has its own logging configuration.

Use `ls -t .logs` to find the latest call, then `tail -f .logs/<job-id>.log`.
For an every-agent update, compare `fleet.coverage` with the `agent.report`
events and check `live.speech` for what the voice model generated. Speech
fragments show generated text, not proof that the caller heard all playback.

Run `.venv/bin/python scripts/smoke_live.py` for a paid synthetic API test of
approvals, backend tool selection, and the Live handshake. It loads `.env`,
executes no Herdr actions, and starts no GPU services.

The 2026-09-27 API smoke run passed 20 approval cases, four backend selection
cases, and a Live handshake using the actual tool schemas. Both Luna roles
used `none` reasoning. Approval latency was 0.75 seconds median and 2.37 seconds
maximum. This small synthetic evaluation supports the initial setting; it does
not establish real-call accuracy or test microphone playback and interruptions.

Audio and delegated fleet/tool data leave the workstation in Live mode.
OpenAI bills voice session duration and backend usage separately. A dropped
connection ends the voice session without automatic reconnection or tool
replay, since an in-flight action's outcome may be unknown. Completed or
already-delivered agent work continues independently.

Protocol references: [Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live),
[delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation),
and [session transcripts](https://developers.openai.com/api/docs/guides/live-conversations).

## Phone/SIP setup (one-time, as deployed)

1. **Telnyx**: Buy a DID. Create a SIP trunk that points to the SIP URI
   of your LiveKit Cloud project. Assign the DID to the trunk.
2. **LiveKit Cloud** (`lk` configured for the project):

   ```sh
   lk sip inbound create trunk.json     # numbers: ["+12025550102"]
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
the agent boots (maximum 120 s). `FleetWatcher` (in `watcher.py`)
announces blocks and completions during the call. It speaks a sanitized
two-sentence summary. A `Delegations` clock blocks "finished"
announcements for panes that never showed activity, because herdr can
still type in them. These panes get one guarded Enter nudge instead.

## herdr notes (`delivery.py`)

`herdr_client.py` speaks the protocol only. Every workaround lives in
`delivery.py`, thus a herdr upgrade has one file to audit. The
workarounds are tested against herdr 0.7.5 (protocol 17). mate never
patches herdr. A test pins each item, and a protocol mismatch causes a
log warning, not a refusal to start.

- **Sanitized names**: The client folds agent names to
  `^[a-z][a-z0-9_-]{0,31}$`.
- **A slow boot is not a failed launch**: Prompt delivery retries
  `agent_not_ready` for a maximum of 120 s. The `timeout_ms` value
  extends the 30 s launch deadline of herdr.
- **Stuck-launch fallback** (0.7.5 bug): A launch can stay in
  `launch_pending` forever while the agent idles. After approximately
  20 s of refusals, the client reads `agent.list`. If an idle agent owns
  the pane, the client types the message directly into the pane. The
  client never types into a bare shell. It also waits while the agent is
  blocked, because the type-in ends with Enter, and Enter answers the
  dialog that blocks the agent.
- **Dropped-prompt fallback** (0.7.5 bug): On worktree panes,
  `agent.prompt` reports success but types nothing. A landed prompt
  increases `state_change_seq`. If the value stays frozen, the client
  uses the same type-in fallback with the same guard.
- **Guarded Enter nudge**: The paste guard of Claude Code sometimes eats
  the Enter from herdr. Thus one bare Enter follows each delivery after
  approximately 2 s. Enter is not safe in all conditions: in a permission
  dialog it accepts the selected option. Therefore `guarded_nudge` first
  reads `agent.list` and the last 40 lines of the pane. It skips the
  Enter if the agent is blocked, if the screen shows a destructive
  action, or if a check fails.

### Other harnesses

herdr hosts many agent kinds (`claude`, `codex`, `gemini`, and more).
Spawns, messages, TUI answers, and status watching work for each kind.
`spawn_task` and `spawn_in_folder` accept an `agent` kind. Transcript
readback (`agent_report` and spoken summaries) needs a per-harness
adapter in the `ADAPTERS` table in `transcripts.py`. Claude and Codex
adapters are included. Missing transcripts automatically fall back to the pane screen. An
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
| `agent.py` | The Mate voice agent: tools, rail, entrypoint. |
| `herdr_client.py` | Async herdr socket client: protocol only. |
| `delivery.py` | Spawn and task delivery: all herdr workarounds, plus the guarded Enter nudge. |
| `watcher.py` | `FleetWatcher` and `Delegations`: spoken block and finish announcements. |
| `screening.py` | Call screening: caller allowlist and passphrase-gate wiring. |
| `folders.py` | Folder-name to path resolution for `spawn_in_folder`. |
| `transcripts.py` | Claude and Codex transcript readers with exact session lookup. |
| `safety.py` | Approval and veto detection for the rail. |
| `allowlist.py` | Caller allowlist: normalization plus fail-closed matching. |
| `passphrase.py` | Spoken-passphrase gate: matching plus launch requirement. |
| `scripts/smoke_llm.py` | Quick sanity test of the local LLM. |

## License

MIT — see `LICENSE`.


## Codex replies

Mate supports Claude and Codex transcript readers. Codex lookup requires an
exact native session ID from Herdr (`agent_session`), including a `pane.get`
lookup when the snapshot omits it. It never chooses a thread by folder or title.
Codex's read-only `state_5.sqlite` index locates the current rollout, including
resumed sessions. Without that index, a unique matching rollout file is used.
`MATE_CODEX_HOME` overrides the state directory (otherwise `CODEX_HOME` or
`~/.codex`). The reader verifies the file's session identity and reads final
assistant answers, excluding reasoning, tools, progress commentary, and duplicate
message IDs. Only the current rollout segment is read; older paginated history
is not reconstructed.

If identity, transcript, or a final answer is unavailable, `agent_report` and
completion announcements automatically use that pane's last 100 screen lines,
clearly marked as a partial screen view. You no longer need to ask it to read
the screen separately. Failed screen reads remain explicit errors. Audit event
`agent.report_fallback` records why the fallback was used.

Existing Codex panes without session metadata use this fallback. Herdr's Codex
SessionStart integration is installed and enabled on this host, but the affected
running pane has not reported an identity; this change does not restart coding
sessions or guess which transcript belongs to them.
