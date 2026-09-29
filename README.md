# Mate

Mate lets you control coding agents through a phone call.
The agents operate in [herdr](https://herdr.dev).
You can start agents, send tasks, ask for status, and hear results.
Mate can read Claude Code and Codex replies.

Mate is an experimental application.
Its security controls have limits.
Read the security section before you connect a phone number.

## System description

Mate has two voice modes.
The `MATE_VOICE_MODE` variable selects the mode.
The default value is `local`.
The current installation uses `live`.

| Mode | Speech and model services |
|---|---|
| `live` | GPT-Live supplies conversation. The Luna model selects tools. Local services supply authentication and confirmation speech. |
| `local` | Whisper changes speech to text. A language model selects tools. Kokoro changes text to speech. |

The following diagram shows the current phone connection.

```text
Phone
  → LiveKit phone number
  → LiveKit room
  → Mate worker
      → GPT-Live conversation
      → Luna tool selection
      → Mate tools
      → herdr Unix socket
      → Coding agents
```

The worker operates on the workstation.
A worker is the process that controls calls and supplies Mate tools.
A pane is a terminal area in herdr.
A transcript is a file that contains conversation records.
A session ID identifies one agent conversation.

The infrastructure project is at `../matebridge-infra`.
Its README gives the service start procedures.
This project does not install an automatic start service.

## Phone connection

Use a phone number assigned to your own LiveKit project or SIP provider.
Keep actual phone numbers, project addresses, and dispatch identifiers in private configuration.
Do not put these values in documentation or test fixtures.
Use reserved example numbers, such as `+1 202 555 0100`, in public examples.

A dispatch rule connects an incoming call to a LiveKit room.
An outbound trunk connects LiveKit to a provider for outgoing calls.
Mate has no callback feature.

### Configure another installation

1. Obtain a phone number from LiveKit or a SIP provider.
2. Configure the incoming call connection for your LiveKit project.
3. Configure a dispatch rule for the Mate worker.
4. Set `MATE_ALLOWED_NUMBERS` in `.env`.
5. Set a passphrase in `.env`.
6. Start the worker.
7. Make a call to make sure that the connection operates correctly.

A separate SIP provider also needs an inbound trunk.
Set its number restrictions to agree with `MATE_ALLOWED_NUMBERS`, if the provider supports this control.
Mate compares the caller number with its own allowlist.
Mate does not need a trunk allowlist to enforce its own caller restrictions.

## Security

**WARNING: Do not give the passphrase to a person without access approval.**
**An authenticated caller can control agents with your workstation permissions.**

Mate compares the caller number with `MATE_ALLOWED_NUMBERS`.
An empty allowlist prevents the phone worker from starting.
Mate rejects a SIP caller whose number is not in the allowlist.
Caller ID is not proof of identity.
An attacker can supply an incorrect caller number.

The passphrase supplies a second check.
Four words are necessary in the default configuration.
There is no default passphrase.
Mate examines the transcript in application code.
Before authentication succeeds, Mate refuses tool actions and gives no agent status.
In Live mode, Mate does not open the GPT-Live connection before authentication succeeds.

The passphrase check has these limits:

- Three incorrect attempts stop the call.
- An attempt longer than 15 seconds counts as a failure.
- Three calls with incorrect authentication, one after the other, stop the worker.
- A state file keeps the failure count between calls.
- The next worker start sets that count to zero.

Confirmation can prevent some transcription errors.
It does not stop an authenticated attacker.
An attacker can give approval for their own request.

The destructive-action detector uses text patterns.
These patterns recognize some Claude Code prompts.
They do not identify all dangerous actions or all Codex permission prompts.
They are not a security boundary.

Live mode sends audio and agent data to OpenAI after authentication.
Local mode can also use a hosted language model.
That model receives the conversation text and tool data.
The local services keep their own logs.

## Configuration

The example file gives the configuration variables.
Git excludes `.env` and `.logs/`.

1. Copy `.env.example` to `.env`.
2. Set the file permissions to `600`.
3. Enter your credentials and configuration values.

```sh
cp .env.example .env
chmod 600 .env
```

| Variable | Function |
|---|---|
| `MATE_ALLOWED_NUMBERS` | Required list of permitted caller numbers. Use E.164 format and commas between numbers. |
| `MATE_PASSPHRASE` | Four-word passphrase. An interactive start can ask for this value if it is missing. |
| `MATE_REQUIRE_PASSPHRASE` | Default: `1`. The value `0` disables the passphrase check. |
| `LIVEKIT_URL` | LiveKit project address. |
| `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit credentials. |
| `MATE_VOICE_MODE` | Voice mode: `local` or `live`. |
| `OPENAI_API_KEY` | OpenAI API key for Live mode. |
| `MATE_LIVE_MODEL` | Live voice model. Default: `gpt-live-1`. |
| `MATE_LIVE_VOICE` | Live voice. Default: `marin`. |
| `MATE_BACKEND_MODEL` | Model that selects Live tools. Default: `gpt-6-luna`. |
| `MATE_BACKEND_REASONING` | Reasoning effort for that model. Default: `none`. |
| `LLM_URL` | Language model address for local mode. Default: `http://localhost:8003/v1`. |
| `STT_URL` | Speech recognition address. Default: `http://localhost:8001/v1`. |
| `TTS_URL` | Speech synthesis address. Default: `http://localhost:8880/v1`. |
| `LLM_MODEL`, `STT_MODEL`, `TTS_VOICE` | Local model and voice selections. See `.env.example` for defaults. |
| `LLM_API_KEY` | Language model credential for local mode. Default: `local`. |
| `HERDR_SOCKET` | herdr socket path. Default: `~/.config/herdr/herdr.sock`. |
| `MATE_SRC_ROOTS` | Project search directories. Use colons between paths. Default: `~/src`. |
| `MATE_CLAUDE_PROJECTS` | Claude transcript directory. Default: `~/.claude/projects`. |
| `MATE_CODEX_HOME` | Codex state directory override. If this value is missing, Mate uses `CODEX_HOME` or `~/.codex`. |

A hosted model in local mode needs three values: `LLM_URL`, `LLM_MODEL`, and `LLM_API_KEY`.
A ChatGPT subscription does not supply API usage for Mate.

The `MATE_APPROVAL_*` variables apply to the separate approval classifier experiment.
Live calls do not use that classifier.
They do not check its service during startup.

## Start Mate

### Prepare the services

1. Start Whisper and Kokoro as specified in `../matebridge-infra`.
2. Make sure that herdr is available.
3. For local mode, start the selected language model service.

Live mode does not need Qwen.
The reference local installation uses the following Qwen service:

```sh
systemctl --user start llama-qwen-long
```

Start herdr only if no instance is available.
The following command starts an instance in tmux:

```sh
tmux new-session -d -s herdr-host herdr
```

### Start the worker

1. Load the configuration.
2. Do the console test with a microphone and speaker.
3. Stop the console test.
4. Start the phone worker.

```sh
set -a
source .env
set +a

.venv/bin/python -m mate.agent console
# Stop the console test before you start the phone worker.
.venv/bin/python -m mate.agent dev
```

The call process examines its necessary services before it starts a call.
Live mode examines OpenAI, Whisper, Kokoro, and herdr.
Local mode examines its language model, Whisper, Kokoro, and herdr.
An unsuccessful check prevents the start of the call process.

### Restart after a change

1. Examine the output of `lk room list`.
2. Make sure that no call is active.
3. Stop the worker.
4. Start the worker again.

Do not restart the worker during a call.
The installed LiveKit CLI does not automatically reload Python changes in `dev` mode.

## Confirmation and delivery

A staged action is a stored request that Mate did not send.
Mate reads the request aloud before it asks for approval.
The tools `tell_agent`, `spawn_task`, and `spawn_in_folder` normally create staged actions.
The `send_staged` tool sends the stored action.
The `discard_staged` tool removes it.

### Live mode

GPT-Live and its backend model interpret your approval.
The backend is the Luna model that selects tools.
After approval, it must call `send_staged`.
A spoken statement that Mate sent a message does not prove delivery.

These conditions are necessary for the application code:

- Authentication succeeded for the caller.
- The voice connection is available.
- An action is staged.
- The spoken confirmation stopped.

Mate removes the staged action before the delivery attempt.
A repeated call to `send_staged` cannot send that action again.
After a correction, a new staged action and spoken confirmation are necessary.

Live mode has no separate approval classifier or transcript timer.
A fixed approval phrase is not necessary.
Transcript fragments supply logs and context, but they do not control delivery.
The models can incorrectly interpret approval.
The application does not independently verify their interpretation.

Live mode always uses confirmation.
The local-mode command "guardrails off" does not disable it.
For a staged destructive prompt, Mate reads the pane again before it sends keys.
The screen must agree with the screen used for confirmation.

### Local mode

The application checks for approval in a subsequent user turn.
It removes the staged action after more than three user turns that follow the request.
The model must then stage the action again.

The command "guardrails off" lets Mate send messages and start agents immediately.
The command "guardrails on" makes confirmation necessary again for those actions.
Neither command bypasses confirmation for a detected destructive prompt.

Before destructive keys, Mate reads the pane again.
It refuses delivery if it cannot read the screen or find the destructive pattern.
Identical screen text is not necessary in local mode.
Thus, this check can accept a different destructive prompt.

### Delivery results and delays

The `delivery.attempt` event identifies an attempt to send a Live action.
The `delivery.result` event contains the returned result.
A delivery result does not mean that the agent completed the task.
If the delivery result is unknown, do not automatically send the action again.

The `wait_for_agent` tool can wait up to 20 seconds for a reply.
This wait can delay the spoken response after delivery.
Long confirmation speech can also delay the next tool result.
The current implementation has these delays.

During a call, `FleetWatcher` announces blocked agents and completed delegated tasks.
It waits for an indication that a delegated agent started work.
If it observes no working state, it can do a completion check after 20 seconds.
These announcements stop when the call stops.
A coding agent can continue its task after the call stops.

## Claude and Codex replies

The `agent_report` tool first tries to read an agent transcript.
Mate has readers for Claude and Codex transcript formats.
It uses the agent type and exact session ID from herdr.
If the snapshot omits the ID, Mate requests the pane details.
Mate does not select a transcript by project directory, title, or modification time.

The Claude reader uses this path:

```text
~/.claude/projects/<encoded-project-path>/<session-id>.jsonl
```

The Codex reader uses the `state_5.sqlite` index to find the current transcript file.
It opens the index for read access only.
This index can identify a new file after a session resumes.
Without a usable index, one file must match the exact session ID.
Mate compares the session ID with the ID inside the file.

The Codex reader returns final assistant answers.
It does not include progress comments, tool records, reasoning records, or duplicate message IDs.
It reads only the current transcript segment.
It does not reconstruct earlier paginated history.

If the session ID, transcript, or final answer is unavailable, Mate reads the last 100 lines from the same pane.
It identifies this text as a partial screen view.
Status requests and completion announcements use this alternative automatically.
You do not need to ask Mate to read the screen separately.
An unsuccessful screen read returns an error.

The `agent.report_fallback` event records why Mate used the screen.
Existing panes without a session ID continue to use this alternative.
The Herdr Codex `SessionStart` integration can supply session IDs.
Its configuration does not guarantee that an existing pane has a recorded ID.
Mate does not restart coding sessions to obtain one.

## Call logs

Each call writes a log at `.logs/<job-id>.log`.
The directory has permission mode `700`.
Each log has permission mode `600`.
Each file has a size limit of 2 MB, with two backup files.
The number of call logs can increase over time.

The logs contain authenticated speech text, tool calls, durations, delivery results, and agent identifiers.
They do not contain complete copies of all terminal results.
Mate removes configured credential values and the passphrase from its audit messages.
It suppresses SDK transcript debug messages.
The separate Whisper service has its own log configuration.

1. List the logs with `ls -t .logs`.
2. Select the required call ID.
3. Read that file with `tail -f .logs/<job-id>.log`.

For a request about every agent, compare `fleet.coverage` and `agent.report` events.
The `live.speech` events show text that GPT-Live generated.
They do not prove that the caller heard all audio.

## Tools and implementation

| Function | Tools |
|---|---|
| Agent status | `fleet_status`, `agent_report`, `read_pane`, `wait_for_agent` |
| Stored agent locations | `list_known_agents`, `forget_agent` |
| Tasks and agent starts | `tell_agent`, `spawn_task`, `spawn_in_folder` |
| Terminal answers | `send_answer` |
| Confirmation | `send_staged`, `discard_staged` |

An agent start returns before task delivery is complete.
A background task waits for the agent to become available.
The delivery retry interval can extend to approximately 120 seconds.

The herdr compatibility code uses protocol 19 as its reference.
This reference is for herdr 0.8.0.
A different protocol version causes a warning.
It does not prevent startup.

The delivery code contains alternatives for delayed launches and missing prompts.
An idle coding agent must be in the pane before direct terminal input.
Mate does not send a task directly into a bare shell or a blocked prompt.
Mate does not send an extra Enter key after delivery.

| File | Function |
|---|---|
| `src/mate/agent.py` | Agent tools, local confirmation, and call setup. |
| `src/mate/live.py` | GPT-Live connection, audio, and delegated tool calls. |
| `src/mate/live_agent.py` | Live tools and confirmation. |
| `src/mate/audit.py` | Call logs and credential removal. |
| `src/mate/herdr_client.py` | herdr socket protocol. |
| `src/mate/delivery.py` | Agent starts and delivery compatibility code. |
| `src/mate/watcher.py` | Agent status observation and spoken updates. |
| `src/mate/screening.py` | Caller checks and passphrase setup. |
| `src/mate/allowlist.py` | Caller number comparison. |
| `src/mate/passphrase.py` | Passphrase checks and failure counts. |
| `src/mate/safety.py` | Local approval checks and destructive-action patterns. |
| `src/mate/folders.py` | Project directory selection. |
| `src/mate/transcripts.py` | Claude and Codex transcript readers. |
| `src/mate/approval.py` | Separate approval classifier experiment. |

## Development checks

Do these checks from the repository directory:

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
```

A GPU, a live herdr instance, and external API calls are not necessary for these tests.
Local Unix sockets and executor threads are necessary for some tests.
A restrictive sandbox can prevent these tests from completing.
The GitHub workflow does the tests and lint checks for pull requests and pushes to `main`.

The following script does a separate OpenAI API test:

```sh
.venv/bin/python scripts/smoke_live.py
```

This script loads `.env` and incurs API charges.
It checks tool selection, the Live connection, and the separate approval classifier experiment.
It sends no herdr actions and starts no GPU services.
Its classifier checks do not use the current Live confirmation path.

Before operational use, do a phone test of these conditions:

- A status request.
- A staged message and an approving reply.
- A correction or refusal after confirmation speech.
- Speech during a confirmation.
- A disconnected call.
- A Codex reply without session metadata.

Compare each requested action with the result in herdr.
Automated tests do not show recognition accuracy during a phone call.

## Technical references

- [Live WebSocket protocol](https://developers.openai.com/api/docs/guides/voice-websockets?api=live)
- [Delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation)
- [Live transcripts](https://developers.openai.com/api/docs/guides/live-conversations)
- [ASD-STE100 specification](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf)

## License

Mate uses the MIT license.
See [LICENSE](LICENSE).
