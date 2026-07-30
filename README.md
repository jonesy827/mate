# mate

Voice bridge: phone-call "Mate", a hands-free assistant that supervises a
fleet of coding agents running in [herdr](https://herdr.dev). Built to be
used from the car — spawn agents, hand them tasks, hear their results, all
over a real phone call.

(The Python package is still named `matebridge`, so commands and module
paths below say `matebridge` — that's the import name, not the project
name.)

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
bring-up. **Nothing in this project starts at boot.**

## Running

Prereqs: infra services up (see `../matebridge-infra`), the llama.cpp unit
started by hand (`systemctl --user start llama-qwen-long`), and herdr
running (`tmux new-session -d -s herdr-host herdr`).

```sh
.venv/bin/python -m matebridge.agent console   # desk test: terminal mic/speaker
set -a && source .env && set +a
.venv/bin/python -m matebridge.agent dev       # real worker: registers with LiveKit
```

The worker preflights llm/stt/tts/herdr and refuses to start with a clear
list of whatever is unreachable. Once registered, an inbound call to the DID
dispatches a job and Mate answers.

Two operational rules learned the hard way:

- **Never restart the worker during a call** — check first:
  `lk room list | grep mate-call-` must be empty.
- Dev-mode job processes re-import the code per call, so edits usually go
  live on the *next* call without a restart; restart anyway (between calls)
  when you need certainty.

## Configuration (`.env`, gitignored, chmod 600)

Copy `.env.example` to `.env` and fill it in — it documents every variable.
The short version:

| var | purpose |
|---|---|
| `MATE_ALLOWED_NUMBERS` | **required** for `dev`/`start`: comma-separated E.164 numbers allowed to call in (e.g. `+14055551234`). The worker refuses to start without it, and any SIP caller not on the list is hung up on before Mate says a word. |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit Cloud project the worker registers with |
| `LLM_URL` `STT_URL` `TTS_URL` | override local endpoints (defaults `:8003` `:8001` `:8880`) |
| `LLM_MODEL` `STT_MODEL` `TTS_VOICE` | model/voice overrides (default voice `af_heart`) |
| `LLM_API_KEY` | default `local` (no auth, qwen-tuned sampling). Set a real key + `LLM_URL=https://api.openai.com/v1` + an `LLM_MODEL` to run the voice brain on the OpenAI API — usage-billed API key only; a ChatGPT subscription has no API access |
| `HERDR_SOCKET` | herdr control socket (default `~/.config/herdr/herdr.sock`) |
| `MATE_SRC_ROOTS` | colon-separated roots `spawn_in_folder` searches (default `~/src`) |
| `MATE_CLAUDE_PROJECTS` | Claude Code transcript dir (default `~/.claude/projects`) |

`.env` holds real credentials — never commit it, never paste it into logs.

## One-time phone/SIP setup (as deployed)

Self-hosted LiveKit was abandoned (it hard-crashed this host twice — see the
infra README); the live path uses **LiveKit Cloud SIP**:

1. **Telnyx**: buy a DID, create a SIP connection/trunk, point its
   destination at your LiveKit Cloud project's SIP URI, assign the DID to it.
2. **LiveKit Cloud** (with `lk` configured for the project):

   ```sh
   # inbound trunk accepting the DID (ours: ST_4yZYpgU6A4cs "telnyx-inbound")
   lk sip inbound create trunk.json     # numbers: ["+14052790756"]

   # dispatch rule: one room per caller (ours: SDR_CZAxhc79FxHU "mate-calls")
   lk sip dispatch create dispatch.json # individual/caller → mate-call-_<caller>_<random>
   ```
3. Run the worker (`... agent dev`). It registers with the Cloud project;
   inbound calls dispatch to it automatically.

Also set `AllowedNumbers` on the inbound trunk to the same list as
`MATE_ALLOWED_NUMBERS` — that rejects unknown callers before a room is even
dispatched. The worker enforces its own allowlist regardless, so a missed
trunk setting is not an open door.

## Threat model

Read this before pointing a phone number at your terminal:

- **The caller allowlist is the only boundary against a hostile caller.**
  The confirmation rail exists to catch lossy *transcription*, not
  attackers — a hostile caller happily says "yes" to their own staged
  action. Everything hinges on who can get a session.
- **Caller ID is not cryptographic identity.** It is asserted by the
  originating carrier, and VoIP origination lets anyone assert anything.
  The allowlist stops strangers who find the DID; it does not stop a
  targeted attacker who knows both your DID and an allowed number and
  spoofs it. If that is in your threat model, don't deploy this as-is (a
  spoken-PIN second factor is on the wishlist).
- **The destructive-action regex (`safety.py`) is a heuristic, not a
  security boundary.** It decides which TUI approvals get the extra
  spoken-confirmation rail; a prompt it fails to flag goes through on one
  utterance. Treat it as a seatbelt for the legitimate user.
- A caller who passes the allowlist can drive coding agents that run
  with your user account's full permissions. There is no sandbox beyond
  whatever the agents themselves enforce.
- **A hosted voice brain ships your data off-box.** With `LLM_API_KEY`
  set, every transcript, fleet status, and agent reply Mate handles goes
  to the provider. The default all-local stack keeps calls on your
  machine (minus ordinary phone carriage).

## How Mate works

`src/matebridge/agent.py` defines the `Mate` agent and its tools:

- **Fleet**: `fleet_status`, `read_pane`, `agent_report` (reads the agent's
  actual replies from its session transcript), `wait_for_agent`,
  `list_known_agents` / `forget_agent`.
- **Acting on agents**: `tell_agent` (message a pane's agent), `spawn_task`
  (new worktree + branch), `spawn_in_folder` (workspace directly in an
  existing folder; empty task means "open ready and wait"), `send_answer`
  (answer TUI prompts; a destructive-looking prompt is staged through the
  rail instead of sent — there is no bypass tool).
- **Rail**: `send_staged` / `discard_staged`.

### The stage-and-confirm rail

Voice transcription is lossy, so nothing outward happens on one utterance.
`tell_agent` / `spawn_task` / `spawn_in_folder` **stage** the action and read
it back word for word; delivery happens only when `send_staged` runs *and*
code (not the LLM) verifies the raw transcript of a **new** user turn
contains a clear yes with no veto words. Saying "guardrails off" (voice
toggle, detected in code) switches to immediate delivery; "guardrails on"
restores the rail.

Approving an agent's pending TUI prompt takes the same rail when the
on-screen action looks destructive (`safety.py`'s heuristic): `send_answer`
stages the keys and delivery needs the same code-verified spoken yes.
"Guardrails off" does **not** bypass this — the toggle covers messaging and
spawn convenience, never destructive approvals.

### Background delivery and the watcher

Spawns return immediately; the task is handed over by a background deliverer
that waits out the agent's whole boot (up to 120 s). During a call,
`watch_fleet` subscribes to pane status events and announces deliveries and
finishes ("Hey mate, the songhaus agent just finished…"), speaking a
sanitized two-sentence summary (`tts_sanitize` / `tts_summary` — so agents'
final replies should front-load the key finding). A `Delegations` clock
guards the race where a pane looks idle only because herdr is still typing:
"finished" is announced only for panes that were actually seen working, and
a pane that never starts gets one Enter nudge instead of a stale
announcement.

## herdr integration notes (`herdr_client.py`)

herdr is an external project — mate **never patches it**; everything
below is a client-side workaround, each pinned by a test.

Tested against **herdr 0.7.5 (protocol 17)**. The worker logs herdr's
version at startup (the preflight ping already carries it) and warns —
without refusing to run — when the protocol differs: the workarounds
degrade to no-ops on a herdr that behaves, and real breakage surfaces
loudly as spoken tool errors, so a hard version gate would only turn
every herdr upgrade into a refusal to start.

Workarounds:

- **Names are sanitized**: herdr agent names must match
  `^[a-z][a-z0-9_-]{0,31}$`; any folder/branch label is folded to fit
  ("RackCoach" → "rackcoach") before `agent.start`.
- **A slow boot is not a failed launch**: the workspace is torn down only if
  `agent.start` itself fails. Prompt delivery retries `agent_not_ready` for
  up to 120 s, and `timeout_ms` stretches herdr's own 30 s launch deadline
  to match.
- **Stuck-launch fallback** (herdr 0.7.5 bug, seen live twice): a managed
  launch can stay `launch_pending` forever while the agent is visibly idle,
  refusing every prompt. After ~20 s of refusals, if `agent.list` proves a
  settled coding agent owns the pane, the message is typed into the pane
  directly. The fallback **never** fires on a bare shell — typed task text
  would execute as a shell command.
- **Enter nudge**: Claude Code's paste guard sometimes swallows the Enter
  herdr sends after typing, leaving the message stuck in the input box.
  Every delivery is followed ~2 s later by one bare Enter — no-op if the
  message submitted, rescue if it didn't.
- Protocol ground truth (one request per connection, events.subscribe
  framing, etc.) is documented in the module docstring and enforced by
  `tests/test_herdr_client.py`'s fake server.

## Development

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/ tests/
```

CI (GitHub Actions) runs both on every push and PR.

Every live-discovered bug gets a pinning test before (or with) its fix; the
suite runs with no network, herdr, or GPU — herdr is faked at the protocol
level (`test_herdr_client.py`) or the client level (recording fakes in the
other files).

| module | role |
|---|---|
| `agent.py` | the Mate voice agent: tools, rail, watcher, entrypoint |
| `herdr_client.py` | async herdr socket client + spawn/delivery logic |
| `folders.py` | folder-name → path resolution for `spawn_in_folder` |
| `transcripts.py` | reading claude session transcripts for `agent_report` |
| `safety.py` | transcript approval / veto detection for the rail |
| `allowlist.py` | caller allowlist: normalization + fail-closed matching |
| `scripts/smoke_llm.py` | quick local-LLM sanity check |

## License

MIT — see `LICENSE`.
