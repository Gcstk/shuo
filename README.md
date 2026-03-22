# shuo 说

A voice agent framework in ~600 lines of Python. 

```bash
python main.py
```

```
🚀 Server starting on port 3040
✓  Ready http://localhost:3040/web
🔌 WebSocket connected
▶  Stream started SID: browser-a1b2c3...
← Flux EndOfTurn "Hey, how's it going?"
◆ LISTENING → RESPONDING
→ Start Agent "Hey, how's it going?"
← Agent turn done
◆ RESPONDING → LISTENING
```

## How it works

Two abstractions, one pure function:

- **Turn service** — Deepgram Flux or remote SoulX-Duplug, providing live transcripts and turn boundaries
- **Agent** — self-contained LLM → TTS → Player pipeline, owns conversation history
- **`process_event(state, event) → (state, actions)`** — the entire state machine in ~30 lines

Everything streams. LLM tokens feed TTS immediately, TTS audio feeds the active transport immediately. If you interrupt (barge-in), the agent cancels everything and clears the audio buffer instantly.

```
LISTENING ──EndOfTurn──→ RESPONDING ──Done──→ LISTENING
    ↑                        │
    └────StartOfTurn─────────┘  (barge-in)
```

## Project structure

```
shuo/
  types.py              # Immutable state, events, actions
  state.py              # Pure state machine (~30 lines)
  conversation.py       # Main event loop
  agent.py              # LLM → TTS → Player pipeline
  log.py                # Colored logging
  server.py             # FastAPI endpoints
  services/
    flux.py             # Deepgram Flux (STT + turns)
    duplug.py           # Remote SoulX-Duplug adapter (turns over /turn ws)
    llm.py              # OpenAI GPT-4o-mini streaming
    tts.py              # ElevenLabs WebSocket streaming
    tts_pool.py         # TTS connection pool (warm spares)
    player.py           # Audio playback to the active transport
    twilio_client.py    # Outbound calls + Twilio message parsing
  static/
    browser_agent.*     # Browser mic / playback / transcript UI
```

## Setup

Requires Python 3.9+ plus:

- a turn service: Deepgram Flux or a remote SoulX-Duplug server
- your OpenAI-compatible LLM
- a streaming TTS provider

### Browser Mode

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in TURN / LLM / TTS
python main.py
```

Then open:
```text
http://localhost:3040/web
```

The default example config is now set up for Qwen realtime TTS via DashScope. If you prefer ElevenLabs, set:

```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
```

To use a remote SoulX-Duplug service instead of Deepgram Flux:

```bash
TURN_PROVIDER=duplug
DUPLUG_WS_URL=ws://your-duplug-host:8000/turn
```

The browser page already supports live streaming transcripts. With `TURN_PROVIDER=duplug`, user speech will be rendered incrementally from `asr_buffer` / `asr_segment` while the user is still speaking.

### Twilio Phone Mode

Requires [ngrok](https://ngrok.com/) and Twilio credentials in addition to the browser-mode keys.

Configure your [ngrok authentication token](https://dashboard.ngrok.com/get-started/your-authtoken):
```bash
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
```

```bash
pip install -r requirements.txt
cp .env.example .env   # also fill in TWILIO_* values
ngrok http 3040        # in another terminal
python main.py +1234567890  # Replace with the phone number the code will call
```

## Tests
Running these tests is not required for executing the application.
It is recommended to run tests after making changes to the codebase.

```bash
python -m pytest tests/ -v   # runs in ~0.03s
```

## License

MIT
