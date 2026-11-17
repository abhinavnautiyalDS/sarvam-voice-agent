# AURA backend (FastAPI + WebSocket)

This backend exposes the existing `CloudVoiceAgent` pipeline to a browser client.

## Endpoints

- `GET /api/health` — non-sensitive service health.
- `WS /ws/voice` — bidirectional voice session.
  - Client sends binary mono signed PCM16 audio at **16 kHz**.
  - Server sends JSON state events and binary mono signed PCM16 TTS audio at **24 kHz**.
  - The server loads the local 12-label BERT-Tiny checkpoint using classifier variant `4.4`.

## Run on EC2

Use the dedicated lightweight requirements file, not the full Streamlit/Pipecat development requirements:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-api.txt
cp .env.example .env
# Edit .env and add the Sarvam and BakBak credentials.
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep port 8000 private. Put Nginx with HTTPS/WSS in front of it. Set
`AURA_ALLOWED_ORIGINS` to the exact deployed frontend origin before public launch.

## Important

- API keys are read only from the backend environment.
- The public websocket accepts audio only; it does not expose credential or model-management controls.
- The KB is not wired into this first voice API yet. The existing UI's read-only KB view remains informational until an owner-side index and retrieval adapter are configured.
- The React UI must connect to `/ws/voice` and send PCM16 audio; the WebSocket service does not make the existing microphone preview live automatically.
