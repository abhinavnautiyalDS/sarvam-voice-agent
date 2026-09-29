# Sarvam Voice Agent

Hindi/Hinglish voice-agent baseline using Pipecat.

## Architecture

Microphone → Saaras v3 Realtime STT → Sarvam 105B → BakBak Raya TTS → Speaker

## Stack

- Pipecat 1.12.0
- Sarvam Saaras v3 Realtime
- Sarvam 105B
- BakBak Raya TTS
- Python

## Local setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your own credentials.

Run:

```bash
python voice_agent.py
```

The local version uses Pipecat LocalAudioTransport/PyAudio.

## Streamlit Cloud

Server-side PyAudio captures the machine running Python, not a visitor's browser microphone.

The cloud version therefore needs:

Browser microphone
→ WebRTC
→ Streamlit Cloud
→ Saaras Realtime STT
→ Sarvam 105B
→ BakBak TTS
→ Browser speaker

The current `app.py` documents this boundary. The WebRTC bridge is the next implementation step.

## Secrets

Never commit `.env` or real API keys.

For Streamlit Cloud add:

```toml
SARVAM_API_KEY="..."
BAKBAK_API_KEY="..."
BAKBAK_VOICE_ID="..."
```
