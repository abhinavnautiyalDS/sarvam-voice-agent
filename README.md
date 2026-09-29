# Sarvam Voice Agent

Hindi/Hinglish browser voice-agent baseline.

## Cloud architecture

Browser microphone
→ WebRTC
→ Sarvam Saaras v3 Realtime STT
→ Sarvam 105B
→ BakBak Raya TTS
→ Browser speaker

The cloud bridge is implemented in `cloud_voice.py`. Network work runs in a background thread so the WebRTC media callback stays lightweight.

## Local architecture

For local hardware testing, `voice_agent.py` keeps the original Pipecat pipeline:

Microphone → Saaras Realtime STT → Sarvam 105B → BakBak TTS → Speaker

The local version uses PyAudio/LocalAudioTransport.

## Run locally

Install:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`.

Run the local Pipecat agent:

```bash
python voice_agent.py
```

Run the browser app:

```bash
streamlit run app.py
```

## Streamlit Cloud

1. Deploy this repository as a Streamlit app.
2. Set the main file to `app.py`.
3. Add these secrets:

```toml
SARVAM_API_KEY="..."
BAKBAK_API_KEY="..."
BAKBAK_VOICE_ID="..."
```

4. Open the deployed URL.
5. Click **Start voice agent**.
6. Allow microphone access.
7. Speak Hindi/Hinglish.

Use headphones if you hear echo.

## Why WebRTC?

PyAudio captures audio on the machine running Python. On Streamlit Cloud, that is the cloud server, not the visitor's laptop.

WebRTC lets the browser provide the microphone stream to the Streamlit application.

## Current baseline

- STT: `saaras:v3-realtime`
- STT language: `hi-IN`
- STT stream: `fast`
- VAD: server-side
- LLM: `sarvam-105b`
- LLM max tokens: 150
- TTS: BakBak Raya `m1`
- TTS sample rate: 24 kHz
- Browser audio is resampled to 16 kHz mono for Saaras

## Security

Never commit `.env` or real API keys. Use Streamlit Secrets for deployed credentials.
