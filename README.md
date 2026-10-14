# Sarvam Voice Agent

Browser-based, real-time Hindi/English voice-agent pipeline with Streamlit and WebRTC.

## Pipeline

Browser microphone → WebRTC → **Sarvam Saaras v3 realtime STT** → **Hindi/English/Hinglish transcript classifier** → **Sarvam LLM** → **BakBak Raya TTS** → browser speaker.

The classifier runs locally on the finalized STT transcript and reports class, confidence, probabilities, and inference latency. It is advisory: explicit Saaras language metadata stays primary. A high-confidence `hindi`/`english` prediction may be used only when Saaras returns a missing or unsupported language; `hinglish` remains a diagnostic class, not a Sarvam locale.

## Install and run

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your API credentials, and start the browser app:

```bash
streamlit run app.py
```

For a microphone connected directly to the Python machine, use the local Pipecat runner:

```bash
python voice_agent.py
```

Note: the live classifier diagnostics/decision fallback is wired into the Streamlit WebRTC bridge. `voice_agent.py` remains the local Pipecat route and currently does not use classifier output to modify Pipecat's internal language context.

## Streamlit Cloud secrets

```toml
SARVAM_API_KEY = "..."
BAKBAK_API_KEY = "..."
BAKBAK_VOICE_ID = "..."
HF_TOKEN = "hf_..."
LANGUAGE_CLASSIFIER_MIN_CONFIDENCE = "0.60"
LANGUAGE_CLASSIFIER_OVERRIDE_CONFIDENCE = "0.80"
```

Never commit API keys or `.env` files.

## Configurable models

- STT defaults to `saaras:v3-realtime`; model ID and endpointing parameters can be configured with the `SARVAM_STT_*` environment variables.
- The Streamlit LLM selector offers `sarvam-105b-conversations` and `sarvam-105b`; use only model IDs enabled for the Sarvam API key.
- TTS defaults to BakBak Raya `m1`; voice ID is supplied by your BakBak account.
- The classifier downloads its trained model from the public [Hindi-English classifier repository](https://github.com/abhinavnautiyalDS/hindi-english-language-classifier) on first use. Override `LANGUAGE_CLASSIFIER_MODEL_URL` or point `LANGUAGE_CLASSIFIER_MODEL_PATH` to a local model file.
- `LANGUAGE_CLASSIFIER_MIN_CONFIDENCE` defaults to 0.60 for diagnostics; `LANGUAGE_CLASSIFIER_OVERRIDE_CONFIDENCE` defaults to 0.80 for fallback language decisions.

The classifier was trained on public text corpora (COMI-LINGUA, PHINC, Tatoeba), not the target production STT transcripts. Its measured 84.03% corpus accuracy and 0.688 Hindi F1 are not a live-call accuracy guarantee. Validate on consented, anonymized transcripts from the target STT system before using this signal for production routing.

## Troubleshooting

The page displays WebRTC audio flow, Saaras transcript and event, classifier prediction/confidence/latency, LLM response/latency, and BakBak output/latency. Check browser microphone permissions and ICE/TURN connectivity if audio is not reaching STT. Check the raw STT event, LLM response, TTS response, and sample-rate handling if speech output is silent.

## Security

Never commit `.env`, Streamlit secrets, personal call transcripts, or API keys.
