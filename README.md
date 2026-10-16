# Sarvam Voice Agent

Browser-based, real-time Hindi/English voice-agent pipeline with Streamlit and WebRTC.

## Pipeline

Browser microphone → WebRTC → **Sarvam Saaras v3 realtime STT** → **fine-tuned BERT-Tiny script classifier (DEVANAGARI vs LATIN)** → **Sarvam LLM** → **BakBak Raya TTS** → browser speaker.

- `english`: update conversation language to English and reply in English.
- `hinglish`: update conversation language to Hinglish and reply in natural Hinglish.
- `neutral`: pass the label to the LLM and explicitly preserve the current conversation language; do not switch on greetings/acknowledgements such as “hello”, “okay”, or “thanks”.
- `uncertain`: preserve the current language and clarify only if needed.
- `other`: bypass the LLM and say: “I can speak Hindi or English only. Please continue in Hindi or English.”

The BERT-Tiny classifier runs on each finalized STT transcript and reports dominant writing script, both softmax scores, top-class score, and inference latency. Its model files are downloaded from the `full-project-backup` branch of [script-switch-classifier](https://github.com/abhinavnautiyalDS/script-switch-classifier) on first use and cached locally. `DEVANAGARI` routes the reply/TTS toward Hindi; `LATIN` routes toward English. This model classifies writing script, not natural language: Latin-script Hindi/Hinglish can be mistaken for English, so treat routing as a script-based heuristic. Every non-empty transcript still proceeds to the LLM.

## Install and run

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your API credentials, and start the browser app. The Streamlit app reads credentials from Streamlit Secrets first, then falls back to environment variables / `.env` for local development:

```bash
streamlit run app.py
```

For a microphone connected directly to the Python machine, use the local Pipecat runner:

```bash
python voice_agent.py
```

The browser WebRTC route uses the classifier for conditional language routing. The separate `voice_agent.py` local Pipecat runner has not been wired to this classifier yet.

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
- The Streamlit app uses only the fine-tuned BERT-Tiny script classifier; the old joblib and Qwen-LoRA backend selectors have been removed from the UI.
- BERT-Tiny reports `DEVANAGARI` and `LATIN` probabilities. These are softmax scores and are not guaranteed to be calibrated confidence estimates.

The script classifier was trained on public text corpora and should be evaluated on consented, anonymized transcripts from your target STT system before relying on production routing. It predicts dominant script only; it does not distinguish English from Hinglish when both are written in Latin characters.

## WebRTC / TURN setup

For hosted deployments, STUN alone may fail on office, mobile, or restrictive NAT/firewall networks. The app now prefers Cloudflare Realtime TURN when both `CLOUDFLARE_TURN_KEY_ID` and `CLOUDFLARE_TURN_KEY_API_TOKEN` are configured, then tries Hugging Face TURN, and finally falls back to Google STUN. You can also configure `TURN_SERVER_URL`, `TURN_SERVER_USERNAME`, and `TURN_SERVER_CREDENTIAL` for another TURN provider. Keep credentials in Streamlit Secrets or environment variables; never commit them.

To configure Cloudflare, create a Realtime TURN key and an API token with permission to generate TURN credentials. Add both values to Streamlit Secrets, redeploy/restart, then click Stop and start a fresh WebRTC session. The dashboard now says whether microphone frames have actually reached Python; “session started” alone does not prove the audio track is flowing. TURN relay traffic may be billable depending on your provider.

## Troubleshooting

The page displays WebRTC audio flow, Saaras transcript and event, classifier prediction/confidence/latency, LLM response/latency, and BakBak output/latency. Check browser microphone permissions and ICE/TURN connectivity if audio is not reaching STT. Check the raw STT event, LLM response, TTS response, and sample-rate handling if speech output is silent.

## Security

Never commit `.env`, Streamlit secrets, personal call transcripts, or API keys.
