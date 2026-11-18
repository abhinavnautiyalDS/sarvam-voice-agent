# Sarvam Voice Agent

Browser-based, real-time Hindi/English voice-agent pipeline with Streamlit and WebRTC.

## Pipeline

Browser microphone → WebRTC → **Sarvam Saaras v3 realtime STT** → **fine-tuned BERT-Tiny script classifier (DEVANAGARI vs LATIN)** → **Sarvam LLM** → **BakBak Raya TTS** → browser speaker.

- `english`: update conversation language to English and reply in English.
- `hinglish`: update conversation language to Hinglish and reply in natural Hinglish.
- `neutral`: pass the label to the LLM and explicitly preserve the current conversation language; do not switch on greetings/acknowledgements such as “hello”, “okay”, or “thanks”.
- `uncertain`: preserve the current language and clarify only if needed.
- `other`: bypass the LLM and say: “I can speak Hindi or English only. Please continue in Hindi or English.”

The BERT-Tiny classifier runs on each finalized STT transcript and reports dominant writing script, both softmax scores, top-class score, and inference latency. Its model files are downloaded from the `script-classifier-dataset` branch of [hinglish-language-switch-llm](https://github.com/abhinavnautiyalDS/hinglish-language-switch-llm/tree/script-classifier-dataset/artifacts/script_classifier_bert_tiny/model) on first use and cached locally. `DEVANAGARI` routes the reply/TTS toward Hindi; `LATIN` routes toward English. This model classifies writing script, not natural language: Latin-script Hindi/Hinglish can be mistaken for English, so treat routing as a script-based heuristic. Every non-empty transcript still proceeds to the LLM.

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

## Voice system prompt

The browser WebRTC app and the local Pipecat runner both load their system prompt from `system_prompt.txt`. This checked-in file takes precedence over the legacy `VOICE_SYSTEM_PROMPT` environment variable; the environment variable is only a fallback if the file is missing or empty.

## Configurable models

- STT defaults to `saaras:v3-realtime`; model ID and endpointing parameters can be configured with the `SARVAM_STT_*` environment variables.
- The Streamlit LLM selector offers `sarvam-105b-conversations` and `sarvam-105b`; use only model IDs enabled for the Sarvam API key.
- TTS defaults to BakBak Raya `m1`; voice ID is supplied by your BakBak account.
- The Streamlit app uses only the fine-tuned BERT-Tiny script classifier; the old joblib and Qwen-LoRA backend selectors have been removed from the UI.
- BERT-Tiny reports `DEVANAGARI` and `LATIN` probabilities. These are softmax scores and are not guaranteed to be calibrated confidence estimates.

The fine-tuned BERT-Tiny weights are maintained in the `hinglish-language-switch-llm` repository and are fetched lazily on first classifier use. The model was evaluated on held-out data, but it should also be evaluated on consented, anonymized transcripts from your target STT system before relying on production routing. It predicts dominant script only; it does not distinguish English from Hinglish when both are written in Latin characters.

## AURA hosted voice backend: reasoning model selection

The AURA browser voice session keeps STT and TTS fixed while allowing the caller/operator to select the reasoning provider before starting a session:

- **Sarvam Gamma** (default): uses the configured `SARVAM_LLM_MODEL` and the Sarvam chat-completions API.
- **Hosted Qwen3-0.6B**: uses the private model-server OpenAI-compatible endpoint over the Backend EC2's Tailscale network. The browser never receives the Qwen API key and never connects directly to the model server.

Configure these variables on the **Backend EC2 only** when enabling Qwen:

```env
QWEN_API_KEY=<same API key configured for the llama.cpp model server>
AURA_QWEN_CHAT_COMPLETIONS_URL=http://100.95.233.66:8081/v1/chat/completions
AURA_QWEN_MODEL=aura-qwen3-0.6b
```

Keep the Qwen API key out of Git, frontend build variables, browser storage, and client requests. The default provider remains Sarvam unless the voice session sends `llm_provider=qwen`.

For a frontend deployed separately on Cloudflare Pages, set these **Pages build environment variables**:

```env
VITE_AURA_API_BASE_URL=https://<backend-tunnel-host>
VITE_AURA_WS_URL=wss://<backend-tunnel-host>/ws/voice
```

Set `AURA_ALLOWED_ORIGINS` on the backend to the exact Cloudflare Pages origin (for example, `https://<project>.pages.dev`). Do not use a wildcard origin for a credentialed browser API. The Quick Tunnel hostname is temporary and can change; update both Pages variables when it changes.

## Knowledge Base storage and inspection

The Knowledge Base's actual content is runtime data, not checked into Git:

- SQLite: `data/knowledge_base/knowledge.sqlite3`
- FAISS index: `data/knowledge_base/knowledge.faiss`
- The directory is excluded by `.gitignore`.

The read-only API exposes the live inventory and stored chunks:

- `GET /api/knowledge/status`
- `GET /api/knowledge/documents`
- `GET /api/knowledge/documents/{document_id}/chunks`

The frontend Knowledge Base view reads those APIs. Set `VITE_AURA_API_BASE_URL` when the frontend and backend have different origins. Existing stored vectors/indexes are not automatically migrated by changing embedding configuration; preserve and back up the runtime SQLite and FAISS files before any embedding-model migration.

## WebRTC / TURN setup

For hosted deployments, STUN alone may fail on office, mobile, or restrictive NAT/firewall networks. The app now prefers Cloudflare Realtime TURN when both `CLOUDFLARE_TURN_KEY_ID` and `CLOUDFLARE_TURN_KEY_API_TOKEN` are configured, then tries Hugging Face TURN, and finally falls back to Google STUN. You can also configure `TURN_SERVER_URL`, `TURN_SERVER_USERNAME`, and `TURN_SERVER_CREDENTIAL` for another TURN provider. Keep credentials in Streamlit Secrets or environment variables; never commit them.

To configure Cloudflare, create a Realtime TURN key and an API token with permission to generate TURN credentials. Add both values to Streamlit Secrets, redeploy/restart, then click Stop and start a fresh WebRTC session. The dashboard now says whether microphone frames have actually reached Python; “session started” alone does not prove the audio track is flowing. TURN relay traffic may be billable depending on your provider.

## Troubleshooting

The page displays WebRTC audio flow, Saaras transcript and event, classifier prediction/confidence/latency, LLM response/latency, and BakBak output/latency. Check browser microphone permissions and ICE/TURN connectivity if audio is not reaching STT. Check the raw STT event, LLM response, TTS response, and sample-rate handling if speech output is silent.

## Security

Never commit `.env`, Streamlit secrets, personal call transcripts, or API keys.
