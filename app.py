import os

import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_pcm_audio_source_track,
    get_hf_ice_servers,
    webrtc_streamer,
)

from cloud_voice import CloudVoiceAgent
from dotenv import load_dotenv

load_dotenv()


def configured_secret(name: str, default: str | None = None) -> str | None:
    """Read a Streamlit secret when configured, otherwise fall back to environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name) or default


st.set_page_config(page_title="Sarvam Voice Agent", page_icon="🎤", layout="wide")
st.title("🎤 Sarvam Voice Agent")
st.caption("Browser Mic → Saaras STT → fine-tuned multilingual BERT-Tiny language classifier → Sarvam LLM → BakBak Raya TTS → Browser Speaker")
st.success("Multilingual BERT-Tiny supports 12 language labels. Choose the classifier model below before starting the agent.")

if "cloud_agent" not in st.session_state:
    st.session_state.cloud_agent = None
if "running" not in st.session_state:
    st.session_state.running = False
if "pcm_output" not in st.session_state:
    st.session_state.pcm_output = None
if "audio_sink" not in st.session_state:
    st.session_state.audio_sink = None
st.subheader("Model settings")
with st.expander("Voice model settings", expanded=True):
    llm_model = st.selectbox(
        "Sarvam LLM model",
        ["sarvam-105b-conversations", "sarvam-105b", "Gemma 4 26B (Google)"],
        index=0,
        disabled=st.session_state.running,
    )
    if llm_model == "Gemma 4 26B (Google)":
        llm_model = "gemma-4-26b-a4b-it"

    tts_model = st.selectbox(
        "BakBak Raya TTS model", ["m1"], index=0,
        disabled=st.session_state.running,
        help="Select a model ID supported by your BakBak account.",
    )
    classifier_variant = st.selectbox(
        "Language classifier model",
        ["4.3", "New E2", "Version 4 (New E2)", "Version 4.1"],
        index=0,
        disabled=st.session_state.running,
        help="Select one of the four multilingual BERT-Tiny checkpoints. Version 4.3 is the latest model from the version4.1 branch. The selected model is cached after its first download.",
    )
    st.caption(f"Selected classifier: {classifier_variant}")
    st.info("Language classification uses the selected 12-label BERT-Tiny model (Indian language labels plus none).")

if st.button("Start voice agent", type="primary", disabled=st.session_state.running, use_container_width=True):
    try:
        st.session_state.webrtc_generation = st.session_state.get("webrtc_generation", 0) + 1
        generation = st.session_state.webrtc_generation
        pcm_output = create_pcm_audio_source_track(
            key=f"sarvam_voice_output_{generation}", sample_rate=24000, ptime=0.020,
        )
        sarvam_key = configured_secret("SARVAM_API_KEY")
        gemma_key = configured_secret("GEMMA_API_KEY") or configured_secret("GEMINI_API_KEY")
        bakbak_key = configured_secret("BAKBAK_API_KEY")
        voice_id = configured_secret("BAKBAK_VOICE_ID")
        missing = [name for name, value in (
            ("SARVAM_API_KEY", sarvam_key),
            ("GEMINI_API_KEY (or GEMMA_API_KEY)", gemma_key) if llm_model == "gemma-4-26b-a4b-it" else ("SARVAM_API_KEY", sarvam_key),
            ("BAKBAK_API_KEY", bakbak_key),
            ("BAKBAK_VOICE_ID", voice_id),
        ) if not value]
        if missing:
            raise ValueError(
                "Missing credentials: " + ", ".join(missing) +
                ". Set them in Streamlit Secrets or your .env file."
            )

        agent = CloudVoiceAgent(
            sarvam_key,
            bakbak_key,
            voice_id,
            output_source=pcm_output,
            llm_model=llm_model,
            gemma_api_key=gemma_key,
            tts_model=tts_model,
            classifier_enabled=True,
            classifier_backend="bert_tiny",
            classifier_model_variant=classifier_variant,
        )

        def audio_sink_callback(frame):
            agent.ingest_webrtc_frame(frame)

        audio_sink = create_audio_sink_track(
            callback=audio_sink_callback, key=f"sarvam_voice_input_{generation}",
        )
        st.session_state.pcm_output = pcm_output
        st.session_state.audio_sink = audio_sink
        agent.start()
        st.session_state.cloud_agent = agent
        st.session_state.running = True
        st.rerun()
    except Exception as exc:
        st.error(f"Could not start: {exc}")

agent = st.session_state.cloud_agent
if st.session_state.running and agent is not None:
    pcm_output = st.session_state.pcm_output
    audio_sink = st.session_state.audio_sink
    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]
    turn_provider = "STUN fallback"
    cloudflare_key_id = configured_secret("CLOUDFLARE_TURN_KEY_ID", "") or ""
    cloudflare_api_token = configured_secret("CLOUDFLARE_TURN_KEY_API_TOKEN", "") or ""
    if cloudflare_key_id and cloudflare_api_token:
        try:
            import httpx
            response = httpx.post(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{cloudflare_key_id}/credentials/generate-ice-servers",
                headers={"Authorization": f"Bearer {cloudflare_api_token}"},
                json={"ttl": 3600},
                timeout=10,
            )
            response.raise_for_status()
            cloudflare_servers = response.json().get("iceServers", [])
            if cloudflare_servers:
                ice_servers = cloudflare_servers + ice_servers
                turn_provider = "Cloudflare TURN + STUN"
            else:
                st.warning("Cloudflare TURN returned no ICE servers; trying Hugging Face TURN.")
        except Exception as exc:
            st.warning(f"Cloudflare TURN unavailable ({type(exc).__name__}: {exc}); trying Hugging Face TURN.")
    if turn_provider == "STUN fallback":
        hf_token = configured_secret("HF_TOKEN", "") or ""
        if hf_token:
            try:
                hf_servers = get_hf_ice_servers(token=hf_token)
                if hf_servers:
                    ice_servers = hf_servers + ice_servers
                    turn_provider = "Hugging Face TURN + STUN"
            except Exception as exc:
                st.warning(f"HF TURN unavailable ({type(exc).__name__}: {exc}). STUN-only fallback is active.")
    custom_turn_url = configured_secret("TURN_SERVER_URL", "") or ""
    custom_turn_username = configured_secret("TURN_SERVER_USERNAME", "") or ""
    custom_turn_credential = configured_secret("TURN_SERVER_CREDENTIAL", "") or ""
    if custom_turn_url and custom_turn_username and custom_turn_credential:
        ice_servers.insert(0, {
            "urls": custom_turn_url,
            "username": custom_turn_username,
            "credential": custom_turn_credential,
        })
        turn_provider = "Custom TURN + STUN"
    rtc_config = {"iceServers": ice_servers}
    webrtc_ctx = webrtc_streamer(
        key=f"sarvam-voice-hf-{st.session_state.webrtc_generation}",
        mode=WebRtcMode.SENDRECV,
        sink_audio_track=audio_sink,
        source_audio_track=pcm_output.track,
        sendback_audio=True,
        media_stream_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            },
            "video": False,
        },
        frontend_rtc_configuration=rtc_config,
        server_rtc_configuration=rtc_config,
    )
    st.caption(f"WebRTC ICE: {turn_provider}. TURN relay is recommended for restrictive networks.")

    @st.fragment(run_every="1s")
    def live_diagnostics():
        playing = bool(webrtc_ctx.state.playing)
        receiving_audio = agent.audio_frames_received > 0
        if receiving_audio:
            st.success("WebRTC: MICROPHONE FRAMES RECEIVED")
        elif playing:
            st.warning("WebRTC session started, but no microphone frames have reached Python yet. Check browser mic permission, ICE/TURN connectivity, and the remote audio track.")
        else:
            st.warning("WebRTC: CONNECTING — waiting for browser audio.")
        if agent.error:
            st.error(f"{agent.error_stage or 'Pipeline'} error: {agent.error}")

        st.subheader("Pipeline status")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("WebRTC", "Audio received" if receiving_audio else ("Session started" if playing else "Waiting"))
        c2.metric("Saaras STT", "Listening" if agent.stt_status in ("Connected — waiting for speech", "Listening") else agent.stt_status)
        c3.metric("Sarvam LLM", "Working" if agent.llm_status == "Generating response" else agent.llm_status)
        c4.metric("BakBak Raya", "Speaking" if agent.tts_status == "Generating speech with BakBak Raya" else agent.tts_status)
        c5.metric("Language classifier", agent.classifier_status)

        st.write(f"**Audio:** {agent.audio_frames_received:,} browser frames / {agent.audio_bytes_sent:,} bytes sent to Saaras")
        st.write(f"**Last Saaras event:** `{agent.last_stt_event or '—'}`")

        st.markdown("### 1. STT transcript")
        if agent.live_transcript:
            st.info(f"**Live transcript:** {agent.live_transcript}")
        else:
            st.caption("Waiting for speech...")
        if agent.last_user_text:
            st.success(f"**Final transcript:** {agent.last_user_text}")
        if agent.last_stt_event == "transcript.final" and not agent.last_user_text:
            st.warning("Saaras sent a final transcript event, but no transcript text was found.")
            st.caption(f"Raw Saaras event: {agent.last_stt_raw}")

        st.markdown("### 2. Context-aware multilingual BERT-Tiny classifier")
        classification = agent.last_classifier_result
        st.caption("Classifier input (current finalized caller turn only):")
        classifier_input = getattr(agent, "last_classifier_input", "") or agent.last_user_text or "Waiting for transcript..."
        st.code(classifier_input, language="text")
        if classification.get("available"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted language", str(classification.get("label", "—")).upper())
            confidence = classification.get("confidence")
            cc2.metric("Top-class score", f"{confidence:.1%}" if confidence is not None else "N/A")
            cc3.metric("Inference latency", f"{classification.get('latency_ms', 0.0):.3f} ms")
            if classification.get("probabilities"):
                st.json(classification["probabilities"])
            st.caption(f"Saaras language metadata: {agent.last_detected_language or 'unknown'}")
            st.caption(f"Comparison with Saaras metadata: {agent.last_language_source}")
        else:
            st.caption(f"Classifier unavailable/disabled. Saaras metadata is shown separately for comparison. {classification.get('reason', '')}")

        st.markdown("### 3. LLM response")
        if agent.last_assistant_text:
            st.success(f"**Sarvam returned:** {agent.last_assistant_text}")
        else:
            st.caption("Waiting for Sarvam response...")
        if agent.last_llm_input:
            st.caption(f"LLM input: {agent.last_llm_input}")
        if agent.llm_latency_ms is not None:
            st.caption(f"LLM latency: {agent.llm_latency_ms} ms")

        st.markdown("### 4. BakBak Raya TTS")
        if agent.last_tts_input:
            st.write(f"**Text sent to TTS:** {agent.last_tts_input}")
        if agent.tts_audio_bytes:
            st.success(f"**Audio generated:** {agent.tts_audio_bytes:,} PCM bytes")
            st.caption(f"TTS latency: {agent.tts_latency_ms} ms")
            st.caption(f"Audio pushed to browser: {agent.output_bytes_pushed:,} bytes")
        else:
            st.caption("Waiting for TTS audio...")

        st.markdown("### Current pipeline")
        st.code(
            f"Browser WebRTC: {'RECEIVING AUDIO' if receiving_audio else 'WAITING'}\n"
            f"↓\nSaaras STT: {agent.stt_status}\n"
            f"Transcript: {agent.last_user_text or '—'}\n"
            f"Classifier: {agent.classifier_status}\n"
            f"↓\nSarvam LLM ({llm_model}): {agent.llm_status}\n"
            f"Response: {agent.last_assistant_text or '—'}\n"
            f"↓\nBakBak Raya ({tts_model}): {agent.tts_status}\n"
            f"Output bytes: {agent.output_bytes_pushed:,}",
            language="text",
        )
    live_diagnostics()

    if st.button("Stop voice agent", use_container_width=True):
        agent.stop()
        try:
            if st.session_state.pcm_output is not None:
                st.session_state.pcm_output.clear()
                st.session_state.pcm_output.track.stop()
            if st.session_state.audio_sink is not None:
                st.session_state.audio_sink.stop()
        except Exception:
            pass
        st.session_state.pcm_output = None
        st.session_state.audio_sink = None
        st.session_state.cloud_agent = None
        st.session_state.running = False
        st.rerun()
else:
    st.info("Configure the models, click Start voice agent, and allow browser microphone access.")

st.divider()
st.subheader("Streamlit secrets")
st.code(
    'SARVAM_API_KEY="..."\n'
    'GEMINI_API_KEY="..."  # needed only when Gemma is selected\n'
    'BAKBAK_API_KEY="..."\n'
    'BAKBAK_VOICE_ID="..."\n'
    'HF_TOKEN="hf_..."',
    language="toml",
)
st.caption("HF_TOKEN obtains temporary TURN credentials. Multilingual BERT-Tiny model files download on first use and are cached locally. Never commit API keys.")
