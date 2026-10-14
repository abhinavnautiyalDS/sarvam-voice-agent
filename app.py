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

st.set_page_config(page_title="Sarvam Voice Agent", page_icon="🎤", layout="wide")
st.title("🎤 Sarvam Voice Agent")
st.caption("Browser Mic → Saaras STT → Hindi/English/Hinglish classifier → Sarvam LLM → BakBak Raya TTS → Browser Speaker")

if "cloud_agent" not in st.session_state:
    st.session_state.cloud_agent = None
if "running" not in st.session_state:
    st.session_state.running = False
if "pcm_output" not in st.session_state:
    st.session_state.pcm_output = None
if "audio_sink" not in st.session_state:
    st.session_state.audio_sink = None
if "primary_language" not in st.session_state:
    st.session_state.primary_language = "Hindi"
if "secondary_language" not in st.session_state:
    st.session_state.secondary_language = "English"

lang_options = {"Hindi": "hi-IN", "English": "en-IN", "Telugu": "te-IN"}

st.subheader("Conversation language settings")
col1, col2 = st.columns(2)
with col1:
    primary_language = st.selectbox(
        "Primary language", list(lang_options),
        index=list(lang_options).index(st.session_state.primary_language),
        key="primary_language_select",
        disabled=st.session_state.running,
    )
with col2:
    secondary_choices = [name for name in lang_options if name != primary_language]
    secondary_default = (
        st.session_state.secondary_language
        if st.session_state.secondary_language in secondary_choices
        else secondary_choices[0]
    )
    secondary_language = st.selectbox(
        "Secondary language", secondary_choices,
        index=secondary_choices.index(secondary_default),
        key="secondary_language_select",
        disabled=st.session_state.running,
    )
st.session_state.primary_language = primary_language
st.session_state.secondary_language = secondary_language
st.caption(f"Allowed conversation languages: {primary_language} + {secondary_language}")

st.subheader("Model settings")
with st.expander("Advanced model and classifier settings", expanded=True):
    classifier_enabled = st.toggle(
        "Enable Hindi / English / Hinglish transcript classifier",
        value=True,
        disabled=st.session_state.running,
        help="Advisory signal only. Saaras language metadata remains primary; a confident Hindi/English classification is fallback only when Saaras has no usable language tag.",
    )
    llm_model = st.selectbox(
        "Sarvam LLM model",
        ["sarvam-105b-conversations", "sarvam-105b"],
        index=0,
        disabled=st.session_state.running,
    )
    tts_model = st.selectbox(
        "BakBak Raya TTS model", ["m1"], index=0,
        disabled=st.session_state.running,
        help="Select a model ID supported by your BakBak account.",
    )
    st.caption("STT: Saaras v3 realtime. TTS: BakBak Raya. The classifier runs locally on each final transcript.")

if st.button("Start voice agent", type="primary", disabled=st.session_state.running, use_container_width=True):
    try:
        st.session_state.webrtc_generation = st.session_state.get("webrtc_generation", 0) + 1
        generation = st.session_state.webrtc_generation
        pcm_output = create_pcm_audio_source_track(
            key=f"sarvam_voice_output_{generation}", sample_rate=24000, ptime=0.020,
        )
        agent = CloudVoiceAgent(
            st.secrets["SARVAM_API_KEY"],
            st.secrets["BAKBAK_API_KEY"],
            st.secrets["BAKBAK_VOICE_ID"],
            primary_language=lang_options[primary_language],
            secondary_language=lang_options[secondary_language],
            output_source=pcm_output,
            llm_model=llm_model,
            tts_model=tts_model,
            classifier_enabled=classifier_enabled,
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
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    ice_servers = []
    if hf_token:
        try:
            ice_servers = get_hf_ice_servers(token=hf_token)
        except Exception as exc:
            st.warning(f"HF TURN unavailable ({exc}). Using STUN only.")
    ice_servers.append({"urls": "stun:stun.l.google.com:19302"})
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
    st.caption("WebRTC ICE: Hugging Face TURN when configured + Google STUN fallback")

    @st.fragment(run_every="1s")
    def live_diagnostics():
        playing = bool(webrtc_ctx.state.playing)
        receiving_audio = agent.audio_frames_received > 0
        if playing or receiving_audio:
            st.success("WebRTC: CONNECTED / RECEIVING AUDIO")
        else:
            st.warning("WebRTC: CONNECTING — waiting for browser audio.")
        if agent.error:
            st.error(f"{agent.error_stage or 'Pipeline'} error: {agent.error}")

        st.subheader("Pipeline status")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("WebRTC", "Receiving" if receiving_audio else "Waiting")
        c2.metric("Saaras STT", "Listening" if agent.stt_status in ("Connected — waiting for speech", "Listening") else agent.stt_status)
        c3.metric("Sarvam LLM", "Working" if agent.llm_status == "Generating response" else agent.llm_status)
        c4.metric("BakBak Raya", "Speaking" if agent.tts_status == "Generating speech with BakBak Raya" else agent.tts_status)
        c5.metric("Classifier", agent.classifier_status)

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

        st.markdown("### 2. Language classifier")
        classification = agent.last_classifier_result
        if classification.get("available"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted class", str(classification.get("label", "—")).upper())
            cc2.metric("Confidence", f"{classification.get('confidence', 0.0):.1%}")
            cc3.metric("Inference latency", f"{classification.get('latency_ms', 0.0):.3f} ms")
            if classification.get("probabilities"):
                st.json(classification["probabilities"])
            st.caption(f"Language decision source: {agent.last_language_source}")
        else:
            st.caption(f"Classifier unavailable/disabled; Saaras remains primary. {classification.get('reason', '')}")

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
    'BAKBAK_API_KEY="..."\n'
    'BAKBAK_VOICE_ID="..."\n'
    'HF_TOKEN="hf_..."\n'
    'LANGUAGE_CLASSIFIER_MIN_CONFIDENCE="0.60"\n'
    'LANGUAGE_CLASSIFIER_OVERRIDE_CONFIDENCE="0.80"',
    language="toml",
)
st.caption("HF_TOKEN obtains temporary TURN credentials. The language classifier downloads its model on first use unless LANGUAGE_CLASSIFIER_MODEL_PATH points to a local copy. Never commit API keys.")
