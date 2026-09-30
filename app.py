import os

import httpx
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_pcm_audio_source_track,
    webrtc_streamer,
)

from cloud_voice import CloudVoiceAgent

st.set_page_config(page_title="Sarvam Voice Agent", page_icon="🎤")
st.title("🎤 Sarvam Voice Agent")
st.caption("Browser Mic → WebRTC → Saaras STT → Sarvam 105B → BakBak TTS → Browser Speaker")

if "cloud_agent" not in st.session_state:
    st.session_state.cloud_agent = None
if "running" not in st.session_state:
    st.session_state.running = False
if "pcm_output" not in st.session_state:
    st.session_state.pcm_output = None
if "audio_sink" not in st.session_state:
    st.session_state.audio_sink = None

if st.button(
    "Start voice agent",
    type="primary",
    disabled=st.session_state.running,
    use_container_width=True,
):
    try:
        pcm_output = create_pcm_audio_source_track(
            key="sarvam_voice_output",
            sample_rate=24000,
            ptime=0.020,
        )

        agent = CloudVoiceAgent(
            st.secrets["SARVAM_API_KEY"],
            st.secrets["BAKBAK_API_KEY"],
            st.secrets["BAKBAK_VOICE_ID"],
            output_source=pcm_output,
        )

        def audio_sink_callback(frame):
            agent.ingest_webrtc_frame(frame)

        audio_sink = create_audio_sink_track(
            callback=audio_sink_callback,
            key="sarvam_voice_input",
        )

        st.session_state.pcm_output = pcm_output
        st.session_state.audio_sink = audio_sink

        agent.start()
        st.session_state.cloud_agent = agent
        st.session_state.running = True
        st.rerun()

    except Exception as exc:
        st.error(f"Could not start: {exc}")

@st.cache_data(ttl=300)
def get_hf_turn_servers(token: str):
    url = "https://fastrtc-turn-service.hf.space/credentials"
    response = httpx.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
        follow_redirects=True,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("iceServers", [])

agent = st.session_state.cloud_agent

if st.session_state.running and agent is not None:
    pcm_output = st.session_state.pcm_output
    audio_sink = st.session_state.audio_sink

    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    ice_servers = []

    if hf_token:
        try:
            ice_servers = get_hf_turn_servers(hf_token)
        except Exception as exc:
            st.warning(f"HF TURN unavailable ({exc}). Falling back to Google STUN.")

    ice_servers.append({"urls": "stun:stun.l.google.com:19302"})
    rtc_config = {"iceServers": ice_servers}

    webrtc_ctx = webrtc_streamer(
        key="sarvam-voice-hf-v4",
        mode=WebRtcMode.SENDRECV,
        sink_audio_track=audio_sink,
        source_audio_track=pcm_output.track,
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

    if len(ice_servers) > 1:
        st.caption("WebRTC ICE: Hugging Face TURN + Google STUN")
    else:
        st.caption("WebRTC ICE: Google STUN fallback")

    @st.fragment(run_every="1s")
    def live_diagnostics():
        playing = bool(webrtc_ctx.state.playing)

        if playing:
            st.success("WebRTC: CONNECTED")
        else:
            st.warning("WebRTC: NOT CONNECTED / CONNECTING — the browser is not sending audio yet.")

        if agent.error:
            st.error(f"{agent.error_stage or 'Pipeline'} error: {agent.error}")

        st.subheader("Pipeline status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WebRTC", "Connected" if playing else "Waiting")
        c2.metric("STT", agent.stt_status)
        c3.metric("LLM", agent.llm_status)
        c4.metric("BakBak", agent.tts_status)

        st.write(f"**Audio:** {agent.audio_frames_received:,} browser frames received / {agent.audio_bytes_sent:,} bytes sent to Saaras")
        st.write(f"**Last Saaras event:** `{agent.last_stt_event or '—'}`")

        st.markdown("### 1. STT — What did Saaras hear?")
        if agent.last_user_text:
            st.info(f"**Transcript:** {agent.last_user_text}")
        else:
            st.caption("No final transcript yet. Speak after WebRTC shows CONNECTED.")

        st.markdown("### 2. LLM — What did Sarvam return?")
        if agent.last_llm_input:
            st.write(f"**Input to LLM:** {agent.last_llm_input}")
        if agent.last_assistant_text:
            st.success(f"**LLM response:** {agent.last_assistant_text}")
        else:
            st.caption("No LLM response yet.")
        if agent.llm_latency_ms is not None:
            st.caption(f"LLM latency: {agent.llm_latency_ms} ms")

        st.markdown("### 3. BakBak TTS — Did speech generation work?")
        if agent.last_tts_input:
            st.write(f"**Text sent to BakBak:** {agent.last_tts_input}")
        if agent.tts_audio_bytes:
            st.success(f"**Audio generated:** {agent.tts_audio_bytes:,} PCM bytes")
            st.caption(f"BakBak latency: {agent.tts_latency_ms} ms | Output pushed to WebRTC: {agent.output_bytes_pushed:,} bytes")
        else:
            st.caption("No BakBak audio generated yet.")

        st.markdown("### Current pipeline")
        st.code(
            f"Browser WebRTC: {'CONNECTED' if playing else 'WAITING'}\n"
            f"↓\n"
            f"Saaras STT: {agent.stt_status}\n"
            f"Transcript: {agent.last_user_text or '—'}\n"
            f"↓\n"
            f"Sarvam LLM: {agent.llm_status}\n"
            f"Response: {agent.last_assistant_text or '—'}\n"
            f"↓\n"
            f"BakBak TTS: {agent.tts_status}\n"
            f"Audio bytes: {agent.tts_audio_bytes:,}\n"
            f"↓\n"
            f"WebRTC output bytes: {agent.output_bytes_pushed:,}",
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
    st.info("Click Start voice agent, then allow microphone access.")

st.divider()
st.subheader("Streamlit secrets")
st.code(
    'SARVAM_API_KEY="..."\n'
    'BAKBAK_API_KEY="..."\n'
    'BAKBAK_VOICE_ID="..."\n'
    'HF_TOKEN="hf_..."',
    language="toml",
)
st.caption("HF_TOKEN is used only to obtain temporary TURN credentials. Never commit it to GitHub.")
