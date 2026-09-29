import os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_webrtc.credentials import get_hf_ice_servers

from cloud_voice import CloudVoiceAgent


st.set_page_config(page_title="Sarvam Voice Agent", page_icon="🎤")
st.title("🎤 Sarvam Voice Agent")
st.caption("Browser Mic → Saaras Realtime → Sarvam 105B → BakBak TTS")


if "cloud_agent" not in st.session_state:
    st.session_state.cloud_agent = None
if "running" not in st.session_state:
    st.session_state.running = False


if st.button(
    "Start voice agent",
    type="primary",
    disabled=st.session_state.running,
    use_container_width=True,
):
    try:
        agent = CloudVoiceAgent(
            st.secrets["SARVAM_API_KEY"],
            st.secrets["BAKBAK_API_KEY"],
            st.secrets["BAKBAK_VOICE_ID"],
        )
        agent.start()
        st.session_state.cloud_agent = agent
        st.session_state.running = True
        st.rerun()
    except Exception as exc:
        st.error(f"Could not start: {exc}")


agent = st.session_state.cloud_agent

if st.session_state.running and agent is not None:
    st.success("Agent is running. Allow microphone access, then speak Hindi/Hinglish.")

    def audio_callback(frame):
        agent.ingest_webrtc_frame(frame)
        return agent.get_output_frame(frame)

    # Hugging Face provides TURN credentials without requiring the
    # Cloudflare TURN billing setup. The token stays server-side in
    # Streamlit secrets.
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    ice_servers = []

    if hf_token:
        try:
            ice_servers = get_hf_ice_servers(hf_token)
            # Add Google's public STUN server as an additional candidate.
            ice_servers.append({"urls": "stun:stun.l.google.com:19302"})
        except Exception as exc:
            st.warning(f"Could not load Hugging Face TURN credentials: {exc}")

    if not ice_servers:
        ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

    rtc_config = {"iceServers": ice_servers}

    webrtc_streamer(
        key="sarvam-voice-hf",
        mode=WebRtcMode.SENDRECV,
        audio_frame_callback=audio_callback,
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
        async_processing=True,
    )

    if hf_token and ice_servers:
        st.caption("WebRTC ICE: Hugging Face TURN + Google STUN")
    else:
        st.caption("WebRTC ICE: Google STUN fallback")

    if agent.error:
        st.error(f"Voice pipeline error: {agent.error}")

    if agent.last_user_text:
        st.write(f"**You:** {agent.last_user_text}")
    if agent.last_assistant_text:
        st.write(f"**Agent:** {agent.last_assistant_text}")

    st.caption(
        "Allow microphone access when the browser asks. "
        "Use headphones if you hear echo."
    )

    if st.button("Stop voice agent", use_container_width=True):
        agent.stop()
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
st.caption(
    "HF_TOKEN is used only by the server to obtain temporary TURN credentials. "
    "Never commit it to GitHub."
)
