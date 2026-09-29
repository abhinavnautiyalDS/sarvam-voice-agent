import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

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

    webrtc_streamer(
        key="sarvam-voice",
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
        # A remote Streamlit deployment needs ICE servers for WebRTC
        # NAT traversal. STUN is enough for many networks; TURN may still
        # be required on restrictive corporate/mobile networks.
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
        async_processing=True,
    )

    if agent.error:
        st.error(f"Voice pipeline error: {agent.error}")

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
st.subheader("Required Streamlit secrets")
st.code(
    'SARVAM_API_KEY="..."\n'
    'BAKBAK_API_KEY="..."\n'
    'BAKBAK_VOICE_ID="..."',
    language="toml",
)
st.caption(
    "The local voice_agent.py remains available for PyAudio testing. "
    "This app uses browser WebRTC instead."
)
