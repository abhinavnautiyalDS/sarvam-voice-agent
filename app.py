import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_webrtc.credentials import get_cloudflare_ice_servers

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

    # Modern streamlit-webrtc uses separate frontend/server ICE configs.
    # Cloudflare TURN is used when its credentials are present in Streamlit
    # secrets; otherwise Google STUN is used as a fallback.
    cloudflare_servers = []
    if (
        "CLOUDFLARE_TURN_KEY_ID" in st.secrets
        and "CLOUDFLARE_TURN_KEY_API_TOKEN" in st.secrets
    ):
        try:
            cloudflare_servers = get_cloudflare_ice_servers(
                turn_key_id=st.secrets["CLOUDFLARE_TURN_KEY_ID"],
                turn_key_api_token=st.secrets["CLOUDFLARE_TURN_KEY_API_TOKEN"],
            )
        except Exception as exc:
            st.warning(f"Cloudflare TURN credentials could not be loaded: {exc}")

    ice_servers = cloudflare_servers or [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
    rtc_config = {"iceServers": ice_servers}

    webrtc_streamer(
        key="sarvam-voice-v2",
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

    if cloudflare_servers:
        st.caption("WebRTC ICE: Cloudflare STUN/TURN")
    else:
        st.caption(
            "WebRTC ICE: Google STUN fallback. Add Cloudflare TURN secrets "
            "below if the connection does not establish."
        )

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
st.subheader("Required Streamlit secrets")
st.code(
    'SARVAM_API_KEY="..."\n'
    'BAKBAK_API_KEY="..."\n'
    'BAKBAK_VOICE_ID="..."\n'
    '\n'
    '# Recommended for Streamlit Cloud WebRTC\n'
    'CLOUDFLARE_TURN_KEY_ID="..."\n'
    'CLOUDFLARE_TURN_KEY_API_TOKEN="..."',
    language="toml",
)
st.caption(
    "Cloudflare TURN is recommended for remote WebRTC deployments. "
    "Never commit these secrets to GitHub."
)
