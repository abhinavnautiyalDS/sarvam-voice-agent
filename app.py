import streamlit as st

st.set_page_config(page_title="Sarvam Voice Agent", page_icon="🎤")
st.title("🎤 Sarvam Voice Agent")
st.write("Saaras v3 Realtime STT → Sarvam 105B → BakBak TTS")

st.info(
    "The local Pipecat agent is in voice_agent.py. It uses PyAudio. "
    "For Streamlit Cloud, browser microphone input needs a WebRTC bridge."
)

st.subheader("Local run")
st.code("python voice_agent.py", language="bash")

st.subheader("Cloud architecture")
st.write(
    "Browser microphone → WebRTC → Streamlit Cloud → Saaras Realtime STT "
    "→ Sarvam 105B → BakBak TTS → Browser speaker"
)

st.subheader("Streamlit secrets")
st.code(
    'SARVAM_API_KEY="..."\nBAKBAK_API_KEY="..."\nBAKBAK_VOICE_ID="..."',
    language="toml",
)
