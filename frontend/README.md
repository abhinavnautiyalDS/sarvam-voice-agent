# AURA frontend

React + TypeScript + Vite voice-playground interface.

## Run locally

    cd frontend
    npm install
    npm run dev

## Production build

    npm run build

## Backend integration boundary

The Start button requests browser microphone permission and manages the local media track. It does not yet stream audio to the existing Python/Streamlit voice pipeline. Diagnostics intentionally report the transport as not connected instead of fabricating live STT/LLM/TTS metrics.

Next: expose a backend session/signalling API and connect browser audio through a supported WebRTC transport. Keep Sarvam, BakBak and other provider credentials server-side; never place them in Vite client variables.