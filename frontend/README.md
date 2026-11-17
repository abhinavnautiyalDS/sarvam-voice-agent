# AURA frontend

React + TypeScript + Vite interface for the AURA Multilingual Voice AI Workbench.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

## Current UI scope

- Workbench dashboard with an end-to-end voice pipeline: microphone → Sarvam Saaras STT → custom BERT-Tiny language classifier → knowledge retrieval (multilingual embeddings + FAISS) → Sarvam Gamma LLM → BakBak Raya TTS.
- Per-turn inspector for transcription, language/script prediction, retrieved context, generated answer, and TTS output.
- Conversation history with selectable turns and a one-sentence summary preview.
- Knowledge Base attachment interface for PDF, DOCX, and TXT files.
- Pipeline/model architecture, evaluation placeholder, and integration settings views.

## Important integration boundary

This is currently a **frontend UI preview**, not a connected live voice product.

- The microphone control only toggles a local UI state. It does not capture or stream audio in this version.
- Conversation examples and processing outputs are illustrative seed data. Typed playground messages are labelled as UI preview responses and do not call an LLM.
- File selection updates the local document list only. It does not upload, parse, chunk, embed, or index documents.
- No classifier inference, RAG retrieval, STT, LLM, TTS, session persistence, or measured latency is performed by the frontend.

## Backend integration plan

1. Add an authenticated session/signalling endpoint and connect browser audio through a supported Pipecat WebRTC transport.
2. Emit structured per-turn events for STT transcription, classifier label/confidence, retrieval references, LLM text, TTS audio, and stage timings.
3. Add document upload and ingestion endpoints. Keep extraction, chunking, multilingual-e5-small embeddings, and FAISS indexing on the backend.
4. Persist full transcript turns and a rolling one-sentence conversation summary separately. Send the summary plus a bounded recent-turn window as conversational context.
5. Store all Sarvam/BakBak provider credentials server-side. Never expose provider keys in Vite frontend variables.

Do not publish sample UI values as measured model accuracy, confidence, latency, or production session telemetry.
