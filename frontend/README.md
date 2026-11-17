# AURA frontend

React + TypeScript + Vite interface for the AURA Multilingual Voice AI Workbench.

## Run locally

\`\`\`bash
cd frontend
npm install
npm run dev
\`\`\`

## UI scope

- Voice-first playground with a connected execution-trace concept.
- Per-turn inspector: STT transcription, custom BERT-Tiny language prediction, selective knowledge decision, LLM answer and TTS status.
- Conversation history and one-sentence session summary.
- Read-only Knowledge Base inventory view. There is intentionally no public upload control.
- Dedicated architecture, evaluation and settings views.

## Knowledge Base operating model

The Knowledge Base is **owner-managed and pre-indexed once**, outside the public user experience.

1. The project owner ingests/replaces source documents through a private script or admin-only workflow.
2. The ingestion job extracts text, chunks it, creates multilingual-e5-small embeddings, and builds/updates the FAISS index.
3. The public frontend can only call read-only endpoints to show approved source metadata and safe content previews.
4. Public users cannot upload documents, trigger embeddings, rebuild the index, or spend ingestion credits.

The current UI contains no actual KB inventory because the backend inventory endpoint is not connected. Do not display fabricated filenames, snippets, counts, or indexing statuses.

## Selective retrieval policy

Do **not** run vector search on every user turn.

Recommended runtime routing:
- General conversational/factual questions that the model can answer directly: skip KB retrieval.
- Questions about this project's private/domain-specific information, explicit references to stored material, or queries flagged as uncertain by a lightweight routing policy: search the existing FAISS index.
- If retrieval returns relevant passages, pass only the top relevant chunks to the LLM.
- If nothing relevant is found, answer from general model knowledge when appropriate and be transparent that the KB did not contain supporting material.

A language classifier predicts language/script; it is not the KB router. Keep these decisions separate.

Important nuance: a model cannot reliably expose a complete list of everything it does not know. Avoid claiming perfect “LLM knows / does not know” detection. Use explicit KB-intent/domain rules and a calibrated lightweight relevance/uncertainty gate. If you add an LLM-based verification call, measure its extra latency and token cost first.

## Backend integration boundary

This is still a **frontend UI preview**, not a connected live voice product:
- Microphone control only toggles local UI state; it does not stream audio.
- Conversation examples are illustrative; typed messages do not call an LLM.
- KB inventory is an empty connection placeholder; no real indexed content is exposed yet.
- No classifier inference, retrieval, STT, LLM, TTS, persistence, or measured telemetry runs in the frontend.

## Backend endpoints to add

- \`GET /api/knowledge/manifest\`: read-only list of indexed sources (id, title, type, description, chunk_count, updated_at).
- \`GET /api/knowledge/sources/{id}/preview\`: return approved excerpts only; never expose private raw files by default.
- \`POST /api/admin/knowledge/ingest\`: private/admin-only ingestion endpoint, or preferably run ingestion offline from a secured script.
- Voice session events should include per-turn stage outputs: transcription, classifier label/confidence, retrieval decision and citations, LLM text, TTS status, timings, and summary update.

Keep all provider credentials and ingestion permissions server-side. Never expose API keys in Vite frontend variables.
