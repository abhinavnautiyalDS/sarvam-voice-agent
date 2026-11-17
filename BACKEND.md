# AURA backend (FastAPI + WebSocket)

The backend implementation is organized under `backend/app/`:

- `main.py` — FastAPI routes and WebSocket entrypoint
- `cloud_voice.py` — voice session orchestration
- `knowledge_base/` — ingestion, embeddings and FAISS retrieval
- `script_classifier.py` — multilingual language/script classifier
- `admin_config.py` — runtime admin configuration
- `system_prompt.py` and `system_prompt.txt` — voice-agent policy

Root-level Python modules are thin compatibility shims for older imports and the existing EC2 command `uvicorn api:app`. New development should target `backend/app/`.

# AURA backend (FastAPI + WebSocket)

AURA exposes the voice pipeline and an owner-managed knowledge base. Keep the API on loopback behind Nginx/HTTPS/WSS; never expose port 8000 directly.

## Voice endpoints

- `GET /api/health` — non-sensitive service health.
- `WS /ws/voice` — bidirectional voice session.
  - Client sends mono signed PCM16 audio at **16 kHz**.
  - Server sends JSON state events and mono signed PCM16 TTS audio at **24 kHz**.
  - The server loads the local 12-label BERT-Tiny checkpoint using classifier variant `4.4`.

## Owner-only knowledge endpoints

All `/api/admin/knowledge/*` routes use the existing `AURA_ADMIN_ALLOWED_IPS` allowlist. They are not exposed on the public voice WebSocket.

- `GET /api/admin/knowledge/status` — enabled state, document/chunk counts, model and index readiness.
- `GET /api/admin/knowledge/documents` — indexed source manifest.
- `GET /api/admin/knowledge/documents/{document_id}/chunks` — inspect chunk text and metadata.
- `POST /api/admin/knowledge/documents` — ingest raw text or pre-chunked JSON.
- `DELETE /api/admin/knowledge/documents/{document_id}` — remove a stale source and rebuild the index.
- `PUT /api/admin/knowledge/enabled` — persist the KB ON/OFF state.
- `POST /api/admin/knowledge/rebuild-index` — rebuild FAISS from durable vectors stored in SQLite.

### Ingestion payload

Pre-chunked content is preserved as supplied:
```json
{
  "title": "Admissions FAQ",
  "source": "Official admissions handbook, 2026",
  "category": "admissions",
  "chunks": [
    {"text": "Verified source passage.", "metadata": {"page": 3, "section": "Eligibility"}}
  ]
}
```

Raw text can be supplied with `content` instead of `chunks`; the backend splits it into overlapping passages. Each document requires a meaningful title and source. Only upload content from authoritative, current sources. Use [the verified content template](knowledge_base/content_template.md) to prepare factual material before ingestion.

## Embeddings and vector index

- Provider: Gemini API.
- Model: `gemini-embedding-2`.
- Dimensions: 768 by default (configurable).
- Similarity: normalized vectors + FAISS inner product, equivalent to cosine similarity.
- Persistence: SQLite stores source/chunk metadata and vector backups; FAISS stores the searchable index.
- Changing embedding model or dimensions requires re-embedding all stored passages; vectors from different embedding spaces must never be mixed.

Set these server environment variables:
- `GEMINI_API_KEY` — embedding key; never send it to the browser.
- `AURA_KB_DATA_DIR` — default `./data/knowledge_base`; mount this directory on persistent storage.
- `AURA_KB_EMBEDDING_MODEL` — default `gemini-embedding-2`.
- `AURA_KB_EMBEDDING_DIMENSIONS` — default `768`.
- `AURA_KB_TOP_K` — default `4`.
- `AURA_KB_SCORE_THRESHOLD` — initial default `0.42`; calibrate against representative queries, not as a probability.

## Retrieval policy

The canonical `system_prompt.txt` says university facts must come only from verified KB information and explicitly forbids inventing fees, programmes, eligibility, dates, placements, or outcomes.

The runtime gate follows that policy:
1. Organisation-specific facts (admissions, courses/programmes, fees, eligibility, placements, deadlines, etc.) trigger retrieval before the LLM call.
2. General questions go directly to the LLM. If it cannot answer confidently, it emits an internal `[AURA_KB_REQUIRED]` marker; the backend retrieves and makes a second grounded LLM call.
3. If the KB is OFF, missing, unavailable, or has no sufficiently relevant passage for an organisation-specific question, the model is instructed not to guess and to offer a counsellor.
4. Retrieved passages are labelled with source/title/category and are treated as untrusted reference data, not instructions.

The system prompt is policy, not a factual source. Do not ingest it as knowledge-base content. Do not populate the index with invented sample university facts.

## EC2 setup

Use the dedicated lightweight requirements file:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-api.txt
cp .env.example .env
```

Add Sarvam, BakBak and Gemini credentials to the server-side `.env`. Restrict `.env` permissions. Keep `AURA_ADMIN_ALLOWED_IPS` set to trusted owner/admin IPs and `AURA_ALLOWED_ORIGINS` to the exact frontend origin. Keep the KB data directory on persistent disk and back it up securely.

## Important limitations

- The initial ingestion API accepts JSON text/pre-chunked passages; PDF/DOCX parsing and automated content authoring are not part of this backend foundation.
- Similarity thresholds are data-dependent and must be tuned using real queries and known relevant/non-relevant examples.
- The IP allowlist is a network-level admin guard, not a substitute for full user authentication if admin access is later exposed through a public multi-user product.
- The existing React KB panel still needs to be wired to these endpoints in a separate frontend change.
