# AURA backend (FastAPI + WebSocket)

The backend implementation is organized under `backend/`. The root `api.py` is retained as a compatibility entry point for the EC2 systemd command `uvicorn api:app`.

## Endpoints

- `GET /api/health` — service health.
- `WS /ws/voice` — browser audio session (PCM16 mono 16 kHz in, PCM16 mono 24 kHz out).
- `GET /api/knowledge/status` — read-only KB status.
- `GET /api/knowledge/documents` — read-only indexed document inventory.
- `GET /api/knowledge/documents/{id}/chunks` — read-only chunk content.
- `PUT /api/admin/knowledge/enabled` — owner IP-allowlisted KB retrieval switch.

## Knowledge Base

Gemini's hosted embedding API creates vectors. FAISS performs similarity search. SQLite stores document metadata, chunk text, and vectors. The KB is disabled by default. The UI exposes browsing only—no document upload, attach, or re-index controls.

Set `GEMINI_API_KEY` on the server (never in frontend variables). Owner-side ingestion:

```bash
.venv/bin/python -m backend.ingest_kb path/to/knowledge.md
```

Only UTF-8 `.txt` and `.md` files are accepted by the CLI. Data is stored in `data/knowledge_base` by default and is excluded from Git.

## EC2

Install dependencies with `.venv/bin/python -m pip install -r requirements-api.txt`. Keep port 8000 private behind Nginx HTTPS/WSS. Do not deploy or restart the service until the frontend build and backend smoke tests pass.
