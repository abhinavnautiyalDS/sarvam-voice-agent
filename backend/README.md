# AURA backend

The FastAPI implementation and voice pipeline modules live in this package. The root `api.py` remains a compatibility entry point for the existing systemd command `uvicorn api:app`.

## Knowledge Base

- Gemini Embedding API generates document/query vectors. Set `GEMINI_API_KEY` in the server environment only.
- FAISS performs vector similarity search.
- SQLite persists document metadata, chunk text, and vectors.
- KB starts disabled. Public endpoints are read-only: status, document list, and per-document chunks.
- The enable/disable endpoint is protected by the existing server IP allowlist.
- There are intentionally no browser upload, attach-document, or re-index controls.

Owner-side ingestion (run from the repository root after installing `requirements-api.txt`):

```bash
.venv/bin/python -m backend.ingest_kb path/to/knowledge.md
```

Supported inputs are UTF-8 `.txt` and `.md`. Data is stored under `AURA_KB_DATA_DIR` (default `data/knowledge_base`) and should be backed up separately from Git.
