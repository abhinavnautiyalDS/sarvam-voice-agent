# AURA Backend

FastAPI voice API and owner-managed knowledge-base service.

## Run from repository root

```bash
python -m pip install -r backend/requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

## Package entrypoint

From `backend/` run `uvicorn app.main:app --host 127.0.0.1 --port 8000`.

Runtime data, credentials and model caches stay outside version control. See the repository-level `BACKEND.md` for API and deployment details.
