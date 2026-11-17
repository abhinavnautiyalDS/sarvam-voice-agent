"""Owner-managed, persistent FAISS knowledge base backed by Gemini embeddings.

The service accepts curated text or pre-chunked passages. It never treats the
voice system prompt as factual source material. All writes are admin-only at
the API layer; runtime callers can only perform retrieval.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np

DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIMENSIONS = 768
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180


class KnowledgeBaseError(RuntimeError):
    pass


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, dimensions: int = DEFAULT_DIMENSIONS):
        if not api_key:
            raise KnowledgeBaseError("GEMINI_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str], task: str) -> np.ndarray:
        """Embed each passage separately so every input retains its own vector."""
        vectors: list[list[float]] = []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=45) as client:
            for text in texts:
                prefix = "task: search_document\n" if task == "document" else "task: search_query\n"
                payload = {
                    "content": {"parts": [{"text": prefix + text}]},
                    "outputDimensionality": self.dimensions,
                }
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                values = data.get("embedding", {}).get("values")
                if not isinstance(values, list) or len(values) != self.dimensions:
                    raise KnowledgeBaseError("Embedding API returned an unexpected vector shape")
                vectors.append(values)
        matrix = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Conservative character chunking that prefers paragraph/sentence boundaries."""
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()
    if not normalized:
        return []
    if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/overlap configuration")
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("\n\n", start, end), normalized.rfind(". ", start, end), normalized.rfind("? ", start, end), normalized.rfind("! ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + (2 if normalized[boundary:boundary + 2] in {". ", "? ", "! "} else 0)
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


class KnowledgeBase:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "knowledge.sqlite3"
        self.index_path = self.data_dir / "knowledge.faiss"
        self.config_path = self.data_dir / "settings.json"
        self.lock = threading.RLock()
        self._index: faiss.IndexIDMap2 | None = None
        self._load_index()

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
            category TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            chunk_count INTEGER NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            text TEXT NOT NULL, metadata_json TEXT NOT NULL, content_hash TEXT NOT NULL,
            vector BLOB NOT NULL, FOREIGN KEY(document_id) REFERENCES documents(id)
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        return db

    def _load_index(self):
        with self.lock:
            if self.index_path.exists():
                try:
                    self._index = faiss.read_index(str(self.index_path))
                    return
                except Exception:
                    # Rebuild from durable vectors in SQLite on a corrupt/missing index.
                    pass
            self._rebuild_index()

    def _rebuild_index(self):
        with self._connect() as db:
            rows = db.execute("SELECT id, vector FROM chunks ORDER BY rowid").fetchall()
        if not rows:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(DEFAULT_DIMENSIONS))
            self._persist_index()
            return
        vectors = np.vstack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows]).astype(np.float32)
        dim = int(vectors.shape[1])
        base = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        ids = np.asarray([int(uuid.UUID(row["id"]).int % (2**63 - 1)) for row in rows], dtype=np.int64)
        base.add_with_ids(vectors, ids)
        self._index = base
        self._persist_index()

    def _persist_index(self):
        if self._index is None:
            return
        fd, tmp_name = tempfile.mkstemp(prefix="faiss-", suffix=".tmp", dir=self.data_dir)
        os.close(fd)
        try:
            faiss.write_index(self._index, tmp_name)
            os.replace(tmp_name, self.index_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def settings(self) -> dict[str, Any]:
        defaults = {"enabled": False, "updated_at": None}
        try:
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                defaults.update(saved)
        except (OSError, json.JSONDecodeError):
            pass
        return defaults

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        state = {"enabled": bool(enabled), "updated_at": time.time()}
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, self.config_path)
        return state

    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            docs = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            **self.settings(), "documents": docs, "chunks": chunks,
            "index_ready": self._index is not None,
            "embedding_model": os.getenv("AURA_KB_EMBEDDING_MODEL", DEFAULT_MODEL),
            "embedding_dimensions": int(os.getenv("AURA_KB_EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS))),
        }

    async def add_document(
        self, *, title: str, source: str, category: str = "general",
        content: str | None = None, chunks: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None, embedder: GeminiEmbedder,
    ) -> dict[str, Any]:
        if chunks is not None:
            passages = [str(item.get("text", "")).strip() for item in chunks if isinstance(item, dict)]
            chunk_meta = [item.get("metadata", {}) if isinstance(item, dict) else {} for item in chunks]
            if len(passages) != len(chunk_meta):
                raise KnowledgeBaseError("Invalid pre-chunked payload")
        else:
            passages = chunk_text(content or "")
            chunk_meta = [{} for _ in passages]
        if not passages or any(not item for item in passages):
            raise KnowledgeBaseError("Document must contain at least one non-empty chunk")
        if len(passages) > 5000:
            raise KnowledgeBaseError("A single document is limited to 5,000 chunks")
        if any(len(item) > 12000 for item in passages):
            raise KnowledgeBaseError("Each chunk must be at most 12,000 characters")
        vectors = await embedder.embed(passages, task="document")
        document_id = str(uuid.uuid4())
        now = time.time()
        chunk_rows = []
        for i, (text, meta, vector) in enumerate(zip(passages, chunk_meta, vectors)):
            chunk_id = str(uuid.uuid4())
            safe_meta = meta if isinstance(meta, dict) else {}
            safe_meta.update(metadata or {})
            chunk_rows.append((chunk_id, document_id, i, text, json.dumps(safe_meta, ensure_ascii=False), hashlib.sha256(text.encode()).hexdigest(), vector.tobytes()))
        with self.lock:
            with self._connect() as db:
                db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", (document_id, title.strip(), source.strip(), category.strip(), now, now, len(chunk_rows)))
                db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", chunk_rows)
            self._rebuild_index()
        return {"document_id": document_id, "title": title, "chunks_created": len(chunk_rows), "status": "indexed"}

    def delete_document(self, document_id: str) -> bool:
        with self.lock:
            with self._connect() as db:
                exists = db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone()
                if not exists:
                    return False
                db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            self._rebuild_index()
        return True

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id, title, source, category, created_at, updated_at, chunk_count FROM documents ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, document_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id, ordinal, text, metadata_json FROM chunks WHERE document_id=? ORDER BY ordinal LIMIT ?", (document_id, min(max(limit, 1), 500))).fetchall()
        return [{"id": r["id"], "ordinal": r["ordinal"], "text": r["text"], "metadata": json.loads(r["metadata_json"])} for r in rows]

    async def retrieve(self, query: str, embedder: GeminiEmbedder, top_k: int = 4, threshold: float = 0.42) -> list[dict[str, Any]]:
        if not query.strip() or self._index is None or self._index.ntotal == 0:
            return []
        vector = await embedder.embed([query], task="query")
        with self.lock:
            scores, ids = self._index.search(vector, min(max(top_k, 1), 10))
        wanted = [int(value) for value in ids[0] if value >= 0]
        if not wanted:
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.id, c.document_id, c.ordinal, c.text, c.metadata_json, d.title, d.source, d.category "
                "FROM chunks c JOIN documents d ON d.id=c.document_id"
            ).fetchall()
        lookup = {int(uuid.UUID(row["id"]).int % (2**63 - 1)): row for row in rows}
        results = []
        for score, chunk_numeric_id in zip(scores[0], ids[0]):
            if chunk_numeric_id < 0 or float(score) < threshold:
                continue
            row = lookup.get(int(chunk_numeric_id))
            if row:
                results.append({
                    "chunk_id": row["id"], "document_id": row["document_id"],
                    "text": row["text"], "score": round(float(score), 4),
                    "source": row["source"], "title": row["title"],
                    "category": row["category"], "metadata": json.loads(row["metadata_json"]),
                })
        return results
