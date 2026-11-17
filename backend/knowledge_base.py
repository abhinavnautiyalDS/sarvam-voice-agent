from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np

DATA_DIR = Path(os.getenv("AURA_KB_DATA_DIR", "data/knowledge_base"))
DB_PATH = DATA_DIR / "knowledge.sqlite3"
INDEX_PATH = DATA_DIR / "knowledge.faiss"
EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
_LOCK = threading.RLock()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL, content TEXT NOT NULL, vector BLOB NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT NOT NULL
    )""")
    conn.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('enabled','false')")
    conn.commit()
    return conn


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the backend")
    payload = {"requests": [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text}]},
        "taskType": task_type, "outputDimensionality": DIMENSIONS} for text in texts]}
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents",
        params={"key": key}, json=payload, timeout=90,
    )
    response.raise_for_status()
    matrix = np.asarray([item["values"] for item in response.json()["embeddings"]], dtype="float32")
    if matrix.ndim != 2 or matrix.shape[1] != DIMENSIONS:
        raise ValueError(f"Embedding dimension mismatch; expected {DIMENSIONS}")
    faiss.normalize_L2(matrix)
    return matrix


def _build_index(conn: sqlite3.Connection):
    rows = conn.execute("SELECT vector FROM chunks ORDER BY id").fetchall()
    index = faiss.IndexFlatIP(DIMENSIONS)
    if rows:
        index.add(np.vstack([np.frombuffer(row["vector"], dtype="float32") for row in rows]))
    return index


def status() -> dict[str, Any]:
    with _LOCK:
        conn = _connect()
        try:
            docs = conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            enabled = conn.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()["value"] == "true"
            return {"enabled": enabled, "documents": docs, "chunks": chunks,
                    "embedding_model": EMBED_MODEL, "dimensions": DIMENSIONS}
        finally:
            conn.close()


def set_enabled(enabled: bool) -> dict[str, Any]:
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("UPDATE settings SET value=? WHERE key='enabled'", ("true" if enabled else "false",))
            conn.commit()
        finally:
            conn.close()
    return status()


def list_documents() -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute("""SELECT d.id,d.title,d.source,d.created_at,COUNT(c.id) chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at DESC""").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def list_chunks(document_id: str) -> list[dict[str, Any]]:
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute("SELECT id,chunk_index,content FROM chunks WHERE document_id=? ORDER BY chunk_index", (document_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def ingest_text(title: str, content: str, source: str = "owner-ingested", chunk_size: int = 1200, overlap: int = 180) -> dict[str, Any]:
    clean = " ".join(content.split())
    if not clean:
        raise ValueError("Document content is empty")
    pieces, start = [], 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        pieces.append(clean[start:end])
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    batches = [_embed(pieces[i:i + 64], "RETRIEVAL_DOCUMENT") for i in range(0, len(pieces), 64)]
    vectors = np.vstack(batches)
    document_id = uuid.uuid4().hex
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("INSERT INTO documents(id,title,source) VALUES(?,?,?)", (document_id, title, source))
            for i, piece in enumerate(pieces):
                conn.execute("INSERT INTO chunks(document_id,chunk_index,content,vector) VALUES(?,?,?,?)",
                             (document_id, i, piece, vectors[i].tobytes()))
            conn.commit()
            index = _build_index(conn)
            temp_path = INDEX_PATH.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(temp_path))
            os.replace(temp_path, INDEX_PATH)
        finally:
            conn.close()
    return {"id": document_id, "title": title, "chunks": len(pieces)}


def retrieve(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    if not query.strip() or not status()["enabled"]:
        return []
    vector = _embed([query], "RETRIEVAL_QUERY")
    with _LOCK:
        conn = _connect()
        try:
            index = _build_index(conn)
            if index.ntotal == 0:
                return []
            scores, ids = index.search(vector, min(max(1, top_k), index.ntotal))
            rows = conn.execute("""SELECT c.id,c.content,d.title,d.source FROM chunks c
                JOIN documents d ON d.id=c.document_id ORDER BY c.id""").fetchall()
            return [{"score": float(score), "content": rows[int(idx)]["content"],
                     "title": rows[int(idx)]["title"], "source": rows[int(idx)]["source"]}
                    for score, idx in zip(scores[0], ids[0]) if idx >= 0]
        finally:
            conn.close()
