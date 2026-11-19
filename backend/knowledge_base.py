from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

import faiss
import httpx
import numpy as np

AGENTS = {"general", "customer_service", "tutor"}
DATA_ROOT = Path(os.getenv("AURA_KB_DATA_DIR", "data/knowledge_base"))
EMBEDDING_URL = os.getenv("AURA_EMBEDDING_API_URL", "http://100.95.233.66:8080/embed").strip()
EMBEDDING_MODEL = "granite-embedding-97m-multilingual-r2"
DIMENSIONS = 384
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
EMBED_BATCH_SIZE = 16
_LOCKS = {agent: threading.RLock() for agent in AGENTS}


def _agent_id(agent_id: str) -> str:
    normalized = (agent_id or "").strip().lower()
    if normalized not in AGENTS:
        raise ValueError(f"Unknown knowledge-base agent: {agent_id}")
    return normalized


def _paths(agent_id: str) -> tuple[Path, Path]:
    directory = DATA_ROOT / _agent_id(agent_id)
    return directory / "knowledge.sqlite3", directory / "knowledge.faiss"


def _connect(agent_id: str) -> sqlite3.Connection:
    db_path, _ = _paths(agent_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL UNIQUE,
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


def _embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, DIMENSIONS), dtype="float32")
    key = os.getenv("AURA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("AURA_API_KEY is not configured on the backend")
    vectors: list[list[float]] = []
    with httpx.Client(timeout=90) as client:
        for offset in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[offset:offset + EMBED_BATCH_SIZE]
            response = client.post(
                EMBEDDING_URL,
                headers={"X-API-Key": key},
                json={"texts": batch},
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("dimension") != DIMENSIONS:
                raise ValueError(
                    f"Granite embedding dimension mismatch: expected {DIMENSIONS}, "
                    f"received {payload.get('dimension')}"
                )
            batch_vectors = payload.get("embeddings")
            if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
                raise ValueError("Granite embedding response count does not match request")
            vectors.extend(batch_vectors)
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.shape != (len(texts), DIMENSIONS):
        raise ValueError(f"Invalid Granite embedding matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Granite returned non-finite embedding values")
    faiss.normalize_L2(matrix)
    return matrix


def _chunk_text(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean = re.sub(r"\s+", " ", content).strip()
    if not clean:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        pieces.append(clean[start:end])
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return pieces


def _build_index(conn: sqlite3.Connection):
    rows = conn.execute("SELECT vector FROM chunks ORDER BY id").fetchall()
    index = faiss.IndexFlatIP(DIMENSIONS)
    if rows:
        vectors = np.vstack([np.frombuffer(row["vector"], dtype="float32") for row in rows])
        if vectors.shape[1] != DIMENSIONS:
            raise ValueError("Stored vector dimensions do not match the Granite index")
        index.add(vectors)
    return index


def _load_index(agent_id: str, conn: sqlite3.Connection):
    _, index_path = _paths(agent_id)
    expected = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            if index.d == DIMENSIONS and index.ntotal == expected:
                return index
        except Exception:
            pass
    index = _build_index(conn)
    temporary = index_path.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, index_path)
    return index


def status(agent_id: str) -> dict[str, Any]:
    agent = _agent_id(agent_id)
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            docs = conn.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            enabled = conn.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()["value"] == "true"
            return {
                "agent_id": agent, "enabled": enabled, "documents": docs, "chunks": chunks,
                "embedding_model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
                "database": str(_paths(agent)[0]), "index": str(_paths(agent)[1]),
            }
        finally:
            conn.close()


def set_enabled(agent_id: str, enabled: bool) -> dict[str, Any]:
    agent = _agent_id(agent_id)
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            conn.execute(
                "UPDATE settings SET value=? WHERE key='enabled'",
                ("true" if enabled else "false",),
            )
            conn.commit()
        finally:
            conn.close()
    return status(agent)


def list_documents(agent_id: str) -> list[dict[str, Any]]:
    agent = _agent_id(agent_id)
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            rows = conn.execute("""SELECT d.id,d.title,d.source,d.created_at,COUNT(c.id) chunk_count
                FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at DESC""").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def list_chunks(agent_id: str, document_id: str) -> list[dict[str, Any]]:
    agent = _agent_id(agent_id)
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            rows = conn.execute(
                "SELECT id,chunk_index,content FROM chunks WHERE document_id=? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def ingest_text(
    agent_id: str,
    title: str,
    content: str,
    source: str = "owner-ingested",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict[str, Any]:
    agent = _agent_id(agent_id)
    pieces = _chunk_text(content, chunk_size, overlap)
    if not pieces:
        raise ValueError("Document content is empty")
    vectors = _embed(pieces)
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            existing = conn.execute("SELECT id FROM documents WHERE source=?", (source,)).fetchone()
            if existing:
                document_id = existing["id"]
                conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            else:
                document_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO documents(id,title,source) VALUES(?,?,?)",
                (document_id, title, source),
            )
            conn.executemany(
                "INSERT INTO chunks(document_id,chunk_index,content,vector) VALUES(?,?,?,?)",
                [(document_id, i, piece, vectors[i].tobytes()) for i, piece in enumerate(pieces)],
            )
            conn.commit()
            index = _build_index(conn)
            _, index_path = _paths(agent)
            temporary = index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(temporary))
            os.replace(temporary, index_path)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return {"id": document_id, "agent_id": agent, "title": title, "chunks": len(pieces)}


def retrieve(agent_id: str, query: str, top_k: int = 4) -> list[dict[str, Any]]:
    agent = _agent_id(agent_id)
    if not query or not query.strip():
        return []
    # Avoid a paid/remote embedding request when the KB is disabled or empty.
    # This is especially important for agents such as "general" that have no documents.
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            enabled = conn.execute(
                "SELECT value FROM settings WHERE key='enabled'"
            ).fetchone()["value"] == "true"
            if not enabled:
                return []
            chunk_count = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks"
            ).fetchone()["n"]
            if chunk_count == 0:
                return []
        finally:
            conn.close()
    vector = _embed([query])
    with _LOCKS[agent]:
        conn = _connect(agent)
        try:
            index = _load_index(agent, conn)
            if index.ntotal == 0:
                return []
            scores, ids = index.search(vector, min(max(1, top_k), index.ntotal))
            rows = conn.execute("""SELECT c.id,c.content,d.title,d.source
                FROM chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id""").fetchall()
            return [
                {"score": float(score), "content": rows[int(idx)]["content"],
                 "title": rows[int(idx)]["title"], "source": rows[int(idx)]["source"]}
                for score, idx in zip(scores[0], ids[0]) if idx >= 0
            ]
        finally:
            conn.close()
