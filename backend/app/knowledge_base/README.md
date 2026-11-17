# AURA Knowledge Base

Owner-managed retrieval for the AURA voice agent. This package stores curated passages in SQLite and a persistent FAISS inner-product index. Gemini Embedding 2 creates normalized 768-dimensional vectors through the Gemini API; the embedding key remains server-side.

## Data flow

`Owner content → validation → chunking (or preserve supplied chunks) → Gemini embeddings → SQLite metadata/vector backup + FAISS index → selective runtime retrieval → LLM context`

The voice system prompt is policy, not a factual knowledge source. It tells the model to use verified KB facts for university-specific claims; it must never be ingested as factual content.

## Input contract

Pre-chunked JSON:
```json
{
  "title": "Admissions FAQ",
  "source": "Admissions handbook, 2026",
  "category": "admissions",
  "chunks": [
    {
      "text": "The verified passage goes here.",
      "metadata": {"page": 4, "section": "Eligibility"}
    }
  ]
}
```

Or raw text:
```json
{
  "title": "Admissions FAQ",
  "source": "Admissions handbook, 2026",
  "category": "admissions",
  "content": "Paste verified source text here."
}
```

Pre-chunked passages are preserved as supplied. Raw text is split on paragraph/sentence boundaries with configurable character size and overlap. Do not upload synthetic or guessed university facts. Record source/version metadata and replace stale material when policies change.

## Environment

- `GEMINI_API_KEY`: required for embedding.
- `AURA_KB_DATA_DIR`: persistent directory; default `./data/knowledge_base`.
- `AURA_KB_EMBEDDING_MODEL`: default `gemini-embedding-2`.
- `AURA_KB_EMBEDDING_DIMENSIONS`: default `768`; changing model/dimensions requires a complete re-embed.
- `AURA_KB_TOP_K`: default `4`.
- `AURA_KB_SCORE_THRESHOLD`: default `0.42`; calibrate on real queries before production.

## Safety and runtime behavior

- Admin routes are protected by the existing `AURA_ADMIN_ALLOWED_IPS` allowlist.
- Public voice WebSocket has no document-write or KB-toggle operation.
- KB defaults to OFF until an admin explicitly enables it.
- OFF means no vector retrieval. The LLM is still instructed not to guess university-specific facts.
- Retrieval is mandatory for organisation-specific factual categories and fallback for uncertain general questions.
- Retrieval scores are cosine similarities because vectors are L2-normalized and FAISS uses inner product. Thresholds are empirical, not probabilities.
- If retrieval returns no sufficiently relevant passage, the assistant must say verified information is unavailable and offer a counsellor.
- SQLite retains embedding vectors to permit index reconstruction if the FAISS file is lost.
