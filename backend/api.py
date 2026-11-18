from __future__ import annotations

import asyncio
import os
import queue
import time

import httpx
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.cloud_voice import CloudVoiceAgent
from backend.admin_config import get_setting, public_config, save_config
from backend import knowledge_base

load_dotenv()


class QueueAudioSource:
    """Thread-safe PCM output queue compatible with CloudVoiceAgent."""
    def __init__(self):
        self.items: queue.Queue[bytes] = queue.Queue(maxsize=32)

    def push(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            self.items.put_nowait(pcm)
        except queue.Full:
            try:
                self.items.get_nowait()
            except queue.Empty:
                pass
            self.items.put_nowait(pcm)


def _required_env(name: str) -> str:
    value = get_setting(name, "")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_admin_ip(request: Request) -> None:
    allowed = {item.strip() for item in os.getenv("AURA_ADMIN_ALLOWED_IPS", "").split(",") if item.strip()}
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client_ip = forwarded or (request.client.host if request.client else "")
    if not allowed or client_ip not in allowed:
        raise HTTPException(status_code=403, detail="Admin access is not enabled for this IP. Configure AURA_ADMIN_ALLOWED_IPS on the server.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="AURA Voice API", version="1.0.0", lifespan=lifespan)
allowed_origins = [
    item.strip() for item in os.getenv("AURA_ALLOWED_ORIGINS", "*").split(",") if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "PUT", "POST"],
    allow_headers=["*"],
)


@app.get("/api/admin/config")
async def admin_get_config(request: Request):
    _require_admin_ip(request)
    return public_config()


@app.put("/api/admin/config")
async def admin_update_config(request: Request):
    _require_admin_ip(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")
    allowed = {"SARVAM_API_KEY", "BAKBAK_API_KEY", "BAKBAK_VOICE_ID", "SARVAM_LLM_MODEL", "BAKBAK_TTS_MODEL", "SARVAM_STT_MODE", "AURA_LLM_PROVIDER", "AURA_TTS_PROVIDER"}
    updates = {key: value.strip() for key, value in body.items() if key in allowed and isinstance(value, str)}
    if not updates:
        raise HTTPException(status_code=400, detail="No supported settings supplied")
    save_config(updates)
    return {"status": "saved", "config": public_config(), "note": "New voice sessions will use the updated settings."}



KNOWLEDGE_SOURCE_AGENTS = {
    "general": {
        "name": "General Assistant",
        "directory": "general",
    },
    "customer_service": {
        "name": "Admission Counsellor",
        "directory": "admission_counsellor",
    },
    "tutor": {
        "name": "AI Tutor",
        "directory": "ai_tutor",
    },
}


@app.get("/api/knowledge/sources")
async def knowledge_sources(agent: str = "general"):
    """Read the checked-in, agent-specific source documents for the KB browser."""
    selected = KNOWLEDGE_SOURCE_AGENTS.get(agent)
    if selected is None:
        raise HTTPException(status_code=400, detail="Unknown knowledge-base agent")
    source_dir = Path(__file__).resolve().parent / "knowledge_sources" / selected["directory"]
    documents = []
    if source_dir.exists():
        for path in sorted(source_dir.glob("*.md")):
            documents.append({
                "id": path.stem,
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "filename": path.name,
                "content": path.read_text(encoding="utf-8"),
                "source": "GitHub · checked-in knowledge source",
            })
    return {
        "agent_id": agent,
        "agent_name": selected["name"],
        "documents": documents,
    }


@app.get("/api/knowledge/status")
async def knowledge_status(agent: str = "general"):
    try:
        return knowledge_base.status(agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/knowledge/documents")
async def knowledge_documents(agent: str = "general"):
    try:
        return {"agent_id": agent, "documents": knowledge_base.list_documents(agent)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/knowledge/documents/{document_id}/chunks")
async def knowledge_document_chunks(document_id: str, agent: str = "general"):
    try:
        documents = knowledge_base.list_documents(agent)
        if document_id not in {item["id"] for item in documents}:
            raise HTTPException(status_code=404, detail="Knowledge document not found for this agent")
        return {"agent_id": agent, "chunks": knowledge_base.list_chunks(agent, document_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/admin/knowledge/enabled")
async def knowledge_enabled(request: Request, agent: str = "general"):
    _require_admin_ip(request)
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        raise HTTPException(status_code=400, detail="Expected {enabled: boolean}")
    try:
        return knowledge_base.set_enabled(agent, body["enabled"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/knowledge/index-sources")
async def knowledge_index_sources(request: Request, agent: str = "general"):
    """Index only the selected agent's checked-in Markdown sources into its private store."""
    _require_admin_ip(request)
    selected = KNOWLEDGE_SOURCE_AGENTS.get(agent)
    if selected is None:
        raise HTTPException(status_code=400, detail="Unknown knowledge-base agent")
    source_dir = Path(__file__).resolve().parent / "knowledge_sources" / selected["directory"]
    if not source_dir.exists():
        raise HTTPException(status_code=404, detail="No source directory exists for this agent")
    files = sorted(source_dir.glob("*.md"))
    if not files:
        raise HTTPException(status_code=404, detail="No Markdown source documents exist for this agent")
    indexed = []
    try:
        for path in files:
            result = knowledge_base.ingest_text(
                agent_id=agent,
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                content=path.read_text(encoding="utf-8"),
                source=f"github:{agent}:{path.name}",
            )
            indexed.append({"filename": path.name, **result})
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=f"Knowledge indexing failed: {type(exc).__name__}: {exc}") from exc
    return {
        "agent_id": agent,
        "agent_name": selected["name"],
        "indexed_documents": len(indexed),
        "documents": indexed,
        "status": knowledge_base.status(agent),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "aura-voice-api",
        "classifier": "local multilingual BERT-Tiny (12 labels)",
        "stt_mode": get_setting("SARVAM_STT_MODE", "verbose"),
    }


def _snapshot(agent: CloudVoiceAgent) -> dict:
    result = agent.last_classifier_result or {}
    label = result.get("label", "pending")
    return {
        "type": "state",
        "pipeline_status": agent.pipeline_status,
        "stt_status": agent.stt_status,
        "llm_status": agent.llm_status,
        "tts_status": agent.tts_status,
        "transcript": agent.live_transcript,
        "user_text": agent.last_user_text,
        "detected_language": label,
        "confidence": result.get("confidence"),
        "classifier_latency_ms": result.get("latency_ms"),
        "assistant_text": agent.last_assistant_text,
        "llm_latency_ms": agent.llm_latency_ms,
        "tts_latency_ms": agent.tts_latency_ms,
        "error": agent.error,
        "error_stage": agent.error_stage,
        "llm_provider": agent.llm_provider,
        "audio_frames_received": agent.audio_frames_received,
    }


@app.post("/api/chat")
async def text_chat(request: Request):
    """Text-only AURA reasoning path. Deliberately bypasses STT and TTS."""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")
    message = body.get("message")
    agent_type = body.get("agent", "general")
    llm_provider = body.get("llm_provider", "qwen")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="message must be a non-empty string")
    if len(message) > 12000:
        raise HTTPException(status_code=413, detail="message is too long")
    if agent_type not in KNOWLEDGE_SOURCE_AGENTS:
        raise HTTPException(status_code=400, detail="Unknown agent selection")
    if llm_provider not in {"sarvam", "qwen"}:
        raise HTTPException(status_code=400, detail="Unknown reasoning model selection")

    from backend.script_classifier import classify_transcript
    started = time.perf_counter()
    classifier = await asyncio.to_thread(classify_transcript, message, "4.4")
    label = classifier.get("label", "none") if classifier.get("available") else "none"
    language_by_label = {
        "hi-IN": "hindi", "en-IN": "english", "pa-IN": "punjabi",
        "bn-IN": "bengali", "gu-IN": "gujarati", "kn-IN": "kannada",
        "ml-IN": "malayalam", "mr-IN": "marathi", "ta-IN": "tamil",
        "te-IN": "telugu", "as-IN": "assamese",
    }
    # Build only the reasoning state needed by _get_llm_response. Do not call
    # CloudVoiceAgent.__init__: it starts the realtime STT worker thread.
    agent = CloudVoiceAgent.__new__(CloudVoiceAgent)
    agent.agent_type = agent_type
    agent.llm_provider = llm_provider
    agent.llm_model = get_setting("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
    agent.sarvam_api_key = get_setting("SARVAM_API_KEY", "")
    agent.qwen_api_key = get_setting("QWEN_API_KEY", "") or None
    agent.qwen_chat_completions_url = get_setting(
        "AURA_QWEN_CHAT_COMPLETIONS_URL",
        "http://100.95.233.66:8081/v1/chat/completions",
    )
    agent.qwen_model = get_setting("AURA_QWEN_MODEL", "aura-qwen3-0.6b")
    agent.gemma_api_key = get_setting("GEMINI_API_KEY", "") or get_setting("GEMMA_API_KEY", "") or None
    agent.current_response_language = language_by_label.get(label, "english")
    agent.previous_turn_context = None
    agent.last_llm_input = message.strip()
    agent.llm_latency_ms = None
    agent.error = ""

    if llm_provider == "qwen" and not agent.qwen_api_key:
        raise HTTPException(status_code=503, detail="Qwen API authentication is not configured on the backend")
    if llm_provider == "sarvam" and not agent.sarvam_api_key:
        raise HTTPException(status_code=503, detail="Sarvam API authentication is not configured on the backend")
    try:
        answer = await agent._get_llm_response(message.strip(), "text-input", classifier)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Reasoning request failed: {type(exc).__name__}: {exc}") from exc
    if not answer:
        raise HTTPException(status_code=502, detail="Reasoning model returned an empty response")
    try:
        kb_status = knowledge_base.status(agent_type)
        retrieval_enabled = bool(kb_status.get("enabled", False)) if isinstance(kb_status, dict) else False
    except Exception:
        retrieval_enabled = False
    return {
        "user_text": message.strip(),
        "assistant_text": answer,
        "detected_language": label,
        "confidence": classifier.get("confidence"),
        "classifier_available": bool(classifier.get("available")),
        "classifier_latency_ms": classifier.get("latency_ms"),
        "llm_provider": llm_provider,
        "llm_latency_ms": agent.llm_latency_ms,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "retrieval_enabled": retrieval_enabled,
        "audio_generated": False,
    }


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    agent_type = websocket.query_params.get("agent", "general")
    llm_provider = websocket.query_params.get(
        "llm_provider", get_setting("AURA_LLM_PROVIDER", "sarvam")
    ).strip().lower()
    if llm_provider not in {"sarvam", "qwen"}:
        await websocket.send_json({"type": "error", "message": "Unknown reasoning model selection"})
        await websocket.close(code=1008)
        return
    voice_ids = {
        "general": "3fe4afbc-3bde-4c97-ab8e-37e3fb8c7ba2",
        "customer_service": "029d327b-987b-4879-8375-454cba8424ec",
        "tutor": "0ffd8c73-7c39-4b22-92f4-ebf33a6fae2c",
    }
    if agent_type not in voice_ids:
        await websocket.send_json({"type": "error", "message": "Unknown agent selection"})
        await websocket.close(code=1008)
        return
    try:
        sarvam_key = _required_env("SARVAM_API_KEY")
        bakbak_key = _required_env("BAKBAK_API_KEY")
        voice_id = voice_ids[agent_type]
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return

    output = QueueAudioSource()
    agent = CloudVoiceAgent(
        sarvam_api_key=sarvam_key,
        bakbak_api_key=bakbak_key,
        bakbak_voice_id=voice_id,
        agent_type=agent_type,
        output_source=output,
        llm_model=get_setting("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
        llm_provider=llm_provider,
        qwen_api_key=get_setting("QWEN_API_KEY", "") or None,
        qwen_chat_completions_url=get_setting(
            "AURA_QWEN_CHAT_COMPLETIONS_URL",
            "http://100.95.233.66:8081/v1/chat/completions",
        ),
        qwen_model=get_setting("AURA_QWEN_MODEL", "aura-qwen3-0.6b"),
        tts_model=get_setting("BAKBAK_TTS_MODEL", "m1"),
        classifier_enabled=True,
        classifier_backend="bert_tiny",
        classifier_model_variant="4.4",
        turn_buffer_enabled=os.getenv("TURN_BUFFER_ENABLED", "false").lower() == "true",
        turn_buffer_timeout_ms=int(os.getenv("TURN_BUFFER_TIMEOUT_MS", "500")),
    )
    agent.start()
    await websocket.send_json({
        "type": "ready",
        "agent": agent_type,
        "voice_id": voice_id,
        "message": "AURA voice session connected. Send mono PCM16 audio at 16000 Hz.",
        "input_sample_rate": 16000,
        "output_sample_rate": 24000,
        "stt_mode": os.getenv("SARVAM_STT_MODE", "verbose"),
        "classifier": "bert_tiny_4.4",
        "llm_provider": llm_provider,
        "reasoning_model": (
            get_setting("AURA_QWEN_MODEL", "aura-qwen3-0.6b")
            if llm_provider == "qwen"
            else get_setting("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
        ),
    })

    async def receive_audio():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            chunk = message.get("bytes")
            if chunk:
                agent.push_audio(chunk)
            elif message.get("text"):
                # Client control messages are JSON; no API secrets are accepted here.
                import json
                control = json.loads(message["text"])
                if control.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

    async def send_audio():
        while True:
            try:
                pcm = await asyncio.to_thread(output.items.get, True, 0.2)
            except queue.Empty:
                continue
            await websocket.send_bytes(pcm)

    async def send_state():
        previous = None
        while True:
            state = _snapshot(agent)
            if state != previous:
                await websocket.send_json(state)
                previous = state
            await asyncio.sleep(0.25)

    tasks = [
        asyncio.create_task(receive_audio()),
        asyncio.create_task(send_audio()),
        asyncio.create_task(send_state()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            if not task.cancelled():
                error = task.exception()
                if error:
                    raise error
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        agent.stop()
