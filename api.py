from __future__ import annotations

import asyncio
import os
import queue
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cloud_voice import CloudVoiceAgent\nfrom admin_config import get_setting, public_config, save_config

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
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/admin/config")\nasync def admin_get_config(request: Request):\n    _require_admin_ip(request)\n    return public_config()\n\n\n@app.put("/api/admin/config")\nasync def admin_update_config(request: Request):\n    _require_admin_ip(request)\n    body = await request.json()\n    if not isinstance(body, dict):\n        raise HTTPException(status_code=400, detail="Expected a JSON object")\n    allowed = {"SARVAM_API_KEY", "BAKBAK_API_KEY", "BAKBAK_VOICE_ID", "SARVAM_LLM_MODEL", "BAKBAK_TTS_MODEL", "SARVAM_STT_MODE", "AURA_LLM_PROVIDER", "AURA_TTS_PROVIDER"}\n    updates = {key: value.strip() for key, value in body.items() if key in allowed and isinstance(value, str)}\n    if not updates:\n        raise HTTPException(status_code=400, detail="No supported settings supplied")\n    save_config(updates)\n    return {"status": "saved", "config": public_config(), "note": "New voice sessions will use the updated settings."}\n\n\n@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "aura-voice-api",
        "classifier": "local multilingual BERT-Tiny (12 labels)",
        "stt_mode": os.getenv("SARVAM_STT_MODE", "verbose"),
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
        "audio_frames_received": agent.audio_frames_received,
    }


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        sarvam_key = _required_env("SARVAM_API_KEY")
        bakbak_key = _required_env("BAKBAK_API_KEY")
        voice_id = _required_env("BAKBAK_VOICE_ID")
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return

    output = QueueAudioSource()
    agent = CloudVoiceAgent(
        sarvam_api_key=sarvam_key,
        bakbak_api_key=bakbak_key,
        bakbak_voice_id=voice_id,
        output_source=output,
        llm_model=get_setting("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
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
        "message": "AURA voice session connected. Send mono PCM16 audio at 16000 Hz.",
        "input_sample_rate": 16000,
        "output_sample_rate": 24000,
        "stt_mode": os.getenv("SARVAM_STT_MODE", "verbose"),
        "classifier": "bert_tiny_4.4",
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
