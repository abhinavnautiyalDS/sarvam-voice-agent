import asyncio
import base64
import json
import queue
import threading
import urllib.parse

import av
import httpx
import numpy as np
import websockets


class CloudVoiceAgent:
    """Bridge browser WebRTC audio to Sarvam STT/LLM and BakBak TTS."""

    def __init__(
        self,
        sarvam_api_key: str,
        bakbak_api_key: str,
        bakbak_voice_id: str,
        output_source=None,
    ):
        self.sarvam_api_key = sarvam_api_key
        self.bakbak_api_key = bakbak_api_key
        self.bakbak_voice_id = bakbak_voice_id
        self.output_source = output_source

        self.audio_in = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None

        self._input_resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

        self._output_buffer = np.zeros(0, dtype=np.int16)
        self._output_lock = threading.Lock()

        self.last_user_text = ""
        self.last_assistant_text = ""
        self.error = ""
        self.stt_status = "Not started"
        self.last_stt_event = ""
        self.audio_frames_received = 0
        self.audio_bytes_sent = 0

    def start(self):
        if self.running:
            return

        self.running = True
        self.error = ""
        self.stt_status = "Connecting to Sarvam STT..."
        self.last_stt_event = ""
        self.audio_frames_received = 0
        self.audio_bytes_sent = 0

        self.thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False
        self.stt_status = "Stopped"

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        self.thread = None

    def push_audio(self, pcm16_mono_16k: bytes):
        if not self.running:
            return

        try:
            self.audio_in.put_nowait(pcm16_mono_16k)
        except queue.Full:
            try:
                self.audio_in.get_nowait()
                self.audio_in.put_nowait(pcm16_mono_16k)
            except queue.Empty:
                pass

    def _run_thread(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stt_status = "STT connection failed"
            self.running = False

    async def _run(self):
        params = urllib.parse.urlencode(
            {
                "language_code": "hi-IN",
                "model": "saaras:v3-realtime",
                "stream_type": "fast",
                "mode": "transcribe",
                "endpointing": "vad",
                "encoding": "linear16",
                "sample_rate": "16000",
                "threshold": "0.1",
                "silence_duration_ms": "700",
                "min_speech_duration_ms": "200",
            }
        )

        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{params}"

        async with websockets.connect(
            url,
            additional_headers={
                "api-subscription-key": self.sarvam_api_key,
            },
            ping_interval=20,
            ping_timeout=20,
            max_size=2**22,
        ) as ws:
            self.stt_status = "Connected to Sarvam STT"

            receiver = asyncio.create_task(self._receive_stt(ws))
            sender = asyncio.create_task(self._send_stt(ws))

            done, pending = await asyncio.wait(
                {receiver, sender},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()

            for task in done:
                exc = task.exception()
                if exc:
                    raise exc

    async def _send_stt(self, ws):
        while self.running:
            try:
                chunk = await asyncio.to_thread(
                    self.audio_in.get,
                    True,
                    0.25,
                )
            except queue.Empty:
                continue

            await ws.send(
                json.dumps(
                    {
                        "event": "audio_input",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

            self.audio_bytes_sent += len(chunk)

    async def _receive_stt(self, ws):
        async for raw in ws:
            if not self.running:
                return

            event = json.loads(raw)
            event_type = event.get("event")
            self.last_stt_event = event_type or str(event)

            if event_type == "transcript.final":
                transcript = event.get("transcript", "").strip()

                if not transcript:
                    continue

                self.last_user_text = transcript
                self.stt_status = "Transcript received — generating response"

                response = await self._get_llm_response(transcript)
                if not response:
                    raise RuntimeError(
                        "Sarvam LLM returned an empty response"
                    )

                self.last_assistant_text = response

                self.stt_status = "Generating speech with BakBak"
                audio = await self._get_bakbak_audio(response)
                if not audio:
                    raise RuntimeError(
                        "BakBak returned empty audio"
                    )

                self._enqueue_output(audio)

                self.stt_status = "Ready"

            elif event_type == "error":
                code = event.get("code", "unknown")
                message = event.get(
                    "message",
                    "Sarvam realtime STT error",
                )
                raise RuntimeError(
                    f"Sarvam STT error {code}: {message}"
                )

    async def _get_llm_response(self, user_text: str) -> str:
        url = "https://api.sarvam.ai/v1/chat/completions"

        headers = {
            "api-subscription-key": self.sarvam_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "model": "sarvam-105b-conversations",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "The user primarily speaks Hindi and may mix English. "
                        "Reply naturally and briefly. "
                        "Do not use markdown, bullet points, or emojis."
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            "max_tokens": 100,
            "reasoning_effort": None,
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    async def _get_bakbak_audio(self, text: str) -> bytes:
        url = "https://hub.getraya.app/v1/text-to-speech"

        headers = {
            "X-API-Key": self.bakbak_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voice_id": self.bakbak_voice_id,
            "model": "m1",
            "language": "hi",
            "codec": "wav",
            "sample_rate": 24000,
            "speed": 1.0,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            audio = response.content

        return audio[44:] if audio[:4] == b"RIFF" else audio

    def _enqueue_output(self, pcm16_24k: bytes):
        """Push 24 kHz mono PCM into streamlit-webrtc's PcmAudioSource."""
        if len(pcm16_24k) % 2:
            raise ValueError("BakBak returned an odd-length PCM payload")

        if self.output_source is None:
            raise RuntimeError("WebRTC output source is not configured")

        self.output_source.push(pcm16_24k)

    def ingest_webrtc_frame(self, frame: av.AudioFrame):
        """Convert browser audio to mono 16 kHz signed PCM for Saaras."""

        self.audio_frames_received += 1

        for converted in self._input_resampler.resample(frame):
            pcm = (
                converted
                .to_ndarray()
                .astype(np.int16)
                .tobytes()
            )
            self.push_audio(pcm)
