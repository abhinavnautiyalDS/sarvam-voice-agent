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

    def __init__(self, sarvam_api_key: str, bakbak_api_key: str, bakbak_voice_id: str):
        self.sarvam_api_key = sarvam_api_key
        self.bakbak_api_key = bakbak_api_key
        self.bakbak_voice_id = bakbak_voice_id

        self.audio_in = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None

        self._output_buffer = np.zeros(0, dtype=np.int16)
        self._output_lock = threading.Lock()
        self.last_user_text = ""
        self.last_assistant_text = ""
        self.error = ""

    def start(self):
        if self.running:
            return
        self.running = True
        self.error = ""
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
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
            self.error = str(exc)
            self.running = False

    async def _run(self):
        params = urllib.parse.urlencode(
            {
                "language_code": "hi-IN",
                "model": "saaras:v3-realtime",
                "stream_type": "fast",
                "mode": "transcribe",
                "endpointing": "vad",
                "threshold": "0.1",
                "silence_duration_ms": "700",
                "min_speech_duration_ms": "200",
            }
        )
        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{params}"

        async with websockets.connect(
            url,
            additional_headers={"api-subscription-key": self.sarvam_api_key},
            ping_interval=20,
            ping_timeout=20,
            max_size=2**22,
        ) as ws:
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
                chunk = await asyncio.to_thread(self.audio_in.get, True, 0.25)
            except queue.Empty:
                continue

            await ws.send(json.dumps({
                "event": "audio_input",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }))

    async def _receive_stt(self, ws):
        async for raw in ws:
            if not self.running:
                return

            event = json.loads(raw)
            event_type = event.get("event")

            if event_type == "transcript.final":
                transcript = event.get("transcript", "").strip()
                if not transcript:
                    continue

                self.last_user_text = transcript
                response = await self._get_llm_response(transcript)
                self.last_assistant_text = response

                audio = await self._get_bakbak_audio(response)
                self._enqueue_output(audio)

            elif event_type == "error":
                raise RuntimeError(
                    event.get("message", "Sarvam realtime STT error")
                )

    async def _get_llm_response(self, user_text: str) -> str:
        url = "https://api.sarvam.ai/v1/chat/completions"
        headers = {
            "api-subscription-key": self.sarvam_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sarvam-105b",
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
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 150,
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, headers=headers, json=payload)
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
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            audio = response.content

        return audio[44:] if audio[:4] == b"RIFF" else audio

    def _enqueue_output(self, pcm16_24k: bytes):
        samples = np.frombuffer(pcm16_24k, dtype=np.int16)
        with self._output_lock:
            self._output_buffer = np.concatenate(
                [self._output_buffer, samples]
            )

    def get_output_frame(self, input_frame: av.AudioFrame) -> av.AudioFrame:
        """Return a TTS frame with the same timing/rate as the browser track."""
        sample_rate = input_frame.sample_rate
        samples_needed = input_frame.samples
        channels = len(input_frame.layout.channels)

        # Consume the correct amount of 24 kHz source audio for this output
        # duration. Example: a 48 kHz browser frame needs twice as many source
        # samples before resampling.
        source_needed = max(
            1, round(samples_needed * 24000 / sample_rate)
        )

        with self._output_lock:
            if len(self._output_buffer) >= source_needed:
                source_samples = self._output_buffer[:source_needed]
                self._output_buffer = self._output_buffer[source_needed:]
            else:
                source_samples = np.zeros(source_needed, dtype=np.int16)
                if len(self._output_buffer):
                    source_samples[: len(self._output_buffer)] = self._output_buffer
                    self._output_buffer = np.zeros(0, dtype=np.int16)

        if sample_rate != 24000:
            source = av.AudioFrame.from_ndarray(
                source_samples.reshape(1, -1),
                format="s16",
                layout="mono",
            )
            source.sample_rate = 24000
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=sample_rate,
            )
            converted = resampler.resample(source)
            mono = (
                converted[0].to_ndarray().reshape(-1).astype(np.int16)
                if converted
                else np.zeros(samples_needed, dtype=np.int16)
            )
        else:
            mono = source_samples

        if len(mono) < samples_needed:
            mono = np.pad(mono, (0, samples_needed - len(mono)))
        else:
            mono = mono[:samples_needed]

        if channels == 1:
            data = mono.reshape(1, -1)
            layout = "mono"
        elif channels == 2:
            data = np.tile(mono, (2, 1))
            layout = "stereo"
        else:
            data = np.tile(mono, (channels, 1))
            layout = input_frame.layout.name

        out = av.AudioFrame.from_ndarray(data, format="s16", layout=layout)
        out.sample_rate = sample_rate
        return out

    def ingest_webrtc_frame(self, frame: av.AudioFrame):
        """Convert browser audio to mono 16 kHz signed PCM for Saaras."""
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000,
        )

        for converted in resampler.resample(frame):
            pcm = converted.to_ndarray().astype(np.int16).tobytes()
            self.push_audio(pcm)
