import asyncio
import base64
import json
import os
import queue
import threading
import time
import urllib.parse

import av
import httpx
import numpy as np
import websockets

from script_classifier import classify_transcript as classify_script_transcript


class CloudVoiceAgent:
    """Bridge browser WebRTC audio to Sarvam STT/LLM and BakBak TTS."""

    def __init__(
        self,
        sarvam_api_key: str,
        bakbak_api_key: str,
        bakbak_voice_id: str,
        output_source=None,
        llm_model: str = "sarvam-105b-conversations",
        tts_model: str = "m1",
        classifier_enabled: bool = True,
        classifier_backend: str = "ml",
    ):
        self.sarvam_api_key = sarvam_api_key
        self.bakbak_api_key = bakbak_api_key
        self.bakbak_voice_id = bakbak_voice_id
        self.output_source = output_source
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.classifier_enabled = classifier_enabled
        self.classifier_backend = "bert_tiny"
        self.current_response_language = "hindi"

        self.audio_in = queue.Queue(maxsize=100)
        self.running = False
        self.thread = None
        self._input_resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

        self.pipeline_status = "Not started"
        self.stt_status = "Not started"
        self.llm_status = "Idle"
        self.tts_status = "Idle"
        self.last_stt_event = ""
        self.last_stt_raw = ""
        self.last_detected_language = ""
        self.last_language_source = "Saaras"
        self.last_classifier_result = {
            "label": "unavailable", "confidence": 0.0, "probabilities": {},
            "latency_ms": 0.0, "available": False, "reason": "not_run",
        }
        self.classifier_status = "BERT-Tiny script classifier enabled" if classifier_enabled else "Disabled"
        self.last_user_text = ""
        self.live_transcript = ""
        self.last_assistant_text = ""
        self.last_llm_input = ""
        self.last_tts_input = ""
        self.llm_latency_ms = None
        self.tts_latency_ms = None
        self.tts_audio_bytes = 0
        self.output_bytes_pushed = 0
        self.error = ""
        self.error_stage = ""
        self.audio_frames_received = 0
        self.audio_bytes_sent = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.error = ""
        self.error_stage = ""
        self.pipeline_status = "Connecting"
        self.stt_status = "Connecting to Sarvam STT..."
        self.llm_status = "Idle"
        self.tts_status = "Idle"
        self.last_stt_event = ""
        self.last_stt_raw = ""
        self.last_detected_language = ""
        self.last_language_source = "Saaras"
        self.last_user_text = ""
        self.live_transcript = ""
        self.audio_frames_received = 0
        self.audio_bytes_sent = 0
        self.output_bytes_pushed = 0
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.pipeline_status = "Stopped"
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
            self.pipeline_status = "Failed"
            if not self.error_stage:
                self.error_stage = "STT/WebSocket"

    async def _run(self):
        params = urllib.parse.urlencode({
            "language_code": "auto",
            "model": os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime"),
            "stream_type": os.getenv("SARVAM_STT_STREAM_TYPE", "fast"),
            "mode": os.getenv("SARVAM_STT_MODE", "codemix"),
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": "16000",
            "threshold": os.getenv("SARVAM_STT_THRESHOLD", "0.3"),
            "silence_duration_ms": os.getenv("SARVAM_SILENCE_DURATION_MS", "900"),
            "min_speech_duration_ms": os.getenv("SARVAM_MIN_SPEECH_DURATION_MS", "300"),
        })
        url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{params}"
        async with websockets.connect(
            url,
            additional_headers={"api-subscription-key": self.sarvam_api_key},
            ping_interval=20,
            ping_timeout=20,
            max_size=2**22,
        ) as ws:
            self.pipeline_status = "Ready"
            self.stt_status = "Connected to Sarvam STT"
            receiver = asyncio.create_task(self._receive_stt(ws))
            sender = asyncio.create_task(self._send_stt(ws))
            done, pending = await asyncio.wait(
                {receiver, sender}, return_when=asyncio.FIRST_EXCEPTION
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
            self.audio_bytes_sent += len(chunk)

    async def _receive_stt(self, ws):
        async for raw in ws:
            if not self.running:
                return
            event = json.loads(raw)
            event_type = event.get("event")
            self.last_stt_event = event_type or str(event)
            self.last_stt_raw = json.dumps(event, ensure_ascii=False)

            if event_type == "session.begin":
                self.stt_status = "Connected — waiting for speech"
            elif event_type == "transcript.partial":
                partial = self._extract_transcript(event)
                if partial:
                    self.live_transcript = partial
                    self.stt_status = "Listening"
            elif event_type == "transcript.final":
                transcript = self._extract_transcript(event)
                if not transcript:
                    self.stt_status = "Final transcript event received — text missing"
                    continue

                self.last_user_text = transcript
                self.live_transcript = transcript
                self.stt_status = "Transcript received"
                self.error = ""
                self.error_stage = ""

                if self.classifier_enabled:
                    self.last_classifier_result = await asyncio.to_thread(classify_script_transcript, transcript)
                    result = self.last_classifier_result
                    confidence = result.get("confidence")
                    if result.get("available"):
                        confidence_text = f" ({confidence:.0%})" if confidence is not None else " (score N/A)"
                        self.classifier_status = f"BERT-Tiny: {result.get('label', 'unknown')}{confidence_text}"
                    else:
                        self.classifier_status = f"BERT-Tiny unavailable ({result.get('reason', 'unknown')})"
                else:
                    self.last_classifier_result = {
                        "label": "disabled", "confidence": 0.0, "probabilities": {},
                        "latency_ms": 0.0, "available": False, "reason": "disabled",
                    }
                    self.classifier_status = "Disabled"

                detected_language = event.get("language")
                if not detected_language:
                    data = event.get("data")
                    if isinstance(data, dict):
                        detected_language = data.get("language")
                detected_language = str(detected_language or "unknown")
                self.last_detected_language = detected_language

                # BERT-Tiny predicts writing script, not natural language.
                # Devanagari routes to Hindi voice output; Latin routes to English voice output.
                # Latin-script Hinglish is therefore a known limitation of this binary classifier.
                result = self.last_classifier_result
                classifier_label = str(result.get("label", "unavailable")).upper()
                if not result.get("available"):
                    self.last_language_source = "BERT-Tiny unavailable"
                elif detected_language == "unknown":
                    self.last_language_source = "Saaras metadata missing; BERT-Tiny script prediction used"
                else:
                    normalized_saaras = detected_language.lower()
                    saaras_says_hindi = normalized_saaras.startswith(("hi", "hindi"))
                    saaras_says_english = normalized_saaras.startswith(("en", "english"))
                    if (saaras_says_hindi and classifier_label == "DEVANAGARI") or (
                        saaras_says_english and classifier_label == "LATIN"
                    ):
                        self.last_language_source = "Saaras metadata broadly agrees with script"
                    else:
                        self.last_language_source = "Saaras metadata and script prediction differ"

                if result.get("available") and classifier_label == "DEVANAGARI":
                    self.current_response_language = "hindi"
                elif result.get("available") and classifier_label == "LATIN":
                    self.current_response_language = "english"

                self.llm_status = "Generating response"
                try:
                    response = await self._get_llm_response(transcript, detected_language, result)
                except Exception as exc:
                    self.error_stage = "LLM"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.llm_status = "Failed"
                    self.pipeline_status = "LLM failed"
                    continue

                # Classifier output controls only the TTS voice language when it is usable.
                # Hinglish is rendered with Hindi voice output; it is not a provider locale.
                tts_language_by_script = {"DEVANAGARI": "hi", "LATIN": "en"}
                if result.get("available") and classifier_label in tts_language_by_script:
                    tts_language = tts_language_by_script[classifier_label]
                else:
                    tts_language = "en" if self.current_response_language == "english" else "hi"

                if not response:
                    self.error_stage = "LLM"
                    self.error = "Sarvam LLM returned an empty response"
                    self.llm_status = "Failed — empty response"
                    self.pipeline_status = "LLM failed"
                    continue

                self.last_assistant_text = response
                self.llm_status = "Completed"
                self.tts_status = "Generating speech with BakBak Raya"
                self.last_tts_input = response
                try:
                    audio = await self._get_bakbak_audio(response, language=tts_language)
                except Exception as exc:
                    self.error_stage = "BakBak TTS"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.tts_status = "Failed"
                    self.pipeline_status = "BakBak failed"
                    continue
                if not audio:
                    self.error_stage = "BakBak TTS"
                    self.error = "BakBak returned empty audio"
                    self.tts_status = "Failed — empty audio"
                    self.pipeline_status = "BakBak failed"
                    continue
                try:
                    self._enqueue_output(audio)
                except Exception as exc:
                    self.error_stage = "WebRTC output"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.tts_status = "Audio generated, but output failed"
                    self.pipeline_status = "WebRTC output failed"
                    continue
                self.tts_status = f"Ready — {len(audio):,} PCM bytes queued"
                self.pipeline_status = "Ready"
                self.stt_status = "Connected — waiting for speech"
            elif event_type == "error":
                code = event.get("code", "unknown")
                message = event.get("message", "Sarvam realtime STT error")
                self.error_stage = "STT"
                raise RuntimeError(f"Sarvam STT error {code}: {message}")

    @staticmethod
    def _extract_transcript(event: dict) -> str:
        candidates = [event.get("text"), event.get("transcript")]
        data = event.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("text"), data.get("transcript")])
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    async def _get_llm_response(
        self, user_text: str, detected_language: str = "unknown", classifier_result: dict | None = None
    ) -> str:
        self.last_llm_input = user_text
        started = time.perf_counter()
        url = os.getenv("SARVAM_LLM_URL", "https://api.sarvam.ai/v1/chat/completions")
        headers = {
            "api-subscription-key": self.sarvam_api_key,
            "Content-Type": "application/json",
        }
        system_prompt = os.getenv(
            "VOICE_SYSTEM_PROMPT",
            "You are a concise, natural voice assistant. Answer directly. Avoid markdown, bullets, and emojis.",
        )
        classifier_result = classifier_result or {}
        classifier_label = str(classifier_result.get("label", "unavailable")).upper()
        classifier_confidence_value = classifier_result.get("confidence")
        classifier_available = bool(classifier_result.get("available"))
        current_language = getattr(self, "current_response_language", "hindi")
        if classifier_available and classifier_confidence_value is not None:
            language_hint = f"Transcript classifier prediction: {classifier_label} (confidence {float(classifier_confidence_value):.1%})."
        elif classifier_available:
            language_hint = f"Transcript classifier prediction: {classifier_label}; confidence is not calibrated for this generative classifier."
        else:
            language_hint = "Transcript classifier is unavailable; use the conversation context."
        if classifier_label == "DEVANAGARI":
            language_instruction = (
                "The transcript classifier predicts DEVANAGARI script. Reply in natural Hindi using Devanagari script. "
                "Keep the response brief and conversational; do not translate unless asked."
            )
        elif classifier_label == "LATIN":
            language_instruction = (
                "The transcript classifier predicts LATIN script. Reply in English using Latin script. "
                "Keep the response brief and conversational; do not translate unless asked. "
                "This is a script prediction, not a guarantee that the spoken language is English."
            )
        else:
            language_instruction = (
                f"The script classifier is unavailable. Preserve the current conversation language ({current_language}) "
                "and use the transcript/context to respond naturally."
            )
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\\n{language_instruction}\\n"
                        f"Latest Saaras language metadata: {detected_language}.\\n"
                        f"{language_hint}"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            "max_tokens": int(os.getenv("SARVAM_LLM_MAX_TOKENS", "120")),
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        self.llm_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (content or "").strip()

    async def _get_bakbak_audio(self, text: str, language: str) -> bytes:
        started = time.perf_counter()
        url = os.getenv("BAKBAK_TTS_URL", "https://hub.getraya.app/v1/text-to-speech")
        headers = {"X-API-Key": self.bakbak_api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "voice_id": self.bakbak_voice_id,
            "model": self.tts_model,
            "language": language,
            "codec": "wav",
            "sample_rate": 24000,
            "speed": float(os.getenv("BAKBAK_TTS_SPEED", "1.0")),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            audio = response.content
        if audio[:4] == b"RIFF":
            import io
            import soundfile as sf
            waveform, sample_rate = sf.read(io.BytesIO(audio), dtype="int16")
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1).astype("int16")
            if sample_rate != 24000:
                raise ValueError(f"Expected 24000 Hz TTS audio, received {sample_rate} Hz")
            audio = waveform.tobytes()
        self.tts_audio_bytes = len(audio)
        self.tts_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return audio

    def _enqueue_output(self, pcm16_24k: bytes):
        """Push 24 kHz mono PCM into streamlit-webrtc's PcmAudioSource."""
        if len(pcm16_24k) % 2:
            raise ValueError("BakBak returned an odd-length PCM payload")
        if self.output_source is None:
            raise RuntimeError("WebRTC output source is not configured")
        self.output_source.push(pcm16_24k)
        self.output_bytes_pushed += len(pcm16_24k)

    def ingest_webrtc_frame(self, frame: av.AudioFrame):
        """Convert browser audio to mono 16 kHz signed PCM for Saaras."""
        self.audio_frames_received += 1
        for converted in self._input_resampler.resample(frame):
            pcm = converted.to_ndarray().astype(np.int16).tobytes()
            self.push_audio(pcm)
