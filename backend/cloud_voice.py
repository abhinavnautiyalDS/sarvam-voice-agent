import asyncio
import base64
import json
import os
import queue
import threading
import time
import urllib.parse

import httpx
import numpy as np
import websockets

from backend.script_classifier import classify_transcript as classify_language_transcript
from backend.system_prompt import get_system_prompt
from backend import knowledge_base


OPENING_GREETINGS = {
    # Frontend agent IDs: general, customer_service (Admission Counsellor), tutor.
    "general": "Hi! I’m Aanya, your AI assistant. I can help answer questions and make things easier. What can I help you with today?",
    "customer_service": "Hello! I’m Arjun, your admission counsellor. I can guide you through courses, eligibility, fees, and the admission process. What would you like to know?",
    "tutor": "Hi! I’m Aanya, your AI tutor. I can explain concepts, work through examples, and help you practise. What would you like to learn today?",
}


class CloudVoiceAgent:
    """Bridge browser WebRTC audio to Sarvam STT/LLM and BakBak TTS."""

    def __init__(
        self,
        sarvam_api_key: str,
        bakbak_api_key: str,
        bakbak_voice_id: str,
        agent_type: str = "general",
        output_source=None,
        llm_model: str = "sarvam-105b-conversations",
        llm_provider: str = "qwen",
        qwen_api_key: str | None = None,
        qwen_chat_completions_url: str = "http://100.95.233.66:8081/v1/chat/completions",
        qwen_model: str = "aura-qwen3-0.6b",
        gemma_api_key: str | None = None,
        tts_model: str = "m1",
        classifier_enabled: bool = True,
        classifier_backend: str = "ml",
        classifier_model_variant: str = "Old E2",
        turn_buffer_enabled: bool = True,
        turn_buffer_timeout_ms: int = 500,
    ):
        self.sarvam_api_key = sarvam_api_key
        self.bakbak_api_key = bakbak_api_key
        self.bakbak_voice_id = bakbak_voice_id
        self.agent_type = agent_type
        self.output_source = output_source
        self.llm_model = llm_model
        self.llm_provider = (llm_provider or "sarvam").strip().lower()
        self.qwen_api_key = qwen_api_key
        self.qwen_chat_completions_url = qwen_chat_completions_url
        self.qwen_model = qwen_model
        self.gemma_api_key = gemma_api_key
        self.tts_model = tts_model
        self.classifier_enabled = classifier_enabled
        self.classifier_backend = "bert_tiny"
        self.classifier_model_variant = classifier_model_variant
        self.turn_buffer_enabled = bool(turn_buffer_enabled)
        self.turn_buffer_timeout_ms = max(200, min(int(turn_buffer_timeout_ms), 1000))
        self.turn_reliability_threshold = float(os.getenv("TURN_RELIABILITY_THRESHOLD", "0.65"))
        self.last_turn_reliability = None
        self.last_turn_reliability_details = {}
        self.turn_buffer_text = ""
        self.turn_buffer_event = {}
        self.turn_buffer_task = None
        self.last_turn_buffer_decision = ""
        self.last_combined_turn_text = ""
        self.current_response_language = "hindi"
        self.current_language_code = "hi-IN"
        self.language_history = []
        # Keep the complete in-memory session transcript for multi-turn reasoning.
        self.conversation_history = []
        self.conversation_turns = []
        self.turn_counter = 0
        self.active_turn_id = None
        self.last_kb_status = "not_checked"
        self.last_kb_used = False
        self.last_kb_chunks = 0
        self.last_kb_agent = agent_type
        self.last_kb_titles = []
        self.last_classifier_input = ""
        self.last_effective_language_code = "hi-IN"
        self.last_effective_language_source = "default"

        self.audio_in = queue.Queue(maxsize=100)
        self.running = False
        self._active_response_task = None
        self._response_generation = 0
        self.thread = None
        self._input_resampler = None

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
        self.classifier_status = "Multilingual BERT-Tiny language classifier enabled" if classifier_enabled else "Disabled"
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
        self.audio_bytes_received = 0
        self.audio_bytes_sent = 0
        self.audio_chunks_dropped = 0

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
        self.last_classifier_input = ""
        self.last_assistant_text = ""
        self.turn_buffer_text = ""
        self.turn_buffer_event = {}
        self.last_turn_buffer_decision = ""
        self.last_combined_turn_text = ""
        self.language_history = []
        self.conversation_history = []
        self.conversation_turns = []
        self.turn_counter = 0
        self.active_turn_id = None
        self.last_kb_status = "not_checked"
        self.last_kb_used = False
        self.last_kb_chunks = 0
        self.last_kb_agent = self.agent_type
        self.last_kb_titles = []
        self.current_language_code = "hi-IN"
        self.last_effective_language_code = "hi-IN"
        self.last_effective_language_source = "default"
        self._opening_greeting_started = False
        self._response_generation += 1
        if self._active_response_task and not self._active_response_task.done():
            self._active_response_task.cancel()
        self._active_response_task = None
        self._clear_output()
        self.current_response_language = "hindi"
        if self.turn_buffer_task and not self.turn_buffer_task.done():
            self.turn_buffer_task.cancel()
        self.turn_buffer_task = None
        self.last_language_source = "Saaras"
        self.last_user_text = ""
        self.live_transcript = ""
        self.audio_frames_received = 0
        self.audio_bytes_received = 0
        self.audio_bytes_sent = 0
        self.audio_chunks_dropped = 0
        self.output_bytes_pushed = 0
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self._response_generation += 1
        if self._active_response_task and not self._active_response_task.done():
            self._active_response_task.cancel()
        self._active_response_task = None
        self._clear_output()
        if self.turn_buffer_task and not self.turn_buffer_task.done():
            self.turn_buffer_task.cancel()
        self.turn_buffer_task = None
        self.pipeline_status = "Stopped"
        self.stt_status = "Stopped"
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None

    def push_audio(self, pcm16_mono_16k: bytes):
        """Queue one mono PCM16/16 kHz chunk and account for actual input audio."""
        if not self.running or not pcm16_mono_16k:
            return
        self.audio_frames_received += 1
        self.audio_bytes_received += len(pcm16_mono_16k)
        try:
            self.audio_in.put_nowait(pcm16_mono_16k)
        except queue.Full:
            try:
                self.audio_in.get_nowait()
                self.audio_chunks_dropped += 1
                self.audio_in.put_nowait(pcm16_mono_16k)
            except queue.Empty:
                self.audio_chunks_dropped += 1

    def _run_thread(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.pipeline_status = "Failed"
            if self.error_stage in {"", "STT/WebSocket"}:
                self.error_stage = "STT/WebSocket"
                self.stt_status = "Failed — see pipeline error"

    async def _run(self):
        params = urllib.parse.urlencode({
            "language_code": "auto",
            "model": os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime"),
            "stream_type": os.getenv("SARVAM_STT_STREAM_TYPE", "fast"),
            "mode": "codemix",
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": "16000",
            "threshold": os.getenv("SARVAM_STT_THRESHOLD", "0.3"),
            "silence_duration_ms": os.getenv("SARVAM_SILENCE_DURATION_MS", "600"),
            "min_speech_duration_ms": os.getenv("SARVAM_MIN_SPEECH_DURATION_MS", "200"),
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
                {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
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
            if self.stt_status in {"Connected — waiting for speech", "Listening"}:
                self.stt_status = "Audio streaming to Sarvam"

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
                if not self._opening_greeting_started:
                    self._opening_greeting_started = True
                    self._response_generation += 1
                    generation = self._response_generation
                    self._active_response_task = asyncio.create_task(
                        self._play_opening_greeting(generation)
                    )
            elif event_type == "transcript.partial":
                partial = self._extract_transcript(event)
                if partial:
                    # A fresh caller utterance interrupts any in-flight reply immediately.
                    self._interrupt_response()
                    # Keep a pending short finalized turn visible while Sarvam streams
                    # the next utterance, so the live transcript reflects the context
                    # that the turn buffer is building.
                    self.live_transcript = (
                        f"{self.turn_buffer_text} {partial}".strip()
                        if self.turn_buffer_text else partial
                    )
                    self.stt_status = (
                        "Listening — building context with buffered turn"
                        if self.turn_buffer_text else "Listening"
                    )
            elif event_type == "transcript.final":
                transcript = self._extract_transcript(event)
                if not transcript:
                    self.stt_status = "Final transcript event received — text missing"
                    continue

                # Run the existing reliability formula on each finalized transcript.
                if self.turn_buffer_text:
                    previous = self.turn_buffer_text
                    if self.turn_buffer_task and not self.turn_buffer_task.done():
                        self.turn_buffer_task.cancel()
                    self.turn_buffer_task = None
                    self.turn_buffer_text = ""
                    self.turn_buffer_event = {}
                    combined = f"{previous} {transcript}".strip()
                    self.live_transcript = combined
                    self.last_turn_buffer_decision = "SECOND TURN → COMBINE_AND_CLASSIFY"
                    self.last_combined_turn_text = combined
                    self._schedule_response(combined, event)
                elif not self.turn_buffer_enabled:
                    self.last_turn_buffer_decision = "BUFFERING DISABLED → CLASSIFY_NOW"
                    self.last_combined_turn_text = transcript
                    self._schedule_response(transcript, event)
                else:
                    probe = await asyncio.to_thread(classify_language_transcript, transcript, self.classifier_model_variant) if self.classifier_enabled else {"available": False, "confidence": 0.0}
                    reliability, details = self._turn_reliability(transcript, probe)
                    self.last_turn_reliability = reliability
                    self.last_turn_reliability_details = details
                    should_wait = reliability < self.turn_reliability_threshold and not self._is_meaningful_short_turn(transcript)
                    self.last_turn_buffer_decision = f"RELIABILITY {reliability:.3f} → {'WAIT_FOR_NEXT_TURN' if should_wait else 'CLASSIFY_NOW'}"
                    if should_wait:
                        self.turn_buffer_text = transcript
                        self.turn_buffer_event = event
                        self.live_transcript = transcript
                        self.classifier_status = f"Waiting for context ({self.turn_buffer_timeout_ms} ms); reliability={reliability:.3f}"
                        self.turn_buffer_task = asyncio.create_task(self._flush_turn_buffer_after_timeout(event))
                    else:
                        self.last_combined_turn_text = transcript
                        self._schedule_response(transcript, event, classifier_result=probe)

            elif event_type == "error":
                code = event.get("code", "unknown")
                message = event.get("message", "Sarvam realtime STT error")
                self.error_stage = "STT"
                raise RuntimeError(f"Sarvam STT error {code}: {message}")


    @staticmethod
    def _turn_reliability(text: str, classifier_result: dict) -> tuple[float, dict]:
        import unicodedata
        words = text.split()
        n = len(words)
        length_score = min(n / 5.0, 1.0)
        fillers = {"um", "umm", "uh", "uhh", "hmm", "hmmm", "erm"}
        meaningful_ratio = sum(w.lower().strip(".,!?;:") not in fillers for w in words) / max(n, 1)
        normalized = text.lower().strip(" .,!?:;")
        known = {"hello", "hi", "hey", "yes", "yeah", "okay", "ok", "haan", "haan bilkul", "हाँ", "हां", "हाँ बिल्कुल", "बिल्कुल", "जी", "thank you", "thanks", "no", "nope", "right"}
        phrase_match = 1.0 if normalized in known else 0.0
        meaning_score = min(max(0.5 * meaningful_ratio + 0.5 * phrase_match, 0.0), 1.0)
        letters = [ch for ch in text if ch.isalpha()]
        scripts = ("LATIN", "DEVANAGARI", "GURMUKHI", "BENGALI", "GUJARATI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "ORIYA", "ODIA")
        recognized = [ch for ch in letters if any(s in unicodedata.name(ch, "") for s in scripts)]
        script_score = len(recognized) / max(len(letters), 1)
        confidence = float(classifier_result.get("confidence") or 0.0)
        score = (0.25 * length_score + 0.20 * meaning_score + 0.20 * confidence + 0.20 * script_score) / 0.85
        return score, {"length": length_score, "meaning": meaning_score, "confidence": confidence, "script": script_score}

    def _clear_output(self):
        """Discard queued PCM so an interrupted/stale reply cannot continue playing."""
        if self.output_source is not None:
            try:
                self.output_source.clear()
            except Exception:
                pass

    def _interrupt_response(self):
        """Invalidate and cancel the current response as soon as caller speech starts."""
        self._response_generation += 1
        task = self._active_response_task
        if task and not task.done():
            task.cancel()
        self._active_response_task = None
        self._clear_output()
        self.tts_status = "Interrupted by caller"
        if self.llm_status == "Generating response":
            self.llm_status = "Interrupted by caller"

    def _schedule_response(self, transcript: str, event: dict, classifier_result: dict | None = None):
        """Run LLM/TTS outside the STT receiver so partial events remain responsive."""
        self._response_generation += 1
        generation = self._response_generation
        previous = self._active_response_task
        if previous and not previous.done():
            previous.cancel()
        self._clear_output()
        self._active_response_task = asyncio.create_task(
            self._process_final_transcript(
                transcript, event, classifier_result=classifier_result, generation=generation
            )
        )

    async def _process_final_transcript(self, transcript: str, event: dict, classifier_result: dict | None = None, generation: int | None = None):
        generation = self._response_generation if generation is None else generation
        # Classify only the current finalized caller turn (single-turn inference).
        classifier_input = transcript
        self.last_classifier_input = classifier_input
        self.last_user_text = transcript
        self.live_transcript = transcript
        self.stt_status = "Transcript received"
        self.error = ""
        self.error_stage = ""

        if self.classifier_enabled:
            self.last_classifier_result = classifier_result or await asyncio.to_thread(
                classify_language_transcript,
                classifier_input,
                self.classifier_model_variant,
            )
            result = self.last_classifier_result
            confidence = result.get("confidence")
            if result.get("available"):
                confidence_text = f" ({confidence:.0%})" if confidence is not None else " (score N/A)"
                self.classifier_status = f"{self.classifier_model_variant}: {result.get('label', 'unknown')}{confidence_text}"
            else:
                self.classifier_status = f"Language classifier unavailable ({result.get('reason', 'unknown')})"
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

        # The fine-tuned model predicts language, not writing script.
        result = self.last_classifier_result
        classifier_label = str(result.get("label", "none")) if result.get("available") else "none"
        if not result.get("available"):
            self.last_language_source = "Language classifier unavailable; using Saaras metadata/context"
        elif detected_language == "unknown":
            self.last_language_source = "Multilingual BERT-Tiny prediction (Saaras metadata missing)"
        else:
            self.last_language_source = "Multilingual BERT-Tiny prediction; Saaras metadata retained for comparison"

        if generation != self._response_generation or not self.running:
            return

        response_language_by_label = {
            "hi-IN": "hindi", "en-IN": "english", "pa-IN": "punjabi",
            "bn-IN": "bengali", "gu-IN": "gujarati", "kn-IN": "kannada",
            "ml-IN": "malayalam", "mr-IN": "marathi", "ta-IN": "tamil",
            "te-IN": "telugu", "as-IN": "assamese",
        }
        if result.get("available") and classifier_label in response_language_by_label:
            effective_language_code = classifier_label
            language_source = "classifier"
        else:
            # A classifier label of none means no language switch. Unavailable
            # predictions also preserve the previous effective language.
            effective_language_code = self.language_history[-1] if self.language_history else "hi-IN"
            language_source = "previous_turn_fallback"
        self.current_language_code = effective_language_code
        self.current_response_language = response_language_by_label.get(effective_language_code, "hindi")
        self.last_effective_language_code = effective_language_code
        self.last_effective_language_source = language_source
        # Store language codes only; never retain prior raw transcripts.
        self.language_history.append(effective_language_code)

        # Map language labels to BakBak's supported locale codes before generation,
        # because the streaming TTS worker starts while Qwen is still generating.
        tts_language_by_label = {
            "hi-IN": "hi", "en-IN": "en", "pa-IN": "pa", "bn-IN": "bn",
            "gu-IN": "gu", "kn-IN": "kn", "ml-IN": "ml", "mr-IN": "mr",
            "ta-IN": "ta", "te-IN": "te", "as-IN": "as",
        }
        tts_language = tts_language_by_label.get(
            effective_language_code,
            "en" if self.current_response_language == "english" else "hi",
        )

        # Start a durable-in-session turn record before generation so the UI can
        # show the caller's turn immediately and update the answer as it streams.
        self.turn_counter += 1
        turn_id = self.turn_counter
        self.active_turn_id = turn_id
        turn = {
            "id": turn_id,
            "user": transcript,
            "assistant": "",
            "status": "processing",
            "language": effective_language_code,
            "kb_status": "pending",
            "kb_used": False,
            "kb_chunks": 0,
            "kb_agent": self.agent_type,
            "kb_titles": [],
            "time": time.strftime("%H:%M:%S"),
        }
        self.conversation_turns.append(turn)
        self.conversation_history.append({"role": "user", "content": transcript})
        self.last_assistant_text = ""
        self.last_kb_status = "pending"
        self.last_kb_used = False
        self.last_kb_chunks = 0
        self.last_kb_agent = self.agent_type
        self.last_kb_titles = []

        streamed_tts = False
        self.llm_status = "Generating response"
        try:
            if self.llm_provider == "qwen" and os.getenv("AURA_STREAMING_VOICE", "true").strip().lower() in {"1", "true", "yes", "on"}:
                streamed_tts = True
                tts_queue = asyncio.Queue()
                self.last_tts_input = ""
                self.tts_status = "Waiting for first Qwen sentence"
                tts_worker = asyncio.create_task(
                    self._consume_streaming_tts(tts_queue, tts_language, generation)
                )
                async def publish_streamed_chunk(chunk: str):
                    current_turn = next(
                        (item for item in reversed(self.conversation_turns) if item["id"] == turn_id),
                        None,
                    )
                    self.last_assistant_text = (
                        f"{self.last_assistant_text} {chunk}".strip()
                    )
                    if current_turn is not None:
                        current_turn["assistant"] = self.last_assistant_text
                        current_turn["status"] = "streaming"
                    await tts_queue.put(chunk)

                try:
                    response = await self._get_llm_response(
                        transcript, detected_language, result,
                        on_text_chunk=publish_streamed_chunk,
                    )
                except Exception:
                    await tts_queue.put(None)
                    try:
                        await tts_worker
                    except Exception:
                        pass
                    raise
                self.llm_status = "Completed"
                await tts_queue.put(None)
                await tts_worker
            else:
                response = await self._get_llm_response(transcript, detected_language, result)
        except asyncio.CancelledError:
            current_turn = next(
                (item for item in reversed(self.conversation_turns) if item["id"] == turn_id),
                None,
            )
            if current_turn is not None:
                current_turn["status"] = "interrupted"
                current_turn["assistant"] = self.last_assistant_text
            if self.last_assistant_text:
                self.conversation_history.append(
                    {"role": "assistant", "content": self.last_assistant_text}
                )
            self.active_turn_id = None
            raise
        except Exception as exc:
            current_turn = next(
                (item for item in reversed(self.conversation_turns) if item["id"] == turn_id),
                None,
            )
            if current_turn is not None:
                current_turn["status"] = "failed"
                current_turn["assistant"] = self.last_assistant_text
            self.active_turn_id = None
            self.error_stage = "LLM" if not streamed_tts else "Streaming LLM/TTS"
            self.error = f"{type(exc).__name__}: {exc}"
            self.llm_status = "Failed"
            self.pipeline_status = "LLM/TTS failed"
            if streamed_tts:
                self.tts_status = "Streaming failed"
            return
        if generation != self._response_generation or not self.running:
            return

        if not response:
            self.error_stage = "LLM"
            self.error = "Sarvam LLM returned an empty response"
            self.llm_status = "Failed — empty response"
            self.pipeline_status = "LLM failed"
            return
        # A voice answer should not become a long monologue. Keep the text
        # bounded even if the model ignores the concise-answer instruction.
        response = self._limit_voice_response(response)
        self.last_assistant_text = response
        current_turn = next(
            (item for item in reversed(self.conversation_turns) if item["id"] == turn_id),
            None,
        )
        if current_turn is not None:
            current_turn["assistant"] = response
            current_turn["status"] = "completed"
        self.conversation_history.append({"role": "assistant", "content": response})
        self.active_turn_id = None

        if streamed_tts:
            # Audio was synthesized and queued concurrently with Qwen generation.
            self.pipeline_status = "Ready"
            self.stt_status = "Connected — waiting for speech"
            return

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
            return
        if generation != self._response_generation or not self.running:
            return
        if not audio:
            self.error_stage = "BakBak TTS"
            self.error = "BakBak returned empty audio"
            self.tts_status = "Failed — empty audio"
            self.pipeline_status = "BakBak failed"
            return
        if generation != self._response_generation or not self.running:
            return
        try:
            await self._enqueue_output(audio, generation=generation)
        except Exception as exc:
            self.error_stage = "WebRTC output"
            self.error = f"{type(exc).__name__}: {exc}"
            self.tts_status = "Audio generated, but output failed"
            self.pipeline_status = "WebRTC output failed"
            return
        self.tts_status = f"Ready — {len(audio):,} PCM bytes queued"
        self.pipeline_status = "Ready"
        self.stt_status = "Connected — waiting for speech"

    async def _play_opening_greeting(self, generation: int):
        """Speak one short agent introduction when the STT session begins."""
        greeting = OPENING_GREETINGS.get(
            self.agent_type, OPENING_GREETINGS["general"]
        )
        self.last_tts_input = greeting
        self.tts_status = "Preparing opening greeting with BakBak Raya"
        try:
            audio = await self._get_bakbak_audio(greeting, language="en")
            if generation != self._response_generation or not self.running:
                return
            if not audio:
                raise ValueError("BakBak returned empty opening-greeting audio")
            await self._enqueue_output(audio, generation=generation)
            if generation == self._response_generation and self.running:
                if not self.conversation_turns:
                    self.turn_counter += 1
                    greeting_turn = {
                        "id": self.turn_counter,
                        "user": "",
                        "assistant": greeting,
                        "status": "completed",
                        "language": "en-IN",
                        "kb_status": "not_needed",
                        "kb_used": False,
                        "kb_chunks": 0,
                        "kb_agent": self.agent_type,
                        "kb_titles": [],
                        "time": time.strftime("%H:%M:%S"),
                        "kind": "greeting",
                    }
                    self.conversation_turns.append(greeting_turn)
                    self.conversation_history.append(
                        {"role": "assistant", "content": greeting}
                    )
                self.tts_status = f"Ready — opening greeting ({len(audio):,} PCM bytes queued)"
                self.pipeline_status = "Ready"
                self.stt_status = "Connected — waiting for speech"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error_stage = "Opening greeting / BakBak TTS"
            self.error = f"{type(exc).__name__}: {exc}"
            self.tts_status = "Opening greeting failed"

    @staticmethod
    def _limit_voice_response(content: str) -> str:
        """Apply a conservative spoken-answer cap after removing model reasoning."""
        import re

        text = re.sub(r"\s+", " ", str(content or "")).strip()
        max_chars = max(250, int(os.getenv("VOICE_MAX_ANSWER_CHARS", "850")))
        max_sentences = max(2, min(int(os.getenv("VOICE_MAX_ANSWER_SENTENCES", "4")), 8))

        # Split only at sentence-ending punctuation followed by whitespace.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > max_sentences:
            text = " ".join(sentences[:max_sentences]).strip()
        if len(text) > max_chars:
            clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
            text = clipped + ("…" if clipped else "")
        return text

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        """Remove Qwen3 private reasoning tags before showing or speaking the answer."""
        import re

        text = str(content or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"^\s*</?think>\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _is_meaningful_short_turn(text: str) -> bool:
        normalized = " ".join(text.lower().strip().split()).strip(" .,!?:;")
        acknowledgements = {
            "hello", "hi", "hey", "yes", "yeah", "yep", "no", "nope",
            "okay", "ok", "thanks", "thank you", "sorry", "right",
            "haan", "han", "haan bilkul", "bilkul", "ji", "achha", "accha",
            "हाँ", "हां", "हाँ बिल्कुल", "हां बिल्कुल", "बिल्कुल", "जी", "अच्छा",
            "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਹਾਂ", "ਠੀਕ ਹੈ", "ঠিক আছে", "হ্যাঁ",
        }
        return normalized in acknowledgements

    @staticmethod
    def _should_buffer_short_turn(text: str) -> bool:
        words = text.split()
        if not words or len(words) > 2:
            return False
        if CloudVoiceAgent._is_meaningful_short_turn(text):
            return False
        # Conservative heuristic: a short finalized ASR chunk may be a name,
        # company, acronym, fragment, or noisy recognition. It is not labelled
        # "none"; we wait briefly for more lexical context.
        return True

    async def _flush_turn_buffer_after_timeout(self, original_event: dict):
        try:
            await asyncio.sleep(self.turn_buffer_timeout_ms / 1000.0)
            if not self.running or not self.turn_buffer_text:
                return
            buffered = self.turn_buffer_text
            buffered_event = self.turn_buffer_event or original_event
            self.turn_buffer_text = ""
            self.turn_buffer_event = {}
            self.turn_buffer_task = None
            self.last_turn_buffer_decision = "TIMEOUT → FLUSH_AND_CLASSIFY"
            self.last_combined_turn_text = buffered
            self.live_transcript = buffered
            self._schedule_response(buffered, buffered_event)
        except asyncio.CancelledError:
            return

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

    def _should_use_knowledge_base(self, user_text: str) -> bool:
        """Avoid KB retrieval for turns that do not need agent-specific facts."""
        text = (user_text or "").lower()

        if self.agent_type == "customer_service":
            admission_terms = (
                "admission", "admissions", "eligibility", "eligible",
                "application process", "how to apply", "apply for",
                "scholarship", "tuition fee", "fees", "cutoff",
                "cut-off", "deadline", "required documents",
                "dv university", "university admission",
                "university fees", "university course",
            )
            return any(term in text for term in admission_terms)

        if self.agent_type == "tutor":
            course_terms = (
                "syllabus", "curriculum", "lesson material",
                "course material", "study material", "course outline",
                "learning path", "chapter in the course",
                "module in the course", "assignment in the course",
                "what does the course cover",
            )
            return any(term in text for term in course_terms)

        if self.agent_type == "general":
            organisation_terms = (
                "our company", "our product", "our service",
                "our organisation", "our organization",
                "company policy", "product features",
                "internal process", "project documentation",
                "our pricing", "our refund policy",
            )
            return any(term in text for term in organisation_terms)

        return False

    async def _get_llm_response(
        self,
        user_text: str,
        detected_language: str = "unknown",
        classifier_result: dict | None = None,
        on_text_chunk=None,
    ) -> str:
        self.last_llm_input = (
            f"Current language code: {self.current_language_code}\n"
            f"Latest transcription: {user_text}"
        )
        started = time.perf_counter()
        url = os.getenv("SARVAM_LLM_URL", "https://api.sarvam.ai/v1/chat/completions")
        headers = {
            "api-subscription-key": self.sarvam_api_key,
            "Content-Type": "application/json",
        }
        system_prompt = get_system_prompt(self.agent_type)
        classifier_result = classifier_result or {}
        classifier_label = str(classifier_result.get("label", "none")) if classifier_result.get("available") else "none"
        classifier_confidence_value = classifier_result.get("confidence")
        classifier_available = bool(classifier_result.get("available"))
        current_language = getattr(self, "current_response_language", "hindi")
        effective_language_code = getattr(self, "current_language_code", "hi-IN")
        language_names = {
            "hi-IN": "Hindi", "en-IN": "English", "pa-IN": "Punjabi", "bn-IN": "Bengali",
            "gu-IN": "Gujarati", "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi",
            "ta-IN": "Tamil", "te-IN": "Telugu", "as-IN": "Assamese",
        }
        if classifier_available and classifier_label in language_names:
            language_instruction = (
                f"Effective response language: {language_names[effective_language_code]} "
                f"({effective_language_code}). This is the current turn's classifier result. "
            )
        else:
            language_instruction = (
                f"Classifier returned None/unavailable. Reuse the previous effective language "
                f"without inferring or switching: {language_names.get(effective_language_code, current_language)} "
                f"({effective_language_code}). "
            )
        language_instruction += (
            "The effective language code is authoritative. Switch languages only when the "
            "classifier returns a supported language code. A none result means no language "
            "switch: reuse the previous effective code. Preserve natural code-mixing when "
            "appropriate, but do not mistake borrowed words for a language switch. Answer the "
            "caller's intent directly; do not translate or merely restate the transcription."
        )
        language_hint = (
            f"Classifier result: {classifier_label}; confidence: "
            f"{classifier_confidence_value if classifier_confidence_value is not None else 'N/A'}; "
            f"language source: {getattr(self, 'last_effective_language_source', 'unknown')}."
        )
        prompt_context = (
            f"{system_prompt}\n\n"
            f"CURRENT TURN LANGUAGE: {language_names.get(effective_language_code, current_language)} "
            f"({effective_language_code}).\n{language_instruction}\n{language_hint}\n"
            "Use the complete conversation messages supplied after this system instruction "
            "to understand references, preferences, decisions, and follow-up questions. "
            "The latest user message is the current request. Never invent missing facts."
        )
        # Selective retrieval avoids embedding calls on ordinary conversational turns.
        # Telemetry is recorded for every turn, including turns that do not need RAG.
        knowledge_context = ""
        self.last_kb_status = "not_needed"
        self.last_kb_used = False
        self.last_kb_chunks = 0
        self.last_kb_agent = self.agent_type
        self.last_kb_titles = []

        if self._should_use_knowledge_base(user_text):
            try:
                kb_status = await asyncio.to_thread(
                    knowledge_base.status, self.agent_type
                )
                if not kb_status.get("enabled"):
                    self.last_kb_status = "disabled"
                elif int(kb_status.get("chunks", 0)) <= 0:
                    self.last_kb_status = "empty"
                else:
                    passages = await asyncio.to_thread(
                        knowledge_base.retrieve, self.agent_type, user_text, 4
                    )
                    if passages:
                        self.last_kb_status = "used"
                        self.last_kb_used = True
                        self.last_kb_chunks = len(passages)
                        self.last_kb_titles = list(dict.fromkeys(
                            str(item.get("title", "Untitled")) for item in passages
                        ))
                        formatted = "\n\n".join(
                            f"[{item['title']}]\n{item['content']}" for item in passages
                        )
                        knowledge_context = (
                            "\n\nKnowledge Base context "
                            "(use only relevant information; do not invent facts):\n"
                            + formatted
                            + "\nIf these passages do not answer the question, "
                            "say so plainly."
                        )
                    else:
                        self.last_kb_status = "no_match"
            except Exception as exc:
                self.last_kb_status = "unavailable"
                self.error = (
                    f"Knowledge Base unavailable: {type(exc).__name__}: {exc}"
                )

        current_turn = next(
            (item for item in reversed(getattr(self, "conversation_turns", []))
             if item.get("id") == getattr(self, "active_turn_id", None)),
            None,
        )
        if current_turn is not None:
            current_turn.update({
                "kb_status": self.last_kb_status,
                "kb_used": self.last_kb_used,
                "kb_chunks": self.last_kb_chunks,
                "kb_agent": self.last_kb_agent,
                "kb_titles": self.last_kb_titles,
            })

        prompt_context += knowledge_context
        conversation_messages = list(getattr(self, "conversation_history", []))
        if (
            not conversation_messages
            or conversation_messages[-1].get("role") != "user"
            or conversation_messages[-1].get("content") != user_text
        ):
            conversation_messages.append({"role": "user", "content": user_text})

        if self.last_kb_status in {"disabled", "empty", "no_match", "unavailable"}:
            if self.agent_type == "customer_service":
                prompt_context += (
                    f"\n\nADMISSION_KB_STATUS: {self.last_kb_status.upper()}. "
                    "The requested institution-specific information was not retrieved. "
                    "Do not guess course availability, fees, duration, eligibility, dates, "
                    "deadlines, required documents, scholarships, or policies. Explain "
                    "that the specific detail cannot be verified from the available "
                    "information and suggest contacting the admissions office."
                )
            elif self.agent_type == "tutor":
                prompt_context += (
                    f"\n\nCOURSE_KB_STATUS: {self.last_kb_status.upper()}. "
                    "Do not invent institution-specific syllabus or course details. "
                    "You may still explain general concepts and clearly label them as "
                    "general knowledge."
                )
            elif self.agent_type == "general":
                prompt_context += (
                    f"\n\nORGANISATION_KB_STATUS: {self.last_kb_status.upper()}. "
                    "The requested organisation-specific information was not retrieved. "
                    "Do not guess or invent company, product, service, pricing, refund, "
                    "or internal-policy facts. Clearly say that the specific information "
                    "is not available or cannot be verified right now."
                )

        max_tokens = int(os.getenv("SARVAM_LLM_MAX_TOKENS", "120"))

        if self.llm_provider == "qwen":
            if not self.qwen_api_key:
                raise ValueError("Qwen selected but QWEN_API_KEY is not configured on the backend.")
            qwen_payload = {
                "model": self.qwen_model,
                "messages": [
                    {"role": "system", "content": prompt_context},
                    *conversation_messages,
                ],
                # Keep Qwen concise and disable private reasoning for voice turns.
                "max_tokens": max(64, min(int(os.getenv("QWEN_MAX_TOKENS", "128")), 256)),
                "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.3")),
                "chat_template_kwargs": {
                    # Never stream private reasoning into the caller's TTS audio.
                    "enable_thinking": False if on_text_chunk is not None else (
                        os.getenv("QWEN_ENABLE_THINKING", "false").strip().lower()
                        in {"1", "true", "yes", "on"}
                    )
                },
            }

            if on_text_chunk is not None:
                qwen_payload["stream"] = True
                collected = []
                pending_text = ""
                max_chars = max(250, int(os.getenv("VOICE_MAX_ANSWER_CHARS", "850")))

                async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
                    async with client.stream(
                        "POST",
                        self.qwen_chat_completions_url,
                        headers={
                            "Authorization": f"Bearer {self.qwen_api_key}",
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                        },
                        json=qwen_payload,
                    ) as response:
                        if response.is_error:
                            body = (await response.aread()).decode("utf-8", errors="replace")
                            raise RuntimeError(f"Qwen HTTP {response.status_code}: {body[:2000]}")

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                packet = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            choices = packet.get("choices") or []
                            if not choices:
                                continue
                            delta = (choices[0].get("delta") or {}).get("content")
                            if not isinstance(delta, str) or not delta:
                                continue

                            collected.append(delta)
                            pending_text += delta

                            # Emit complete spoken sentences; also split very long
                            # clauses so the TTS request does not wait for the full turn.
                            while pending_text:
                                import re
                                boundary = re.search(r"[.!?।](?:\s|$)", pending_text)
                                cut = boundary.end() if boundary else 0
                                if not cut and len(pending_text) >= 180:
                                    split_at = pending_text.rfind(" ", 0, 180)
                                    if split_at >= 70:
                                        cut = split_at
                                if not cut:
                                    break
                                speakable = pending_text[:cut].strip()
                                pending_text = pending_text[cut:].lstrip()
                                if speakable:
                                    await on_text_chunk(speakable)

                            # Bound speech to the same voice-answer limit as the
                            # non-streaming path. Qwen may finish generating extra text,
                            # but excess text is not sent to the caller.
                            emitted_chars = sum(len(part) for part in collected)
                            if emitted_chars >= max_chars:
                                break

                if pending_text.strip():
                    await on_text_chunk(pending_text.strip())
                self.llm_latency_ms = round((time.perf_counter() - started) * 1000, 1)
                answer = self._extract_final_answer("".join(collected))
                if not answer:
                    raise RuntimeError("Hosted Qwen returned no final answer after streaming.")
                return answer

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.qwen_chat_completions_url,
                    headers={
                        "Authorization": f"Bearer {self.qwen_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=qwen_payload,
                )
                if response.is_error:
                    raise RuntimeError(
                        f"Qwen HTTP {response.status_code}: {response.text[:2000]}"
                    )
                data = response.json()
            self.llm_latency_ms = round((time.perf_counter() - started) * 1000, 1)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            answer = self._extract_final_answer(content)
            if not answer:
                raise RuntimeError("Hosted Qwen returned no final answer after reasoning.")
            return answer

        if self.llm_model == "gemma-4-26b-a4b-it":
            if not self.gemma_api_key:
                raise ValueError(
                    "Gemma selected but GEMINI_API_KEY (or GEMMA_API_KEY) is missing from Streamlit secrets."
                )
            # Send instructions through the API's dedicated systemInstruction field.
            # The caller's utterance is the only user content, so the selected model
            # performs reasoning on the request instead of echoing internal context.
            gemma_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.llm_model}:generateContent"
            )
            gemma_payload = {
                "systemInstruction": {
                    "parts": [{"text": prompt_context}]
                },
                "contents": [
                    {
                        "role": "model" if item.get("role") == "assistant" else "user",
                        "parts": [{"text": item.get("content", "")}],
                    }
                    for item in conversation_messages
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.4,
                },
            }
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(
                    gemma_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.gemma_api_key,
                    },
                    json=gemma_payload,
                )
                response.raise_for_status()
                data = response.json()
            self.llm_latency_ms = round((time.perf_counter() - started) * 1000, 1)
            candidates = data.get("candidates") or []
            if not candidates:
                feedback = data.get("promptFeedback", {})
                raise RuntimeError(
                    "Gemma returned no response candidate. "
                    f"Prompt feedback: {feedback or 'not provided'}"
                )
            parts = candidates[0].get("content", {}).get("parts", [])
            answer = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            ).strip()
            if not answer:
                raise RuntimeError("Gemma returned an empty text response.")
            return answer

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": prompt_context},
                *conversation_messages,
            ],
            "max_tokens": max_tokens,
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

    async def _stream_bakbak_audio(self, text: str, language: str):
        """Yield PCM16/24 kHz chunks from BakBak Raya's SSE streaming endpoint."""
        stream_url = os.getenv("BAKBAK_TTS_STREAM_URL", "").strip()
        if not stream_url:
            base_url = os.getenv(
                "BAKBAK_TTS_URL",
                "https://hub.getraya.app/v1/text-to-speech",
            ).rstrip("/")
            if base_url.endswith("/text-to-speech"):
                stream_url = base_url + "/stream"
            else:
                stream_url = "https://hub.getraya.app/v1/text-to-speech/stream"

        headers = {
            "X-API-Key": self.bakbak_api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "text": text,
            "voice_id": self.bakbak_voice_id,
            "model": self.tts_model,
            "language": language,
            "sample_rate": 24000,
            "speed": float(os.getenv("BAKBAK_TTS_SPEED", "1.0")),
        }
        received_audio = False
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            async with client.stream("POST", stream_url, headers=headers, json=payload) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"BakBak streaming HTTP {response.status_code}: {body[:1500]}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        packet = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    encoded = packet.get("data")
                    if not encoded:
                        if packet.get("type") == "error":
                            raise RuntimeError(f"BakBak streaming error: {packet}")
                        continue

                    audio_f32 = base64.b64decode(encoded)
                    if len(audio_f32) < 4:
                        continue
                    usable_length = len(audio_f32) - (len(audio_f32) % 4)
                    samples = np.frombuffer(audio_f32[:usable_length], dtype="<f4")
                    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    if pcm16:
                        received_audio = True
                        self.tts_audio_bytes += len(pcm16)
                        yield pcm16

                    if packet.get("done") or packet.get("type") == "done":
                        break

        if not received_audio:
            raise RuntimeError("BakBak streaming completed without returning audio chunks.")

    async def _consume_streaming_tts(self, text_queue, language: str, generation: int):
        """Synthesize sentence chunks while Qwen generates and pace PCM to WebRTC."""
        started = time.perf_counter()
        pending_pcm = bytearray()
        total_pcm_bytes = 0
        self.tts_audio_bytes = 0
        self.output_bytes_pushed = 0

        async def push_available(final: bool = False):
            nonlocal total_pcm_bytes
            while len(pending_pcm) >= 960 or (final and pending_pcm):
                if generation != self._response_generation or not self.running:
                    return
                valid = min(960, len(pending_pcm))
                frame = bytes(pending_pcm[:valid])
                del pending_pcm[:valid]
                if valid < 960:
                    frame += bytes(960 - valid)
                if self.output_source is None:
                    raise RuntimeError("WebRTC output source is not configured")
                self.output_source.push(frame)
                total_pcm_bytes += valid
                self.output_bytes_pushed += valid
                await asyncio.sleep(0.020)

        while True:
            text_chunk = await text_queue.get()
            if text_chunk is None:
                break
            if generation != self._response_generation or not self.running:
                return
            self.last_tts_input = (self.last_tts_input + " " + text_chunk).strip()
            self.tts_status = "Streaming speech with BakBak Raya"

            received_for_sentence = False
            try:
                async for pcm_chunk in self._stream_bakbak_audio(text_chunk, language):
                    received_for_sentence = True
                    pending_pcm.extend(pcm_chunk)
                    await push_available()
            except Exception:
                # Safe fallback only when the stream failed before delivering any
                # audio for this sentence; never replay a partially spoken sentence.
                if received_for_sentence:
                    raise
                fallback_audio = await self._get_bakbak_audio(text_chunk, language=language)
                pending_pcm.extend(fallback_audio)
                await push_available()

        await push_available(final=True)
        self.tts_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.tts_status = f"Ready — {total_pcm_bytes:,} streamed PCM bytes queued"

    async def _enqueue_output(self, pcm16_24k: bytes, generation: int):
        """Play PCM in paced 20 ms frames so interruption can pre-empt playback."""
        if len(pcm16_24k) % 2:
            raise ValueError("BakBak returned an odd-length PCM payload")
        if self.output_source is None:
            raise RuntimeError("WebRTC output source is not configured")

        # 24 kHz * 20 ms * 2 bytes/sample = 960 bytes per mono PCM16 frame.
        frame_bytes = 960
        for offset in range(0, len(pcm16_24k), frame_bytes):
            if generation != self._response_generation or not self.running:
                return
            frame = pcm16_24k[offset:offset + frame_bytes]
            if len(frame) < frame_bytes:
                frame += bytes(frame_bytes - len(frame))
            self.output_source.push(frame)
            self.output_bytes_pushed += min(frame_bytes, len(pcm16_24k) - offset)
            # Yield at the playback cadence. STT partial events can now cancel
            # this task and clear the source instead of waiting for the full
            # response audio to be queued in one synchronous operation.
            await asyncio.sleep(0.020)

    def ingest_webrtc_frame(self, frame):
        """Convert browser audio to mono 16 kHz signed PCM for Saaras."""
        import av
        if self._input_resampler is None:
            self._input_resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for converted in self._input_resampler.resample(frame):
            pcm = converted.to_ndarray().astype(np.int16).tobytes()
            self.push_audio(pcm)
