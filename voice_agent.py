import os
import asyncio
import httpx

from dotenv import load_dotenv
from system_prompt import get_system_prompt
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

load_dotenv()

SARVAM_API_KEY = os.environ["SARVAM_API_KEY"]
BAKBAK_API_KEY = os.environ["BAKBAK_API_KEY"]
BAKBAK_VOICE_ID = os.environ["BAKBAK_VOICE_ID"]


class BakBakTTSService(TTSService):
    def __init__(
        self, api_key: str, voice_id: str, model: str = "m1",
        language: str = "hi", sample_rate: int = 24000,
        speed: float = 1.0, **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.language = language
        self._bakbak_sample_rate = sample_rate
        self.speed = speed

    async def run_tts(self, text: str):
        url = os.getenv("BAKBAK_TTS_URL", "https://hub.getraya.app/v1/text-to-speech")
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "language": self.language,
            "codec": "wav",
            "sample_rate": self._bakbak_sample_rate,
            "speed": self.speed,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            audio_data = response.content
        if audio_data[:4] == b"RIFF":
            import io
            import soundfile as sf
            waveform, rate = sf.read(io.BytesIO(audio_data), dtype="int16")
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1).astype("int16")
            if rate != self._bakbak_sample_rate:
                raise ValueError(f"Expected {self._bakbak_sample_rate} Hz TTS audio, got {rate} Hz")
            audio_data = waveform.tobytes()
        await self.push_frame(TTSAudioRawFrame(
            audio=audio_data,
            sample_rate=self._bakbak_sample_rate,
            num_channels=1,
        ))


async def run_voice_agent():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            input_device_index=int(os.getenv("AUDIO_INPUT_DEVICE_INDEX", "4")),
        )
    )
    stt = SarvamRealtimeSTTService(
        api_key=SARVAM_API_KEY,
        model=os.getenv("SARVAM_STT_MODEL", "saaras:v3-realtime"),
        language_code=os.getenv("SARVAM_STT_LANGUAGE", "unknown"),
        endpointing=os.getenv("SARVAM_STT_ENDPOINTING", "vad"),
        stream_type=os.getenv("SARVAM_STT_STREAM_TYPE", "fast"),
        mode=os.getenv("SARVAM_STT_MODE", "codemix"),
        threshold=float(os.getenv("SARVAM_STT_THRESHOLD", "0.3")),
        silence_duration_ms=int(os.getenv("SARVAM_SILENCE_DURATION_MS", "700")),
        min_speech_duration_ms=int(os.getenv("SARVAM_MIN_SPEECH_DURATION_MS", "200")),
    )
    llm = SarvamLLMService(
        api_key=SARVAM_API_KEY,
        base_url=os.getenv("SARVAM_LLM_BASE_URL", "https://api.sarvam.ai/v1"),
        settings=SarvamLLMService.Settings(
            model=os.getenv("SARVAM_LLM_MODEL", "sarvam-105b"),
            system_instruction=get_system_prompt(),
            max_tokens=int(os.getenv("SARVAM_LLM_MAX_TOKENS", "150")),
        ),
    )
    tts = BakBakTTSService(
        api_key=BAKBAK_API_KEY,
        voice_id=BAKBAK_VOICE_ID,
        model=os.getenv("BAKBAK_TTS_MODEL", "m1"),
        language=os.getenv("BAKBAK_TTS_LANGUAGE", "hi"),
        sample_rate=int(os.getenv("BAKBAK_TTS_SAMPLE_RATE", "24000")),
        speed=float(os.getenv("BAKBAK_TTS_SPEED", "1.0")),
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams()
    )
    pipeline = Pipeline([
        transport.input(), stt, user_aggregator, llm, tts, assistant_aggregator, transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
    )
    print("Voice agent started. Speak into the selected microphone.")
    print("Note: transcript classifier diagnostics currently run in the Streamlit WebRTC path.")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(run_voice_agent())
