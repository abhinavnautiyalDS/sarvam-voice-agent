export type AuraVoiceState = {
  pipeline_status?: string;
  stt_status?: string;
  llm_status?: string;
  tts_status?: string;
  transcript?: string;
  user_text?: string;
  detected_language?: string;
  confidence?: number | null;
  classifier_latency_ms?: number | null;
  assistant_text?: string;
  llm_latency_ms?: number | null;
  tts_latency_ms?: number | null;
  error?: string;
  error_stage?: string;
};

type VoiceClientOptions = {
  onState: (state: AuraVoiceState) => void;
  onStatus: (status: string) => void;
  onAudioLevel: (level: number) => void;
};

export class AuraVoiceClient {
  private socket?: WebSocket;
  private stream?: MediaStream;
  private context?: AudioContext;
  private processor?: ScriptProcessorNode;
  private source?: MediaStreamAudioSourceNode;
  private playbackAt = 0;
  private stopped = false;
  private readonly options: VoiceClientOptions;
  private analyser?: AnalyserNode;
  private animationFrame = 0;
  private readonly agent: string;

  constructor(options: VoiceClientOptions, agent: string = "general") {
    this.options = options;
    this.agent = agent;
  }

  async start() {
    this.stopped = false;
    const secure = window.location.protocol === 'https:';
    const configured = import.meta.env.VITE_AURA_WS_URL as string | undefined;
    const baseUrl = configured || `${secure ? 'wss' : 'ws'}://${window.location.host}/ws/voice`;
    const separator = baseUrl.includes("?") ? "&" : "?";
    const wsUrl = `${baseUrl}${separator}agent=${encodeURIComponent(this.agent)}`;
    this.options.onStatus('Connecting to AURA backend…');

    this.socket = new WebSocket(wsUrl);
    this.socket.binaryType = 'arraybuffer';
    this.socket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const payload = JSON.parse(event.data);
        if (payload.type === 'state') this.options.onState(payload);
        if (payload.type === 'ready') this.options.onStatus('Connected · microphone is live');
        if (payload.type === 'error') this.options.onStatus(payload.message || 'Backend error');
      } else {
        this.playPcm(event.data as ArrayBuffer);
      }
    };
    this.socket.onerror = () => this.options.onStatus('WebSocket connection failed');
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error('Backend connection timed out')), 10000);
      this.socket!.addEventListener('open', () => { window.clearTimeout(timeout); resolve(); }, { once: true });
      this.socket!.addEventListener('error', () => { window.clearTimeout(timeout); reject(new Error('Could not connect to AURA backend')); }, { once: true });
    });

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.context = new AudioContext();
    await this.context.resume();
    this.source = this.context.createMediaStreamSource(this.stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 256;
    this.source.connect(this.analyser);
    this.trackInputLevel();
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN || this.stopped) return;
      const input = event.inputBuffer.getChannelData(0);
      const ratio = this.context!.sampleRate / 16000;
      const length = Math.floor(input.length / ratio);
      const pcm = new Int16Array(length);
      for (let i = 0; i < length; i++) {
        const position = i * ratio;
        const left = Math.floor(position);
        const fraction = position - left;
        const sample = input[left] * (1 - fraction) + (input[Math.min(left + 1, input.length - 1)] || 0) * fraction;
        pcm[i] = Math.max(-1, Math.min(1, sample)) * (sample < 0 ? 32768 : 32767);
      }
      this.socket.send(pcm.buffer);
    };
    this.source.connect(this.processor);
    // Keep the processor active without routing microphone audio to speakers.
    const mute = this.context.createGain();
    mute.gain.value = 0;
    this.processor.connect(mute);
    mute.connect(this.context.destination);
    this.options.onStatus('Microphone connected · listening');
  }

  private trackInputLevel() {
    if (!this.analyser || this.stopped) return;
    const samples = new Uint8Array(this.analyser.fftSize);
    this.analyser.getByteTimeDomainData(samples);
    let energy = 0;
    for (const sample of samples) { const centered = (sample - 128) / 128; energy += centered * centered; }
    const rms = Math.sqrt(energy / samples.length);
    const level = Math.min(1, Math.max(0, (rms - 0.012) * 7));
    this.options.onAudioLevel(level);
    this.animationFrame = window.requestAnimationFrame(() => this.trackInputLevel());
  }

  private playPcm(data: ArrayBuffer) {
    if (!this.context || !data.byteLength) return;
    const pcm = new Int16Array(data);
    const audio = this.context.createBuffer(1, pcm.length, 24000);
    const samples = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
    const node = this.context.createBufferSource();
    node.buffer = audio;
    node.connect(this.context.destination);
    const startAt = Math.max(this.context.currentTime + 0.03, this.playbackAt);
    node.start(startAt);
    this.playbackAt = startAt + audio.duration;
  }

  stop() {
    this.stopped = true;
    if (this.animationFrame) window.cancelAnimationFrame(this.animationFrame);
    this.options.onAudioLevel(0);
    this.analyser?.disconnect();
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.context?.close();
    if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close();
    this.options.onStatus('Voice session stopped');
  }
}
