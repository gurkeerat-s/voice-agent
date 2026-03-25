from pydantic_settings import BaseSettings
from pydantic import Field


class VADConfig(BaseSettings):
    """Voice Activity Detection thresholds."""
    # Silero VAD confidence threshold (0-1). Higher = less sensitive.
    speech_threshold: float = 0.5
    # Silence longer than this = end of turn (ms)
    end_of_turn_ms: int = 800
    # Silence in this range = short pause (backchannel candidate)
    short_pause_min_ms: int = 300
    short_pause_max_ms: int = 600
    # Minimum speech duration to count as barge-in (ms)
    barge_in_min_ms: int = 150
    # Audio chunk size fed to VAD (ms)
    chunk_duration_ms: int = 200


class STTConfig(BaseSettings):
    """Speech-to-text settings."""
    model_size: str = "distil-large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    # Emit partial transcript every N ms
    partial_interval_ms: int = 500
    # Beam size for final transcription (higher = more accurate, slower)
    beam_size: int = 5
    # Beam size for partial/streaming transcription (fast)
    partial_beam_size: int = 1


class LLMConfig(BaseSettings):
    """LLM inference settings."""
    # vLLM serves an OpenAI-compatible API on this URL
    api_base: str = "http://localhost:8000/v1"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ"
    max_tokens: int = 256
    temperature: float = 0.7
    # System prompt for the voice agent
    system_prompt: str = (
        "You are a friendly, helpful voice assistant. Keep responses concise "
        "and conversational — 1-3 sentences max unless the user asks for detail. "
        "Do not use markdown, bullet points, or formatting. Speak naturally."
    )
    # Enable prefix caching in vLLM for KV-cache warming
    enable_prefix_caching: bool = True


class TTSConfig(BaseSettings):
    """Text-to-speech settings."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cuda"
    # Path to reference audio for voice cloning (6-10s WAV)
    reference_audio_path: str = "voice/reference.wav"
    # Sample rate of generated audio
    sample_rate: int = 24000
    # Streaming chunk size (in model tokens, not audio samples)
    stream_chunk_size: int = 20
    # Crossfade duration when transitioning filler -> real response (ms)
    crossfade_ms: int = 100
    # Language for XTTS
    language: str = "en"


class AudioConfig(BaseSettings):
    """Audio I/O settings."""
    # Sample rate from browser mic (WebAudio default)
    input_sample_rate: int = 48000
    # Internal processing sample rate (Whisper wants 16kHz)
    stt_sample_rate: int = 16000
    # Output sample rate (XTTS outputs 24kHz)
    output_sample_rate: int = 24000
    # Chunk size sent over WebSocket (ms)
    ws_chunk_ms: int = 200
    # Target RMS for volume normalization
    target_rms: float = 0.05


class ServerConfig(BaseSettings):
    """WebSocket server settings."""
    host: str = "0.0.0.0"
    port: int = 8765


class Config(BaseSettings):
    """Root configuration — single source of truth."""
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


# Global config instance
config = Config()
