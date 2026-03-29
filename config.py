from pydantic_settings import BaseSettings
from pydantic import Field


class VADConfig(BaseSettings):
    """Voice Activity Detection thresholds."""
    speech_threshold: float = 0.5
    end_of_turn_ms: int = 800
    short_pause_min_ms: int = 300
    short_pause_max_ms: int = 600
    barge_in_min_ms: int = 150
    chunk_duration_ms: int = 200


class STTConfig(BaseSettings):
    """Speech-to-text settings."""
    model_size: str = "distil-large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    partial_interval_ms: int = 500
    beam_size: int = 5
    partial_beam_size: int = 1


class LLMConfig(BaseSettings):
    """LLM inference settings."""
    api_base: str = "http://localhost:11434/v1"
    model: str = "llama3.1:8b"
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: str = (
        "You are a friendly, helpful voice assistant. Keep responses concise "
        "and conversational — 1-3 sentences max unless the user asks for detail. "
        "Do not use markdown, bullet points, or formatting. Speak naturally."
    )
    enable_prefix_caching: bool = True


class TTSConfig(BaseSettings):
    """Text-to-speech settings (Orpheus TTS)."""
    # Available voices: tara, leah, jess, leo, dan, mia, zac, zoe
    voice: str = "tara"
    # Sample rate (Orpheus/SNAC outputs 24kHz)
    sample_rate: int = 24000
    # Crossfade duration when transitioning filler -> real response (ms)
    crossfade_ms: int = 100


class AudioConfig(BaseSettings):
    """Audio I/O settings."""
    input_sample_rate: int = 48000
    stt_sample_rate: int = 16000
    output_sample_rate: int = 24000
    ws_chunk_ms: int = 200
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
