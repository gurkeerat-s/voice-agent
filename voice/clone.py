"""
Voice setup for Kokoro TTS.

Kokoro uses preset voices (no cloning needed).
This module just initializes the TTS pipeline.

Usage:
    tts = setup_voice()
"""

from pipeline.tts import StreamingTTS
from config import config


def setup_voice(reference_audio_path: str | None = None) -> StreamingTTS:
    """
    Initialize TTS. Reference audio path is ignored (Kokoro uses preset voices).

    Returns:
        A StreamingTTS instance ready to synthesize speech.
    """
    tts = StreamingTTS()
    print(f"Loading Kokoro TTS (voice: {config.tts.voice})...")
    tts.load_model()
    print("Kokoro TTS ready.")
    return tts
