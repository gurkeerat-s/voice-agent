"""
Voice setup for Orpheus TTS.

Orpheus uses preset voices (no cloning needed).
This module just initializes the TTS pipeline.

Usage:
    tts = setup_voice()
"""

from pipeline.tts import StreamingTTS
from config import config


def setup_voice(reference_audio_path: str | None = None) -> StreamingTTS:
    """
    Initialize TTS with Orpheus.

    Returns:
        A StreamingTTS instance ready to synthesize speech.
    """
    tts = StreamingTTS()
    tts.load_model()
    return tts
