"""
Voice setup for CSM-1B with LoRA adapter.

Loads the base model + fine-tuned adapter and returns
a ready-to-use TTS instance.

Usage:
    tts = setup_voice()
"""

from pipeline.tts import StreamingTTS
from config import config


def setup_voice(reference_audio_path: str | None = None) -> StreamingTTS:
    """
    Initialize TTS with the fine-tuned CSM-1B model.

    Returns:
        A StreamingTTS instance ready to synthesize speech.
    """
    tts = StreamingTTS()
    print(f"Loading CSM-1B + LoRA adapter ({config.tts.adapter_path})...")
    tts.load_model()
    print("CSM-1B TTS ready.")
    return tts
