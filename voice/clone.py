"""
Voice cloning setup.

Handles loading a reference audio file and initializing the TTS
with the cloned voice. Also validates the reference audio quality.

Usage:
    tts = setup_voice("voice/reference.wav")
    # tts is now ready to synthesize in the cloned voice
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.tts import StreamingTTS
from config import config


def validate_reference_audio(path: str) -> dict:
    """
    Check that the reference audio is suitable for voice cloning.

    Returns a dict with validation results and audio info.
    """
    audio_path = Path(path)
    issues = []

    if not audio_path.exists():
        return {"valid": False, "issues": ["File not found"], "info": {}}

    if audio_path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
        issues.append(f"Unexpected format: {audio_path.suffix}. WAV recommended.")

    # Load and check properties
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        issues.append("Audio is stereo — will be mixed to mono.")

    duration_s = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))

    info = {
        "duration_s": round(duration_s, 1),
        "sample_rate": sr,
        "rms": round(rms, 4),
        "channels": 1 if audio.ndim == 1 else audio.shape[1],
    }

    if duration_s < 3:
        issues.append(f"Too short ({duration_s:.1f}s). Need at least 6s for good cloning.")
    elif duration_s < 6:
        issues.append(f"Short ({duration_s:.1f}s). 6-10s recommended for best quality.")
    elif duration_s > 30:
        issues.append(f"Long ({duration_s:.1f}s). Will use first 30s only.")

    if rms < 0.01:
        issues.append("Audio is very quiet. Consider normalizing volume.")
    elif rms > 0.5:
        issues.append("Audio may be clipping. Consider reducing volume.")

    return {
        "valid": len([i for i in issues if "Too short" in i or "not found" in i]) == 0,
        "issues": issues,
        "info": info,
    }


def setup_voice(reference_audio_path: str | None = None) -> StreamingTTS:
    """
    Initialize TTS with a cloned voice.

    Args:
        reference_audio_path: Path to reference WAV. Defaults to config value.

    Returns:
        A StreamingTTS instance ready to synthesize speech.
    """
    ref_path = reference_audio_path or config.tts.reference_audio_path

    # Validate reference audio
    validation = validate_reference_audio(ref_path)

    if not validation["valid"]:
        raise ValueError(
            f"Reference audio not usable: {validation['issues']}"
        )

    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"  Warning: {issue}")

    info = validation["info"]
    print(f"Reference audio: {info['duration_s']}s, {info['sample_rate']}Hz, RMS={info['rms']}")

    # Initialize TTS
    tts = StreamingTTS()
    print("Loading XTTS-v2 model...")
    tts.load_model()
    print("Cloning voice from reference audio...")
    tts.load_voice(ref_path)
    print("Voice cloning complete.")

    return tts
