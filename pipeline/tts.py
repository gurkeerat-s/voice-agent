"""
TTS using Kokoro-82M.

High-quality, fast text-to-speech with preset voices.
Outputs 24kHz float32 mono audio. Supports sentence-level streaming.

Usage:
    tts = StreamingTTS()
    tts.load_model()

    audio = tts.synthesize_full("Hello, how can I help?")
    blended = tts.crossfade(filler_audio, first_real_chunk)
"""

import numpy as np
from kokoro import KPipeline

from config import config


class StreamingTTS:
    """
    Kokoro-82M TTS wrapper.
    """

    def __init__(self):
        cfg = config.tts
        self.sample_rate = cfg.sample_rate  # 24000
        self.crossfade_ms = cfg.crossfade_ms
        self.voice = cfg.voice
        self.speed = cfg.speed

        self.pipeline: KPipeline | None = None

    def load_model(self):
        """Load the Kokoro pipeline. Call once at startup."""
        # Language code: first letter of voice ID (e.g. 'af_heart' -> 'a')
        lang_code = self.voice[0] if self.voice else 'a'
        self.pipeline = KPipeline(lang_code=lang_code)
        print(f"Kokoro loaded — voice: {self.voice}, lang: {lang_code}")

    def load_voice(self, reference_audio_path: str):
        """No-op for Kokoro (uses preset voices, no cloning needed)."""
        pass

    def synthesize_full(self, text: str) -> np.ndarray:
        """
        Generate speech for the full text at once.

        Returns:
            float32 numpy array of audio at self.sample_rate
        """
        if self.pipeline is None:
            raise RuntimeError("Call load_model() first")

        chunks = []
        for _gs, _ps, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            chunks.append(audio)

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32)

    def crossfade(
        self,
        filler_audio: np.ndarray,
        response_audio: np.ndarray,
    ) -> np.ndarray:
        """
        Crossfade from filler audio into the start of the real response.
        """
        fade_samples = int(self.sample_rate * self.crossfade_ms / 1000)

        if len(filler_audio) < fade_samples or len(response_audio) < fade_samples:
            return np.concatenate([filler_audio, response_audio])

        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        filler_tail = filler_audio[-fade_samples:] * fade_out
        response_head = response_audio[:fade_samples] * fade_in
        blended_region = filler_tail + response_head

        result = np.concatenate([
            filler_audio[:-fade_samples],
            blended_region,
            response_audio[fade_samples:],
        ])

        return result
