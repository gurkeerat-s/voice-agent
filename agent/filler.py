"""
Filler word management during THINKING state.

When the user stops talking but the LLM/TTS response isn't ready yet,
plays a pre-generated filler clip ("um...", "so...") to mask the latency.
Then crossfades into the real TTS response when it arrives.

Usage:
    filler = FillerManager(audio_cache, tts)

    # User stopped talking — start the response race
    filler.start()

    # Check if we should play a filler (call after delay_ms)
    clip = filler.get_filler_if_needed()

    # When real TTS audio arrives, blend it with the filler
    audio = filler.blend_with_response(first_tts_chunk)
"""

import time

import numpy as np

from voice.cache import AudioCache
from pipeline.tts import StreamingTTS


class FillerManager:
    """
    Manages filler playback and crossfade into real responses.
    """

    def __init__(
        self,
        cache: AudioCache,
        tts: StreamingTTS,
        delay_ms: float = 200.0,
    ):
        """
        Args:
            cache: AudioCache with pre-generated filler clips.
            tts: StreamingTTS for crossfade utility.
            delay_ms: Wait this long after end-of-turn before playing filler.
                      If TTS is ready within this window, no filler needed.
        """
        self.cache = cache
        self.tts = tts
        self.delay_ms = delay_ms

        self._thinking_start: float | None = None
        self._filler_audio: np.ndarray | None = None
        self._filler_phrase: str | None = None
        self._filler_played = False

    def start(self):
        """
        Call when entering THINKING state (user just stopped talking).
        Starts the timer for filler injection.
        """
        self._thinking_start = time.monotonic()
        self._filler_audio = None
        self._filler_phrase = None
        self._filler_played = False

    def get_filler_if_needed(self) -> tuple[str, np.ndarray] | None:
        """
        Check if enough time has passed to warrant a filler.

        Returns (phrase, audio) if a filler should play, None otherwise.
        Should be called repeatedly after start() until it returns a filler
        or the real response arrives.
        """
        if self._thinking_start is None or self._filler_played:
            return None

        elapsed_ms = (time.monotonic() - self._thinking_start) * 1000

        if elapsed_ms < self.delay_ms:
            return None

        # Time's up — LLM is slow, play a filler
        result = self.cache.get_filler()
        if result is None:
            return None
        self._filler_phrase, self._filler_audio = result
        self._filler_played = True
        return self._filler_phrase, self._filler_audio

    def blend_with_response(self, first_tts_chunk: np.ndarray) -> np.ndarray:
        """
        Blend the filler audio (if playing) with the first real TTS chunk.

        If no filler was played, returns the TTS chunk unchanged.

        Args:
            first_tts_chunk: First audio chunk from the real TTS response.

        Returns:
            Audio to play — either the raw chunk or a crossfaded blend.
        """
        if self._filler_audio is None or not self._filler_played:
            return first_tts_chunk

        blended = self.tts.crossfade(self._filler_audio, first_tts_chunk)

        # Reset — filler is consumed
        self._filler_audio = None
        return blended

    @property
    def is_filler_playing(self) -> bool:
        return self._filler_played and self._filler_audio is not None

    def reset(self):
        """Reset for a new turn."""
        self._thinking_start = None
        self._filler_audio = None
        self._filler_phrase = None
        self._filler_played = False
