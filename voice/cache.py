"""
Pre-generated audio cache for fillers and backchannels.

At startup, generates all filler and backchannel phrases in the cloned voice
and stores them as numpy arrays for instant playback (zero TTS latency).

Usage:
    cache = AudioCache()
    cache.generate_all(tts)          # run once at startup
    audio = cache.get_filler()       # instant random filler
    audio = cache.get_backchannel()  # instant random backchannel
"""

import random
from dataclasses import dataclass, field

import numpy as np

from pipeline.tts import StreamingTTS


# Phrases to pre-generate. Short, natural, conversational.
FILLER_PHRASES = [
    "Um...",
    "Uh...",
    "So...",
    "Let me see...",
    "Hmm...",
    "Well...",
    "OK so...",
    "Right, so...",
]

BACKCHANNEL_PHRASES = [
    "Mhm.",
    "Yeah.",
    "OK.",
    "Right.",
    "Yep.",
    "Uh huh.",
    "Got it.",
    "Sure.",
]


class AudioCache:
    """
    Stores pre-generated audio clips for instant playback.
    """

    def __init__(self):
        self.fillers: dict[str, np.ndarray] = {}
        self.backchannels: dict[str, np.ndarray] = {}
        self.sample_rate: int = 24000

        # Track recent picks to avoid immediate repeats
        self._recent_fillers: list[str] = []
        self._recent_backchannels: list[str] = []

    def generate_all(self, tts: StreamingTTS):
        """
        Pre-generate all filler and backchannel audio clips.

        Args:
            tts: A StreamingTTS instance with model and voice already loaded.
        """
        self.sample_rate = tts.sample_rate

        print("Generating filler audio clips...")
        for phrase in FILLER_PHRASES:
            audio = tts.synthesize_full(phrase)
            self.fillers[phrase] = audio
            print(f"  '{phrase}' -> {len(audio)/self.sample_rate:.1f}s")

        print("Generating backchannel audio clips...")
        for phrase in BACKCHANNEL_PHRASES:
            audio = tts.synthesize_full(phrase)
            self.backchannels[phrase] = audio
            print(f"  '{phrase}' -> {len(audio)/self.sample_rate:.1f}s")

        print(f"Audio cache ready: {len(self.fillers)} fillers, {len(self.backchannels)} backchannels")

    def get_filler(self) -> tuple[str, np.ndarray]:
        """
        Get a random filler clip, avoiding immediate repeats.

        Returns:
            (phrase, audio_array) tuple
        """
        return self._pick_random(
            self.fillers, self._recent_fillers, max_recent=3
        )

    def get_backchannel(self) -> tuple[str, np.ndarray]:
        """
        Get a random backchannel clip, avoiding immediate repeats.

        Returns:
            (phrase, audio_array) tuple
        """
        return self._pick_random(
            self.backchannels, self._recent_backchannels, max_recent=3
        )

    def _pick_random(
        self,
        pool: dict[str, np.ndarray],
        recent: list[str],
        max_recent: int,
    ) -> tuple[str, np.ndarray]:
        """Pick a random clip, avoiding recently used ones."""
        available = [k for k in pool if k not in recent]
        if not available:
            # All used recently — reset and pick any
            recent.clear()
            available = list(pool.keys())

        phrase = random.choice(available)
        recent.append(phrase)
        if len(recent) > max_recent:
            recent.pop(0)

        return phrase, pool[phrase].copy()

    @property
    def is_ready(self) -> bool:
        return len(self.fillers) > 0 and len(self.backchannels) > 0
