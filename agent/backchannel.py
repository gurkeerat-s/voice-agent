"""
Backchannel injection during user speech.

While the user is talking, occasionally plays short acknowledgment
sounds ("mhm", "yeah", "ok") on brief pauses. This makes the agent
feel like it's actively listening.

Rules:
  - Only inject on SHORT_PAUSE events (300-600ms silence mid-speech)
  - ~30% probability per eligible pause (not every one)
  - Minimum 3 second cooldown between backchannels
  - Never backchannel in the first 1 second of user speech
  - Randomized clip selection with no immediate repeats

Usage:
    bc = BackchannelInjector(audio_cache)
    audio = bc.maybe_inject(vad_event, speech_start_time)
    if audio is not None:
        play(audio)
"""

import time
import random

import numpy as np

from voice.cache import AudioCache


class BackchannelInjector:
    """
    Decides when to inject backchannel audio during user speech.
    """

    def __init__(
        self,
        cache: AudioCache,
        probability: float = 0.3,
        cooldown_s: float = 3.0,
        min_speech_before_bc_s: float = 1.0,
    ):
        """
        Args:
            cache: AudioCache with pre-generated backchannel clips.
            probability: Chance of injecting on any eligible pause (0-1).
            cooldown_s: Minimum seconds between backchannels.
            min_speech_before_bc_s: Don't backchannel in the first N seconds
                                     of user speech.
        """
        self.cache = cache
        self.probability = probability
        self.cooldown_s = cooldown_s
        self.min_speech_before_bc_s = min_speech_before_bc_s

        self._last_bc_time: float = 0.0

    def maybe_inject(
        self,
        speech_duration_ms: float,
    ) -> tuple[str, np.ndarray] | None:
        """
        Decide whether to inject a backchannel right now.

        Call this when a SHORT_PAUSE event is received from VAD.

        Args:
            speech_duration_ms: How long the user has been speaking in this turn.

        Returns:
            (phrase, audio_array) if injecting, None otherwise.
        """
        now = time.monotonic()

        # Don't backchannel too early in the user's turn
        if speech_duration_ms < self.min_speech_before_bc_s * 1000:
            return None

        # Respect cooldown
        if now - self._last_bc_time < self.cooldown_s:
            return None

        # Probabilistic gate
        if random.random() > self.probability:
            return None

        # All checks passed — inject a backchannel
        self._last_bc_time = now
        phrase, audio = self.cache.get_backchannel()
        return phrase, audio

    def reset(self):
        """Reset for a new conversation."""
        self._last_bc_time = 0.0
