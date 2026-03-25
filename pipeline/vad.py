"""
Silero VAD wrapper.

Classifies incoming audio chunks into events:
  SPEECH_START  — user started talking
  SPEECH_ONGOING — user is still talking
  SHORT_PAUSE   — brief pause mid-speech (backchannel candidate)
  END_OF_TURN   — user finished talking
  BARGE_IN      — user started talking while agent is speaking
  SILENCE       — no speech detected (idle)
"""

import enum
import time
from dataclasses import dataclass

import torch
import numpy as np

from config import config


class VadEventType(enum.Enum):
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH_ONGOING = "speech_ongoing"
    SHORT_PAUSE = "short_pause"
    END_OF_TURN = "end_of_turn"
    BARGE_IN = "barge_in"


@dataclass
class VadEvent:
    type: VadEventType
    timestamp: float  # time.monotonic() when event was produced
    speech_duration_ms: float = 0.0  # how long user has been speaking
    silence_duration_ms: float = 0.0  # how long silence has lasted


class VADProcessor:
    """
    Processes audio chunks through Silero VAD and emits classified events.

    Usage:
        vad = VADProcessor()
        event = vad.process_chunk(audio_chunk_np, is_agent_speaking=False)
    """

    def __init__(self):
        cfg = config.vad
        self.speech_threshold = cfg.speech_threshold
        self.end_of_turn_ms = cfg.end_of_turn_ms
        self.short_pause_min_ms = cfg.short_pause_min_ms
        self.short_pause_max_ms = cfg.short_pause_max_ms
        self.barge_in_min_ms = cfg.barge_in_min_ms
        self.chunk_ms = cfg.chunk_duration_ms

        # Silero VAD operates at 16kHz
        self.sample_rate = 16000

        # Load Silero VAD model
        self.model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

        # State tracking
        self._is_speaking = False
        self._speech_start_time: float | None = None
        self._silence_start_time: float | None = None
        self._short_pause_emitted = False

    def reset(self):
        """Reset internal state for a new conversation."""
        self.model.reset_states()
        self._is_speaking = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._short_pause_emitted = False

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        is_agent_speaking: bool = False,
    ) -> VadEvent:
        """
        Process a single audio chunk and return a VadEvent.

        Args:
            audio_chunk: float32 numpy array, 16kHz mono, values in [-1, 1]
            is_agent_speaking: True if the agent is currently playing audio
                               (used to detect barge-in)

        Returns:
            VadEvent indicating what's happening in the audio.
        """
        now = time.monotonic()

        # Run Silero VAD on the chunk
        tensor = torch.from_numpy(audio_chunk).float()
        confidence = self.model(tensor, self.sample_rate).item()
        speech_detected = confidence >= self.speech_threshold

        if speech_detected:
            return self._handle_speech(now, is_agent_speaking)
        else:
            return self._handle_silence(now, is_agent_speaking)

    def _handle_speech(self, now: float, is_agent_speaking: bool) -> VadEvent:
        """Handle a chunk where speech was detected."""
        self._silence_start_time = None
        self._short_pause_emitted = False

        if not self._is_speaking:
            # Transition: silence -> speech
            self._is_speaking = True
            self._speech_start_time = now

            if is_agent_speaking:
                return VadEvent(
                    type=VadEventType.BARGE_IN,
                    timestamp=now,
                )
            else:
                return VadEvent(
                    type=VadEventType.SPEECH_START,
                    timestamp=now,
                )
        else:
            # Continuing speech
            speech_dur = (now - self._speech_start_time) * 1000
            return VadEvent(
                type=VadEventType.SPEECH_ONGOING,
                timestamp=now,
                speech_duration_ms=speech_dur,
            )

    def _handle_silence(self, now: float, is_agent_speaking: bool) -> VadEvent:
        """Handle a chunk where no speech was detected."""
        if not self._is_speaking:
            # Was already silent — still idle
            return VadEvent(type=VadEventType.SILENCE, timestamp=now)

        # User was speaking, now there's silence
        if self._silence_start_time is None:
            self._silence_start_time = now

        silence_ms = (now - self._silence_start_time) * 1000
        speech_dur = (now - self._speech_start_time) * 1000 if self._speech_start_time else 0

        # Check: end of turn?
        if silence_ms >= self.end_of_turn_ms:
            self._is_speaking = False
            self._speech_start_time = None
            self._silence_start_time = None
            self._short_pause_emitted = False
            return VadEvent(
                type=VadEventType.END_OF_TURN,
                timestamp=now,
                speech_duration_ms=speech_dur,
                silence_duration_ms=silence_ms,
            )

        # Check: short pause (backchannel candidate)?
        if (
            not self._short_pause_emitted
            and self.short_pause_min_ms <= silence_ms <= self.short_pause_max_ms
        ):
            self._short_pause_emitted = True
            return VadEvent(
                type=VadEventType.SHORT_PAUSE,
                timestamp=now,
                speech_duration_ms=speech_dur,
                silence_duration_ms=silence_ms,
            )

        # In between — still considered "speaking" (just a pause)
        return VadEvent(
            type=VadEventType.SPEECH_ONGOING,
            timestamp=now,
            speech_duration_ms=speech_dur,
            silence_duration_ms=silence_ms,
        )
