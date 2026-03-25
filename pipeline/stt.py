"""
Streaming Speech-to-Text using faster-whisper.

Accumulates audio chunks and periodically runs transcription to produce
partial (interim) transcripts. Emits a final transcript when signaled
that the user's turn has ended.

Usage:
    stt = StreamingSTT()
    stt.add_audio(chunk)          # feed 200ms chunks continuously
    partial = stt.get_partial()   # poll for latest partial transcript
    final = stt.finalize()        # get final transcript & reset buffer
"""

import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

from config import config


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    timestamp: float
    # Cumulative audio duration that produced this transcript (seconds)
    audio_duration_s: float = 0.0


class StreamingSTT:
    """
    Wraps faster-whisper for streaming transcription.

    Audio is buffered internally. Partial transcripts are generated
    on demand (non-blocking). Final transcript uses higher beam size
    for accuracy.
    """

    def __init__(self):
        cfg = config.stt
        self.sample_rate = config.audio.stt_sample_rate  # 16kHz
        self.partial_interval_ms = cfg.partial_interval_ms
        self.beam_size = cfg.beam_size
        self.partial_beam_size = cfg.partial_beam_size

        # Load model
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        self.language = cfg.language

        # Audio buffer (accumulates chunks between finalizations)
        self._buffer: list[np.ndarray] = []
        self._buffer_samples: int = 0
        self._last_partial_time: float = 0.0
        self._last_partial_text: str = ""

    def reset(self):
        """Clear buffer for a new turn."""
        self._buffer.clear()
        self._buffer_samples = 0
        self._last_partial_time = 0.0
        self._last_partial_text = ""

    def add_audio(self, chunk: np.ndarray):
        """
        Add an audio chunk to the buffer.

        Args:
            chunk: float32 numpy array, 16kHz mono
        """
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)

    def get_partial(self) -> TranscriptEvent | None:
        """
        Get a partial transcript if enough time has passed since the last one.

        Returns None if it's too soon or there's not enough audio.
        Returns a TranscriptEvent with is_final=False otherwise.
        """
        now = time.monotonic()
        elapsed_ms = (now - self._last_partial_time) * 1000

        if elapsed_ms < self.partial_interval_ms:
            return None

        if self._buffer_samples < self.sample_rate * 0.5:
            # Need at least 0.5s of audio for a meaningful partial
            return None

        audio = self._get_buffer_audio()
        text = self._transcribe(audio, beam_size=self.partial_beam_size)

        self._last_partial_time = now
        self._last_partial_text = text

        if not text.strip():
            return None

        return TranscriptEvent(
            text=text,
            is_final=False,
            timestamp=now,
            audio_duration_s=len(audio) / self.sample_rate,
        )

    def finalize(self) -> TranscriptEvent:
        """
        Produce the final transcript for the current turn and reset the buffer.

        Uses higher beam size for better accuracy.
        """
        audio = self._get_buffer_audio()
        now = time.monotonic()

        if len(audio) < self.sample_rate * 0.1:
            # Too short — return whatever partial we had
            text = self._last_partial_text
        else:
            text = self._transcribe(audio, beam_size=self.beam_size)

        event = TranscriptEvent(
            text=text.strip(),
            is_final=True,
            timestamp=now,
            audio_duration_s=len(audio) / self.sample_rate,
        )

        self.reset()
        return event

    def get_buffered_duration_s(self) -> float:
        """Return how many seconds of audio are in the buffer."""
        return self._buffer_samples / self.sample_rate

    def _get_buffer_audio(self) -> np.ndarray:
        """Concatenate all buffered chunks into one array."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def _transcribe(self, audio: np.ndarray, beam_size: int) -> str:
        """Run whisper transcription on audio array."""
        if len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=beam_size,
            vad_filter=False,  # We handle VAD ourselves
            without_timestamps=True,
        )

        return " ".join(seg.text for seg in segments).strip()
