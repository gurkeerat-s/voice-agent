"""
Audio I/O utilities.

Handles:
  - Sample rate conversion (browser 48kHz <-> STT 16kHz <-> TTS 24kHz)
  - Volume normalization
  - WebSocket audio frame encoding/decoding
  - Audio format conversion (float32 <-> int16 PCM)
"""

import struct

import numpy as np
from scipy import signal

from config import config


class AudioIO:
    """
    Audio conversion and framing utilities for the WebSocket pipeline.
    """

    def __init__(self):
        cfg = config.audio
        self.input_sr = cfg.input_sample_rate    # 48000 (browser)
        self.stt_sr = cfg.stt_sample_rate        # 16000 (whisper)
        self.output_sr = cfg.output_sample_rate   # 24000 (xtts)
        self.target_rms = cfg.target_rms
        self.ws_chunk_ms = cfg.ws_chunk_ms

    # ── Incoming audio (browser → server) ──────────────────────────

    def decode_ws_audio(self, data: bytes) -> np.ndarray:
        """
        Decode raw bytes from WebSocket into float32 numpy array.

        Expects 16-bit signed PCM (little-endian) from the browser,
        which is the most common WebSocket audio format.
        """
        pcm_int16 = np.frombuffer(data, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    def resample_for_stt(self, audio: np.ndarray) -> np.ndarray:
        """Resample from browser sample rate (48kHz) to STT rate (16kHz)."""
        if self.input_sr == self.stt_sr:
            return audio
        return self._resample(audio, self.input_sr, self.stt_sr)

    def resample_for_vad(self, audio: np.ndarray) -> np.ndarray:
        """Resample to 16kHz for Silero VAD (same as STT)."""
        return self.resample_for_stt(audio)

    # ── Outgoing audio (server → browser) ──────────────────────────

    def encode_ws_audio(self, audio: np.ndarray) -> bytes:
        """
        Encode float32 audio to 16-bit PCM bytes for WebSocket transmission.

        The audio should already be at the output sample rate (24kHz).
        Browser will be configured to play at matching rate.
        """
        # Clip to [-1, 1] to avoid overflow
        audio = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (audio * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def resample_for_output(self, audio: np.ndarray, from_sr: int) -> np.ndarray:
        """Resample audio to the output sample rate for browser playback."""
        if from_sr == self.output_sr:
            return audio
        return self._resample(audio, from_sr, self.output_sr)

    # ── Volume normalization ───────────────────────────────────────

    def normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target RMS level.

        Prevents volume jumps between filler audio and real TTS output.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio  # silence, don't amplify noise
        gain = self.target_rms / rms
        # Limit gain to avoid amplifying quiet noise excessively
        gain = min(gain, 10.0)
        return audio * gain

    # ── Chunking ───────────────────────────────────────────────────

    def chunk_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[np.ndarray]:
        """
        Split audio into WebSocket-sized chunks.

        Args:
            audio: Full audio array
            sample_rate: Sample rate of the audio

        Returns:
            List of audio chunks, each ws_chunk_ms long (last may be shorter).
        """
        chunk_samples = int(sample_rate * self.ws_chunk_ms / 1000)
        chunks = []
        for start in range(0, len(audio), chunk_samples):
            chunks.append(audio[start : start + chunk_samples])
        return chunks

    # ── Internal ───────────────────────────────────────────────────

    @staticmethod
    def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample using scipy's polyphase resampler."""
        if from_sr == to_sr:
            return audio
        # Calculate the number of output samples
        num_samples = int(len(audio) * to_sr / from_sr)
        return signal.resample(audio, num_samples).astype(np.float32)
