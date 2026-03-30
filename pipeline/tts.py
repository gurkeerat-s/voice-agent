"""
TTS client that calls the Orpheus TTS server via HTTP.

The Orpheus model runs in a separate process (scripts/orpheus_server.py)
to avoid vLLM/asyncio event loop conflicts with our FastAPI server.

Usage:
    tts = StreamingTTS()
    tts.load_model()  # just verifies the server is running

    audio = tts.synthesize_full("Hello, how can I help?")
"""

import time
import numpy as np
import requests

from config import config


class StreamingTTS:
    """
    HTTP client for the Orpheus TTS server.
    """

    def __init__(self):
        cfg = config.tts
        self.sample_rate = cfg.sample_rate  # 24000
        self.crossfade_ms = cfg.crossfade_ms
        self.voice = cfg.voice
        self.server_url = cfg.server_url

    def load_model(self):
        """Verify the Orpheus TTS server is running."""
        print(f"Connecting to Orpheus TTS server at {self.server_url}...")

        # Wait for server to be ready (it may still be loading the model)
        for i in range(120):  # wait up to 2 minutes
            try:
                resp = requests.get(f"{self.server_url}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"Orpheus TTS server connected (voice: {self.voice}).")
                    return
            except requests.ConnectionError:
                pass
            if i % 10 == 0 and i > 0:
                print(f"  Waiting for Orpheus server... ({i}s)")
            time.sleep(1)

        raise RuntimeError(
            f"Orpheus TTS server not responding at {self.server_url}. "
            "Start it with: python scripts/orpheus_server.py"
        )

    def load_voice(self, reference_audio_path: str):
        """No-op — Orpheus uses preset voices."""
        pass

    def synthesize_full(self, text: str) -> np.ndarray:
        """
        Generate speech by calling the Orpheus TTS server.

        Returns:
            float32 numpy array of audio at self.sample_rate
        """
        resp = requests.post(
            f"{self.server_url}/synthesize",
            json={"text": text, "voice": self.voice},
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"TTS server error: {resp.status_code}")
            return np.array([], dtype=np.float32)

        raw_audio = resp.content
        if not raw_audio:
            return np.array([], dtype=np.float32)

        # Convert int16 PCM bytes to float32
        audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32767.0

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
