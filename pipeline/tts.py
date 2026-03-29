"""
TTS using Orpheus-TTS (3B, Llama-based).

Fast, expressive speech synthesis with emotion tags.
Outputs 24kHz int16 PCM via SNAC codec, converted to float32 for pipeline.
Native streaming support (~85ms per chunk).

Usage:
    tts = StreamingTTS()
    tts.load_model()

    audio = tts.synthesize_full("Hello, how can I help?")
    blended = tts.crossfade(filler_audio, first_real_chunk)
"""

import numpy as np

from config import config


class StreamingTTS:
    """
    Orpheus TTS wrapper with streaming output.
    """

    def __init__(self):
        cfg = config.tts
        self.sample_rate = cfg.sample_rate  # 24000
        self.crossfade_ms = cfg.crossfade_ms
        self.voice = cfg.voice

        self.model = None

    def load_model(self):
        """Load Orpheus TTS model. Call once at startup."""
        import os
        os.environ["SNAC_DEVICE"] = "cuda"

        from orpheus_tts import OrpheusModel

        print(f"Loading Orpheus TTS (voice: {self.voice})...")
        self.model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
        )
        print("Orpheus TTS ready.")

    def load_voice(self, reference_audio_path: str):
        """No-op — Orpheus uses preset voices."""
        pass

    def synthesize_full(self, text: str) -> np.ndarray:
        """
        Generate speech for the full text at once.

        Returns:
            float32 numpy array of audio at self.sample_rate
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        import concurrent.futures

        def _generate():
            chunks = []
            for audio_bytes in self.model.generate_speech(
                prompt=text,
                voice=self.voice,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.3,
                max_tokens=1200,
            ):
                chunk_np = np.frombuffer(audio_bytes, dtype=np.int16)
                chunks.append(chunk_np)
            return chunks

        # Run in a separate thread to avoid async event loop conflicts
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            chunks = future.result(timeout=30)

        if not chunks:
            return np.array([], dtype=np.float32)

        audio_int16 = np.concatenate(chunks)
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
