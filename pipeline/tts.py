"""
Streaming TTS using XTTS-v2 with voice cloning.

Clones a voice from a short reference audio clip, then generates
speech in that voice with streaming chunk output. Supports crossfading
from a filler audio clip into the real response.

Usage:
    tts = StreamingTTS()
    tts.load_voice("voice/reference.wav")

    # Stream audio chunks for a sentence
    async for chunk in tts.synthesize_stream("Hello, how can I help?"):
        send_audio(chunk)

    # Crossfade from filler to real response
    blended = tts.crossfade(filler_audio, first_real_chunk)
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from config import config


@dataclass
class AudioChunk:
    """A chunk of generated audio."""
    audio: np.ndarray  # float32 mono
    sample_rate: int
    is_last: bool = False


class StreamingTTS:
    """
    XTTS-v2 wrapper with voice cloning and streaming output.
    """

    def __init__(self):
        cfg = config.tts
        self.sample_rate = cfg.sample_rate
        self.stream_chunk_size = cfg.stream_chunk_size
        self.crossfade_ms = cfg.crossfade_ms
        self.language = cfg.language
        self.device = cfg.device

        self.model: Xtts | None = None
        self.gpt_cond_latent: torch.Tensor | None = None
        self.speaker_embedding: torch.Tensor | None = None

    def load_model(self):
        """Load the XTTS-v2 model. Call once at startup."""
        model_path = Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"

        # If the model hasn't been downloaded yet, use the TTS API to trigger download
        if not model_path.exists():
            from TTS.api import TTS as TTSApi
            tts_api = TTSApi(model_name=config.tts.model_name)
            # Extract the underlying XTTS model
            self.model = tts_api.synthesizer.tts_model
            self.model.to(self.device)
            return

        # Load directly for faster startup
        xtts_config = XttsConfig()
        xtts_config.load_json(str(model_path / "config.json"))
        self.model = Xtts.init_from_config(xtts_config)
        self.model.load_checkpoint(xtts_config, checkpoint_dir=str(model_path))
        self.model.to(self.device)

    def load_voice(self, reference_audio_path: str):
        """
        Clone a voice from a reference audio file.

        Args:
            reference_audio_path: Path to a 6-10 second WAV file of the
                                  target voice speaking clearly.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before load_voice()")

        self.gpt_cond_latent, self.speaker_embedding = (
            self.model.get_conditioning_latents(
                audio_path=[reference_audio_path],
            )
        )

    def synthesize_stream(self, text: str):
        """
        Generate speech for the given text, yielding AudioChunks.

        This is a synchronous generator (XTTS inference is synchronous).
        Wrap in asyncio.to_thread() or run in an executor for async use.

        Yields:
            AudioChunk with audio data and metadata.
        """
        if self.model is None or self.gpt_cond_latent is None:
            raise RuntimeError("Call load_model() and load_voice() first")

        chunks = self.model.inference_stream(
            text,
            self.language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            stream_chunk_size=self.stream_chunk_size,
        )

        for i, chunk_tensor in enumerate(chunks):
            # chunk_tensor is a 1D torch tensor on GPU
            audio_np = chunk_tensor.cpu().numpy().astype(np.float32)

            yield AudioChunk(
                audio=audio_np,
                sample_rate=self.sample_rate,
                is_last=False,  # We don't know until iteration ends
            )

        # Mark the last chunk (caller can use this to know TTS is done)
        # Note: this is handled by the generator naturally exhausting

    def synthesize_full(self, text: str) -> np.ndarray:
        """
        Generate speech for the full text at once (non-streaming).
        Useful for pre-generating fillers and backchannels.

        Returns:
            float32 numpy array of audio at self.sample_rate
        """
        if self.model is None or self.gpt_cond_latent is None:
            raise RuntimeError("Call load_model() and load_voice() first")

        output = self.model.inference(
            text,
            self.language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            temperature=0.65,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.8,
            enable_text_splitting=True,
        )

        # output["wav"] may be a torch tensor or numpy array depending on version
        wav = output["wav"]
        if hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
        return np.asarray(wav, dtype=np.float32)

    def crossfade(
        self,
        filler_audio: np.ndarray,
        response_audio: np.ndarray,
    ) -> np.ndarray:
        """
        Crossfade from filler audio into the start of the real response.

        The filler's tail fades out while the response's head fades in,
        creating a smooth transition.

        Args:
            filler_audio: The filler clip currently playing (full clip)
            response_audio: The first chunk of the real TTS response

        Returns:
            Blended audio array ready for playback.
        """
        fade_samples = int(self.sample_rate * self.crossfade_ms / 1000)

        # If either clip is shorter than the fade, just concatenate
        if len(filler_audio) < fade_samples or len(response_audio) < fade_samples:
            return np.concatenate([filler_audio, response_audio])

        # Fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        # The crossfade region
        filler_tail = filler_audio[-fade_samples:] * fade_out
        response_head = response_audio[:fade_samples] * fade_in
        blended_region = filler_tail + response_head

        # Assemble: filler (minus tail) + blended + response (minus head)
        result = np.concatenate([
            filler_audio[:-fade_samples],
            blended_region,
            response_audio[fade_samples:],
        ])

        return result
