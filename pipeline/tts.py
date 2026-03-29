"""
TTS using fine-tuned CSM-1B with LoRA adapter.

Loads sesame/csm-1b base model + LoRA adapter (trained on Zara voice),
merges weights, and generates speech.
Outputs 24kHz float32 mono audio.

Usage:
    tts = StreamingTTS()
    tts.load_model()

    audio = tts.synthesize_full("Hello, how can I help?")
    blended = tts.crossfade(filler_audio, first_real_chunk)
"""

import numpy as np
import torch

from config import config


class StreamingTTS:
    """
    CSM-1B TTS with LoRA adapter for voice cloning.
    """

    def __init__(self):
        cfg = config.tts
        self.sample_rate = cfg.sample_rate  # 24000
        self.crossfade_ms = cfg.crossfade_ms
        self.base_model_id = cfg.base_model
        self.adapter_path = cfg.adapter_path
        self.speaker_id = cfg.speaker_id

        self.model = None
        self.processor = None
        self.device = None

    def load_model(self):
        """Load CSM-1B base model + LoRA adapter. Call once at startup."""
        from transformers import CsmForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        print(f"Loading CSM-1B base model: {self.base_model_id}")
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        base_model = CsmForConditionalGeneration.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float32,
        )

        print(f"Loading LoRA adapter: {self.adapter_path}")
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model = model.merge_and_unload()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"CSM-1B ready on {self.device}")

    def load_voice(self, reference_audio_path: str):
        """No-op — voice is baked into the LoRA adapter."""
        pass

    def synthesize_full(self, text: str) -> np.ndarray:
        """
        Generate speech for the full text at once.

        Returns:
            float32 numpy array of audio at self.sample_rate
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        # Format text with speaker ID prefix
        prompt = f"[{self.speaker_id}]{text}"
        inputs = self.processor(prompt, add_special_tokens=True).to(self.device)

        with torch.no_grad():
            audio = self.model.generate(**inputs, output_audio=True)

        # Extract audio tensor — handle different output formats
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        elif hasattr(audio, 'audio_values'):
            audio_tensor = audio.audio_values
        elif isinstance(audio, (tuple, list)):
            audio_tensor = audio[0]
        else:
            audio_tensor = audio

        audio_np = audio_tensor.cpu().float().numpy().astype(np.float32)

        # Flatten if needed
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        return audio_np

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
