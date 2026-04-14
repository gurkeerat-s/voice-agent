"""
Standalone Orpheus TTS server using vLLM Python API directly.

Skips the orpheus-speech wrapper (which has version conflicts) and uses
vLLM directly for fast inference, plus SNAC for audio decoding.

Usage:
    python scripts/orpheus_server.py --model-dir ./models/orpheus-zara

API:
    POST /synthesize  {"text": "Hello", "voice": "zara"}  → raw PCM audio bytes
    GET  /health      → {"status": "ok"}
"""

import argparse
import os

import numpy as np
import torch


# Orpheus special token IDs (must match training format)
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263
END_OF_TEXT = 128009
AUDIO_BASE = 128266
CODEBOOK_SIZE = 4096
SAMPLE_RATE = 24000


def build_prompt(text, voice, tokenizer):
    """Build Orpheus prompt in the exact training format."""
    full_text = f"{voice}: {text}"
    text_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    prompt = (
        [START_OF_HUMAN]
        + text_tokens
        + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    )
    return prompt


def decode_audio_tokens(audio_tokens, snac_model):
    """Decode interleaved audio tokens back to waveform."""
    num_frames = len(audio_tokens) // 7
    audio_tokens = audio_tokens[: num_frames * 7]

    codes_0, codes_1, codes_2 = [], [], []
    for i in range(0, len(audio_tokens), 7):
        t = audio_tokens[i : i + 7]
        codes_0.append(t[0] - AUDIO_BASE)
        codes_1.append(t[1] - (AUDIO_BASE + CODEBOOK_SIZE))
        codes_2.append(t[2] - (AUDIO_BASE + 2 * CODEBOOK_SIZE))
        codes_2.append(t[3] - (AUDIO_BASE + 3 * CODEBOOK_SIZE))
        codes_1.append(t[4] - (AUDIO_BASE + 4 * CODEBOOK_SIZE))
        codes_2.append(t[5] - (AUDIO_BASE + 5 * CODEBOOK_SIZE))
        codes_2.append(t[6] - (AUDIO_BASE + 6 * CODEBOOK_SIZE))

    codes_0 = [max(0, min(4095, c)) for c in codes_0]
    codes_1 = [max(0, min(4095, c)) for c in codes_1]
    codes_2 = [max(0, min(4095, c)) for c in codes_2]

    c0 = torch.tensor(codes_0, dtype=torch.long).unsqueeze(0).to("cuda")
    c1 = torch.tensor(codes_1, dtype=torch.long).unsqueeze(0).to("cuda")
    c2 = torch.tensor(codes_2, dtype=torch.long).unsqueeze(0).to("cuda")

    with torch.no_grad():
        audio = snac_model.decode([c0, c1, c2])

    return audio.squeeze().cpu().numpy()


def create_app(model_name):
    from flask import Flask, request, Response, jsonify
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import snac as snac_lib

    app = Flask(__name__)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading Orpheus TTS model via vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,  # leave room for LLM + STT + SNAC
        max_model_len=2048,
    )

    print("Loading SNAC decoder...")
    snac_model = snac_lib.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        repetition_penalty=1.1,
        max_tokens=1200,
        stop_token_ids=[END_OF_SPEECH],
    )

    print("Orpheus TTS ready.")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/synthesize", methods=["POST"])
    def synthesize():
        data = request.get_json()
        text = data.get("text", "")
        voice = data.get("voice", "zara")

        if not text.strip():
            return Response(b"", content_type="application/octet-stream")

        prompt_token_ids = build_prompt(text, voice, tokenizer)

        outputs = llm.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        generated = list(outputs[0].outputs[0].token_ids)

        audio_tokens = [tok for tok in generated if tok >= AUDIO_BASE]

        if not audio_tokens:
            return Response(b"", content_type="application/octet-stream")

        audio_np = decode_audio_tokens(audio_tokens, snac_model)
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        return Response(audio_int16.tobytes(), content_type="application/octet-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Server (vLLM)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model-dir", required=True,
                        help="Path to fine-tuned model (local dir or HF repo)")
    args = parser.parse_args()

    app = create_app(model_name=args.model_dir)
    print(f"Orpheus server listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
