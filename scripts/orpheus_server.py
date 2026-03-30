"""
Standalone Orpheus TTS server.

Runs as a separate process from the main voice agent to avoid
vLLM/asyncio event loop conflicts. Exposes a simple HTTP API.

Usage:
    python scripts/orpheus_server.py              # starts on port 8766
    python scripts/orpheus_server.py --port 8766

API:
    POST /synthesize  {"text": "Hello", "voice": "tara"}  → raw PCM audio bytes
    GET  /health      → {"status": "ok"}
"""

import argparse
import io
import sys
import os
import json

import numpy as np

# Set SNAC device before importing orpheus
os.environ["SNAC_DEVICE"] = "cuda"


def create_app():
    from flask import Flask, request, Response, jsonify
    from orpheus_tts import OrpheusModel

    app = Flask(__name__)

    print("Loading Orpheus TTS model...")
    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
    print("Orpheus TTS ready.")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/synthesize", methods=["POST"])
    def synthesize():
        data = request.get_json()
        text = data.get("text", "")
        voice = data.get("voice", "tara")

        if not text.strip():
            return Response(b"", content_type="application/octet-stream")

        # Generate audio
        chunks = []
        for audio_bytes in model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.3,
            max_tokens=1200,
        ):
            chunks.append(audio_bytes)

        if not chunks:
            return Response(b"", content_type="application/octet-stream")

        # Return raw PCM bytes (int16, 24kHz, mono)
        raw_audio = b"".join(chunks)
        return Response(raw_audio, content_type="application/octet-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Server")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app()
    print(f"Orpheus server listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
