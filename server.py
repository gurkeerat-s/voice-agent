"""
WebSocket server — main entry point.

On startup:
  1. Loads XTTS-v2 model
  2. Clones voice from reference audio
  3. Pre-generates filler & backchannel audio cache

Each WebSocket connection gets its own VoiceAgent that runs the
full-duplex conversation loop.

Run:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8765
"""

import asyncio
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import config
from voice.clone import setup_voice
from voice.cache import AudioCache
from agent.state_machine import VoiceAgent
from pipeline.tts import StreamingTTS


# Shared resources (initialized at startup, shared across connections)
tts_instance: StreamingTTS | None = None
audio_cache: AudioCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models and generate audio cache."""
    global tts_instance, audio_cache

    print("=" * 60)
    print("  Voice Agent — Starting up")
    print("=" * 60)

    # 1. Load TTS model
    tts_instance = setup_voice()

    # 2. Pre-generate filler & backchannel audio
    audio_cache = AudioCache()
    audio_cache.generate_all(tts_instance)

    # 3. Warm up LLM (first call is slow due to model loading)
    print("Warming up LLM...")
    from pipeline.llm import LLMClient
    llm = LLMClient()
    try:
        warmup_response = await llm.generate_full("Say hi.", [])
        print(f"  LLM warm: {warmup_response[:50]}")
    except Exception as e:
        print(f"  LLM warmup failed (will retry on first request): {e}")

    print("=" * 60)
    print(f"  Ready — listening on ws://0.0.0.0:{config.server.port}/ws")
    print("=" * 60)

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Each connection gets its own VoiceAgent.
    Binary messages = audio frames (16-bit PCM).
    JSON messages = control/state updates.
    """
    await websocket.accept()
    print("Client connected")

    agent = VoiceAgent(
        websocket=websocket,
        tts=tts_instance,
        audio_cache=audio_cache,
    )

    try:
        await agent.run()
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Connection error: {e}")


@app.get("/")
async def serve_client():
    """Serve the browser test client."""
    return FileResponse("client/index.html")


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Server")
    parser.add_argument("--host", default=config.server.host)
    parser.add_argument("--port", type=int, default=config.server.port)
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
