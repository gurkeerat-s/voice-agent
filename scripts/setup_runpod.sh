#!/bin/bash
# ──────────────────────────────────────────────────────────
# RunPod Setup Script — Voice Agent with Orpheus TTS
#
# Usage:
#   cd /workspace && git clone https://github.com/gurkeerat-s/voice-agent.git && cd voice-agent && chmod +x scripts/setup_runpod.sh && ./scripts/setup_runpod.sh
# ──────────────────────────────────────────────────────────

set -e

echo "=============================================="
echo "  Voice Agent — RunPod Setup"
echo "=============================================="

# ── 1. System deps ─────────────────────────────────────────
echo "[1/4] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg zstd > /dev/null 2>&1
echo "  Done."

# ── 2. Python packages ────────────────────────────────────
echo "[2/4] Installing Python packages..."
pip install -q orpheus-speech faster-whisper openai fastapi "uvicorn[standard]" websockets scipy soundfile pydantic pydantic-settings
echo "  Done."

# ── 3. Install Ollama ─────────────────────────────────────
echo "[3/4] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "  Done."

# ── 4. Start Ollama and pull model ────────────────────────
echo "[4/4] Starting Ollama and pulling Llama 3.1 8B..."
ollama serve &
sleep 5
ollama pull llama3.1:8b
echo "  Done."

echo ""
echo "=============================================="
echo "  Setup complete! Run:"
echo "    python server.py"
echo "=============================================="
