#!/bin/bash
# ──────────────────────────────────────────────────────────
# RunPod Setup Script — One-shot setup
#
# Usage:
#   cd /workspace && git clone https://github.com/gurkeerat-s/voice-agent.git && cd voice-agent && chmod +x scripts/setup_runpod.sh && ./scripts/setup_runpod.sh
# ──────────────────────────────────────────────────────────

set -e

echo "=============================================="
echo "  Voice Agent — RunPod Setup"
echo "=============================================="

# ── 1. System deps ─────────────────────────────────────────
echo "[1/5] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg zstd espeak-ng > /dev/null 2>&1
echo "  Done."

# ── 2. Python packages ────────────────────────────────────
echo "[2/5] Installing Python packages..."
pip install -q "kokoro>=0.9" soundfile faster-whisper openai fastapi "uvicorn[standard]" websockets scipy pydantic pydantic-settings
echo "  Done."

# ── 3. Install Ollama ─────────────────────────────────────
echo "[3/5] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "  Done."

# ── 4. Start Ollama and pull model ────────────────────────
echo "[4/5] Starting Ollama and pulling Llama 3.1 8B..."
ollama serve &
sleep 5
ollama pull llama3.1:8b
echo "  Done."

# ── 5. Download Kokoro model ──────────────────────────────
echo "[5/5] Downloading Kokoro model..."
python -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); print('Kokoro ready.')"
echo "  Done."

echo ""
echo "=============================================="
echo "  Setup complete! Run:"
echo "    python server.py"
echo "=============================================="
