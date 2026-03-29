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
echo "[1/7] System dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg zstd > /dev/null 2>&1
echo "  Done."

# ── 2. Install vllm first (pins torch==2.4.0) ─────────────
echo "[2/7] Installing vllm..."
pip install -q vllm==0.6.2
echo "  Done."

# ── 3. Install TTS and other deps ─────────────────────────
echo "[3/7] Installing TTS and dependencies..."
pip install -q --ignore-installed blinker
pip install -q faster-whisper TTS openai fastapi "uvicorn[standard]" websockets scipy soundfile pydantic-settings
echo "  Done."

# ── 4. Pin transformers for TTS compatibility ──────────────
echo "[4/7] Pinning transformers..."
pip install -q transformers==4.45.0
echo "  Done."

# ── 5. Fix torch CUDA version (TTS pulls wrong one) ───────
echo "[5/7] Fixing torch CUDA versions..."
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps -q
echo "  Done."

# ── 6. Install Ollama ─────────────────────────────────────
echo "[6/7] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "  Done."

# ── 7. Start Ollama and pull model ────────────────────────
echo "[7/7] Starting Ollama and pulling Llama 3.1 8B..."
ollama serve &
sleep 5
ollama pull llama3.1:8b
echo "  Done."

echo ""
echo "=============================================="
echo "  Setup complete! Run:"
echo "    python server.py"
echo "=============================================="
