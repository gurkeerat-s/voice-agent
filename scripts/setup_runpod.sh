#!/bin/bash
# ──────────────────────────────────────────────────────────
# RunPod Setup Script
#
# Sets up the full voice agent environment on a fresh RunPod pod.
# Tested on: RunPod PyTorch 2.1+ template (CUDA 12.x)
#
# Usage:
#   chmod +x scripts/setup_runpod.sh
#   ./scripts/setup_runpod.sh [path/to/reference_voice.wav]
# ──────────────────────────────────────────────────────────

set -e

REFERENCE_AUDIO="${1:-voice/reference.wav}"
VLLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ"
VLLM_PORT=8000

echo "=============================================="
echo "  Voice Agent — RunPod Setup"
echo "=============================================="

# ── 1. System dependencies ─────────────────────────────────
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "  Done."

# ── 2. Python dependencies ─────────────────────────────────
echo ""
echo "[2/6] Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Done."

# ── 3. Download models ─────────────────────────────────────
echo ""
echo "[3/6] Downloading models (this may take a few minutes)..."

# Faster-Whisper model (auto-downloads on first use, but let's warm it)
python -c "
from faster_whisper import WhisperModel
print('  Downloading Whisper distil-large-v3...')
model = WhisperModel('distil-large-v3', device='cpu', compute_type='int8')
print('  Whisper ready.')
"

# XTTS-v2 model
python -c "
from TTS.api import TTS
print('  Downloading XTTS-v2...')
tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2')
print('  XTTS-v2 ready.')
"

# Silero VAD
python -c "
import torch
print('  Downloading Silero VAD...')
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
print('  Silero VAD ready.')
"

echo "  All models downloaded."

# ── 4. Start vLLM server ───────────────────────────────────
echo ""
echo "[4/6] Starting vLLM server (Llama 3.1 8B AWQ)..."

# Check for HuggingFace token (needed for Llama)
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set. You may need to set it for gated models."
    echo "  Run: export HF_TOKEN=your_token_here"
fi

# Start vLLM in the background
python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --dtype half \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.35 \
    > /tmp/vllm.log 2>&1 &

VLLM_PID=$!
echo "  vLLM starting (PID: $VLLM_PID), waiting for it to be ready..."

# Wait for vLLM to be ready
for i in $(seq 1 60); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  vLLM is ready."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: vLLM failed to start. Check /tmp/vllm.log"
        cat /tmp/vllm.log | tail -20
        exit 1
    fi
    sleep 2
done

# ── 5. Generate audio cache ────────────────────────────────
echo ""
echo "[5/6] Generating filler & backchannel audio cache..."

if [ -f "$REFERENCE_AUDIO" ]; then
    python scripts/generate_cache.py --reference "$REFERENCE_AUDIO"
else
    echo "  WARNING: Reference audio not found at $REFERENCE_AUDIO"
    echo "  Skipping cache generation. You'll need to run this manually:"
    echo "    python scripts/generate_cache.py --reference /path/to/your/voice.wav"
fi

# ── 6. Start voice agent server ────────────────────────────
echo ""
echo "[6/6] Starting voice agent server..."
echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  vLLM:        http://0.0.0.0:${VLLM_PORT}"
echo "  Voice Agent: http://0.0.0.0:8765"
echo "  Web Client:  http://0.0.0.0:8765/"
echo ""
echo "  To start the voice agent:"
echo "    python server.py --reference $REFERENCE_AUDIO"
echo ""
echo "  To check vLLM logs:"
echo "    tail -f /tmp/vllm.log"
echo "=============================================="
