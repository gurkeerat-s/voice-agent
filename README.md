# Voice Agent — self-hosted, real-time, custom voice

A full-duplex **real-time voice agent** you can talk to and interrupt live, with a
**custom fine-tuned TTS voice**, built to run on a **single GPU with no external
inference APIs**. Speech-to-text, the LLM, and the text-to-speech voice all run
locally — nothing is sent to a hosted speech API.

The goal was to get as close as possible to the "feels real" quality of commercial
voice agents (Sesame, etc.) on a solo budget. The interesting part isn't raw TTS
quality — it's the **conversational layer**: turn-taking, barge-in, backchannels
("mhm", "right"), filler timing, and LLM KV-cache warming so replies start fast.

## The custom voice (this part is done and shipped)

The TTS voice is a **full fine-tune of [Orpheus 3B](https://github.com/canopyai/Orpheus-TTS)**
on a self-collected **850-clip dataset**, tokenized through the **SNAC** neural audio
codec in Orpheus's official token format.

- Trained on an A100 80GB, 3 epochs, **~19 minutes**, loss **4.8 → 0.93**
- The trained model is published publicly: **[`thunderringz/orpheus-zara`](https://huggingface.co/thunderringz/orpheus-zara)**
- Orpheus (over CSM-1B) was chosen specifically because it has real vLLM support and a
  flat token stream, which gets TTS time-to-first-byte down to the ~200–500ms range
  needed for real-time conversation.

## Architecture

Two processes, so vLLM's async loop doesn't fight FastAPI's event loop:

```
  mic ──▶ ┌──────────────────────── server.py (FastAPI, ws://:8765) ─────────────────────┐
          │  VAD (Silero)  ─▶  STT (faster-whisper, streaming)  ─▶  LLM (local, OpenAI    │
          │       │                                                  API-compatible)      │
          │       │   barge-in / end-of-turn detection                     │             │
          │       ▼                                                         ▼             │
          │  agent/state_machine.py  ──── backchannels · fillers · turn-taking ───┐       │
          │                                                                       ▼       │
          │                                            TTS client ─── HTTP ──▶ Orpheus    │
          └───────────────────────────────────────────────────────────────────┬─────────┘
                                                                                ▼
                              scripts/orpheus_server.py  (Flask + vLLM, http://:8766)
                              loads the fine-tuned Zara model, streams 24kHz audio
```

- **`pipeline/`** — the swappable stages: `vad.py`, `stt.py`, `llm.py`, `tts.py`, `audio_io.py`
- **`agent/`** — the conversational brain: `state_machine.py` (the full-duplex loop),
  `backchannel.py`, `filler.py`, `conversation.py`
- **`scripts/orpheus_server.py`** — standalone Orpheus + vLLM TTS server
- **`finetune/` + `scripts/finetune_orpheus.py` + `scripts/prepare_dataset.py`** — the
  training pipeline that produced the Zara voice
- **`config.py`** — all thresholds and model choices (pydantic settings)

## Running it

TTS server (separate process), then the agent:

```bash
# TTS: loads the fine-tuned voice, listens on :8766
python scripts/orpheus_server.py --model-dir thunderringz/orpheus-zara

# Agent: VAD → STT → LLM → TTS over a WebSocket on :8765
python server.py          # then open http://localhost:8765 for the browser client
```

The LLM stage talks to any OpenAI-API-compatible endpoint (defaults to a local
`llama3.1:8b` via Ollama on `:11434`).

## Status — honest

- **Training pipeline: complete and verified.** It produced the published Zara model
  end-to-end (dataset prep → SNAC tokenization → fine-tune → inference test).
- **Real-time serving stack: fully built, bring-up in progress.** All the code above is
  written and wired together. The last clean GPU run was the training; standing the
  full live loop up on a fresh GPU box hit CUDA-container/dependency issues, and the
  known-working serving setup (container image + atomic install order) is documented for
  the next deploy. So treat the live agent as "built and runnable," not "hosted demo."

## Stack

PyTorch · PEFT / full fine-tune · vLLM · Orpheus 3B · SNAC · faster-whisper · Silero VAD ·
FastAPI · Flask · Ollama · RunPod (A100 / L40S)
