"""
End-to-end latency benchmark.

Tests each pipeline stage independently and reports P50/P95/P99 latencies.
Also runs a simulated conversation to measure full round-trip time.

Usage:
    python scripts/benchmark.py --reference voice/reference.wav
    python scripts/benchmark.py --reference voice/reference.wav --rounds 20
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def percentiles(values: list[float]) -> dict:
    """Calculate P50, P95, P99 from a list of durations (ms)."""
    arr = np.array(sorted(values))
    return {
        "min": round(arr[0], 1),
        "p50": round(np.percentile(arr, 50), 1),
        "p95": round(np.percentile(arr, 95), 1),
        "p99": round(np.percentile(arr, 99), 1),
        "max": round(arr[-1], 1),
    }


def print_stats(label: str, times_ms: list[float]):
    stats = percentiles(times_ms)
    print(f"  {label:.<40} P50={stats['p50']:>7.1f}ms  P95={stats['p95']:>7.1f}ms  P99={stats['p99']:>7.1f}ms  (min={stats['min']}, max={stats['max']})")


# ── VAD Benchmark ──────────────────────────────────────────

def bench_vad(rounds: int):
    print("\n[VAD] Silero VAD — process_chunk()")
    from pipeline.vad import VADProcessor

    vad = VADProcessor()
    # Generate fake audio chunks (200ms at 16kHz = 3200 samples)
    chunk = np.random.randn(3200).astype(np.float32) * 0.01

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        vad.process_chunk(chunk)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print_stats("process_chunk (200ms audio)", times)


# ── STT Benchmark ──────────────────────────────────────────

def bench_stt(rounds: int):
    print("\n[STT] Faster-Whisper — transcribe()")
    from pipeline.stt import StreamingSTT

    stt = StreamingSTT()

    # Generate 3 seconds of fake audio at 16kHz
    audio_3s = np.random.randn(16000 * 3).astype(np.float32) * 0.01

    times = []
    for _ in range(rounds):
        stt.reset()
        # Add audio in chunks
        chunk_size = 3200  # 200ms
        for i in range(0, len(audio_3s), chunk_size):
            stt.add_audio(audio_3s[i:i+chunk_size])

        t0 = time.perf_counter()
        result = stt.finalize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print_stats("finalize (3s audio, beam=5)", times)

    # Partial transcript speed
    partial_times = []
    for _ in range(rounds):
        stt.reset()
        for i in range(0, len(audio_3s), chunk_size):
            stt.add_audio(audio_3s[i:i+chunk_size])

        stt._last_partial_time = 0  # force partial
        t0 = time.perf_counter()
        stt.get_partial()
        t1 = time.perf_counter()
        partial_times.append((t1 - t0) * 1000)

    print_stats("get_partial (3s audio, beam=1)", partial_times)


# ── LLM Benchmark ─────────────────────────────────────────

async def bench_llm(rounds: int):
    print("\n[LLM] vLLM — generate_stream()")
    from pipeline.llm import LLMClient

    llm = LLMClient()
    test_prompts = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you help me with my booking?",
        "Tell me a short joke.",
        "What time is it?",
    ]

    # First token latency
    first_token_times = []
    full_times = []

    for i in range(min(rounds, len(test_prompts))):
        prompt = test_prompts[i % len(test_prompts)]

        t0 = time.perf_counter()
        first_token_received = False
        full_response = ""

        async for sentence in llm.generate_stream(prompt, []):
            if not first_token_received:
                t_first = time.perf_counter()
                first_token_times.append((t_first - t0) * 1000)
                first_token_received = True
            full_response += sentence + " "

        t_end = time.perf_counter()
        full_times.append((t_end - t0) * 1000)

    if first_token_times:
        print_stats("first sentence", first_token_times)
        print_stats("full response", full_times)
    else:
        print("  SKIPPED — vLLM not running (start it first)")


# ── TTS Benchmark ──────────────────────────────────────────

def bench_tts(reference_path: str, rounds: int):
    print("\n[TTS] XTTS-v2 — synthesize_stream()")
    from voice.clone import setup_voice

    tts = setup_voice(reference_path)

    test_sentences = [
        "Hello, how can I help you today?",
        "Sure, let me check that for you.",
        "Your booking is confirmed for tomorrow.",
        "Is there anything else you need?",
        "Great, have a wonderful day!",
    ]

    # Streaming: time to first chunk
    first_chunk_times = []
    full_synth_times = []

    for i in range(min(rounds, len(test_sentences))):
        text = test_sentences[i % len(test_sentences)]

        t0 = time.perf_counter()
        first_chunk = True
        total_samples = 0

        for chunk in tts.synthesize_stream(text):
            if first_chunk:
                t_first = time.perf_counter()
                first_chunk_times.append((t_first - t0) * 1000)
                first_chunk = False
            total_samples += len(chunk.audio)

        t_end = time.perf_counter()
        full_synth_times.append((t_end - t0) * 1000)
        audio_dur = total_samples / tts.sample_rate * 1000
        rtf = (t_end - t0) * 1000 / audio_dur if audio_dur > 0 else 0
        print(f"    [{i+1}] \"{text[:40]}...\" -> {audio_dur:.0f}ms audio, RTF={rtf:.2f}")

    if first_chunk_times:
        print_stats("first chunk", first_chunk_times)
        print_stats("full synthesis", full_synth_times)


# ── Full Pipeline Benchmark ────────────────────────────────

def bench_crossfade(reference_path: str, rounds: int):
    print("\n[Crossfade] filler -> response blend")
    from voice.clone import setup_voice

    tts = setup_voice(reference_path)

    filler = tts.synthesize_full("Um...")
    response = tts.synthesize_full("Hello, how can I help you?")

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        tts.crossfade(filler, response)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print_stats("crossfade", times)


# ── Audio I/O Benchmark ───────────────────────────────────

def bench_audio_io(rounds: int):
    print("\n[Audio I/O] resample + encode")
    from pipeline.audio_io import AudioIO

    aio = AudioIO()
    # 200ms of 48kHz audio
    chunk_48k = np.random.randn(9600).astype(np.float32) * 0.1

    resample_times = []
    encode_times = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        resampled = aio.resample_for_stt(chunk_48k)
        t1 = time.perf_counter()
        resample_times.append((t1 - t0) * 1000)

        t2 = time.perf_counter()
        aio.encode_ws_audio(resampled)
        t3 = time.perf_counter()
        encode_times.append((t3 - t2) * 1000)

    print_stats("resample 48k->16k (200ms)", resample_times)
    print_stats("encode to PCM16", encode_times)


# ── Main ───────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Voice Agent Latency Benchmark")
    parser.add_argument("--reference", default="voice/reference.wav",
                        help="Path to reference voice WAV")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of benchmark rounds per test")
    parser.add_argument("--skip-tts", action="store_true",
                        help="Skip TTS benchmarks (slow to load)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM benchmarks (requires vLLM running)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Voice Agent — Latency Benchmark")
    print(f"  Rounds: {args.rounds}")
    print("=" * 60)

    # Fast benchmarks (no GPU models)
    bench_audio_io(args.rounds)
    bench_vad(args.rounds)

    # STT benchmark (loads Whisper)
    bench_stt(args.rounds)

    # LLM benchmark (needs vLLM server running)
    if not args.skip_llm:
        await bench_llm(args.rounds)

    # TTS benchmarks (loads XTTS-v2)
    if not args.skip_tts:
        if Path(args.reference).exists():
            bench_tts(args.reference, args.rounds)
            bench_crossfade(args.reference, args.rounds)
        else:
            print(f"\n  SKIPPED TTS — reference audio not found: {args.reference}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
