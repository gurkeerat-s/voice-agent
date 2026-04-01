"""
Prepare Zara training data for Orpheus fine-tuning.

Converts manifest.json + MP3 files into a HuggingFace Dataset with
pre-tokenized input_ids, matching the official Orpheus data format exactly.

Output dataset has 3 columns:
  - input_ids: list[int]  (full token sequence)
  - labels: list[int]     (same as input_ids)
  - attention_mask: list[int] (all 1s)

Usage:
    python scripts/prepare_dataset.py --data-dir data/training --output-dir data/orpheus-dataset
    python scripts/prepare_dataset.py --data-dir data/training --push-to-hub username/zara-orpheus
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import numpy as np


# ── Orpheus token constants (official spec) ────────────────
TOKENIZER_LENGTH = 128256
BOS_TOKEN = 128000         # <|begin_of_text|>
END_OF_TEXT = 128009       # <|end_of_text|>
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263
AUDIO_BASE = 128266        # tokenizer_length + 10
CODEBOOK_SIZE = 4096
SAMPLE_RATE = 24000

VOICE_NAME = "zara"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
ORPHEUS_MODEL = "canopylabs/orpheus-tts-0.1-finetune-prod"


def encode_audio_to_snac_tokens(waveform, snac_model, device="cuda"):
    """Encode a waveform into interleaved SNAC audio token IDs.

    Returns list of ints — 7 tokens per coarsest time step.
    """
    audio = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

    with torch.no_grad():
        codes = snac_model.encode(audio)

    # SNAC 24kHz: 3 codebooks at 1:2:4 temporal ratio
    c0 = codes[0][0].cpu()  # [T0]
    c1 = codes[1][0].cpu()  # [2*T0]
    c2 = codes[2][0].cpu()  # [4*T0]

    T0 = c0.shape[0]
    token_ids = []
    for t in range(T0):
        token_ids.append(int(c0[t].item()) + AUDIO_BASE)
        token_ids.append(int(c1[2 * t].item()) + AUDIO_BASE + CODEBOOK_SIZE)
        token_ids.append(int(c2[4 * t].item()) + AUDIO_BASE + 2 * CODEBOOK_SIZE)
        token_ids.append(int(c2[4 * t + 1].item()) + AUDIO_BASE + 3 * CODEBOOK_SIZE)
        token_ids.append(int(c1[2 * t + 1].item()) + AUDIO_BASE + 4 * CODEBOOK_SIZE)
        token_ids.append(int(c2[4 * t + 2].item()) + AUDIO_BASE + 5 * CODEBOOK_SIZE)
        token_ids.append(int(c2[4 * t + 3].item()) + AUDIO_BASE + 6 * CODEBOOK_SIZE)

    return token_ids


def remove_duplicate_frames(audio_tokens):
    """Remove consecutive SNAC frames where codebook-0 is identical.

    This is what the official Orpheus preprocessing does to compress
    silence and repeated segments.
    """
    if len(audio_tokens) < 14:
        return audio_tokens

    cleaned = audio_tokens[:7]  # keep first frame
    for i in range(7, len(audio_tokens), 7):
        frame = audio_tokens[i:i + 7]
        if len(frame) < 7:
            break
        prev_frame = audio_tokens[i - 7:i]
        # Compare codebook-0 token (first in each group of 7)
        if frame[0] != prev_frame[0]:
            cleaned.extend(frame)

    return cleaned


def build_token_sequence(text, audio_tokens, tokenizer):
    """Build the full Orpheus training sequence.

    Official format:
      [start_of_human] + [BOS] + text_tokens + [end_of_text] + [end_of_human]
      + [start_of_ai] + [start_of_speech] + audio_tokens + [end_of_speech] + [end_of_ai]
    """
    prompt = f"{VOICE_NAME}: {text}"
    # add_special_tokens=True prepends BOS (128000)
    text_ids = tokenizer.encode(prompt, add_special_tokens=True)

    sequence = (
        [START_OF_HUMAN]
        + text_ids
        + [END_OF_TEXT, END_OF_HUMAN]
        + [START_OF_AI, START_OF_SPEECH]
        + audio_tokens
        + [END_OF_SPEECH, END_OF_AI]
    )
    return sequence


def main():
    parser = argparse.ArgumentParser(description="Prepare Orpheus training dataset from Zara audio")
    parser.add_argument("--data-dir", default="data/training", help="Directory with manifest.json + MP3s")
    parser.add_argument("--output-dir", default="data/orpheus-dataset", help="Where to save the HF dataset")
    parser.add_argument("--push-to-hub", default=None, help="Optional: push to HF Hub (e.g. username/dataset-name)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Skip clips longer than this (seconds)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate frame removal")
    args = parser.parse_args()

    import snac as snac_lib
    from transformers import AutoTokenizer
    from datasets import Dataset

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"Loading SNAC model: {SNAC_MODEL_NAME}")
    snac_model = snac_lib.SNAC.from_pretrained(SNAC_MODEL_NAME).to("cuda").eval()

    print(f"Loading tokenizer: {ORPHEUS_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)

    rows = {"input_ids": [], "labels": [], "attention_mask": []}
    lines = manifest["lines"]
    skipped = 0

    for i, item in enumerate(lines):
        audio_path = data_dir / item["audio"]
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            # Load and resample audio
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Skip clips that are too long
            duration = waveform.shape[1] / SAMPLE_RATE
            if duration > args.max_duration:
                skipped += 1
                continue

            # Encode to SNAC tokens
            audio_tokens = encode_audio_to_snac_tokens(waveform, snac_model)

            # Remove duplicate frames (official preprocessing step)
            if not args.no_dedup:
                before = len(audio_tokens)
                audio_tokens = remove_duplicate_frames(audio_tokens)
                after = len(audio_tokens)
                if before != after and (i + 1) % 100 == 0:
                    print(f"    dedup: {before} → {after} tokens ({(before - after) / before * 100:.0f}% reduction)")

            # Build full sequence
            sequence = build_token_sequence(item["text"], audio_tokens, tokenizer)

            rows["input_ids"].append(sequence)
            rows["labels"].append(sequence)  # official: labels = input_ids
            rows["attention_mask"].append([1] * len(sequence))

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i + 1}/{len(lines)}] {item['text'][:50]}... "
                      f"({len(sequence)} tokens, {duration:.1f}s audio)")

        except Exception as e:
            print(f"  [{i + 1}/{len(lines)}] ERROR: {e}")
            skipped += 1
            continue

    print(f"\nProcessed {len(rows['input_ids'])} samples ({skipped} skipped)")

    # Token length stats
    lengths = [len(ids) for ids in rows["input_ids"]]
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Create HF dataset
    dataset = Dataset.from_dict(rows)
    print(f"\nDataset: {dataset}")

    # Save locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"Saved to {output_dir}")

    # Optionally push to Hub
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=True)
        print("Done!")


if __name__ == "__main__":
    main()
