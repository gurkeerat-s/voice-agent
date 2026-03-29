"""
Generate training data for CSM-1B fine-tuning using ElevenLabs TTS.

Usage:
    python scripts/generate_training_data.py --api-key YOUR_KEY
    python scripts/generate_training_data.py --api-key YOUR_KEY --voice-id VOICE_ID --output data/training
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from training_lines import TRAINING_LINES


def generate_audio(api_key: str, voice_id: str, text: str, output_path: str) -> bool:
    """Call ElevenLabs API to synthesize one line."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.4,
            "use_speaker_boost": True,
        },
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"  ERROR {response.status_code}: {response.text[:200]}")
        return False


def get_remaining_credits(api_key: str) -> dict:
    """Check remaining ElevenLabs credits."""
    url = "https://api.elevenlabs.io/v1/user/subscription"
    headers = {"xi-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        used = data.get("character_count", 0)
        limit = data.get("character_limit", 0)
        return {"used": used, "limit": limit, "remaining": limit - used}
    return {"used": 0, "limit": 0, "remaining": 0}


def main():
    parser = argparse.ArgumentParser(description="Generate TTS training data via ElevenLabs")
    parser.add_argument("--api-key", required=True, help="ElevenLabs API key")
    parser.add_argument("--voice-id", default="jqcCZkN6Knx8BJ5TBdYR", help="ElevenLabs voice ID (default: Zara)")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from line number (0-indexed)")
    parser.add_argument("--max-lines", type=int, default=None, help="Max lines to generate (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check credits
    credits = get_remaining_credits(args.api_key)
    total_chars = sum(len(line) for line in TRAINING_LINES[args.start_from:])
    print(f"ElevenLabs credits: {credits['remaining']:,} remaining of {credits['limit']:,}")
    print(f"Lines: {len(TRAINING_LINES)}, Characters needed: {total_chars:,}")
    print(f"Estimated audio: ~{total_chars/1100:.0f} minutes")

    if credits['remaining'] < total_chars:
        can_do = credits['remaining']
        count = 0
        for line in TRAINING_LINES[args.start_from:]:
            if can_do < len(line):
                break
            can_do -= len(line)
            count += 1
        print(f"WARNING: Only enough credits for ~{count} lines. Will generate what we can.")

    lines = TRAINING_LINES[args.start_from:]
    if args.max_lines:
        lines = lines[:args.max_lines]

    # Generate audio for each line
    manifest = []
    chars_used = 0
    skipped = 0

    for i, text in enumerate(lines):
        idx = i + args.start_from
        filename = f"line_{idx:04d}.mp3"
        filepath = output_dir / filename

        # Skip if already generated
        if filepath.exists():
            skipped += 1
            manifest.append({"id": idx, "text": text, "audio": filename})
            continue

        print(f"  [{idx+1}/{len(TRAINING_LINES)}] {text[:60]}...")

        success = generate_audio(args.api_key, args.voice_id, text, str(filepath))

        if success:
            manifest.append({"id": idx, "text": text, "audio": filename})
            chars_used += len(text)
        else:
            print(f"    FAILED — resume with: --start-from {idx}")
            break

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "voice_id": args.voice_id,
            "total_lines": len(manifest),
            "total_chars": chars_used,
            "lines": manifest,
        }, f, indent=2)

    print(f"\nDone! Generated {len(manifest) - skipped} new files ({skipped} skipped)")
    print(f"Total: {len(manifest)} audio files in {output_dir}/")
    print(f"Characters used: {chars_used:,}")

    credits_after = get_remaining_credits(args.api_key)
    print(f"Credits remaining: {credits_after['remaining']:,}")


if __name__ == "__main__":
    main()
