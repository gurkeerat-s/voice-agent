"""
Pre-generate filler and backchannel audio clips in the cloned voice.

Saves them as .npy files so the server can load them instantly at startup
without re-running TTS.

Usage:
    python scripts/generate_cache.py --reference voice/reference.wav
    python scripts/generate_cache.py --reference voice/reference.wav --output voice/cache
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf

from voice.clone import setup_voice, validate_reference_audio
from voice.cache import FILLER_PHRASES, BACKCHANNEL_PHRASES


def main():
    parser = argparse.ArgumentParser(description="Generate filler & backchannel audio cache")
    parser.add_argument("--reference", required=True, help="Path to reference voice WAV (6-10s)")
    parser.add_argument("--output", default="voice/cache", help="Output directory for cached audio")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    print(f"Reference audio: {args.reference}")
    validation = validate_reference_audio(args.reference)
    if not validation["valid"]:
        print(f"ERROR: {validation['issues']}")
        sys.exit(1)
    for issue in validation["issues"]:
        print(f"  Warning: {issue}")

    # Load TTS and clone voice
    tts = setup_voice(args.reference)

    # Generate fillers
    print("\nGenerating fillers...")
    for phrase in FILLER_PHRASES:
        audio = tts.synthesize_full(phrase)
        filename = f"filler_{phrase.lower().replace(' ', '_').replace('.', '').replace(',', '')}.npy"
        filepath = output_dir / filename
        np.save(filepath, audio)

        # Also save as WAV for manual listening
        wav_path = filepath.with_suffix(".wav")
        sf.write(str(wav_path), audio, tts.sample_rate)
        print(f"  '{phrase}' -> {len(audio)/tts.sample_rate:.2f}s  [{filepath}]")

    # Generate backchannels
    print("\nGenerating backchannels...")
    for phrase in BACKCHANNEL_PHRASES:
        audio = tts.synthesize_full(phrase)
        filename = f"bc_{phrase.lower().replace(' ', '_').replace('.', '').replace(',', '')}.npy"
        filepath = output_dir / filename
        np.save(filepath, audio)

        wav_path = filepath.with_suffix(".wav")
        sf.write(str(wav_path), audio, tts.sample_rate)
        print(f"  '{phrase}' -> {len(audio)/tts.sample_rate:.2f}s  [{filepath}]")

    # Save a manifest
    manifest = {
        "fillers": {p: f"filler_{p.lower().replace(' ', '_').replace('.', '').replace(',', '')}.npy"
                    for p in FILLER_PHRASES},
        "backchannels": {p: f"bc_{p.lower().replace(' ', '_').replace('.', '').replace(',', '')}.npy"
                         for p in BACKCHANNEL_PHRASES},
        "sample_rate": tts.sample_rate,
    }

    import json
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(FILLER_PHRASES) + len(BACKCHANNEL_PHRASES)
    print(f"\nDone. {total} clips saved to {output_dir}/")
    print(f"Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
