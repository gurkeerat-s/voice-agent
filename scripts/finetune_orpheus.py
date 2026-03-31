"""
Fine-tune Orpheus 3B TTS on Zara voice data.

Steps:
  1. Encode Zara audio files into SNAC audio tokens
  2. LoRA fine-tune Orpheus to produce Zara's voice
  3. Merge adapter into base model
  4. Test with a sample

Usage:
    python scripts/finetune_orpheus.py encode --data-dir data/training
    python scripts/finetune_orpheus.py train --data-dir data/training
    python scripts/finetune_orpheus.py merge --adapter-dir checkpoints/orpheus-zara --output-dir models/orpheus-zara
    python scripts/finetune_orpheus.py test --model-dir models/orpheus-zara --text "Hey, how's it going?"
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio


# ── Constants ──────────────────────────────────────────────
ORPHEUS_MODEL = "canopylabs/orpheus-tts-0.1-finetune-prod"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
VOICE_NAME = "zara"
SAMPLE_RATE = 24000


def get_token_config(tokenizer):
    """Get audio token offsets from the Orpheus tokenizer."""
    start_token = tokenizer.encode("<custom_token_10>", add_special_tokens=False)[0]
    end_token_1 = tokenizer.encode("<custom_token_11>", add_special_tokens=False)[0]
    end_token_2 = tokenizer.encode("<custom_token_12>", add_special_tokens=False)[0]

    # Audio tokens start after the custom tokens
    # In Orpheus, audio codes for each of 7 "virtual codebooks" are offset by 4096
    audio_base = start_token + 1  # first audio token ID

    return {
        "start_token": start_token,
        "end_token_1": end_token_1,
        "end_token_2": end_token_2,
        "audio_base": audio_base,
        "codebook_size": 4096,
    }


def encode_audio_to_tokens(audio_path, snac_model, token_config, device="cuda"):
    """Convert an audio file to Orpheus audio token IDs."""
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

    with torch.no_grad():
        codes = snac_model.encode(audio)

    # SNAC 24kHz returns 3 code groups at ratios 1:2:4
    # Shape: [1, T] for each group
    c0 = codes[0][0].cpu()  # coarsest, [T0]
    c1 = codes[1][0].cpu()  # middle, [2*T0]
    c2 = codes[2][0].cpu()  # finest, [4*T0]

    T0 = c0.shape[0]
    base = token_config["audio_base"]
    cs = token_config["codebook_size"]

    # Interleave: 7 tokens per coarsest time step
    token_ids = []
    for t in range(T0):
        token_ids.append(int(c0[t].item()) + base)
        token_ids.append(int(c1[2 * t].item()) + base + cs)
        token_ids.append(int(c2[4 * t].item()) + base + 2 * cs)
        token_ids.append(int(c2[4 * t + 1].item()) + base + 3 * cs)
        token_ids.append(int(c1[2 * t + 1].item()) + base + 4 * cs)
        token_ids.append(int(c2[4 * t + 2].item()) + base + 5 * cs)
        token_ids.append(int(c2[4 * t + 3].item()) + base + 6 * cs)

    return token_ids


# ── Encode command ─────────────────────────────────────────
def cmd_encode(args):
    """Encode all training audio files to SNAC token sequences."""
    import snac as snac_lib
    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading SNAC model: {SNAC_MODEL}")
    snac_model = snac_lib.SNAC.from_pretrained(SNAC_MODEL).to("cuda").eval()

    print(f"Loading tokenizer: {ORPHEUS_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    token_config = get_token_config(tokenizer)
    print(f"  Audio base token: {token_config['audio_base']}")
    print(f"  Start token: {token_config['start_token']}")

    encoded_samples = []
    lines = manifest["lines"]

    for i, item in enumerate(lines):
        audio_path = data_dir / item["audio"]
        if not audio_path.exists():
            continue

        try:
            audio_tokens = encode_audio_to_tokens(audio_path, snac_model, token_config)

            # Build the full token sequence:
            # text_tokens + start_audio + audio_tokens + end_audio
            text = f"{VOICE_NAME}: {item['text']}"
            text_tokens = tokenizer.encode(text, add_special_tokens=True)

            full_sequence = (
                text_tokens
                + [token_config["start_token"]]
                + audio_tokens
                + [token_config["end_token_1"], token_config["end_token_2"]]
            )

            encoded_samples.append({
                "id": item["id"],
                "text": item["text"],
                "num_text_tokens": len(text_tokens),
                "num_audio_tokens": len(audio_tokens),
                "token_ids": full_sequence,
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i + 1}/{len(lines)}] {item['text'][:50]}... "
                      f"({len(text_tokens)} text + {len(audio_tokens)} audio tokens)")

        except Exception as e:
            print(f"  [{i + 1}/{len(lines)}] ERROR: {e}")
            continue

    # Save encoded data
    output_path = data_dir / "encoded_tokens.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": ORPHEUS_MODEL,
            "voice": VOICE_NAME,
            "token_config": token_config,
            "num_samples": len(encoded_samples),
            "samples": encoded_samples,
        }, f)

    total_audio_tokens = sum(s["num_audio_tokens"] for s in encoded_samples)
    print(f"\nEncoded {len(encoded_samples)} samples")
    print(f"Total audio tokens: {total_audio_tokens:,}")
    print(f"Saved to {output_path}")


# ── Train command ──────────────────────────────────────────
def cmd_train(args):
    """LoRA fine-tune Orpheus on encoded Zara data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset

    data_dir = Path(args.data_dir)
    encoded_path = data_dir / "encoded_tokens.json"

    if not encoded_path.exists():
        print("No encoded_tokens.json found. Run 'encode' first.")
        sys.exit(1)

    with open(encoded_path) as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Training samples: {len(samples)}")

    # ── Load model ─────────────────────────────────────────
    print(f"Loading model: {ORPHEUS_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ORPHEUS_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Apply LoRA ─────────────────────────────────────────
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────
    class OrpheusDataset(Dataset):
        def __init__(self, samples, max_len=2048):
            self.samples = []
            for s in samples:
                ids = s["token_ids"]
                if len(ids) <= max_len:
                    self.samples.append(ids)
                else:
                    # Truncate (keep text + start + truncated audio + end)
                    self.samples.append(ids[:max_len - 2] + ids[-2:])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            ids = self.samples[idx]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long),
            }

    dataset = OrpheusDataset(samples, max_len=args.max_len)
    print(f"Dataset: {len(dataset)} samples")

    # ── Data collator (pad to same length) ─────────────────
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad
        max_len = max(ids.shape[0] for ids in input_ids)
        padded_ids = []
        padded_labels = []

        for ids, lbl in zip(input_ids, labels):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100)]))

        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack([
                (ids != tokenizer.pad_token_id).long() for ids in padded_ids
            ]),
        }

    # ── Training ───────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print(f"\nStarting training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print()

    trainer.train()

    # Save adapter
    final_path = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nSaved adapter to {final_path}")


# ── Merge command ──────────────────────────────────────────
def cmd_merge(args):
    """Merge LoRA adapter into base model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {ORPHEUS_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        ORPHEUS_MODEL,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done!")


# ── Test command ───────────────────────────────────────────
def cmd_test(args):
    """Generate a test audio sample."""
    import snac as snac_lib
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = args.model_dir or ORPHEUS_MODEL
    print(f"Loading model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    token_config = get_token_config(tokenizer)

    print(f"Loading SNAC decoder: {SNAC_MODEL}")
    snac_model = snac_lib.SNAC.from_pretrained(SNAC_MODEL).to("cuda").eval()

    # Generate
    text = f"{VOICE_NAME}: {args.text}"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    print(f"Generating: \"{args.text}\"")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.3,
            do_sample=True,
        )

    # Extract generated audio tokens (skip input tokens)
    generated = output[0][input_ids.shape[1]:].cpu().tolist()

    # Filter to audio tokens only (between start and end)
    audio_tokens = []
    in_audio = False
    for tok in generated:
        if tok == token_config["start_token"]:
            in_audio = True
            continue
        if tok in (token_config["end_token_1"], token_config["end_token_2"]):
            break
        if in_audio:
            audio_tokens.append(tok)

    if not audio_tokens:
        print("No audio tokens generated!")
        return

    print(f"Generated {len(audio_tokens)} audio tokens")

    # Decode: token IDs → SNAC codes → audio
    base = token_config["audio_base"]
    cs = token_config["codebook_size"]

    # Reshape into groups of 7
    num_frames = len(audio_tokens) // 7
    audio_tokens = audio_tokens[:num_frames * 7]

    codes_0, codes_1, codes_2 = [], [], []
    for i in range(0, len(audio_tokens), 7):
        t = audio_tokens[i:i + 7]
        codes_0.append(t[0] - base)
        codes_1.append(t[1] - (base + cs))
        codes_2.append(t[2] - (base + 2 * cs))
        codes_2.append(t[3] - (base + 3 * cs))
        codes_1.append(t[4] - (base + 4 * cs))
        codes_2.append(t[5] - (base + 5 * cs))
        codes_2.append(t[6] - (base + 6 * cs))

    # Clamp to valid range
    codes_0 = [max(0, min(4095, c)) for c in codes_0]
    codes_1 = [max(0, min(4095, c)) for c in codes_1]
    codes_2 = [max(0, min(4095, c)) for c in codes_2]

    # Convert to tensors for SNAC — shape [1, T] (batch, time)
    c0 = torch.tensor(codes_0, dtype=torch.long).unsqueeze(0).to("cuda")
    c1 = torch.tensor(codes_1, dtype=torch.long).unsqueeze(0).to("cuda")
    c2 = torch.tensor(codes_2, dtype=torch.long).unsqueeze(0).to("cuda")

    with torch.no_grad():
        audio = snac_model.decode([c0, c1, c2])

    audio_np = audio.squeeze().cpu().numpy()

    output_path = args.output or "test_orpheus_zara.wav"
    sf.write(output_path, audio_np, SAMPLE_RATE)
    print(f"Saved to {output_path} ({len(audio_np) / SAMPLE_RATE:.1f}s)")


# ── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Orpheus 3B Fine-Tuning for Zara Voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode
    enc = subparsers.add_parser("encode", help="Encode audio files to SNAC tokens")
    enc.add_argument("--data-dir", default="data/training")

    # Train
    trn = subparsers.add_parser("train", help="LoRA fine-tune Orpheus")
    trn.add_argument("--data-dir", default="data/training")
    trn.add_argument("--output-dir", default="checkpoints/orpheus-zara")
    trn.add_argument("--epochs", type=int, default=5)
    trn.add_argument("--batch-size", type=int, default=1)
    trn.add_argument("--grad-accum", type=int, default=8)
    trn.add_argument("--lr", type=float, default=2e-5)
    trn.add_argument("--lora-rank", type=int, default=16)
    trn.add_argument("--lora-alpha", type=int, default=32)
    trn.add_argument("--max-len", type=int, default=2048)

    # Merge
    mrg = subparsers.add_parser("merge", help="Merge adapter into base model")
    mrg.add_argument("--adapter-dir", required=True)
    mrg.add_argument("--output-dir", default="models/orpheus-zara")

    # Test
    tst = subparsers.add_parser("test", help="Generate test audio")
    tst.add_argument("--model-dir", default=None, help="Merged model dir (or uses base model)")
    tst.add_argument("--text", default="Hey, how's it going? I haven't seen you in a while!")
    tst.add_argument("--output", default="test_orpheus_zara.wav")
    tst.add_argument("--max-tokens", type=int, default=1200)

    args = parser.parse_args()

    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
