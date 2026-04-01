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


def get_token_config():
    """Orpheus special token IDs (hardcoded per official spec).

    Layout: base Llama tokenizer has 128256 tokens (IDs 0–128255).
    Orpheus adds structured control tokens and 7×4096 audio tokens:
      128256 = tokenizer_length
      128257 = start_of_speech
      128258 = end_of_speech
      128259 = start_of_human
      128260 = end_of_human
      128261 = start_of_ai
      128262 = end_of_ai
      128263 = pad_token
      128264 = (reserved)
      128265 = (reserved)
      128266 = audio_tokens_start (= tokenizer_length + 10)
      128266 + 7*4096 - 1 = last audio token
    """
    return {
        "start_of_speech": 128257,
        "end_of_speech": 128258,
        "start_of_human": 128259,
        "end_of_human": 128260,
        "start_of_ai": 128261,
        "end_of_ai": 128262,
        "pad_token": 128263,
        "audio_base": 128266,  # first audio token ID
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
    tc = get_token_config()
    print(f"  Audio base: {tc['audio_base']}")
    print(f"  start_of_speech: {tc['start_of_speech']}")
    print(f"  end_of_speech: {tc['end_of_speech']}")

    encoded_samples = []
    lines = manifest["lines"]

    for i, item in enumerate(lines):
        audio_path = data_dir / item["audio"]
        if not audio_path.exists():
            continue

        try:
            audio_tokens = encode_audio_to_tokens(audio_path, snac_model, tc)

            # Orpheus training sequence format:
            # [start_of_human] + text_tokens + [end_of_human, start_of_ai, start_of_speech]
            #   + audio_tokens + [end_of_speech, end_of_ai]
            text = f"{VOICE_NAME}: {item['text']}"
            text_tokens = tokenizer.encode(text, add_special_tokens=False)

            prompt_tokens = (
                [tc["start_of_human"]]
                + text_tokens
                + [tc["end_of_human"], tc["start_of_ai"], tc["start_of_speech"]]
            )
            full_sequence = (
                prompt_tokens
                + audio_tokens
                + [tc["end_of_speech"], tc["end_of_ai"]]
            )

            encoded_samples.append({
                "id": item["id"],
                "text": item["text"],
                "num_prompt_tokens": len(prompt_tokens),
                "num_audio_tokens": len(audio_tokens),
                "token_ids": full_sequence,
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i + 1}/{len(lines)}] {item['text'][:50]}... "
                      f"({len(prompt_tokens)} prompt + {len(audio_tokens)} audio tokens)")

        except Exception as e:
            print(f"  [{i + 1}/{len(lines)}] ERROR: {e}")
            continue

    # Save encoded data
    output_path = data_dir / "encoded_tokens.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": ORPHEUS_MODEL,
            "voice": VOICE_NAME,
            "token_config": tc,
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
    tc = get_token_config()
    tokenizer.pad_token_id = tc["pad_token"]  # 128263

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
        def __init__(self, samples, max_len=4096):
            self.samples = []
            self.prompt_lengths = []  # how many tokens to mask in labels
            skipped = 0
            for s in samples:
                ids = s["token_ids"]
                prompt_len = s["num_prompt_tokens"]

                if len(ids) <= max_len:
                    self.samples.append(ids)
                    self.prompt_lengths.append(prompt_len)
                else:
                    # Truncate audio to fit, keeping 7-token SNAC alignment
                    # Sequence ends with [end_of_speech, end_of_ai] = 2 tokens
                    audio_start = prompt_len
                    end_tokens = ids[-2:]  # [end_of_speech, end_of_ai]
                    max_audio = ((max_len - prompt_len - 2) // 7) * 7
                    if max_audio <= 0:
                        skipped += 1
                        continue
                    truncated = ids[:audio_start] + ids[audio_start:audio_start + max_audio] + end_tokens
                    self.samples.append(truncated)
                    self.prompt_lengths.append(prompt_len)

            if skipped:
                print(f"  Skipped {skipped} samples (too long even after truncation)")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            ids = self.samples[idx]
            prompt_len = self.prompt_lengths[idx]

            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = torch.tensor(ids, dtype=torch.long)
            # Mask the text prompt + start_audio token — only train on audio output
            labels[:prompt_len] = -100

            return {"input_ids": input_ids, "labels": labels}

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
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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

    tc = get_token_config()

    print(f"Loading SNAC decoder: {SNAC_MODEL}")
    snac_model = snac_lib.SNAC.from_pretrained(SNAC_MODEL).to("cuda").eval()

    # Build prompt matching the exact training format from prepare_dataset.py:
    # [start_of_human] + [BOS] + text_tokens + [end_of_text, end_of_human]
    #   + [start_of_ai, start_of_speech]
    text = f"{VOICE_NAME}: {args.text}"
    # add_special_tokens=True prepends BOS (128000), matching training data
    text_tokens = tokenizer.encode(text, add_special_tokens=True)
    prompt = (
        [tc["start_of_human"]]
        + text_tokens
        + [128009, tc["end_of_human"], tc["start_of_ai"], tc["start_of_speech"]]
    )
    input_ids = torch.tensor([prompt], dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    print(f"Generating: \"{args.text}\" ({len(prompt)} prompt tokens)")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_tokens,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tc["pad_token"],
        )

    # Extract generated tokens (after the prompt)
    generated = output[0][input_ids.shape[1]:].cpu().tolist()

    # Collect audio tokens until end_of_speech
    audio_tokens = []
    for tok in generated:
        if tok == tc["end_of_speech"]:
            break
        if tok >= tc["audio_base"]:
            audio_tokens.append(tok)

    if not audio_tokens:
        print("No audio tokens generated!")
        return

    print(f"Generated {len(audio_tokens)} audio tokens")

    # Decode: token IDs → SNAC codes → audio
    base = tc["audio_base"]
    cs = tc["codebook_size"]

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
    trn.add_argument("--epochs", type=int, default=8)
    trn.add_argument("--batch-size", type=int, default=1)
    trn.add_argument("--grad-accum", type=int, default=8)
    trn.add_argument("--lr", type=float, default=1e-5)
    trn.add_argument("--lora-rank", type=int, default=64)
    trn.add_argument("--lora-alpha", type=int, default=128)
    trn.add_argument("--max-len", type=int, default=4096)

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
