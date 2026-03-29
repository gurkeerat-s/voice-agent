"""
LoRA fine-tuning script for Sesame CSM-1B using HuggingFace Transformers Trainer.

Expects:
  - A folder of MP3 audio files (e.g. data/training/line_0000.mp3)
  - A manifest.json in the same folder mapping text to audio filenames:
      {
        "voice_id": "...",
        "total_lines": 255,
        "lines": [
          {"id": 0, "text": "Hey! How's it going?", "audio": "line_0000.mp3"},
          ...
        ]
      }

Usage:
    # Generate the manifest first (if not already done by generate_training_data.py):
    python scripts/finetune_csm.py --build-manifest --data-dir data/training

    # Run training:
    python scripts/finetune_csm.py --data-dir data/training --output-dir checkpoints/csm-lora

    # Resume from checkpoint:
    python scripts/finetune_csm.py --data-dir data/training --output-dir checkpoints/csm-lora --resume

    # Inference with fine-tuned model:
    python scripts/finetune_csm.py --inference --adapter-path checkpoints/csm-lora --text "Hello there!"

Requirements:
    pip install transformers>=4.52.1 peft>=0.15.0 datasets torchaudio soundfile accelerate
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# HuggingFace imports  -- these are the REAL class names from
# transformers.models.csm (available since transformers 4.52.1)
# ---------------------------------------------------------------------------
from transformers import (
    CsmForConditionalGeneration,
    CsmProcessor,
    TrainingArguments,
    Trainer,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24_000  # CSM / Mimi codec expects 24 kHz
SPEAKER_ID = 0        # single-speaker data; we always use speaker 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CSMFineTuneDataset(Dataset):
    """
    Loads a manifest.json + MP3 folder into a simple map-style dataset.
    Each item is a dict with "text" (str) and "audio" (np.ndarray at 24 kHz).
    The Trainer's data_collator handles tokenization + label creation.
    """

    def __init__(self, manifest_path: str | Path, split: str = "train", val_ratio: float = 0.05, seed: int = 42):
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        lines = manifest["lines"]
        audio_dir = manifest_path.parent

        # Deterministic train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(lines))
        n_val = max(1, int(len(lines) * val_ratio))

        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.samples = []
        for idx in indices:
            entry = lines[idx]
            audio_path = audio_dir / entry["audio"]
            if audio_path.exists():
                self.samples.append({
                    "text": entry["text"],
                    "audio_path": str(audio_path),
                })

        print(f"[{split}] Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio and resample to 24 kHz
        waveform, sr = torchaudio.load(sample["audio_path"])
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)

        # Convert to mono float32 numpy array
        audio_array = waveform.squeeze(0).numpy().astype(np.float32)

        return {
            "text": sample["text"],
            "audio": audio_array,
        }


# ---------------------------------------------------------------------------
# Data collator -- this is where the magic happens
# ---------------------------------------------------------------------------
def make_data_collator(processor: CsmProcessor):
    """
    Returns a data collator function that converts raw samples into the format
    CsmForConditionalGeneration.forward() expects.

    For single-speaker data (no multi-turn conversation), each sample is a
    single turn: speaker says text with corresponding audio. We format this as
    a 1-turn "conversation" using the processor's apply_chat_template with
    output_labels=True.
    """

    def data_collator(samples: list[dict]) -> dict:
        conversations = []

        for sample in samples:
            # Single-speaker, single-turn: one conversation with one message
            # The message has both text (what was said) and audio (how it sounded)
            conversation = [
                {
                    "role": f"{SPEAKER_ID}",
                    "content": [
                        {"type": "text", "text": sample["text"]},
                        {"type": "audio", "audio": sample["audio"]},
                    ],
                }
            ]
            conversations.append(conversation)

        # apply_chat_template with output_labels=True creates:
        #   - input_ids: token sequence with <|AUDIO|> placeholders expanded
        #   - input_values: raw audio waveforms for the codec to encode
        #   - input_values_cutoffs: boundaries of audio segments
        #   - attention_mask: padding mask
        #   - labels: audio_token_id where audio frames are, -100 for text/padding
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            output_labels=True,
        )

        return inputs

    return data_collator


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------
def get_lora_config() -> LoraConfig:
    """
    LoRA config targeting the attention and MLP projections in both the
    backbone (16-layer LLaMA-1B) and depth decoder (4-layer LLaMA-100M).

    Target modules (matched by suffix across the full model):
      - q_proj, v_proj  -- attention query & value projections
      - gate_proj, down_proj -- MLP gating & down projections

    These match in both backbone_model.layers.* and depth_decoder.model.layers.*.
    The lm_head (codebook-0 prediction) is also fine-tuned via modules_to_save.

    We use rank=16 and alpha=32 as a solid starting point for voice cloning.
    """
    # PEFT matches target_modules as suffixes against all named modules.
    # Using short names like "q_proj" will match in BOTH the backbone and
    # depth decoder, which is exactly what we want. All attention + MLP
    # linear layers in the model end with these names:
    #   backbone_model.layers.{i}.self_attn.{q,k,v,o}_proj
    #   backbone_model.layers.{i}.mlp.{gate,up,down}_proj
    #   depth_decoder.model.layers.{i}.self_attn.{q,k,v,o}_proj
    #   depth_decoder.model.layers.{i}.mlp.{gate,up,down}_proj
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            "gate_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["lm_head"],  # also fine-tune the codebook-0 prediction head
    )


# ---------------------------------------------------------------------------
# Freeze the codec model
# ---------------------------------------------------------------------------
def freeze_codec(model: CsmForConditionalGeneration):
    """
    The codec model (Mimi) is used only to encode audio into codebook tokens
    during forward. It should NEVER be trained -- freeze all its parameters
    and set it to eval mode.
    """
    for param in model.codec_model.parameters():
        param.requires_grad = False
    model.codec_model.eval()
    print(f"Froze codec model ({sum(p.numel() for p in model.codec_model.parameters()):,} params)")


# ---------------------------------------------------------------------------
# Print trainable parameter summary
# ---------------------------------------------------------------------------
def print_trainable_summary(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# Build manifest from a folder of MP3s + training_lines.py
# ---------------------------------------------------------------------------
def build_manifest(data_dir: str):
    """
    If the generate_training_data.py script didn't create a manifest
    (e.g. because it was interrupted), rebuild it from the MP3 files
    on disk + the TRAINING_LINES list.
    """
    data_dir = Path(data_dir)

    # Import the training lines
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from training_lines import TRAINING_LINES

    mp3_files = sorted(data_dir.glob("line_*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files in {data_dir}")

    lines = []
    for mp3_path in mp3_files:
        # Extract index from filename: line_0042.mp3 -> 42
        idx = int(mp3_path.stem.split("_")[1])
        if idx < len(TRAINING_LINES):
            lines.append({
                "id": idx,
                "text": TRAINING_LINES[idx],
                "audio": mp3_path.name,
            })
        else:
            print(f"  WARNING: {mp3_path.name} has index {idx} but only {len(TRAINING_LINES)} training lines")

    manifest = {
        "voice_id": "elevenlabs",
        "total_lines": len(lines),
        "total_chars": sum(len(l["text"]) for l in lines),
        "lines": lines,
    }

    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest with {len(lines)} entries to {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    manifest_path = Path(args.data_dir) / "manifest.json"

    # Auto-build manifest if missing
    if not manifest_path.exists():
        print("No manifest.json found -- building from MP3 files + training_lines.py...")
        build_manifest(args.data_dir)

    # ---- Load model + processor ----
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = CsmForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # use Flash Attention if available
    )

    # ---- Freeze codec ----
    freeze_codec(model)

    # ---- Put model in training mode (but codec stays eval) ----
    model.train()
    model.codec_model.eval()  # re-enforce after .train()

    # ---- Apply LoRA ----
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    print_trainable_summary(model)

    # ---- Load datasets ----
    train_ds = CSMFineTuneDataset(manifest_path, split="train", val_ratio=args.val_ratio)
    val_ds = CSMFineTuneDataset(manifest_path, split="val", val_ratio=args.val_ratio)

    # ---- Training arguments ----
    # Compute steps
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # -- Batch size --
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # -- Schedule --
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # -- Precision --
        bf16=True,
        bf16_full_eval=True,

        # -- Memory --
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # -- Eval + saving --
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 2),  # eval twice per epoch
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch),  # save every epoch
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # -- Logging --
        logging_steps=10,
        logging_first_step=True,
        report_to="none",  # change to "wandb" if you want W&B

        # -- Misc --
        remove_unused_columns=False,  # IMPORTANT: we pass raw dicts
        dataloader_num_workers=0,     # audio loading is fast enough
        seed=42,

        # -- Resume --
        resume_from_checkpoint=args.resume,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_data_collator(processor),
    )

    # ---- Train ----
    print(f"\nStarting training:")
    print(f"  Samples: {len(train_ds)} train, {len(val_ds)} val")
    print(f"  Effective batch: {effective_batch} (bs={args.batch_size} x accum={args.grad_accum})")
    print(f"  Epochs: {args.epochs}, Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    print(f"  LR: {args.lr}, Warmup: {warmup_steps} steps")
    print(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    print()

    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    # ---- Save final adapter ----
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nSaved final LoRA adapter + processor to {final_path}")


# ---------------------------------------------------------------------------
# Inference with fine-tuned adapter
# ---------------------------------------------------------------------------
def inference(args):
    """Load base model + LoRA adapter and generate speech."""
    import soundfile as sf

    print(f"Loading base model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = CsmForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()  # merge LoRA weights for faster inference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Format as single-speaker prompt (text only, no audio context)
    text = f"[{SPEAKER_ID}]{args.text}"
    inputs = processor(text, add_special_tokens=True).to(device)

    print(f"Generating speech for: {args.text}")
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)

    # Save output
    output_path = Path(args.output_file)
    processor.save_audio(audio, str(output_path))
    print(f"Saved audio to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CSM-1B LoRA fine-tuning")
    subparsers = parser.add_subparsers(dest="command")

    # -- Build manifest --
    build_parser = subparsers.add_parser("build-manifest", help="Build manifest.json from MP3 files")
    build_parser.add_argument("--data-dir", default="data/training", help="Directory with MP3 files")

    # -- Train --
    train_parser = subparsers.add_parser("train", help="Fine-tune CSM-1B with LoRA")
    train_parser.add_argument("--model-id", default="sesame/csm-1b")
    train_parser.add_argument("--data-dir", default="data/training")
    train_parser.add_argument("--output-dir", default="checkpoints/csm-lora")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=1,
                              help="Per-device batch size. CSM is memory-heavy; start with 1.")
    train_parser.add_argument("--grad-accum", type=int, default=8,
                              help="Gradient accumulation steps (effective batch = bs * accum)")
    train_parser.add_argument("--lr", type=float, default=5e-5,
                              help="Learning rate. 1e-5 to 1e-4 works well for LoRA.")
    train_parser.add_argument("--val-ratio", type=float, default=0.05)
    train_parser.add_argument("--resume", nargs="?", const=True, default=False,
                              help="Resume from latest checkpoint (or specify path)")

    # -- Inference --
    infer_parser = subparsers.add_parser("inference", help="Generate speech with fine-tuned model")
    infer_parser.add_argument("--model-id", default="sesame/csm-1b")
    infer_parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    infer_parser.add_argument("--text", required=True, help="Text to synthesize")
    infer_parser.add_argument("--output-file", default="output.wav")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "build-manifest":
        build_manifest(args.data_dir)
    elif args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    else:
        print("Usage:")
        print("  python scripts/finetune_csm.py build-manifest --data-dir data/training")
        print("  python scripts/finetune_csm.py train --data-dir data/training")
        print("  python scripts/finetune_csm.py inference --adapter-path checkpoints/csm-lora/final --text 'Hello!'")
        sys.exit(1)


if __name__ == "__main__":
    main()
