"""
Full fine-tune Orpheus TTS on a pre-tokenized dataset.

Based on the official Orpheus finetune/train.py. Requires ~80GB VRAM.
For LoRA (lower VRAM), use lora.py instead.

Usage:
    accelerate launch finetune/train.py
"""

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import os

config_file = "finetune/config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]
model_name = config["model_name"]
epochs = config["epochs"]
batch_size = config["batch_size"]
pad_token = config["pad_token"]
save_steps = config["save_steps"]
learning_rate = config["learning_rate"]
save_folder = config["save_folder"]
project_name = config["project_name"]
run_name = config["run_name"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
)

# Load dataset — local path or HF Hub
if os.path.isdir(dsn):
    ds = load_from_disk(dsn)
else:
    ds = load_dataset(dsn, split="train")

# Optional: WandB logging
try:
    import wandb
    wandb.init(project=project_name, name=run_name)
    report_to = "wandb"
except ImportError:
    report_to = "none"

training_args = TrainingArguments(
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{save_folder}",
    report_to=report_to,
    save_steps=save_steps,
    save_total_limit=1,
    remove_unused_columns=True,
    learning_rate=learning_rate,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

# Save
output_dir = f"./{save_folder}/final"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
