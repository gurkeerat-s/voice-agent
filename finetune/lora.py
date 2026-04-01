"""
LoRA fine-tune Orpheus TTS on a pre-tokenized dataset.

Based on the official Orpheus finetune/lora.py with minor adjustments
for local dataset loading.

Usage:
    accelerate launch finetune/lora.py
    # or just:
    python finetune/lora.py
"""

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
)

# Apply LoRA (official Orpheus config)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    modules_to_save=["lm_head", "embed_tokens"],  # critical for audio token generation
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{save_folder}",
    report_to=report_to,
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

trainer.train()

# Merge LoRA into base model and save
print("Merging LoRA adapter into base model...")
merged_model = model.merge_and_unload()

output_dir = f"./{save_folder}/merged"
os.makedirs(output_dir, exist_ok=True)
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Merged model saved to {output_dir}")
