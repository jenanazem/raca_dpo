#!/usr/bin/env python3
"""
DPO Fine-tuning for RACA Legal LLM
=====================================
Trains the v3 model using Direct Preference Optimization (DPO)
on the generated (chosen, rejected) pairs.

Usage:
    python train_dpo.py
    python train_dpo.py --output_dir ./checkpoints/ft_raca_v4
"""

import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

MODEL_PATH  = "./checkpoints/ft_raca_v5/merged"
DPO_DATA    = "./data/processed/llama_matched_surface_dataset.jsonl"

def load_dpo_data(path: str) -> Dataset:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            pairs.append({
                "prompt":   ex["prompt"],
                "chosen":   ex["chosen"],
                "rejected": ex["rejected"],
            })
    print(f"Loaded {len(pairs)} DPO pairs")
    return Dataset.from_list(pairs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./checkpoints/ft_llama_matched_surface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_reg", action="store_true", help="Enable length regularization (SimPO)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Apply LoRA for efficient DPO training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dpo_data(DPO_DATA)

    # Split train/val
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    dpo_config = DPOConfig(
        output_dir=str(output_dir / "dpo_run"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        fp16=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        dataloader_num_workers=0,
        beta=0.1,  # DPO temperature — controls how much to deviate from reference
        loss_type="sigmoid_norm" if args.length_reg else "sigmoid",
        max_length=512,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting DPO training...")
    trainer.train()

    print("\nSaving LoRA adapter...")
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved → {adapter_path}")

    # Merge adapter into base model
    print("\nMerging LoRA into base model...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
    )
    merged = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = merged.merge_and_unload()
    merged_path = output_dir / "merged"
    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"Merged model saved → {merged_path}")
    print("\n✓ DPO training complete.")

if __name__ == "__main__":
    main()
