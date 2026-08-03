"""
Phase 3 — QLoRA Fine-Tuning
==============================
Trains the base model on RACA Arabic legal data using LoRA / QLoRA.

Key decisions:
  - Uses PEFT (parameter-efficient fine-tuning) via LoRA adapters
  - Targets all linear projection layers for maximum expressiveness
  - Gradient checkpointing to reduce memory footprint
  - Saves full merged checkpoint at the end (required for Phase 4 SAE)

Install:
    pip install transformers peft bitsandbytes accelerate trl datasets flash-attn

Usage:
    python phase3_finetune.py --config ./data/project_config.json \
                              --data_dir ./data \
                              --output_dir ./checkpoints/ft_raca
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig


# ─────────────────────────────────────────────
# 1. LOAD CONFIG
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────

def load_sft_data(data_dir: Path) -> DatasetDict:
    """Load the SFT dataset produced by Phase 1."""
    dataset_path = data_dir / 'sft_dataset'
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Run phase1_data_prep.py first."
        )
    print(f"  Loading SFT dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return dataset


# ─────────────────────────────────────────────
# 3. BUILD TOKENIZER + MODEL
# ─────────────────────────────────────────────

def build_model_and_tokenizer(config: dict):
    """Load tokenizer and model with QLoRA config."""
    model_id = config['model_id']
    use_qlora = config['use_qlora']
    dtype = torch.bfloat16 if config['dtype'] == 'bfloat16' else torch.float16

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side='right', trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype if not use_qlora else None,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='eager',
    )

    # Prepare for k-bit training (adds cast hooks for QLoRA)
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model, tokenizer


# ─────────────────────────────────────────────
# 4. APPLY LORA
# ─────────────────────────────────────────────

def apply_lora(model, config: dict):
    """
    Wrap model with LoRA adapters.
    
    Target modules: all linear projections in the attention and MLP blocks.
    This gives ~full expressiveness while keeping adapter params small (~1% of total).
    
    Rank (r) selection guide:
      r=8   → minimal, fastest; good for narrow domain adaptation
      r=16  → standard; good balance for legal domain
      r=32  → more capacity; use if validation loss plateaus at r=16
      r=64  → diminishing returns; only if you have very large data (10k+ docs)
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],                      # rank
        lora_alpha=config['lora_alpha'],          # scaling = alpha / r
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias='none',
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# 5. TRAINING ARGUMENTS
# ─────────────────────────────────────────────

def build_training_args(config: dict, output_dir: str) -> SFTConfig:
    """
    Build SFTConfig (extends TrainingArguments with SFT-specific options).
    
    Effective batch size = batch_size × gradient_accumulation_steps × num_GPUs
    With batch=2, accum=8, 1 GPU → effective batch = 16
    
    Learning rate:
      2e-4 is standard for LoRA. If loss is unstable, try 1e-4.
      Cosine schedule decays to 10% of peak by the end.
    
    Epochs:
      1-3 epochs typical for domain adaptation on a small corpus.
      Monitor val loss — stop before it starts rising (overfitting).
    """
    return SFTConfig(
        output_dir=output_dir,

        # Training loop
        num_train_epochs=config.get('num_train_epochs_sft', 3),  # overridden per stage
        per_device_train_batch_size=config['training_batch_size'],
        per_device_eval_batch_size=config['training_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        gradient_checkpointing=True,

        # Optimizer
        optim='paged_adamw_32bit',      # memory-efficient AdamW for QLoRA
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Precision
        bf16=config['dtype'] == 'bfloat16',
        fp16=False,

        # Sequence length
        max_length=config['max_seq_length'],
        dataset_text_field='text',       # column in the dataset containing the full ChatML string
        packing=False,                   # set True to pack multiple short samples into one sequence

        # Logging & saving
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,              # keep only 3 checkpoints to save disk space
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # Misc
        report_to='tensorboard',        # change to 'wandb' if you use Weights & Biases
        dataloader_num_workers=0,
        seed=42,
    )


# ─────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────

def load_cpt_data(data_dir: Path):
    """Load CPT dataset if available (used for small-corpus two-stage training)."""
    cpt_path = data_dir / 'cpt_dataset'
    if cpt_path.exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(cpt_path))
        print(f"  CPT dataset: {len(ds['train'])} train docs")
        return ds
    return None


def train(config: dict, data_dir: Path, output_dir: Path):
    print("\n[1/5] Loading data...")
    sft_dataset = load_sft_data(data_dir)
    cpt_dataset = load_cpt_data(data_dir)

    print("\n[2/5] Building model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(config)

    print("\n[3/5] Applying LoRA adapters...")
    model = apply_lora(model, config)

    # ── Stage A: CPT (1 epoch on raw documents) ──────────────────────────
    # Only runs if a CPT dataset exists (produced by Phase 1 for small corpora).
    # Teaches the model Arabic legal vocabulary before it sees Q&A pairs.
    if cpt_dataset is not None:
        n_epochs_cpt = config.get('num_train_epochs_cpt', 1)
        print(f"\n[4/5] Stage A — CPT pre-training ({n_epochs_cpt} epoch)...")
        cpt_args = build_training_args(config, str(output_dir / 'cpt_run'))
        # Override epochs for CPT stage
        cpt_args.num_train_epochs = n_epochs_cpt
        cpt_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=cpt_args,
            train_dataset=cpt_dataset['train'],
            eval_dataset=cpt_dataset['validation'],
        )
        print(f"  CPT docs: {len(cpt_dataset['train'])} train")
        cpt_trainer.train()
        print("  CPT stage complete.")
    else:
        print("\n[4/5] No CPT dataset found — skipping CPT stage.")

    # ── Stage B: SFT (instruction fine-tuning) ───────────────────────────
    n_epochs_sft = config.get('num_train_epochs_sft', 3)
    print(f"\n[5/5] Stage B — SFT instruction fine-tuning ({n_epochs_sft} epochs)...")
    sft_args = build_training_args(config, str(output_dir / 'sft_run'))
    sft_args.num_train_epochs = n_epochs_sft

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['validation'],
    )

    print(f"  Effective batch size: "
          f"{config['training_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  SFT examples: {len(sft_dataset['train'])} train")
    print()

    trainer.train()
    trainer.save_model(str(output_dir / 'adapter'))
    tokenizer.save_pretrained(str(output_dir / 'adapter'))
    print(f"\n  LoRA adapter saved → {output_dir / 'adapter'}")


# ─────────────────────────────────────────────
# 7. MERGE AND SAVE FULL MODEL
# ─────────────────────────────────────────────

def merge_and_save(config: dict, adapter_path: Path, merged_path: Path):
    """
    Merge LoRA adapter weights back into the base model and save the full model.
    
    WHY THIS MATTERS FOR PHASE 4:
    The SAE needs to hook into activation tensors of a single unified model.
    LoRA adapters that sit on top of a quantized base do not produce the same
    activation geometry as the merged model. Always merge before Phase 4.
    
    Note: merge requires loading the base model in fp16/bf16 (not 4-bit).
    This temporarily needs ~16GB VRAM or can be done on CPU (slow but works).
    """
    from peft import PeftModel

    print("\n  Loading base model in fp16 for merge (ignoring quantization)...")
    dtype = torch.bfloat16 if config['dtype'] == 'bfloat16' else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        torch_dtype=dtype,
        device_map='cpu',       # load on CPU to avoid VRAM issues during merge
        trust_remote_code=True,
    )

    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("  Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model → {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"  ✓ Merged model ready at {merged_path}")


# ─────────────────────────────────────────────
# 8. QUICK INFERENCE TEST
# ─────────────────────────────────────────────

def test_inference(model_path: str, tokenizer_path: str):
    """Run a quick Arabic legal question to verify the fine-tuned model works."""
    print("\n  Running inference test...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map='auto'
    )

    prompt = """<|im_start|>system
أنت مساعد قانوني متخصص في تشريعات هيئة تنظيم الأعمال الخيرية في دولة قطر.<|im_end|>
<|im_start|>user
ما هي اختصاصات قسم الامتثال في هيئة تنظيم الأعمال الخيرية؟<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n  Prompt: ما هي اختصاصات قسم الامتثال؟")
    print(f"  Response:\n{response}\n")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: QLoRA fine-tuning")
    parser.add_argument('--config', default='./data/project_config.json')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default='./checkpoints/ft_raca')
    parser.add_argument('--skip_merge', action='store_true',
                        help='Skip merging (useful for quick iteration; merge before Phase 4)')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training; test inference on existing merged model')
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Phase 3: QLoRA Fine-Tuning ===\n")
    print(f"  Model:  {config['model_id']}")
    print(f"  QLoRA:  {config['use_qlora']}")
    print(f"  LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")

    if not args.test_only:
        train(config, data_dir, output_dir)

        if not args.skip_merge:
            print("\n  Merging LoRA adapter into base model...")
            merge_and_save(
                config,
                adapter_path=output_dir / 'adapter',
                merged_path=output_dir / 'merged',
            )

    # Test
    merged_path = output_dir / 'merged'
    if merged_path.exists():
        test_inference(str(merged_path), str(merged_path))

    print("\n✓ Phase 3 complete.")
    print(f"  Fine-tuned model: {output_dir / 'merged'}")
    print("  Next step: Phase 4 — SAE training on the merged checkpoint.\n")


if __name__ == '__main__':
    main()
