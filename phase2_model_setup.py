"""
Phase 2 — Model Selection & Setup
===================================
Handles:
  - Selecting and downloading the base model
  - Tokenizer configuration for Arabic legal text
  - Environment and GPU memory check
  - Verifying the model can handle Arabic input correctly

Recommended model: meta-llama/Meta-Llama-3.1-8B-Instruct
  - Strong multilingual/Arabic capability
  - 128k context window
  - SAE support via SAELens (EleutherAI)
  - 4-bit QLoRA fits in ~12GB VRAM

Alternatives:
  - mistralai/Mistral-7B-Instruct-v0.3  (good Arabic, smaller SAE ecosystem)
  - inceptionai/jais-13b-chat           (Arabic-native, but larger, limited SAE support)
  - aubmindlab/aragpt2-mega              (Arabic GPT-2, weak for instruction following)

Usage:
    python phase2_model_setup.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ─────────────────────────────────────────────
# 1. GPU ENVIRONMENT CHECK
# ─────────────────────────────────────────────

def check_environment() -> dict:
    """
    Check GPU availability, VRAM, and compute capability.
    Prints a recommendation for quantization level.
    """
    info = {}
    info['cuda_available'] = torch.cuda.is_available()
    
    if not info['cuda_available']:
        print("⚠  No CUDA GPU detected. Fine-tuning will be extremely slow on CPU.")
        print("   Use a cloud instance: A100 40GB (best), A10G 24GB, or T4 16GB (QLoRA only)")
        return info
    
    info['gpu_count'] = torch.cuda.device_count()
    info['gpus'] = []
    
    total_vram_gb = 0
    for i in range(info['gpu_count']):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        total_vram_gb += vram_gb
        gpu_info = {
            'index': i,
            'name': props.name,
            'vram_gb': round(vram_gb, 1),
            'compute_capability': f"{props.major}.{props.minor}",
        }
        info['gpus'].append(gpu_info)
        print(f"  GPU {i}: {props.name} — {vram_gb:.1f} GB VRAM")
    
    info['total_vram_gb'] = round(total_vram_gb, 1)
    
    # Recommend quantization strategy
    if total_vram_gb >= 80:
        rec = "Full fine-tune (bf16) — no quantization needed"
        info['recommended_dtype'] = 'bfloat16'
        info['use_qlora'] = False
    elif total_vram_gb >= 40:
        rec = "Full fine-tune with bf16, or LoRA for faster iteration"
        info['recommended_dtype'] = 'bfloat16'
        info['use_qlora'] = False
    elif total_vram_gb >= 24:
        rec = "LoRA with bf16, or QLoRA (4-bit) for 13B+ models"
        info['recommended_dtype'] = 'bfloat16'
        info['use_qlora'] = False  # only needed for larger models
    elif total_vram_gb >= 12:
        rec = "QLoRA (4-bit quantization) — required for 7-8B models"
        info['recommended_dtype'] = 'float16'
        info['use_qlora'] = True
    else:
        rec = "⚠ Less than 12GB VRAM — consider a smaller model or multi-GPU setup"
        info['recommended_dtype'] = 'float16'
        info['use_qlora'] = True
    
    print(f"\n  Recommendation: {rec}")
    return info


# ─────────────────────────────────────────────
# 2. TOKENIZER SETUP
# ─────────────────────────────────────────────

def setup_tokenizer(model_id: str, cache_dir: Optional[str] = None):
    """
    Load and configure the tokenizer for Arabic legal fine-tuning.
    
    Key settings:
    - padding_side='right': Required for causal LM training
    - pad_token: Many models lack a pad token; we add one
    - trust_remote_code: Needed for some models (e.g. Jais)
    """
    from typing import Optional, Optional  # local import to avoid top-level issue
    
    print(f"\n  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        padding_side='right',   # right-pad for causal LM
    )
    
    # Add pad token if missing (common in Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Added pad_token = eos_token")
    
    # Verify Arabic encoding
    test_text = "هيئة تنظيم الأعمال الخيرية"
    tokens = tokenizer(test_text, return_tensors='pt')
    decoded = tokenizer.decode(tokens['input_ids'][0])
    print(f"  Arabic encoding test:")
    print(f"    Input:   {test_text}")
    print(f"    Tokens:  {tokens['input_ids'][0].tolist()}")
    print(f"    Decoded: {decoded}")
    
    # Warn if Arabic text got fragmented into many tokens (sign of weak tokenizer)
    arabic_words = len(test_text.split())
    token_count = tokens['input_ids'].shape[1]
    tokens_per_word = token_count / arabic_words
    if tokens_per_word > 5:
        print(f"  ⚠ High tokens/word ratio ({tokens_per_word:.1f}x) — "
              "model may have poor Arabic tokenization. Consider Jais or AraT5.")
    else:
        print(f"  ✓ Tokens per Arabic word: {tokens_per_word:.1f}x (acceptable)")
    
    return tokenizer


# ─────────────────────────────────────────────
# 3. MODEL LOADING (with optional quantization)
# ─────────────────────────────────────────────

def load_base_model(
    model_id: str,
    use_qlora: bool = True,
    dtype: str = 'float16',
    cache_dir: Optional[str] = None,
):
    """
    Load the base model with the appropriate quantization config.
    
    For QLoRA (use_qlora=True):
      - 4-bit NF4 quantization via bitsandbytes
      - Double quantization for memory efficiency
      - Loads ~4-5GB for a 7-8B model
    
    For full precision (use_qlora=False):
      - bf16 or fp16
      - Loads ~14-16GB for a 7-8B model
    """
    from typing import Optional, Optional
    
    torch_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    
    if use_qlora:
        print("\n  Loading model with 4-bit QLoRA quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',          # NF4 is best for LLMs
            bnb_4bit_compute_dtype=torch_dtype,  # compute in fp16/bf16
            bnb_4bit_use_double_quant=True,      # saves ~0.4 bits/param extra
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='auto',                  # automatically splits across GPUs
            cache_dir=cache_dir,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',  # requires flash-attn package
        )
    else:
        print(f"\n  Loading model in {dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map='auto',
            cache_dir=cache_dir,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
        )
    
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model loaded:")
    print(f"    Total parameters:    {total_params / 1e9:.2f}B")
    print(f"    Trainable (pre-LoRA): {trainable_params / 1e9:.2f}B")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"    GPU memory used:     {mem:.1f} GB")
    
    return model


# ─────────────────────────────────────────────
# 4. SAE COMPATIBILITY CHECK
# ─────────────────────────────────────────────

# Models with known SAE support (as of 2025)
SAE_SUPPORT_MAP = {
    'meta-llama/Meta-Llama-3.1-8B':           'SAELens (EleutherAI) — full layer coverage',
    'meta-llama/Meta-Llama-3.1-8B-Instruct':  'SAELens (EleutherAI) — use base model SAE',
    'meta-llama/Meta-Llama-3-8B':             'SAELens (EleutherAI) — multiple releases',
    'mistralai/Mistral-7B-v0.1':              'Partial — community SAEs via Neuronpedia',
    'EleutherAI/pythia-70m':                  'SAELens reference implementation',
    'EleutherAI/pythia-160m':                 'SAELens reference implementation',
    'google/gemma-2-9b':                      'Goodfire SDK — limited',
    'inceptionai/jais-13b-chat':              '⚠ No public SAE — must train from scratch',
    'aubmindlab/aragpt2-mega':                '⚠ No public SAE — must train from scratch',
}

def check_sae_support(model_id: str):
    """Print known SAE availability for the chosen model."""
    # Check exact match first, then prefix match
    support = SAE_SUPPORT_MAP.get(model_id)
    if not support:
        for key in SAE_SUPPORT_MAP:
            if model_id.startswith(key) or key.startswith(model_id):
                support = SAE_SUPPORT_MAP[key]
                break
    
    if support:
        print(f"\n  SAE support for {model_id}:")
        print(f"    {support}")
        if '⚠' in support:
            print("    → You will need to train an SAE from scratch in Phase 4.")
            print("      This adds ~2-4 hours of compute on top of fine-tuning.")
        else:
            print("    → You can use the pre-trained SAE as a starting point in Phase 4.")
            print("      Re-training on your fine-tuned checkpoint is still recommended.")
    else:
        print(f"\n  ⚠ No known SAE for {model_id}.")
        print("    → You will train an SAE from scratch in Phase 4.")
        print("      SAELens supports any HuggingFace causal LM — this is fine.")


# ─────────────────────────────────────────────
# 5. SAVE CONFIG FOR DOWNSTREAM PHASES
# ─────────────────────────────────────────────

def save_model_config(env_info: dict, model_id: str, output_dir: Path):
    """
    Save a project config JSON that downstream phases (3, 4, 5) will read.
    This avoids repeating model selection decisions in every script.
    """
    config = {
        'model_id': model_id,
        'use_qlora': env_info.get('use_qlora', True),
        'dtype': env_info.get('recommended_dtype', 'float16'),
        'total_vram_gb': env_info.get('total_vram_gb', 0),
        'gpu_count': env_info.get('gpu_count', 0),
        # Phase 3 defaults (can be overridden)
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                 'gate_proj', 'up_proj', 'down_proj'],
        'max_seq_length': 2048,
        'training_batch_size': 2,
        'gradient_accumulation_steps': 8,
        # Phase 4 defaults
        'sae_hook_layer': 16,          # Mid-model layer — good for semantic features
        'sae_expansion_factor': 8,     # Feature dict size = hidden_dim × expansion_factor
        'sae_l1_coefficient': 0.0002,  # Sparsity penalty
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'project_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Saved project config → {config_path}")
    return config


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Model selection and setup")
    parser.add_argument('--model_id', default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='HuggingFace model ID')
    parser.add_argument('--cache_dir', default=None,
                        help='Directory to cache model weights')
    parser.add_argument('--output_dir', default='./data',
                        help='Directory to write project config')
    parser.add_argument('--skip_model_load', action='store_true',
                        help='Skip loading model weights (just check env and config)')
    args = parser.parse_args()

    from typing import Optional, Optional  # ensure available in main scope
    
    print("\n=== Phase 2: Model Selection & Setup ===\n")
    
    # Check environment
    print("[1/4] Checking GPU environment...")
    env_info = check_environment()
    
    # SAE compatibility
    print("\n[2/4] Checking SAE compatibility...")
    check_sae_support(args.model_id)
    
    # Save config
    print("\n[3/4] Saving project config...")
    output_dir = Path(args.output_dir)
    config = save_model_config(env_info, args.model_id, output_dir)
    
    # Load tokenizer and model
    if not args.skip_model_load:
        print("\n[4/4] Loading tokenizer and model...")
        tokenizer = setup_tokenizer(args.model_id, cache_dir=args.cache_dir)
        model = load_base_model(
            args.model_id,
            use_qlora=config['use_qlora'],
            dtype=config['dtype'],
            cache_dir=args.cache_dir,
        )
        print("\n✓ Phase 2 complete. Model ready for fine-tuning.")
        print("  Next step: Phase 3 — QLoRA fine-tuning.\n")
    else:
        print("\n[4/4] Skipped model load (--skip_model_load).")
        print("✓ Phase 2 config written. Run without --skip_model_load to verify model.\n")


if __name__ == '__main__':
    main()
