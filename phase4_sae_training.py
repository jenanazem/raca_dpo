"""
Phase 4 — Sparse Autoencoder (SAE) Training
=============================================
Trains a Sparse Autoencoder on the internal activations of the fine-tuned LLM.

What this does:
  1. Hooks into a specific layer of the fine-tuned model (residual stream)
  2. Collects activation vectors as the model processes the legal corpus
  3. Trains an SAE to decompose those activations into sparse, interpretable features
  4. Identifies features that correspond to domain concepts (law articles, authority names)
     and — critically — features that activate during hallucination

SAE architecture:
  - Encoder: Linear(hidden_dim → feature_dim) + ReLU
  - Decoder: Linear(feature_dim → hidden_dim) — columns kept unit-norm
  - Loss: MSE reconstruction + L1 sparsity penalty
  - feature_dim = hidden_dim × expansion_factor (typically 8×–16×)

Install:
    pip install sae-lens transformer-lens

Usage:
    python phase4_sae_training.py --config ./data/project_config.json \
                                  --model_path ./checkpoints/ft_raca/merged \
                                  --data_dir ./data \
                                  --output_dir ./checkpoints/sae_raca
"""

import json

def load_config(path):
    with open(path) as f:
        return json.load(f)
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. ACTIVATION COLLECTION
# ─────────────────────────────────────────────

class ActivationCollector:
    """
    Hooks into a transformer layer's residual stream and collects activations.
    
    We hook AFTER the layer norm at the start of the transformer block
    (the 'residual stream pre-MLP' position). This is the standard hook point
    used by SAELens and interpretability research.
    
    For Llama 3, hook at model.layers[layer_idx] — the full block output.
    The residual stream at this point contains the accumulated representations
    of all previous layers, which is where semantic features tend to live.
    """
    
    def __init__(self, model, layer_idx: int):
        self.activations = []
        self.hook = None
        self._register_hook(model, layer_idx)
    
    def _register_hook(self, model, layer_idx: int):
        """Register a forward hook on the target layer."""
        # Access transformer blocks (works for Llama, Mistral, Gemma)
        layers = model.model.layers
        if layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
        
        target_layer = layers[layer_idx]
        
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state tensor
            hidden = output[0]               # shape: [batch, seq_len, hidden_dim]
            self.activations.append(hidden.detach().float().cpu())  # store full tensor
        
        self.hook = target_layer.register_forward_hook(hook_fn)
        print(f"  Hooked layer {layer_idx}: {target_layer.__class__.__name__}")
    
    def get_activations(self) -> torch.Tensor:
        """Stack and return all collected activations as [N, hidden_dim]."""
        if not self.activations:
            return torch.empty(0)
        all_acts = torch.cat([a.view(-1, a.shape[-1]) for a in self.activations], dim=0)  # [batch*seq, hidden_dim]
        return all_acts
    
    def clear(self):
        self.activations = []
    
    def remove(self):
        if self.hook:
            self.hook.remove()


def collect_activations(
    model_path: str,
    data_dir: Path,
    layer_idx: int,
    max_samples: int = 5000,
    batch_size: int = 4,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, int]:
    """
    Run the fine-tuned model over the CPT corpus and collect layer activations.
    
    Returns:
        activations: Tensor of shape [N_tokens, hidden_dim]
        hidden_dim: The model's hidden dimension (needed for SAE construction)
    
    Note:
        max_samples controls how many documents to process.
        For a 7B model at layer 16, expect ~768 tokens/doc × 5000 docs = 3.8M vectors.
        Each fp32 vector at 4096-dim = 16KB → ~60GB total. Use fp16 to halve this.
        In practice, 500k–2M tokens is sufficient for SAE training.
    """
    print(f"\n  Loading fine-tuned model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.eval()
    
    hidden_dim = model.config.hidden_size
    print(f"  Hidden dimension: {hidden_dim}")
    
    # Register activation hook
    collector = ActivationCollector(model, layer_idx)
    
    # Load CPT dataset (plain documents, no Q&A format needed for activation collection)
    cpt_path = data_dir / 'cpt_dataset'
    if cpt_path.exists():
        dataset = load_from_disk(str(cpt_path))['train']
    else:
        # Fall back to SFT dataset
        dataset = load_from_disk(str(data_dir / 'sft_dataset'))['train']
    
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  Collecting activations from {len(dataset)} documents...")
    
    all_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_texts = dataset[i:i+batch_size]['text']
            
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_seq_len,
                padding='max_length',
            ).to(model.device)
            
            _ = model(**inputs)
            
            # Get activations, mask padding tokens
            acts = collector.get_activations()  # [batch*seq, hidden]
            attention_mask = inputs['attention_mask'].bool().cpu().reshape(-1)[:acts.shape[0]]  # [batch*seq]
            print(f"DEBUG acts shape: {acts.shape}, mask shape: {attention_mask.shape}")
            acts = acts[attention_mask]  # remove padding positions
            
            all_activations.append(acts.half())  # store in fp16 to save memory
            collector.clear()
    
    collector.remove()
    
    all_activations = torch.cat(all_activations, dim=0)
    print(f"  Total activation vectors collected: {all_activations.shape[0]:,}")
    print(f"  Shape: {all_activations.shape}")
    
    return all_activations, hidden_dim


# ─────────────────────────────────────────────
# 2. SPARSE AUTOENCODER MODEL
# ─────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) — learns interpretable features from LLM activations.
    
    Architecture (following Anthropic and EleutherAI convention):
        encode: x → pre_acts = W_enc(x - b_dec) + b_enc
                  → acts = ReLU(pre_acts)          ← feature activations
        decode: x_hat = W_dec @ acts + b_dec
    
    Key design choices:
    - Decoder columns are kept unit-norm during training (normalization step)
      This prevents the trivial solution where one huge decoder column absorbs everything.
    - The encoder takes (x - b_dec) as input, which centers the activations
      around the decoder bias. This is the 'pre-encoder bias' trick.
    - L1 penalty on activations enforces sparsity — each input should activate
      only a small fraction of features.
    
    The resulting feature dictionary has features that fire for specific concepts:
    e.g., "feature 427 fires when the text mentions a board of directors vote"
    """
    
    def __init__(self, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(hidden_dim, feature_dim))
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))
        
        # Decoder (note: transposed W_enc is not used — decoder is independent)
        self.W_dec = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        
        # Initialize
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder columns immediately
        self._normalize_decoder()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, hidden_dim] → feature_acts: [batch, feature_dim]"""
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """feature_acts: [batch, feature_dim] → x_hat: [batch, hidden_dim]"""
        return feature_acts @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_acts = self.encode(x)
        x_hat = self.decode(feature_acts)
        return x_hat, feature_acts
    
    @torch.no_grad()
    def _normalize_decoder(self):
        """Keep decoder weight columns unit-norm."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms
    
    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        feature_acts: torch.Tensor,
        l1_coefficient: float,
    ) -> tuple[torch.Tensor, dict]:
        """
        Combined reconstruction + sparsity loss.
        
        mse_loss: How well the SAE reconstructs the original activations.
                  Lower is better. Target: similar scale to input variance.
        
        l1_loss:  Mean absolute activation per sample. Enforces sparsity.
                  The l1_coefficient controls the trade-off.
                  Too high → all features dead (never activate)
                  Too low  → dense activations, features aren't interpretable
        
        l0_approx: Average number of active features per token (diagnostic only).
                   Target: 10–100 active features out of feature_dim.
        """
        mse_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        l1_loss = feature_acts.abs().sum(dim=-1).mean()
        total_loss = mse_loss + l1_coefficient * l1_loss
        
        # Diagnostics
        l0_approx = (feature_acts > 0).float().sum(dim=-1).mean()
        dead_features = (feature_acts.max(dim=0).values == 0).float().mean()
        
        return total_loss, {
            'loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'l0_approx': l0_approx.item(),
            'dead_features_frac': dead_features.item(),
        }


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train_sae(
    activations: torch.Tensor,
    hidden_dim: int,
    config: dict,
    output_dir: Path,
) -> SparseAutoencoder:
    """
    Train the SAE on collected activations.
    
    Training details:
    - AdamW optimizer with learning rate warmup
    - Decoder normalization after every gradient step
    - Ghost gradients to revive dead features (optional but recommended)
    - Checkpoint every 1000 steps
    """
    feature_dim = hidden_dim * config['sae_expansion_factor']
    l1_coeff = config['sae_l1_coefficient']
    
    print(f"\n  SAE architecture: {hidden_dim}d → {feature_dim}d features")
    print(f"  Training on {activations.shape[0]:,} activation vectors")
    print(f"  L1 coefficient: {l1_coeff}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sae = SparseAutoencoder(hidden_dim, feature_dim).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=4e-4, weight_decay=0.0)
    
    # Learning rate schedule: warmup for first 5% of steps
    dataset = TensorDataset(activations.float())
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True)
    
    n_steps = len(loader) * 10  # 10 epochs
    warmup_steps = n_steps // 20
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (n_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item()))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'training_log.jsonl'
    
    step = 0
    best_loss = float('inf')
    
    print("\n  Starting SAE training...")
    for epoch in range(10):
        epoch_losses = []
        
        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}/10"):
            batch = batch.to(device)
            
            x_hat, feature_acts = sae(batch)
            loss, metrics = sae.loss(batch, x_hat, feature_acts, l1_coeff)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Renormalize decoder after every step
            sae._normalize_decoder()
            
            epoch_losses.append(metrics['loss'])
            step += 1
            
            # Log
            if step % 100 == 0:
                metrics['step'] = step
                metrics['epoch'] = epoch + 1
                metrics['lr'] = scheduler.get_last_lr()[0]
                with open(log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
            
            # Save checkpoint
            if step % 1000 == 0:
                ckpt_path = output_dir / f'sae_step_{step}.pt'
                torch.save(sae.state_dict(), ckpt_path)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, "
              f"l0≈{metrics['l0_approx']:.1f}, "
              f"dead_features={metrics['dead_features_frac']:.2%}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(sae.state_dict(), output_dir / 'sae_best.pt')
    
    print(f"\n  SAE training complete. Best loss: {best_loss:.4f}")
    print(f"  Saved to {output_dir / 'sae_best.pt'}")
    return sae


# ─────────────────────────────────────────────
# 4. FEATURE INTERPRETATION
# ─────────────────────────────────────────────

def interpret_features(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    texts: list[str],
    top_k_features: int = 20,
    output_dir: Path = None,
):
    """
    Find the most activating text examples for each feature.
    
    This is the core interpretability step:
    - For each SAE feature, find which texts caused it to fire most strongly
    - Human reviewers label these features: "this feature = law article citations"
    - Features linked to hallucination are identified and tagged for Phase 5
    
    Output: JSON file mapping feature_id → {max_activation, top_examples}
    """
    print("\n  Interpreting top features...")
    device = next(sae.parameters()).device
    
    with torch.no_grad():
        _, all_feature_acts = sae(activations[:10000].float().to(device))  # sample 10k
    
    # For each feature, find the top activating examples
    feature_max = all_feature_acts.max(dim=0).values.cpu()
    top_feature_ids = feature_max.argsort(descending=True)[:top_k_features]
    
    feature_info = {}
    for fid in top_feature_ids.tolist():
        acts_for_feature = all_feature_acts[:, fid].cpu()
        top_indices = acts_for_feature.argsort(descending=True)[:5]
        
        # Map token indices back to document indices (approximate)
        doc_indices = [min(idx.item() // 256, len(texts) - 1) for idx in top_indices]
        
        feature_info[fid] = {
            'feature_id': fid,
            'max_activation': feature_max[fid].item(),
            'mean_activation': acts_for_feature.mean().item(),
            'activation_frequency': (acts_for_feature > 0.1).float().mean().item(),
            'top_activating_docs': [
                {
                    'doc_index': doc_idx,
                    'text_snippet': texts[doc_idx][:200] if doc_idx < len(texts) else '',
                    'activation': acts_for_feature[top_indices[i]].item(),
                }
                for i, doc_idx in enumerate(doc_indices)
            ],
            'label': None,  # To be filled in manually by human reviewers
            'is_hallucination_feature': None,  # To be determined in Phase 5
        }
    
    if output_dir:
        interp_path = output_dir / 'feature_interpretations.json'
        with open(interp_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        print(f"  Saved feature interpretations → {interp_path}")
        print("  Next step: manually review this file and label:")
        print("    - 'label': describe what concept this feature encodes")
        print("    - 'is_hallucination_feature': true/false/null")
    
    return feature_info


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: SAE training")
    parser.add_argument('--config', default='./data/project_config.json')
    parser.add_argument('--model_path', default='./checkpoints/ft_raca/merged')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', default='./checkpoints/sae_raca')
    parser.add_argument('--skip_collection', action='store_true',
                        help='Skip activation collection (use cached activations)')
    parser.add_argument('--max_samples', type=int, default=5000)
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_idx = config['sae_hook_layer']
    acts_cache = output_dir / 'activations.pt'
    
    print("\n=== Phase 4: SAE Training ===\n")
    print(f"  Fine-tuned model: {args.model_path}")
    print(f"  Hook layer: {layer_idx}")
    print(f"  Expansion factor: {config['sae_expansion_factor']}×")
    
    # Collect activations
    if not args.skip_collection or not acts_cache.exists():
        print("\n[1/3] Collecting activations...")
        activations, hidden_dim = collect_activations(
            args.model_path, data_dir, layer_idx,
            max_samples=args.max_samples,
        )
        torch.save({'activations': activations, 'hidden_dim': hidden_dim}, acts_cache)
        print(f"  Cached activations → {acts_cache}")
    else:
        print(f"\n[1/3] Loading cached activations from {acts_cache}...")
        cache = torch.load(acts_cache)
        activations = cache['activations']
        hidden_dim = cache['hidden_dim']
    
    # Train SAE
    print("\n[2/3] Training SAE...")
    sae = train_sae(activations, hidden_dim, config, output_dir)
    
    # Interpret features
    print("\n[3/3] Identifying top features...")
    dataset = load_from_disk(str(data_dir / 'cpt_dataset'))['train']
    texts = [r['text'] for r in dataset][:args.max_samples]
    interpret_features(sae, activations, texts, top_k_features=50, output_dir=output_dir)
    
    print("\n✓ Phase 4 complete.")
    print(f"  SAE weights: {output_dir / 'sae_best.pt'}")
    print(f"  Feature file: {output_dir / 'feature_interpretations.json'}")
    print("\n  ACTION REQUIRED before Phase 5:")
    print("  → Open feature_interpretations.json")
    print("  → For each feature, review the top_activating_docs and fill in:")
    print("     'label': what concept does this feature encode?")
    print("     'is_hallucination_feature': does this fire on hallucinated outputs?")
    print("\n  Next step: Phase 5 — HAL reduction using labeled features.\n")


if __name__ == '__main__':
    main()
