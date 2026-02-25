"""
config.py — Model and Training Hyperparameters
================================================
Dataclass-based configs for two model sizes (~50M, ~120M params).
All hyperparameters are set to research-validated defaults.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    """Transformer model architecture configuration."""
    # ----- Vocabulary & Sequence -----
    vocab_size: int = 32_000          # BPE vocabulary size (power-of-2 friendly)
    max_seq_len: int = 512            # Maximum context length

    # ----- Transformer Dimensions -----
    d_model: int = 768                # Hidden / embedding dimension
    n_heads: int = 12                 # Number of attention heads
    n_layers: int = 12                # Number of transformer blocks
    d_ff: int = 3072                  # Feed-forward inner dimension (4 × d_model)

    # ----- Positional Encoding -----
    pos_encoding: str = "rope"        # "learned" | "sinusoidal" | "rope"
    rope_theta: float = 10_000.0      # RoPE base frequency

    # ----- Regularization -----
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # ----- Normalization -----
    norm_type: str = "pre_ln"         # "pre_ln" (GPT-2 style) or "post_ln"
    norm_eps: float = 1e-5

    # ----- Weight Tying -----
    tie_weights: bool = True          # Tie input embedding ↔ output projection

    # ----- Activation -----
    activation: str = "gelu"          # "gelu" | "relu" | "swiglu"

    @property
    def d_head(self) -> int:
        """Per-head dimension: d_model / n_heads."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    def param_count_estimate(self) -> dict:
        """
        Analytical estimate of parameter count.

        Embedding:  vocab_size × d_model  (+ max_seq_len × d_model if learned pos)
        Per-layer Attention:  4 × d_model²  (Wq, Wk, Wv, Wo)
        Per-layer FFN:  2 × d_model × d_ff  (up + down projections)
        Per-layer Norms:  2 × 2 × d_model   (2 LayerNorms, each has γ and β)
        LM Head:  d_model × vocab_size  (0 if weight-tied)
        """
        emb = self.vocab_size * self.d_model
        pos = self.max_seq_len * self.d_model if self.pos_encoding == "learned" else 0

        attn_per_layer = 4 * (self.d_model ** 2) + 4 * self.d_model  # weights + biases
        ffn_per_layer = 2 * self.d_model * self.d_ff + self.d_model + self.d_ff  # weights + biases
        norm_per_layer = 4 * self.d_model  # 2 LayerNorms × (γ + β)

        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        all_layers = self.n_layers * per_layer

        final_norm = 2 * self.d_model
        lm_head = 0 if self.tie_weights else self.vocab_size * self.d_model

        total = emb + pos + all_layers + final_norm + lm_head

        return {
            "embedding": emb,
            "positional": pos,
            "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "all_layers": all_layers,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": total,
            "total_M": total / 1e6,
        }


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # ----- Optimization -----
    learning_rate: float = 3e-4           # Peak LR (after warmup)
    min_lr: float = 3e-5                  # Minimum LR (cosine floor = 10% of peak)
    weight_decay: float = 0.1             # AdamW decoupled weight decay
    beta1: float = 0.9                    # Adam β₁
    beta2: float = 0.95                   # Adam β₂ (lower for stability)
    eps: float = 1e-8                     # Adam ε
    max_grad_norm: float = 1.0            # Gradient clipping threshold

    # ----- Schedule -----
    warmup_steps: int = 2000              # Linear warmup steps
    max_steps: int = 100_000              # Total training steps

    # ----- Batching -----
    batch_size: int = 32                  # Micro-batch size per GPU
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size × accum × n_gpus

    # ----- Mixed Precision -----
    use_amp: bool = True                  # Automatic Mixed Precision (fp16/bf16)
    amp_dtype: str = "bfloat16"           # "float16" or "bfloat16"

    # ----- Logging & Checkpointing -----
    log_interval: int = 10                # Steps between loss logging
    eval_interval: int = 500              # Steps between validation
    save_interval: int = 5000             # Steps between checkpoint saves
    eval_steps: int = 50                  # Number of validation batches per eval

    # ----- Paths -----
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tokenizer_path: str = "tokenizer/sp_bpe.model"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ─────────────────────────────────────────────────────────
#  Preset Configurations
# ─────────────────────────────────────────────────────────

def small_config() -> tuple[ModelConfig, TrainingConfig]:
    """
    ~49M parameter model.
    Suitable for single-GPU training on 1–2GB of text.
    """
    model = ModelConfig(
        vocab_size=32_000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,          # 4 × 512
        dropout=0.1,
    )
    train = TrainingConfig(
        learning_rate=6e-4,
        warmup_steps=1000,
        max_steps=20_000,
        batch_size=16,                     # Reduced for 16GB Mac (was 64)
        gradient_accumulation_steps=4,     # Effective batch = 16 × 4 = 64
    )
    return model, train


def medium_config() -> tuple[ModelConfig, TrainingConfig]:
    """
    ~124M parameter model (GPT-2 small equivalent).
    Suitable for multi-GPU training on 3–5GB of text.
    """
    model = ModelConfig(
        vocab_size=32_000,
        max_seq_len=512,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,          # 4 × 768
        dropout=0.1,
    )
    train = TrainingConfig(
        learning_rate=3e-4,
        warmup_steps=2000,
        max_steps=100_000,
        batch_size=32,
        gradient_accumulation_steps=4,
    )
    return model, train


if __name__ == "__main__":
    for name, fn in [("Small", small_config), ("Medium", medium_config)]:
        model_cfg, _ = fn()
        counts = model_cfg.param_count_estimate()
        print(f"\n{'='*50}")
        print(f" {name} Model Configuration")
        print(f"{'='*50}")
        print(f"  d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
              f"n_layers={model_cfg.n_layers}, d_ff={model_cfg.d_ff}")
        for k, v in counts.items():
            if k == "total_M":
                print(f"  {k}: {v:.1f}M parameters")
            else:
                print(f"  {k}: {v:,}")
