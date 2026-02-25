"""
train.py — Entry-point training script
========================================
Wires together config -> data -> model -> trainer.

Usage:
    # Small model, quick sanity check
    python training/train.py --config small --sanity

    # Medium model, full training
    python training/train.py --config medium --data-path data/processed/train.bin \
                             --val-path data/processed/val.bin
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import small_config, medium_config, ModelConfig, TrainingConfig
from model.transformer import GPT
from data.dataset import TokenizedDataset, ShuffledTokenDataset
from training.trainer import Trainer
from torch.utils.data import DataLoader


def create_synthetic_data(seq_len: int, vocab_size: int, n_tokens: int = 100_000):
    """Create synthetic token data for sanity checking."""
    print(f"Creating synthetic data ({n_tokens:,} tokens)...")
    os.makedirs("/tmp/llm_train_test", exist_ok=True)
    train_path = "/tmp/llm_train_test/train.bin"
    val_path = "/tmp/llm_train_test/val.bin"

    # Generate random token sequences
    rng = np.random.default_rng(42)
    train_tokens = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    val_tokens = rng.integers(0, vocab_size, size=n_tokens // 10, dtype=np.uint16)

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "medium"],
                        help="Model configuration preset")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to tokenized training data (.bin)")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to tokenized validation data (.bin)")
    parser.add_argument("--sanity", action="store_true",
                        help="Quick sanity check with synthetic data")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(e.g. checkpoints/checkpoint_step5000.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    # Load configuration
    config_fn = small_config if args.config == "small" else medium_config
    model_config, train_config = config_fn()

    # Sanity check overrides
    if args.sanity:
        train_config.max_steps = 100
        train_config.log_interval = 5
        train_config.eval_interval = 25
        train_config.save_interval = 50
        train_config.batch_size = 8
        train_config.gradient_accumulation_steps = 1
        train_config.warmup_steps = 10
        train_config.use_amp = False

    # Print config
    print(f"\n{'='*50}")
    print(f" Model: {args.config}")
    print(f"{'='*50}")
    counts = model_config.param_count_estimate()
    print(f"  Estimated parameters: {counts['total_M']:.1f}M")
    print(f"  d_model={model_config.d_model}, n_heads={model_config.n_heads}, "
          f"n_layers={model_config.n_layers}")
    print(f"  seq_len={model_config.max_seq_len}, vocab={model_config.vocab_size}")

    # Prepare data
    if args.sanity or (args.data_path is None):
        train_path, val_path = create_synthetic_data(
            model_config.max_seq_len, model_config.vocab_size
        )
    else:
        train_path = args.data_path
        val_path = args.val_path or args.data_path  # Use same file if no val

    train_ds = TokenizedDataset(train_path, seq_len=model_config.max_seq_len)
    val_ds = TokenizedDataset(val_path, seq_len=model_config.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=train_config.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_config.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    # Create model
    model = GPT(model_config)

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.load_checkpoint(args.resume)
        print(f"Resuming training from step {trainer.step}")

    trainer.train()

    if args.sanity:
        print("\n--- Sanity Check Results ---")
        if trainer.train_losses:
            first = trainer.train_losses[0]["loss"]
            last = trainer.train_losses[-1]["loss"]
            print(f"  First loss: {first:.4f}")
            print(f"  Last loss:  {last:.4f}")
            if last < first:
                print("  PASS: Loss is decreasing")
            else:
                print("  WARNING: Loss is not decreasing — check hyperparameters")


if __name__ == "__main__":
    main()
