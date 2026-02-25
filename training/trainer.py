"""
trainer.py — Custom Training Loop
====================================
Full training loop with:
  - AdamW optimizer (decoupled weight decay)
  - Cosine learning rate schedule with linear warmup
  - Gradient clipping (max norm)
  - Automatic Mixed Precision (AMP)
  - Gradient accumulation for large effective batch sizes
  - Periodic evaluation, logging, and checkpointing

AdamW Update Rule (mathematically):
    For parameter theta, gradient g_t at step t:

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           (first moment)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2          (second moment)
    m_hat = m_t / (1 - beta1^t)                           (bias correction)
    v_hat = v_t / (1 - beta2^t)                           (bias correction)

    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
                                  Adam update (adaptive gradient)    Decoupled weight decay

    Why decoupled weight decay (AdamW vs Adam + L2):
    In Adam + L2, the regularization gradient is also adapted by Adam's second moment,
    effectively reducing its impact for parameters with large gradients. AdamW applies
    weight decay DIRECTLY to the parameters, independent of the adaptive gradient.
    This gives more uniform regularization across parameters.

Learning Rate Schedule:
    Phase 1 — Linear Warmup (steps 0 to warmup_steps):
        lr(t) = peak_lr * (t / warmup_steps)

        Why: Large initial LR with random weights causes gradient explosion.
        Warmup lets the optimizer "calibrate" its moment estimates.

    Phase 2 — Cosine Decay (steps warmup_steps to max_steps):
        lr(t) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
        where progress = (t - warmup_steps) / (max_steps - warmup_steps)

        Why cosine > linear: Smoother decay, spends more time at higher LR
        (better exploration), then quickly drops (fine-tuning behavior).

Gradient Clipping:
    Clips gradient global norm to max_grad_norm (typically 1.0).

    total_norm = sqrt(sum(||grad_i||^2))
    if total_norm > max_norm:
        grad_i *= max_norm / total_norm

    Why: Prevents gradient explosion from rare high-loss batches.
    Without clipping, a single bad batch can destroy training progress.
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Callable

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, TrainingConfig


class Trainer:
    """Custom training loop for GPT models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        device: str = "auto",
    ):
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Training device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.config = train_config

        # Optimizer: AdamW with decoupled weight decay
        # Don't apply weight decay to biases and LayerNorm parameters
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            eps=train_config.eps,
        )

        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in no_decay_params)
        print(f"Param groups: {n_decay/1e6:.1f}M decay, {n_nodecay/1e6:.1f}M no-decay")

        # AMP (Automatic Mixed Precision)
        self.use_amp = train_config.use_amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if train_config.amp_dtype == "bfloat16" else torch.float16
        )
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # State
        self.step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        os.makedirs(train_config.checkpoint_dir, exist_ok=True)
        os.makedirs(train_config.log_dir, exist_ok=True)

    def get_lr(self, step: int) -> float:
        """Compute learning rate for given step (warmup + cosine decay)."""
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        peak_lr = self.config.learning_rate
        min_lr = self.config.min_lr

        if step < warmup:
            # Linear warmup
            return peak_lr * (step + 1) / warmup
        elif step >= max_steps:
            return min_lr
        else:
            # Cosine decay
            progress = (step - warmup) / (max_steps - warmup)
            return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

    def set_lr(self, lr: float):
        """Update optimizer learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.use_amp):
                _, loss, _ = self.model(x, targets=y)

            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir, f"checkpoint_step{self.step}.pt"
            )
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": vars(self.model_config),
            "train_config": vars(self.config),
        }
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from step {self.step}")

    def train(self):
        """
        Main training loop.

        Implements gradient accumulation:
        For effective_batch = micro_batch * accumulation_steps:
            1. Forward pass on micro_batch, compute loss / accumulation_steps
            2. Backward pass (gradients accumulate)
            3. After accumulation_steps micro-batches:
               a. Clip gradients
               b. Optimizer step
               c. Zero gradients
        """
        self.model.train()
        accum_steps = self.config.gradient_accumulation_steps
        data_iter = iter(self.train_loader)

        print(f"\n{'='*60}")
        print(f" Training GPT Model")
        print(f"{'='*60}")
        print(f"  Max steps:     {self.config.max_steps:,}")
        print(f"  Batch size:    {self.config.batch_size} x {accum_steps} = "
              f"{self.config.effective_batch_size}")
        print(f"  Peak LR:       {self.config.learning_rate}")
        print(f"  Warmup:        {self.config.warmup_steps} steps")
        print(f"  AMP:           {self.use_amp} ({self.amp_dtype})")
        print(f"{'='*60}\n")

        t0 = time.time()
        running_loss = 0.0
        micro_step = 0

        while self.step < self.config.max_steps:
            # Update learning rate
            lr = self.get_lr(self.step)
            self.set_lr(lr)

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for micro in range(accum_steps):
                # Get next batch (cycle through data)
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass with AMP
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                    enabled=self.use_amp):
                    _, loss, _ = self.model(x, targets=y)
                    # Scale loss by accumulation steps
                    scaled_loss = loss / accum_steps

                # Backward pass
                self.scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.item()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            else:
                grad_norm = None

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track loss
            avg_loss = accumulated_loss / accum_steps
            running_loss += avg_loss
            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - t0
                tokens_per_sec = (
                    self.config.log_interval * self.config.effective_batch_size *
                    self.model_config.max_seq_len / elapsed
                )
                avg_running = running_loss / self.config.log_interval
                grad_info = f"grad_norm={grad_norm:.2f}" if grad_norm is not None else ""
                print(f"  step {self.step:>6d} | loss {avg_running:.4f} | "
                      f"lr {lr:.2e} | {grad_info} | "
                      f"{tokens_per_sec/1e3:.1f}K tok/s")
                self.train_losses.append({"step": self.step, "loss": avg_running})
                running_loss = 0.0
                t0 = time.time()

            # Evaluation
            if self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                perplexity = math.exp(min(val_loss, 20))  # Cap to avoid overflow
                print(f"  >>> Eval: val_loss={val_loss:.4f}, ppl={perplexity:.2f}")
                self.val_losses.append({
                    "step": self.step,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                })

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, "best.pt")
                    )

            # Periodic checkpoint
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint()

        # Save final checkpoint and training log
        self.save_checkpoint(
            os.path.join(self.config.checkpoint_dir, "final.pt")
        )
        log_path = os.path.join(self.config.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }, f, indent=2)
        print(f"\nTraining complete. Log saved to {log_path}")
