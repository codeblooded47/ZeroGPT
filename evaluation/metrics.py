"""
metrics.py — Evaluation Metrics
=================================
Validation loss tracking and perplexity calculation.

Perplexity Derivation:
    Given a language model P and test sequence w_1, ..., w_N:

    Perplexity = exp( -1/N * sum_{i=1}^{N} log P(w_i | w_1, ..., w_{i-1}) )

    This is the exponentiated average negative log-likelihood (cross-entropy).

    Equivalently:  PPL = exp(H)  where H is the cross-entropy loss.

    Interpretation:
    - PPL = 1:     Perfect prediction (model always assigns prob 1 to correct token)
    - PPL = V:     Random guessing over vocabulary V (~32000 for GPT)
    - PPL = 20:    Good language model (equivalent to choosing from ~20 options)

    Lower perplexity = better model. It represents the "effective branching factor"
    — how many tokens the model is effectively choosing between at each step.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


@torch.no_grad()
def compute_validation_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Compute validation loss and perplexity.

    Returns:
        dict with keys: val_loss, perplexity, n_batches, n_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss, _ = model(x, targets=y)

        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    model.train()
    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "n_batches": n_batches,
        "n_tokens": total_tokens,
    }


def overfitting_sanity_check(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    steps: int = 100,
    lr: float = 1e-3,
) -> list[float]:
    """
    Verify the model can memorize a single batch.

    If the model CANNOT overfit a single batch, something is fundamentally
    wrong with the architecture or training pipeline.

    Expected behavior: loss should drop to near 0 within 100 steps.
    """
    model.train()
    model = model.to(device)
    x, y = batch
    x, y = x.to(device), y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            print(f"  Overfit check step {step+1}: loss={loss.item():.4f}")

    final_loss = losses[-1]
    initial_loss = losses[0]
    passed = final_loss < initial_loss * 0.1  # Should drop by 90%

    print(f"  Overfit check: {initial_loss:.4f} -> {final_loss:.4f} "
          f"({'PASS' if passed else 'FAIL'})")
    return losses
