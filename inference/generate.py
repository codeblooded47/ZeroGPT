"""
generate.py — KV-Cached Autoregressive Generation
=====================================================
Production-grade generation loop with:
  - KV caching for O(1) per-token cost (vs O(n) without cache)
  - Batch decoding support
  - Throughput and latency measurement
  - Multiple stopping criteria (max tokens, EOS, custom)

KV Cache Explained:
    Without cache: At step t, the model recomputes attention for ALL t tokens.
    Total compute for generating N tokens: O(N^2 * d)

    With cache: At step t, we store K and V from all previous steps.
    Only the new token's Q is computed. We attend to cached K/V.
    Total compute: O(N * d) — linear instead of quadratic.

    Memory cost: O(N * n_layers * 2 * d_model) for storing K and V
    For 512 context, 12 layers, 768 d_model: ~9.4 MB per sequence (float16)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.sampling import sample_logits


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    eos_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Autoregressive generation with KV caching.

    Args:
        model: GPT model
        input_ids: Prompt tokens (B, T)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        greedy: Greedy decoding
        eos_token_id: Stop generation on this token
        use_cache: Enable KV caching

    Returns:
        output_ids: Generated token ids (B, T + max_new_tokens)
        stats: Dictionary with timing info
    """
    model.eval()
    device = input_ids.device
    B = input_ids.shape[0]
    kv_caches = None
    generated_tokens = 0

    t_start = time.perf_counter()
    t_first_token = None

    for step in range(max_new_tokens):
        if use_cache and kv_caches is not None:
            # Incremental: only process last token
            current_input = input_ids[:, -1:]
            position_offset = input_ids.shape[1] - 1
        else:
            # Full pass (first step or no cache)
            current_input = input_ids
            position_offset = 0

        logits, _, new_kv_caches = model(
            current_input,
            kv_caches=kv_caches if use_cache else None,
            position_offset=position_offset,
        )

        if use_cache:
            kv_caches = new_kv_caches

        # Get last-position logits and sample
        next_logits = logits[:, -1, :]
        next_token = sample_logits(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )

        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens += 1

        if t_first_token is None:
            t_first_token = time.perf_counter()

        # Check EOS
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                break

    t_end = time.perf_counter()

    stats = {
        "total_time_s": t_end - t_start,
        "time_to_first_token_s": (t_first_token - t_start) if t_first_token else 0,
        "tokens_generated": generated_tokens,
        "tokens_per_second": generated_tokens / (t_end - t_start) if (t_end - t_start) > 0 else 0,
        "batch_size": B,
        "used_cache": use_cache,
    }

    return input_ids, stats


@torch.no_grad()
def batch_generate(
    model: nn.Module,
    prompts: list[torch.Tensor],
    max_new_tokens: int = 100,
    pad_token_id: int = 0,
    **kwargs,
) -> list[torch.Tensor]:
    """
    Generate text for multiple prompts with padding.

    Left-pads shorter prompts to create a batch, generates,
    then strips padding from results.
    """
    # Find max prompt length
    max_len = max(p.shape[0] for p in prompts)

    # Left-pad to create batch
    padded = []
    for p in prompts:
        pad_len = max_len - p.shape[0]
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_token_id, dtype=p.dtype, device=p.device)
            padded.append(torch.cat([padding, p]))
        else:
            padded.append(p)

    batch = torch.stack(padded)  # (B, max_len)

    # Generate (note: padding may affect results; for production use,
    # consider using attention masks or generating individually)
    output, stats = generate(model, batch, max_new_tokens=max_new_tokens, **kwargs)

    # Strip padding from results
    results = []
    for i, p in enumerate(prompts):
        pad_len = max_len - p.shape[0]
        results.append(output[i, pad_len:])

    return results, stats


if __name__ == "__main__":
    from config import small_config
    from model.transformer import GPT

    model_config, _ = small_config()
    model_config.vocab_size = 256
    model_config.max_seq_len = 64
    model_config.d_model = 128
    model_config.n_heads = 4
    model_config.n_layers = 2
    model_config.d_ff = 512

    model = GPT(model_config)

    # Test generation with and without cache
    prompt = torch.randint(0, 256, (1, 8))

    print("Generating with KV cache...")
    out_cached, stats_cached = generate(model, prompt, max_new_tokens=50, use_cache=True,
                                        temperature=0.8, top_k=50)
    print(f"  Time: {stats_cached['total_time_s']:.3f}s")
    print(f"  Tokens/s: {stats_cached['tokens_per_second']:.1f}")
    print(f"  Output length: {out_cached.shape[1]}")

    print("\nGenerating without KV cache...")
    out_uncached, stats_uncached = generate(model, prompt, max_new_tokens=50, use_cache=False,
                                            temperature=0.8, top_k=50)
    print(f"  Time: {stats_uncached['total_time_s']:.3f}s")
    print(f"  Tokens/s: {stats_uncached['tokens_per_second']:.1f}")

    speedup = stats_uncached['total_time_s'] / max(stats_cached['total_time_s'], 1e-6)
    print(f"\nKV cache speedup: {speedup:.1f}x")
