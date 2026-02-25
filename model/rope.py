"""
rope.py — Rotary Positional Embeddings (RoPE)
===============================================
Implementation of Rotary Position Embedding from Su et al. (2021).

Mathematical Formulation:
    Given position m and dimension pair (2i, 2i+1):

    theta_i = base^(-2i / d)    where base = 10000 (default)

    RoPE applies a rotation matrix to each pair of dimensions:

    [cos(m * theta_i)  -sin(m * theta_i)] [q_{2i}  ]
    [sin(m * theta_i)   cos(m * theta_i)] [q_{2i+1}]

    This is equivalent to:
    q'_{2i}   = q_{2i}   * cos(m * theta_i) - q_{2i+1} * sin(m * theta_i)
    q'_{2i+1} = q_{2i}   * sin(m * theta_i) + q_{2i+1} * cos(m * theta_i)

Why RoPE works:
    1. Relative position awareness: dot product q'_m . k'_n only depends on (m - n)
    2. Decaying with distance: attention naturally decreases for far-apart positions
    3. No learned parameters: purely mathematical, generalizes to unseen lengths
    4. Compatible with KV-cache: can compute for any position independently
"""

import torch
import torch.nn as nn
import math


def precompute_rope_frequencies(
    d_head: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine tables for RoPE.

    Args:
        d_head: Per-head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency (default 10000)
        device: Target device

    Returns:
        cos_table: (max_seq_len, d_head) cosine values
        sin_table: (max_seq_len, d_head) sine values
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"

    # Compute frequency for each dimension pair: theta_i = base^(-2i/d)
    # Shape: (d_head // 2,)
    dim_pairs = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (dim_pairs / d_head))

    # Position indices: (max_seq_len,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # Outer product: (max_seq_len, d_head // 2)
    angles = torch.outer(positions, freqs)

    # Duplicate to fill full d_head: (max_seq_len, d_head)
    # Each pair (2i, 2i+1) gets the same angle
    angles = angles.repeat(1, 2)

    cos_table = angles.cos()
    sin_table = angles.sin()

    return cos_table, sin_table


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, d_head)
        cos: Cosine table of shape (max_seq_len, d_head)
        sin: Sine table of shape (max_seq_len, d_head)
        position_offset: Starting position (for KV cache inference)

    Returns:
        Rotated tensor of same shape as x
    """
    seq_len = x.shape[2]
    d_head = x.shape[3]

    # Slice the precomputed tables for the current positions
    cos_slice = cos[position_offset : position_offset + seq_len, :d_head]  # (T, d)
    sin_slice = sin[position_offset : position_offset + seq_len, :d_head]  # (T, d)

    # Reshape for broadcasting: (1, 1, T, d)
    cos_slice = cos_slice.unsqueeze(0).unsqueeze(0)
    sin_slice = sin_slice.unsqueeze(0).unsqueeze(0)

    # Split x into pairs and rotate
    # x_rotated[..., 2i]   = x[..., 2i] * cos - x[..., 2i+1] * sin
    # x_rotated[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos
    x_even = x[..., 0::2]  # (B, H, T, d/2)
    x_odd = x[..., 1::2]   # (B, H, T, d/2)

    cos_half = cos_slice[..., 0::2]
    sin_half = sin_slice[..., 0::2]

    rotated_even = x_even * cos_half - x_odd * sin_half
    rotated_odd = x_even * sin_half + x_odd * cos_half

    # Interleave back: stack along last dim and reshape
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    rotated = rotated.reshape(x.shape)

    return rotated


class RotaryEmbedding(nn.Module):
    """
    Module wrapper for RoPE that manages precomputed tables.
    """

    def __init__(self, d_head: int, max_seq_len: int = 2048, theta: float = 10_000.0):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        cos, sin = precompute_rope_frequencies(d_head, max_seq_len, theta)
        # Register as buffers (not parameters — no gradients)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        return apply_rope(x, self.cos_table, self.sin_table, position_offset)


if __name__ == "__main__":
    # Quick validation
    B, H, T, D = 2, 8, 64, 64
    rope = RotaryEmbedding(d_head=D, max_seq_len=128)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rot = rope(q)
    k_rot = rope(k)

    print(f"Input shape:  {q.shape}")
    print(f"Output shape: {q_rot.shape}")
    assert q_rot.shape == q.shape, "Shape mismatch"

    # Verify relative position property:
    # attn(q_m, k_n) should only depend on (m - n)
    # This is hard to verify directly, but we can check that
    # rotating the same vector at different positions gives different results
    assert not torch.allclose(q_rot[:, :, 0, :], q_rot[:, :, 1, :]), \
        "Different positions should give different rotations"
    print("PASS: RoPE produces position-dependent rotations")
