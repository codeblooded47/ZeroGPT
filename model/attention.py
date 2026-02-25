"""
attention.py — Multi-Head Causal Self-Attention
=================================================
Pure PyTorch implementation of scaled dot-product attention
with causal masking, multi-head projections, and optional RoPE.

Mathematical formulation:
    Q = X @ W_q + b_q    shape: (B, T, d_model) -> (B, T, d_model)
    K = X @ W_k + b_k
    V = X @ W_v + b_v

    Reshape to heads: (B, T, d_model) -> (B, n_heads, T, d_head)

    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) + M ) @ V

    where M is the causal mask:
        M[i,j] = 0      if j <= i  (can attend)
        M[i,j] = -inf   if j > i   (cannot attend to future)

    Why divide by sqrt(d_k):
        Without scaling, the dot products Q @ K^T grow in magnitude proportional
        to d_k (each element is a sum of d_k terms). Large dot products push
        softmax into saturated regions where gradients vanish. Dividing by
        sqrt(d_k) normalizes the variance of the dot products to ~1, keeping
        softmax in its sensitive (non-saturated) regime.

    Multi-head motivation:
        Multiple heads allow the model to jointly attend to information from
        different representation subspaces at different positions. A single
        head can only compute one attention pattern; multiple heads learn
        complementary patterns (e.g., syntactic structure, semantic similarity,
        positional proximity).

    Output projection:
        After concatenating all heads: (B, T, n_heads * d_head) = (B, T, d_model)
        Final: Output = Concat(head_1, ..., head_h) @ W_o + b_o
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Complexity: O(n^2 * d) where n = sequence length, d = d_model
    Memory:     O(n^2 * h) for attention weights (h = n_heads)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        pos_encoding: str = "rope",
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)  # 1 / sqrt(d_k)

        # Combined Q, K, V projection (more efficient than 3 separate layers)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Causal mask: upper-triangular matrix of -inf
        # Registered as buffer (not a parameter — no gradients)
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        # Rotary Positional Embeddings
        self.pos_encoding = pos_encoding
        self.rope = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(self.d_head, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            kv_cache: Optional (cached_K, cached_V) for autoregressive inference
            position_offset: Starting position for RoPE (used with KV cache)

        Returns:
            output: (B, T, d_model)
            new_kv_cache: Updated (K, V) cache or None
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, d_model)

        # Reshape to multi-head: (B, T, d_model) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (before attention computation)
        if self.rope is not None:
            q = self.rope(q, position_offset=position_offset)
            k = self.rope(k, position_offset=position_offset)

        # KV cache handling for autoregressive generation
        new_kv_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # (B, H, T_cached + T, d_head)
            v = torch.cat([cached_v, v], dim=2)
            new_kv_cache = (k, v)
        elif self.training is False:
            # During inference, always store KV cache
            new_kv_cache = (k, v)

        # Scaled dot-product attention
        # attn_weights = Q @ K^T / sqrt(d_k)
        # Shape: (B, H, T_q, d_head) @ (B, H, d_head, T_k) -> (B, H, T_q, T_k)
        T_k = k.shape[2]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        # The mask ensures position i can only attend to positions <= i
        if T == 1 and kv_cache is not None:
            # Single-token generation step: no masking needed
            # (the single query token can attend to all cached keys)
            pass
        else:
            mask = self.causal_mask[:T, :T_k]
            attn_weights = attn_weights + mask

        # Softmax + dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # (B, H, T_q, T_k) @ (B, H, T_k, d_head) -> (B, H, T_q, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads: (B, H, T, d_head) -> (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output, new_kv_cache


if __name__ == "__main__":
    # Quick test
    B, T, D, H = 2, 64, 512, 8
    attn = CausalSelfAttention(d_model=D, n_heads=H, max_seq_len=128)

    x = torch.randn(B, T, D)
    out, cache = attn(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Cache K shape: {cache[0].shape}")
    assert out.shape == x.shape, "Output shape mismatch"

    # Test with KV cache (single token inference)
    single = torch.randn(B, 1, D)
    out2, cache2 = attn(single, kv_cache=cache, position_offset=T)
    print(f"Single-token output: {out2.shape}")
    print(f"Updated cache K: {cache2[0].shape}")
    assert out2.shape == (B, 1, D)
    assert cache2[0].shape[2] == T + 1, "Cache should grow by 1"
    print("PASS: Attention with KV cache works correctly")
