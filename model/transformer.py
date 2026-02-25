"""
transformer.py — Full GPT-style Decoder-Only Transformer
==========================================================
A complete implementation of the GPT architecture:

Architecture (Pre-LN, GPT-2 style):
    Input -> Token Embedding + Positional Embedding -> Dropout
    -> N x TransformerBlock:
        -> LayerNorm -> Multi-Head Causal Self-Attention -> Residual Add
        -> LayerNorm -> Feed-Forward Network -> Residual Add
    -> Final LayerNorm -> Linear (LM Head) -> Logits

Why Pre-LN (vs Post-LN):
    Post-LN (original Transformer):  x + Sublayer(LayerNorm(x))  -- NO, it's:
                                     LayerNorm(x + Sublayer(x))
    Pre-LN (GPT-2):                  x + Sublayer(LayerNorm(x))

    Pre-LN places LayerNorm BEFORE each sublayer. This ensures:
    1. The residual path is "clean" (identity), improving gradient flow
    2. Gradients flow directly through the residual stream without norm scaling
    3. Training is more stable — no need for careful LR warmup
    4. The final output is NOT normalized, so we add a final LayerNorm

Feed-Forward Network:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Where W1 is (d_model, d_ff) and W2 is (d_ff, d_model).

    Why d_ff = 4 * d_model:
    - The FFN acts as a key-value memory (Keys et al., 2021)
    - Each row of W1 is a "key pattern" and corresponding row of W2 is a "value"
    - 4x expansion gives sufficient capacity for memorizing patterns
    - Empirically validated: 4x is the sweet spot across model scales

    GELU activation (Gaussian Error Linear Unit):
    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal
    Approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Why GELU over ReLU:
    - Smooth (differentiable everywhere), better gradient flow
    - Probabilistic interpretation: stochastic regularization
    - Empirically better for language modeling at all scales

Weight Tying:
    The input embedding matrix and output LM head share weights.
    Reasoning: Both operate in the same semantic space (token -> vector -> token).
    Saves d_model * vocab_size parameters (~25M for d=768, V=32K).
    Press & Wolf (2016) showed this improves perplexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.attention import CausalSelfAttention
from config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = GELU(x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)   # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block (Pre-LN architecture).

    Forward pass:
        h = x + Attention(LayerNorm(x))
        out = h + FFN(LayerNorm(h))

    Residual connections:
        The "+" operations are residual connections. They ensure:
        1. Gradient flows directly through the residual stream (identity path)
        2. Each layer only needs to learn a "delta" (refinement)
        3. Deep networks can be trained without vanishing gradients
        4. Information from early layers is preserved
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.dropout,
            pos_encoding=config.pos_encoding,
            rope_theta=config.rope_theta,
        )
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Attention with residual
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), kv_cache=kv_cache, position_offset=position_offset
        )
        x = x + attn_out

        # Pre-LN FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x, new_kv_cache


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer language model.

    Architecture:
        1. Token embedding:     (vocab_size, d_model)
        2. Positional embedding: (max_seq_len, d_model) [if learned]
        3. Dropout
        4. N x TransformerBlock
        5. Final LayerNorm
        6. LM Head:             (d_model, vocab_size) [weight-tied with token embedding]

    The model outputs logits over the vocabulary for each position.
    During training, these logits are compared with next-token targets
    using cross-entropy loss.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding (learned) — only used if not using RoPE
        if config.pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        elif config.pos_encoding == "sinusoidal":
            pe = self._sinusoidal_encoding(config.max_seq_len, config.d_model)
            self.register_buffer("pos_embedding", pe)
        # RoPE is handled inside the attention module

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final LayerNorm (necessary for Pre-LN architecture)
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # Language model head (projects from d_model -> vocab_size)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with LM head
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_embed = n_params - self.token_embedding.weight.numel()
        print(f"GPT model initialized: {n_params/1e6:.1f}M parameters "
              f"({n_params_no_embed/1e6:.1f}M non-embedding)")

    def _init_weights(self):
        """
        Initialize weights following GPT-2 conventions:
        - Linear layers: N(0, 0.02)
        - Embeddings: N(0, 0.02)
        - Output projections in residual path: N(0, 0.02 / sqrt(2 * n_layers))
          This scales down the residual contributions so they don't explode
          when summed across many layers.
        - Biases: zero
        - LayerNorm: gamma=1, beta=0
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Scale down residual-path projections
                if "out_proj" in name or "fc2" in name:
                    std *= (2 * self.config.n_layers) ** -0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass.

        Args:
            input_ids: Token ids of shape (B, T)
            targets: Target token ids of shape (B, T), for computing loss
            kv_caches: List of (K, V) caches per layer (for inference)
            position_offset: Starting position (for KV cache inference)

        Returns:
            logits: (B, T, vocab_size)
            loss: Scalar cross-entropy loss (if targets provided)
            new_kv_caches: Updated KV caches (if inference)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # (B, T, d_model)

        # Add positional embeddings (if not using RoPE)
        if self.config.pos_encoding == "learned":
            positions = torch.arange(
                position_offset, position_offset + T, device=device
            )
            x = x + self.pos_embedding(positions)  # (B, T, d_model)
        elif self.config.pos_encoding == "sinusoidal":
            x = x + self.pos_embedding[position_offset:position_offset + T]

        x = self.dropout(x)

        # Pass through transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=layer_cache, position_offset=position_offset)
            new_kv_caches.append(new_cache)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Cross-entropy loss for next-token prediction
            # Reshape: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding tokens
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with optional KV caching.

        Args:
            input_ids: Prompt token ids (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = sharper, >1 = flatter)
            top_k: Keep only top-k logits
            top_p: Nucleus sampling threshold
            use_cache: Whether to use KV caching

        Returns:
            Generated token ids (B, T + max_new_tokens)
        """
        self.eval()
        kv_caches = None

        for step in range(max_new_tokens):
            if use_cache and kv_caches is not None:
                # Only process the last token (all previous are cached)
                current_input = input_ids[:, -1:]
                position_offset = input_ids.shape[1] - 1
            else:
                # Process entire sequence (first step or no caching)
                current_input = input_ids
                if input_ids.shape[1] > self.config.max_seq_len:
                    current_input = input_ids[:, -self.config.max_seq_len:]
                position_offset = 0

            logits, _, new_kv_caches = self(
                current_input,
                kv_caches=kv_caches if use_cache else None,
                position_offset=position_offset,
            )

            if use_cache:
                kv_caches = new_kv_caches

            # Get logits for the last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back to original indexing
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    from config import small_config

    model_cfg, _ = small_config()
    model = GPT(model_cfg)

    # Test forward pass
    B, T = 2, 64
    x = torch.randint(0, model_cfg.vocab_size, (B, T))
    targets = torch.randint(0, model_cfg.vocab_size, (B, T))

    logits, loss, _ = model(x, targets=targets)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Loss:   {loss.item():.4f}")
    assert logits.shape == (B, T, model_cfg.vocab_size)

    # Test generation
    prompt = torch.randint(0, model_cfg.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated: {generated.shape}")
    assert generated.shape == (1, 28)  # 8 prompt + 20 generated
    print("PASS: All transformer tests passed")
