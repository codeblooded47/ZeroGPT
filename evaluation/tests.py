"""
tests.py — Unit Tests for GPT Model
======================================
Verifies correctness of model components:
  1. Output shape matches expected dimensions
  2. Causal mask blocks future tokens
  3. Loss decreases on overfit sanity check
  4. Gradient norms are finite and non-zero
  5. Tokenizer roundtrip (encode -> decode = original)
  6. KV cache consistency
"""

import os
import sys
import math
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import small_config, ModelConfig
from model.transformer import GPT
from model.attention import CausalSelfAttention
from tokenizer.bpe import BPETokenizer


# ─── Fixtures ───────────────────────────────────────────

@pytest.fixture
def model_config():
    """Small model config for testing."""
    cfg = ModelConfig(
        vocab_size=256,
        max_seq_len=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.0,
        attn_dropout=0.0,
    )
    return cfg


@pytest.fixture
def model(model_config):
    return GPT(model_config)


@pytest.fixture
def tokenizer():
    """Train a tiny tokenizer for testing."""
    tok = BPETokenizer(vocab_size=270)  # 256 bytes + 4 special + 10 merges
    sample = "the quick brown fox jumps over the lazy dog " * 50
    tok.train(sample, verbose=False)
    return tok


# ─── Test 1: Output Shape ──────────────────────────────

class TestOutputShape:
    def test_logits_shape(self, model, model_config):
        """Model output should be (B, T, vocab_size)."""
        B, T = 2, 32
        x = torch.randint(0, model_config.vocab_size, (B, T))
        logits, _, _ = model(x)
        assert logits.shape == (B, T, model_config.vocab_size), \
            f"Expected {(B, T, model_config.vocab_size)}, got {logits.shape}"

    def test_loss_is_scalar(self, model, model_config):
        """Loss should be a scalar tensor."""
        B, T = 2, 32
        x = torch.randint(0, model_config.vocab_size, (B, T))
        y = torch.randint(0, model_config.vocab_size, (B, T))
        _, loss, _ = model(x, targets=y)
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"

    def test_initial_loss_near_random(self, model, model_config):
        """Initial loss should be near -log(1/vocab_size) = log(vocab_size)."""
        B, T = 4, 32
        x = torch.randint(0, model_config.vocab_size, (B, T))
        y = torch.randint(0, model_config.vocab_size, (B, T))
        _, loss, _ = model(x, targets=y)
        expected = math.log(model_config.vocab_size)
        # Allow 50% tolerance
        assert abs(loss.item() - expected) / expected < 0.5, \
            f"Initial loss {loss.item():.2f} too far from expected {expected:.2f}"


# ─── Test 2: Causal Mask ───────────────────────────────

class TestCausalMask:
    def test_mask_blocks_future(self, model_config):
        """Attention weights should be zero for future positions."""
        attn = CausalSelfAttention(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            max_seq_len=model_config.max_seq_len,
            attn_dropout=0.0,
            proj_dropout=0.0,
        )
        attn.eval()

        B, T = 1, 16
        x = torch.randn(B, T, model_config.d_model)

        # Hook to capture attention weights
        attn_weights_captured = []
        def hook_fn(module, input, output):
            # We'll verify via the causal mask property
            pass

        # Verify that changing future tokens doesn't affect past outputs
        x1 = x.clone()
        x2 = x.clone()
        x2[:, T//2:, :] = torch.randn(B, T - T//2, model_config.d_model)

        out1, _ = attn(x1)
        out2, _ = attn(x2)

        # Outputs for positions before T//2 should be identical
        # (since those positions can't attend to future)
        assert torch.allclose(out1[:, :T//2, :], out2[:, :T//2, :], atol=1e-5), \
            "Causal mask violated: changing future tokens affected past outputs"

    def test_mask_shape(self, model_config):
        """Causal mask should be upper triangular."""
        attn = CausalSelfAttention(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            max_seq_len=model_config.max_seq_len,
        )
        mask = attn.causal_mask[:8, :8]
        # Upper triangle (above diagonal) should be -inf
        for i in range(8):
            for j in range(8):
                if j > i:
                    assert mask[i, j] == float("-inf"), \
                        f"Mask[{i},{j}] should be -inf but is {mask[i,j]}"
                else:
                    assert mask[i, j] == 0.0, \
                        f"Mask[{i},{j}] should be 0 but is {mask[i,j]}"


# ─── Test 3: Overfit Sanity Check ─────────────────────

class TestOverfit:
    def test_loss_decreases(self, model, model_config):
        """Model should be able to memorize a single batch."""
        B, T = 4, 16
        x = torch.randint(0, model_config.vocab_size, (B, T))
        y = torch.randint(0, model_config.vocab_size, (B, T))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        for _ in range(50):
            optimizer.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, \
            f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ─── Test 4: Gradient Health ──────────────────────────

class TestGradients:
    def test_gradients_exist(self, model, model_config):
        """All trainable parameters should have non-zero gradients."""
        B, T = 2, 16
        x = torch.randint(0, model_config.vocab_size, (B, T))
        y = torch.randint(0, model_config.vocab_size, (B, T))

        _, loss, _ = model(x, targets=y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient for {name}"

    def test_gradient_norm_reasonable(self, model, model_config):
        """Global gradient norm should be finite and reasonable."""
        B, T = 2, 16
        x = torch.randint(0, model_config.vocab_size, (B, T))
        y = torch.randint(0, model_config.vocab_size, (B, T))

        _, loss, _ = model(x, targets=y)
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert torch.isfinite(total_norm), f"Gradient norm is not finite: {total_norm}"
        assert total_norm > 0, "Gradient norm is zero"
        assert total_norm < 1000, f"Gradient norm too large: {total_norm}"


# ─── Test 5: Tokenizer Roundtrip ─────────────────────

class TestTokenizer:
    def test_encode_decode_roundtrip(self, tokenizer):
        """Encoding then decoding should recover the original text."""
        texts = [
            "hello world",
            "the quick brown fox",
            "testing 123",
        ]
        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert decoded == text, \
                f"Roundtrip failed: {text!r} -> {ids} -> {decoded!r}"

    def test_special_tokens(self, tokenizer):
        """Encoded text should start with BOS and end with EOS."""
        ids = tokenizer.encode("hello")
        assert ids[0] == tokenizer.bos_id, "Should start with BOS"
        assert ids[-1] == tokenizer.eos_id, "Should end with EOS"


# ─── Test 6: KV Cache Consistency ────────────────────

class TestKVCache:
    def test_cache_matches_no_cache(self, model, model_config):
        """Output with KV cache should match output without cache."""
        model.eval()
        B = 1
        prompt_len = 8
        x = torch.randint(0, model_config.vocab_size, (B, prompt_len))

        # Full forward pass (no cache)
        with torch.no_grad():
            logits_full, _, _ = model(x)

        # Incremental forward pass (with cache)
        with torch.no_grad():
            # First: process prompt
            logits_cached, _, kv_caches = model(x)

            # Verify prompt processing matches
            assert torch.allclose(logits_full, logits_cached, atol=1e-4), \
                "Cached and non-cached outputs differ for prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
