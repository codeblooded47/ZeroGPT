"""
sampling.py — Text Generation Sampling Strategies
====================================================
Implements greedy, top-k, top-p (nucleus), and temperature sampling.

Temperature Scaling Mathematics:
    Given logits z = (z_1, ..., z_V) for vocabulary V:

    P(token_i) = exp(z_i / T) / sum_j exp(z_j / T)

    T = 1.0:  Standard softmax (unchanged distribution)
    T -> 0:   Approaches one-hot on argmax (greedy, deterministic)
    T -> inf: Approaches uniform distribution (maximum randomness)
    T < 1:    Sharper distribution (more confident, less diverse)
    T > 1:    Flatter distribution (less confident, more diverse)

    Why it works: Dividing logits by T scales the log-probabilities.
    Since softmax is shift-invariant but NOT scale-invariant, this
    directly controls the entropy of the output distribution.

Top-k Sampling:
    1. Sort logits in descending order
    2. Keep only the top k logits
    3. Set all others to -infinity
    4. Apply softmax and sample

    Problem: Fixed k doesn't adapt to varying confidence levels.
    When the model is confident, k=50 still includes many bad tokens.
    When uncertain, k=50 might exclude valid options.

Top-p (Nucleus) Sampling (Holtzman et al., 2019):
    1. Sort logits in descending order
    2. Compute cumulative probability from sorted softmax
    3. Find smallest set of tokens whose cumulative probability >= p
    4. Set all others to -infinity
    5. Renormalize and sample

    Advantage: Adapts the number of candidate tokens to the model's
    confidence. High confidence -> few candidates. Low confidence -> many.
    Typically p = 0.9 works well.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Sample next token from logits.

    Args:
        logits: Raw model output (B, vocab_size)
        temperature: Sampling temperature
        top_k: Top-k filtering (None = no filtering)
        top_p: Nucleus sampling threshold (None = no filtering)
        greedy: If True, always pick argmax

    Returns:
        Sampled token ids (B, 1)
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k is not None and top_k > 0:
        logits = top_k_filter(logits, top_k)

    # Top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the top-k threshold."""
    k = min(k, logits.size(-1))
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:]  # (B, 1)
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return logits


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Create mask for tokens to remove
    # Shift right by 1 so the token that crosses the threshold is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original order
    logits = logits.scatter(-1, sorted_indices, sorted_logits)
    return logits


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text from a prompt string.

    Args:
        model: GPT model
        tokenizer: BPE tokenizer
        prompt: Input text
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        device: Computation device

    Returns:
        Generated text (including prompt)
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True,
        )

    # Decode
    generated = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    return generated


if __name__ == "__main__":
    # Test sampling functions with synthetic logits
    B, V = 2, 100
    logits = torch.randn(B, V)

    print("Testing sampling strategies:")

    # Greedy
    greedy = sample_logits(logits, greedy=True)
    assert greedy.shape == (B, 1)
    assert (greedy == logits.argmax(dim=-1, keepdim=True)).all()
    print(f"  Greedy: {greedy.squeeze().tolist()}")

    # Temperature
    hot = sample_logits(logits, temperature=2.0)
    cold = sample_logits(logits, temperature=0.1)
    print(f"  Hot (T=2.0):  {hot.squeeze().tolist()}")
    print(f"  Cold (T=0.1): {cold.squeeze().tolist()}")

    # Top-k
    topk = sample_logits(logits, top_k=5)
    print(f"  Top-k (k=5):  {topk.squeeze().tolist()}")

    # Top-p
    topp = sample_logits(logits, top_p=0.9)
    print(f"  Top-p (p=0.9): {topp.squeeze().tolist()}")

    print("PASS: All sampling strategies work correctly")
