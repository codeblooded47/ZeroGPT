# BLUEPRINT.md — GPT from Scratch: A Complete Technical Reference

> A research-grade technical blueprint for building a decoder-only transformer
> language model (50M–150M parameters) entirely from scratch in PyTorch.

---

## Table of Contents

1. [Dataset Engineering](#section-1--dataset-engineering)
2. [Tokenizer](#section-2--tokenizer)
3. [Transformer Architecture](#section-3--transformer-architecture)
4. [Training Dynamics](#section-4--training-dynamics)
5. [Evaluation](#section-5--evaluation)
6. [Systems & Optimization](#section-6--systems--optimization)
7. [Scaling Laws](#section-7--scaling-laws)
8. [Inference Engineering](#section-8--inference-engineering)
9. [Failure Modes & Debugging](#section-9--failure-modes--debugging)
10. [Final Checklists](#final-checklists)

---

## Section 1 — Dataset Engineering

### 1.1 Dataset Sources

| Dataset               | Size   | Description                                       | URL                                                                                                                      |
| --------------------- | ------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **WikiText-103**      | ~500MB | Clean Wikipedia articles, long-range dependencies | `https://huggingface.co/datasets/Salesforce/wikitext` (use `load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')`) |
| **OpenWebText**       | ~40GB  | Reddit-filtered web text (WebText replication)    | `https://huggingface.co/datasets/Skylion007/openwebtext`                                                                 |
| **The Pile** (subset) | ~200GB | Diverse corpus: books, code, web, academic        | `https://huggingface.co/datasets/monology/pile-uncopyrighted`                                                            |
| **GitHub Code**       | ~1TB   | Permissively-licensed source code                 | `https://huggingface.co/datasets/codeparrot/github-code`                                                                 |

**Recommended**: Start with WikiText-103 for prototyping, then scale to OpenWebText for real training.

### 1.2 Data Cleaning Techniques

**Step 1 — Deduplication**:

- **Exact dedup**: Hash each document (SHA-256), remove duplicates. O(N) time, O(N) space.
- **Near-dedup (MinHash + LSH)**: Compute MinHash signatures (128-256 hashes per document). Use Locality-Sensitive Hashing to find candidate pairs with Jaccard similarity > 0.8. Removes paraphrased or slightly modified copies. Lee et al. (2022) showed dedup improves perplexity by 3-7%.
- **Substring dedup**: Remove documents containing long repeated n-grams (n=13-50). Catches boilerplate like navigation bars, privacy policies.

**Step 2 — Quality Filtering**:

- Remove documents shorter than 50 characters (noise, titles, etc.)
- Remove documents with alphanumeric ratio < 30% (boilerplate HTML/navigation)
- Remove documents with excessive special character density
- Language filtering: use fastText language ID to keep English-only (or target language)
- Perplexity filtering: Score documents with a small reference LM; remove extreme outliers

**Step 3 — Boilerplate Removal**:

- Strip HTML tags, JavaScript, CSS
- Remove cookie consent banners, navigation menus
- Strip "Share on Twitter/Facebook" blocks
- Remove "Copyright ©" footers
- trafilatura or resiliparse for web text extraction

**Step 4 — Document Boundary Handling**:

- Separate documents with a special `<|endofdoc|>` marker
- During tokenization, this becomes the `<|eos|>` token
- The model learns to recognize document boundaries

### 1.3 Token Distribution Statistics

**Zipf's Law**: Token frequency follows a power law: `f(r) ∝ 1/r^α` where `r` is rank and `α ≈ 1`.

Implications:

- The top 100 tokens (~0.3% of vocab) account for ~50% of all tokens
- The bottom 50% of vocabulary tokens are extremely rare (< 0.01% each)
- This creates severe class imbalance in the cross-entropy loss
- Rare tokens receive very few gradient updates → poor quality

**Vocabulary Coverage**: With 32K BPE vocab on English text:

- Average tokens per word: ~1.3
- Average characters per token: ~3.8
- 99.9%+ of text is covered without `<|unk|>` (byte-level BPE guarantees 100%)

### 1.4 Scaling Laws: Data vs Parameters

**Chinchilla Optimal** (Hoffmann et al., 2022):

```
Optimal tokens ≈ 20 × Parameters
```

| Model Size  | Optimal Tokens | Optimal Data Size (~4 chars/token) |
| ----------- | -------------- | ---------------------------------- |
| 50M params  | 1B tokens      | ~4GB                               |
| 120M params | 2.4B tokens    | ~10GB                              |
| 350M params | 7B tokens      | ~28GB                              |

Training with **less data** than optimal → underfitting (model can learn more but data is exhausted).
Training with **more data** than optimal → diminishing returns (compute better spent on larger model).

---

## Section 2 — Tokenizer

### 2.1 Byte Pair Encoding: Formal Algorithm

**Input**: Corpus C (string of characters), target vocabulary size V

**Initialization**: Set vocabulary `vocab = {all unique bytes}` (256 entries)

**Algorithm**:

```
for merge_step = 1 to (V - 256):
    1. Count all adjacent token pairs (a, b) across the corpus
       counts = {(a, b): frequency for all adjacent pairs}

    2. Select the most frequent pair:
       (a*, b*) = argmax_{(a,b)} counts[(a, b)]

    3. Create new token: t = concat(a*, b*)

    4. Replace all occurrences of (a*, b*) with t in the corpus

    5. Add t to vocab, record merge rule: (a*, b*) → t
```

**Encoding** (applying learned merges to new text):

```
tokens = [byte for byte in text.encode("utf-8")]
for (a, b) in merge_rules:  # in order they were learned
    merge all adjacent (a, b) → merged_token in tokens
return tokens
```

**Decoding**: Each token maps to a byte sequence; concatenate and decode as UTF-8.

### 2.2 Complexity Analysis

| Operation | Time     | Space     |
| --------- | -------- | --------- |
| Training  | O(N × M) | O(N + V²) |
| Encoding  | O(n × M) | O(n)      |
| Decoding  | O(n)     | O(n)      |

Where N = corpus size (bytes), M = number of merges, n = input length, V = vocab size.

Training is the bottleneck: for a 1GB corpus with 32K merges, this is ~32 trillion operations in the naive implementation. Practical implementations use indexed data structures (suffix arrays, priority queues) to reduce this.

### 2.3 Why 32K Vocabulary

Trade-offs:

- **Too small** (1K): Many tokens per word → wastefully long sequences, model must spend capacity on subword composition
- **Too large** (100K+): Each token is rarer → less training signal per token, larger embedding matrix (wastes parameters)
- **Sweet spot** (32K–64K): ~1.3 tokens/word for English, good subword granularity, manageable embedding size

Embedding size: `vocab_size × d_model`. At 32K × 768 = 24.6M params (≈25% of a 100M model). At 100K × 768 = 76.8M params (dominant cost for small models).

### 2.4 Special Tokens & OOV Handling

```
<|pad|>  (id=0)  — Padding for batching variable-length sequences
<|unk|>  (id=1)  — Unknown token (unused in byte-level BPE)
<|bos|>  (id=2)  — Beginning of sequence
<|eos|>  (id=3)  — End of sequence / document boundary
```

**OOV Handling**: Byte-level BPE guarantees zero OOV tokens because any Unicode character decomposes into bytes, and all 256 bytes are in the base vocabulary. This is a key advantage over word-level or character-level tokenization.

### 2.5 Tokenizer Impact on Model Quality

- Better tokenization → shorter sequences → more context per sample → better long-range modeling
- Consistent tokenization of morphological variants (e.g., "running", "runs", "ran" sharing "run") improves generalization
- Code-aware tokenization (preserving indentation, operators) is critical for code models
- Multi-lingual tokenization requires balancing vocabulary across languages (avoid English bias)

---

## Section 3 — Transformer Architecture

### 3.1 Self-Attention: Full Mathematical Formulation

Given input sequence `X ∈ ℝ^{T × d_model}` (T tokens, d_model dimensions):

**Step 1 — Linear Projections**:

```
Q = X × W_q + b_q    where W_q ∈ ℝ^{d_model × d_model}, b_q ∈ ℝ^{d_model}
K = X × W_k + b_k    where W_k ∈ ℝ^{d_model × d_model}, b_k ∈ ℝ^{d_model}
V = X × W_v + b_v    where W_v ∈ ℝ^{d_model × d_model}, b_v ∈ ℝ^{d_model}
```

**Step 2 — Scaled Dot-Product Attention**:

```
Attention(Q, K, V) = softmax( Q × K^T / √d_k + M ) × V
```

Where:

- `Q × K^T ∈ ℝ^{T × T}` is the attention score matrix
- `√d_k` is the scaling factor (d_k = d_model / n_heads)
- `M` is the causal mask
- `softmax` is applied row-wise (over key dimension)

**Why divide by √d_k**:

Consider `q · k = Σ_i q_i × k_i` where `q_i, k_i ~ N(0, 1)` independently. Then:

```
E[q · k] = 0
Var[q · k] = d_k    (sum of d_k independent products, each with variance 1)
```

So `q · k ~ O(√d_k)`. Without scaling, the dot products grow with `√d_k`, pushing softmax into saturation where:

- One element approaches 1, all others approach 0
- Gradients of softmax vanish in saturated regions: `∂softmax/∂z_i ≈ 0`

Dividing by `√d_k` normalizes variance to 1, keeping softmax in its sensitive regime.

**Step 3 — Causal Mask** (for autoregressive generation):

```
M[i, j] = 0      if j ≤ i    (token i CAN attend to token j)
M[i, j] = -∞     if j > i    (token i CANNOT attend to future token j)
```

After adding M to the attention scores, `softmax(-∞) = 0`, effectively zeroing out future positions. This ensures the model can only condition on past tokens, which is essential for autoregressive generation where tokens are produced left-to-right.

**Complexity**: O(T² × d_model) time, O(T² × n_heads) memory for attention weights.

### 3.2 Multi-Head Attention

Instead of one attention function with d_model dimensions, use `h` heads each with `d_k = d_model / h` dimensions:

```
head_i = Attention(Q_i, K_i, V_i)    where Q_i ∈ ℝ^{T × d_k}
MultiHead = Concat(head_1, ..., head_h) × W_o
```

Where `W_o ∈ ℝ^{d_model × d_model}` is the output projection.

**Why multiple heads?**

A single head computes ONE attention pattern — one way to weigh which tokens are relevant. But language has multiple simultaneous relationships:

- **Head A** might learn syntactic structure (subject ↔ verb agreement)
- **Head B** might learn semantic similarity (synonyms, co-references)
- **Head C** might learn positional proximity (local context)
- **Head D** might learn quotation/dialogue patterns

Each head independently learns which positions to attend to. The output projection combines these different "views" of the sequence. Empirically, 8-16 heads work well for d_model=512-1024.

### 3.3 Positional Encoding Comparison

#### Sinusoidal (Vaswani et al., 2017)

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

- ✅ No learnable parameters
- ✅ Theoretically generalizes to any length
- ❌ In practice, generalization to unseen lengths is poor
- ❌ Fixed representation, cannot adapt to data

#### Learned Positional Embeddings (GPT-2)

```
PE = Embedding(max_seq_len, d_model)
x = token_embedding(tokens) + PE(positions)
```

- ✅ Data-adaptive, can learn task-specific position patterns
- ✅ Simple implementation
- ❌ Cannot generalize beyond training length
- ❌ Uses max_seq_len × d_model additional parameters

#### Rotary Positional Embeddings (RoPE, Su et al., 2021)

```
θ_i = base^{-2i/d}

For position m, dimension pair (2i, 2i+1):
R(m, θ_i) = [cos(m·θ_i)  -sin(m·θ_i)] [q_{2i}  ]
             [sin(m·θ_i)   cos(m·θ_i)] [q_{2i+1}]
```

Key property: `<R(m)q, R(n)k> = <q, R(n-m)k>` — the dot product only depends on _relative_ position (m - n).

- ✅ Encodes relative position in attention scores directly
- ✅ No learnable parameters
- ✅ Naturally decays with distance (high-frequency dimensions decay faster)
- ✅ Compatible with KV caching (position computed per-token)
- ✅ Can be extended to longer sequences (NTK-aware scaling)
- This is the **recommended choice** (used in LLaMA, Mistral, GPT-NeoX)

### 3.4 Feed-Forward Network (FFN)

```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2
```

Where `W_1 ∈ ℝ^{d_model × d_ff}`, `W_2 ∈ ℝ^{d_ff × d_model}`, and typically **d_ff = 4 × d_model**.

**Why d_ff = 4 × d_model?**

The FFN acts as a **key-value memory** (Geva et al., 2021). Each row of W_1 is a "key pattern" that matches certain input representations. The corresponding row of W_2 is the "value" that gets added to the representation. The 4× expansion gives d_ff = 4 × d_model memory slots per layer. Empirically, this ratio is the sweet spot: smaller ratios underfit, larger ratios have diminishing returns.

**GELU Activation**:

```
GELU(x) = x × Φ(x)    where Φ is the standard normal CDF

Approximation: GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

Why GELU over ReLU:

- ReLU has a hard zero for `x < 0` → dead neurons, high-variance gradients
- GELU is smooth everywhere → better gradient flow
- GELU has a probabilistic interpretation: stochastic regularization
- Empirically gives ~0.5-1% better perplexity across scales

### 3.5 Residual Connections & Layer Normalization

**Pre-LN Architecture** (GPT-2, recommended):

```
h = x + Attention(LayerNorm(x))
out = h + FFN(LayerNorm(h))
```

**Post-LN** (Original Transformer):

```
h = LayerNorm(x + Attention(x))
out = LayerNorm(h + FFN(h))
```

**Why Pre-LN is better for training**:

In Pre-LN, the residual path is a **clean identity function**: `x_{L} = x_0 + Σ_{l=1}^{L} F_l(LayerNorm(x_{l-1}))`.

Gradient through the residual stream: `∂L/∂x_0 = ∂L/∂x_L × (I + ∂/∂x_0 Σ F_l(...))`.

The identity term `I` ensures gradients flow directly from loss to input without any normalization scaling, preventing vanishing/exploding gradients. Post-LN applies normalization ON the residual path, which can scale gradients unpredictably.

**LayerNorm** (Ba et al., 2016):

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```

Where `μ = mean(x)`, `σ² = var(x)` computed over the feature dimension, and `γ, β` are learnable scale and bias. Unlike BatchNorm, LayerNorm normalizes over features (not batch), making it independent of batch size.

### 3.6 Parameter Count Formulas

For a transformer with `L` layers, `d` = d_model, `h` = n_heads, `d_ff`, `V` = vocab_size, `T` = max_seq_len:

| Component             | Formula                   | Notes                              |
| --------------------- | ------------------------- | ---------------------------------- |
| Token Embedding       | `V × d`                   | Shared with LM head if weight-tied |
| Positional Embedding  | `T × d`                   | Only if learned (0 for RoPE)       |
| Attention (per layer) | `4d² + 4d`                | W_q, W_k, W_v, W_o + biases        |
| FFN (per layer)       | `2 × d × d_ff + d + d_ff` | W_1, W_2 + biases                  |
| LayerNorm (per layer) | `4d`                      | 2 norms × (γ + β)                  |
| Final LayerNorm       | `2d`                      | γ + β                              |
| LM Head               | `d × V` or 0              | 0 if weight-tied                   |

**Total** = `V×d + T×d + L×(4d² + 4d + 2×d×d_ff + d + d_ff + 4d) + 2d + LM_head`

### 3.7 Configuration Examples

#### Small Config (~49M parameters)

```python
d_model=512, n_heads=8, n_layers=8, d_ff=2048, vocab=32K, seq_len=256
```

- Embedding: 16.4M | Attention (8 layers): 8.4M | FFN: 16.8M | Total: ~42M (+ norms)

#### Medium Config (~124M parameters)

```python
d_model=768, n_heads=12, n_layers=12, d_ff=3072, vocab=32K, seq_len=512
```

- Embedding: 24.6M | Attention (12 layers): 28.3M | FFN: 56.6M | Total: ~110M (+ norms)

Run `python scripts/count_params.py` for exact counts.

---

## Section 4 — Training Dynamics

### 4.1 Cross-Entropy Loss for Next-Token Prediction

Given model output logits `z ∈ ℝ^V` and true next token `y`:

```
L = -log P(y | z) = -log( exp(z_y) / Σ_j exp(z_j) ) = -z_y + log(Σ_j exp(z_j))
```

This is the negative log-likelihood of the correct token under the model's predicted distribution.

**Why cross-entropy?** It is the KL divergence between the true distribution (one-hot on correct token) and the model's predicted softmax distribution, up to a constant:

```
CE(p, q) = H(p) + D_KL(p || q)
```

Since H(p) = 0 for one-hot distributions, `CE = D_KL`. Minimizing CE = finding the distribution closest to the true one.

### 4.2 Teacher Forcing

During training, we always use the **true previous tokens** as input, not the model's own predictions:

```
Input:  [BOS]  "The"  "cat"  "sat"
Target: "The"  "cat"  "sat"  [EOS]
```

**Why it works**: Without teacher forcing (free-running), errors compound — one wrong prediction shifts all subsequent inputs. Teacher forcing provides a stable optimization signal at every position.

**Exposure bias**: The model never sees its own errors during training but encounters them at inference. This is mitigated by:

1. High-quality sampling (top-k, top-p) at inference
2. Scheduled sampling (gradually replacing ground truth with predictions during training)
3. For GPT-scale models, exposure bias is empirically minor

### 4.3 Perplexity

```
PPL = exp(H) = exp( -1/N × Σ_{i=1}^{N} log P(w_i | w_{<i}) )
```

Where H is the average cross-entropy loss over N tokens.

**Interpretation**: PPL = K means the model is "as confused as if choosing uniformly between K options at each step."

| PPL        | Quality                |
| ---------- | ---------------------- |
| 1          | Perfect prediction     |
| 20-30      | Good LM (GPT-2 level)  |
| 50-100     | Decent, early training |
| V (~32000) | Random guessing        |

### 4.4 AdamW Optimizer

**Update rule** at step t for parameter θ:

```
g_t = ∇L(θ_{t-1})                           # Gradient
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # First moment (momentum)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       # Second moment (RMSProp)
m̂_t = m_t / (1 - β₁^t)                      # Bias correction
v̂_t = v_t / (1 - β₂^t)                      # Bias correction
θ_t = θ_{t-1} - α × (m̂_t / (√v̂_t + ε) + λ × θ_{t-1})
                       ──────────────────   ──────────────
                       Adaptive gradient     Weight decay
```

Standard values: `β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1`

**Why β₂=0.95 (not 0.999)?** Lower β₂ makes the second moment more responsive to recent gradients, which improves stability for LLM training where gradient distribution shifts rapidly.

**Why decoupled weight decay (AdamW)?** In L2 regularization with Adam, the penalty gradient `2λθ` is also divided by `√v̂`. This means parameters with large gradient variance get less regularization. AdamW applies weight decay directly: `θ = θ - α×λ×θ`, uniformly across all parameters.

### 4.5 Learning Rate Schedule

**Phase 1: Linear Warmup** (steps 0 to W):

```
lr(t) = lr_peak × t / W
```

**Why warmup?** At initialization (step 0):

- Weights are random → gradients are noisy
- Adam's moment estimates (m, v) are zero → bias correction gives large effective LR
- Large LR + noisy gradients = gradient explosion

Warmup lets the optimizer "calibrate" (build up moment estimates) before reaching full learning rate.

**Phase 2: Cosine Decay** (steps W to T_max):

```
lr(t) = lr_min + 0.5 × (lr_peak - lr_min) × (1 + cos(π × (t - W) / (T_max - W)))
```

**Why cosine?** Compared to linear decay:

- Cosine starts slow (gentle initial decrease)
- Spends more time at higher LR (exploration)
- Drops quickly at end (fine-tuning behavior)
- Empirically gives 1-3% better final perplexity

### 4.6 Gradient Stability

**Gradient Explosion**: Gradients grow exponentially through layers. In Post-LN: `||g_l|| ∝ Π_{l} ||W_l||`. If `||W_l|| > 1`, gradients explode.

**Vanishing Gradients**: Gradient shrink to zero. Especially problematic for long sequences where the loss at position T must propagate through T attention steps.

**Gradient Clipping**:

```
total_norm = √(Σ ||∇θ_i||²)
if total_norm > max_norm:
    ∇θ_i ← ∇θ_i × (max_norm / total_norm)    ∀i
```

Typical `max_norm = 1.0`. This prevents rare high-loss batches from destroying training. Only activates when needed — normal gradient magnitudes pass through unchanged.

**Mixed Precision Training**:

- Forward pass: FP16 or BF16 (2x memory savings, faster matmuls)
- Loss + backward: FP16/BF16
- Parameter update: FP32 (for numerical precision in optimizer state)
- **Loss scaling**: Multiply loss by a large factor S before backward, divide gradients by S before optimizer step. Prevents small FP16 gradients from underflowing to zero.
- BF16 is preferred over FP16: same range as FP32 (avoids overflow), no loss scaling needed.

---

## Section 5 — Evaluation

### 5.1 Validation Tracking

Evaluate every N training steps on a held-out validation set:

1. Compute average cross-entropy loss over validation batches
2. Convert to perplexity: `PPL = exp(loss)`
3. Track for overfitting detection

**Overfitting signature**: Train loss decreasing while val loss increasing. If this occurs:

- Increase dropout (0.1 → 0.2)
- Increase weight decay (0.1 → 0.3)
- Add more training data
- Reduce model size

### 5.2 Sanity Checks

1. **Initial loss** ≈ `log(vocab_size)` ≈ 10.4 for V=32K. If significantly different, there's a bug.
2. **Overfit one batch**: Train on 8 sequences for 100 steps. Loss should drop to near 0. If not, the model can't learn at all.
3. **Gradient norms**: Should be finite, non-zero, and typically 0.1–10.0 after clipping.

### 5.3 Sampling Strategy Mathematics

See [sampling.py](file://llm-train/evaluation/sampling.py) for implementations.

**Temperature**: `P(token_i) = exp(z_i / T) / Σ exp(z_j / T)`. Entropy of distribution is monotonically increasing in T.

**Top-k**: Keep only k highest logits, renormalize. Simple but inflexible — k=50 keeps too many low-probability tokens when the model is confident.

**Top-p (Nucleus)**: Adaptively selects the smallest set of tokens whose cumulative probability ≥ p. When confident, might keep only 3 tokens. When uncertain, might keep 500. This adaptivity makes it superior to top-k in practice.

**Recommended**: Temperature=0.8, top-p=0.9 for general text. Temperature=0.3, top-k=10 for factual/deterministic output.

---

## Section 6 — Systems & Optimization

### 6.1 Memory Breakdown

GPU memory during training consists of:

| Component           | Formula                    | 50M model | 120M model |
| ------------------- | -------------------------- | --------- | ---------- |
| **Parameters**      | P × 4 bytes (FP32)         | 200 MB    | 480 MB     |
| **Gradients**       | P × 4 bytes                | 200 MB    | 480 MB     |
| **Optimizer State** | P × 8 bytes (AdamW: m + v) | 400 MB    | 960 MB     |
| **Activations**     | ~BxTxdx12xL bytes (FP16)   | ~1-4 GB   | ~3-10 GB   |
| **Total**           |                            | ~2-5 GB   | ~5-12 GB   |

**Activations dominate** for large batch sizes and long sequences. Activations scale as O(B × T × d_model × L).

### 6.2 VRAM Estimation Formula

```
VRAM ≈ Params_bytes + Grads_bytes + Optimizer_bytes + Activations_bytes

Params_bytes = P × bytes_per_param  (4 for FP32, 2 for FP16/BF16)
Grads_bytes  = P × bytes_per_grad   (same as params)
Optim_bytes  = P × 8               (AdamW stores m and v in FP32)
Act_bytes    ≈ B × T × d_model × L × 12 × 2  (FP16, rule of thumb)
```

For the medium config (120M params, B=32, T=512, d=768, L=12):

```
Params:  120M × 4 = 480 MB
Grads:   120M × 4 = 480 MB
Optim:   120M × 8 = 960 MB
Acts:    32 × 512 × 768 × 12 × 12 × 2 ≈ 3.6 GB
Total:   ≈ 5.5 GB
```

### 6.3 Gradient Checkpointing

**Problem**: Activations memory grows linearly with number of layers (must store intermediate outputs of every layer for backward pass).

**Solution**: Only store activations at checkpointed layers. During backward, recompute activations of non-checkpointed layers from the nearest checkpoint.

**Trade-off**: Saves ~60-70% activation memory at the cost of ~33% more compute (each layer's forward pass runs twice: once in forward, once in backward for recomputation).

**PyTorch**:

```python
from torch.utils.checkpoint import checkpoint
# In TransformerBlock.forward:
x = checkpoint(self.attn, self.ln1(x))  # Recompute on backward
```

### 6.4 FlashAttention (Dao et al., 2022)

Standard attention materializes the full T×T attention matrix → O(T²) memory.

FlashAttention computes attention **in tiles** using the online softmax trick:

1. Split Q, K, V into blocks that fit in SRAM (fast on-chip memory)
2. Compute partial softmax within each block
3. Merge results using the online softmax correction factor
4. Never materialize the full T×T matrix in HBM (slow global memory)

**Result**: O(T) memory instead of O(T²), and 2-4x faster due to better memory access patterns.

**PyTorch** (v2.0+):

```python
from torch.nn.functional import scaled_dot_product_attention
# Automatically uses FlashAttention when available
attn_output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 6.5 Parallelism Strategies

| Strategy                | What's Parallelized                          | When to Use                        |
| ----------------------- | -------------------------------------------- | ---------------------------------- |
| **Data Parallel (DDP)** | Replicate model on each GPU, split batch     | Model fits on 1 GPU                |
| **FSDP (ZeRO Stage 3)** | Shard params + grads + optimizer across GPUs | Model barely fits on 1 GPU         |
| **Tensor Parallel**     | Shard individual weight matrices across GPUs | Very large layers (d_model > 4096) |
| **Pipeline Parallel**   | Assign different layers to different GPUs    | Very deep models (>100 layers)     |

**FSDP / DeepSpeed ZeRO**:

- **Stage 1**: Shard optimizer state → saves 4× per GPU
- **Stage 2**: + Shard gradients → saves 2× more
- **Stage 3**: + Shard parameters → saves 2× more. Requires all-gather before each forward.

**Recommendation for 50-150M models**: DDP is sufficient. A single A100 (80GB) can train a 120M model with B=32, T=512. Two consumer GPUs (24GB each) with DDP can train with B=16 each.

---

## Section 7 — Scaling Laws

### 7.1 Chinchilla Scaling Laws (Hoffmann et al., 2022)

The compute-optimal frontier is described by:

```
L(N, D) = A/N^α + B/D^β + L_∞
```

Where:

- `L` = validation loss (cross-entropy)
- `N` = number of parameters
- `D` = number of training tokens
- `A, B, α, β, L_∞` = fitted constants
- `α ≈ 0.34, β ≈ 0.28`

**Key insight**: For a given compute budget C ∝ 6ND (FLOPs ≈ 6 × params × tokens):

```
Optimal N* ∝ C^0.50
Optimal D* ∝ C^0.50
```

Both model size and data should scale equally with compute. The "Chinchilla optimal" ratio is approximately **20 tokens per parameter**.

### 7.2 Compute-Optimal Sizing

Given a GPU budget:

| GPU Hours (A100) | Compute (FLOPs) | Optimal Params | Optimal Tokens |
| ---------------- | --------------- | -------------- | -------------- |
| 10               | 3.1 × 10¹⁸      | ~25M           | 500M           |
| 100              | 3.1 × 10¹⁹      | ~80M           | 1.6B           |
| 1000             | 3.1 × 10²⁰      | ~250M          | 5B             |

**FLOPs per A100 hour**: ~312 TFLOPs × 3600s ≈ 1.1 × 10¹⁸ (at ~30% utilization)

### 7.3 Underfitting vs Overfitting Detection

| Signal                                          | Diagnosis                | Action                        |
| ----------------------------------------------- | ------------------------ | ----------------------------- |
| Train loss plateaus, val loss also plateau high | **Underfitting**         | Increase model size or LR     |
| Train loss decreasing, val loss increasing      | **Overfitting**          | Add data, increase dropout/wd |
| Train loss = val loss, both decreasing          | **On track**             | Continue training             |
| Train loss spikes then recovers                 | **Gradient instability** | Reduce LR, increase grad clip |
| Train loss goes to NaN                          | **Training collapse**    | See Section 9                 |

---

## Section 8 — Inference Engineering

### 8.1 KV Caching

See [generate.py](file://llm-train/inference/generate.py).

**Without cache**: At step t, recompute attention for ALL t tokens. Cost per step: O(t × d²). Total for N tokens: O(N² × d²).

**With cache**: Store K and V matrices from all previous steps. Only compute Q for the new token. Cost per step: O(d²) for QKV projection + O(t × d) for attention. Total: O(N × d²).

**Memory**: Cache stores (K, V) for each layer, each of shape (batch, n_heads, seq_len, d_head). Total: `2 × L × B × H × T × d_h × bytes_per_element`.

### 8.2 Throughput vs Latency

- **Latency** = time to generate first token (TTFT) + time per subsequent token
- **Throughput** = total tokens generated per second across all requests

These conflict:

- Lower latency → smaller batches, less GPU utilization
- Higher throughput → larger batches, higher latency per request
- Production: typically optimize for P99 latency with maximum throughput

### 8.3 Quantization (INT8)

```python
# Post-training quantization concept:
scale = max(|W|) / 127
W_int8 = round(W / scale)  # 4x memory reduction
# During inference: output = (W_int8 × scale) @ x
```

INT8 quantization typically has <1% perplexity degradation for models >100M params. For smaller models, the degradation can be 2-5%.

### 8.4 FastAPI Server

See [server.py](file://llm-train/inference/server.py) and [Dockerfile](file://llm-train/inference/Dockerfile).

```bash
# Run locally
python inference/server.py

# Build and run with Docker
docker build -t gpt-server -f inference/Dockerfile .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints gpt-server

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 50, "temperature": 0.8}'
```

---

## Section 9 — Failure Modes & Debugging

### 9.1 Training Collapse

**Symptoms**: Loss spikes to very high values and never recovers, or drops to 0 (degenerate solution).

**Common causes**:

1. Learning rate too high → reduce by 3-10x
2. No warmup → add 1-2K warmup steps
3. Numerical overflow in softmax → check for inf/nan in attention weights
4. Weight initialization too large → use std=0.02

### 9.2 Repeating Tokens

**Symptoms**: Model generates the same token or phrase repeatedly.

**Causes**:

1. Temperature too low → increase to 0.7-1.0
2. No top-p/top-k filtering → enable nucleus sampling
3. Model undertrained → train longer
4. Degenerate attention patterns → check attention entropy

### 9.3 Mode Collapse

**Symptoms**: Model only generates a few stereotypical phrases regardless of prompt.

**Causes**:

1. Dataset too small or repetitive → more diverse data
2. Overfitting → increase regularization
3. Learning rate too low (stuck in local minimum) → restart with higher LR

### 9.4 NaN Loss Debugging

**Step-by-step checklist**:

1. Check for NaN in inputs: `assert torch.isfinite(x).all()`
2. Check for NaN in embeddings: `assert torch.isfinite(model.token_embedding.weight).all()`
3. Check attention scores before softmax: look for very large values (> 1e4)
4. Check loss value: `assert torch.isfinite(loss)`
5. After backward, check gradients: `for p in model.parameters(): assert torch.isfinite(p.grad).all()`
6. Check optimizer state for NaN: `for group in optimizer.param_groups: for p in group['params']: assert torch.isfinite(optimizer.state[p]['exp_avg']).all()`

**Common fixes**: Reduce LR, add gradient clipping, switch to BF16, check data for garbage tokens.

### 9.5 Gradient Verification

```python
# Numerical gradient check (expensive, use for debugging only)
# For parameter p, loss L:
eps = 1e-5
for i in range(min(10, p.numel())):  # Check first 10 elements
    p_flat = p.view(-1)
    original = p_flat[i].item()

    p_flat[i] = original + eps
    L_plus = compute_loss()

    p_flat[i] = original - eps
    L_minus = compute_loss()

    p_flat[i] = original

    numerical_grad = (L_plus - L_minus) / (2 * eps)
    analytical_grad = p.grad.view(-1)[i].item()

    relative_error = abs(numerical_grad - analytical_grad) / max(abs(numerical_grad), abs(analytical_grad), 1e-8)
    assert relative_error < 1e-4, f"Gradient mismatch: {relative_error}"
```

---

## Final Checklists

### Step-by-Step Roadmap

1. **Environment Setup**: `pip install -r requirements.txt`
2. **Download Data**: `python data/download.py` → download WikiText-103
3. **Train Tokenizer**: `python tokenizer/train_tokenizer.py --input data/raw/wiki.train.raw --vocab-size 32000`
4. **Tokenize Data**: Use `data/dataset.py:tokenize_and_save()` to create binary token files
5. **Sanity Check**: `python training/train.py --config small --sanity` → verify loss decreases
6. **Run Tests**: `python -m pytest evaluation/tests.py -v`
7. **Full Training**: `python training/train.py --config medium --data-path data/processed/train.bin --val-path data/processed/val.bin`
8. **Generate Text**: Use `evaluation/sampling.py:generate_text()`
9. **Serve**: `python inference/server.py`

### Debugging Checklist

- [ ] Initial loss ≈ log(vocab_size)?
- [ ] Can overfit one batch to loss < 0.1?
- [ ] Gradient norms are finite and non-zero?
- [ ] Learning rate schedule looks correct (warmup → cosine)?
- [ ] Validation loss is tracked and not increasing (no overfitting)?
- [ ] Generated text is coherent (not random or repeating)?
- [ ] Memory usage is within GPU limits?
- [ ] Checkpoints are being saved?

### Final Readiness Checklist

- [ ] Model architecture matches design (param count, dimensions)
- [ ] Tokenizer roundtrips correctly (encode → decode = original)
- [ ] Training completes without errors or NaN
- [ ] Validation perplexity is reasonable (< 100 for WikiText-103)
- [ ] Generated text is coherent and grammatically correct
- [ ] KV cache produces same outputs as non-cached
- [ ] FastAPI server responds to requests
- [ ] Docker image builds and runs
