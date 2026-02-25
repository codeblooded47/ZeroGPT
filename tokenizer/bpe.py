"""
bpe.py — Byte Pair Encoding Tokenizer from Scratch
====================================================
A complete, research-grade BPE tokenizer built in pure Python.

Algorithm (formalized):
  1. Initialize vocabulary V = set of all unique bytes (256 entries)
  2. Represent corpus as sequences of byte tokens
  3. Repeat until |V| = target_vocab_size:
     a. Count all adjacent token pairs (bigrams) in corpus
     b. Select pair (a, b) with maximum frequency: (a*, b*) = argmax_{(a,b)} count(a, b)
     c. Merge all occurrences of (a*, b*) -> new token t = a*b*
     d. Add t to V
     e. Record merge rule: (a*, b*) -> t

Time complexity:  O(N x M) where N = corpus size in bytes, M = num merges
Space complexity: O(N + |V|^2) for pair counts

Why BPE works:
  - Morphologically aware: common subwords ("ing", "tion") merge early
  - Open vocabulary: any byte sequence can be tokenized (no OOV)
  - Compression: frequent words become single tokens, reducing sequence length
  - Zipf-aligned: merge frequency follows Zipf's law naturally
"""

import json
import os
import regex as re
from collections import Counter, OrderedDict
from typing import Optional


# GPT-style pre-tokenization regex (splits on word boundaries, whitespace, etc.)
# This prevents merges across word boundaries (e.g., "the" + " dog" won't merge)
GPT_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Number of special tokens reserved at the start of vocabulary
NUM_SPECIAL_TOKENS = 4
SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|bos|>": 2,
    "<|eos|>": 3,
}


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Special tokens:
        <|pad|>  (id=0)  Padding token
        <|unk|>  (id=1)  Unknown token (fallback, rarely used in byte-level BPE)
        <|bos|>  (id=2)  Beginning of sequence
        <|eos|>  (id=3)  End of sequence

    Byte tokens occupy ids [4, 259] (256 bytes).
    Merged tokens start at id 260 and go up to vocab_size - 1.
    """

    def __init__(self, vocab_size: int = 32_000):
        assert vocab_size >= 256 + NUM_SPECIAL_TOKENS, \
            f"vocab_size must be >= {256 + NUM_SPECIAL_TOKENS}"
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256 - NUM_SPECIAL_TOKENS

        # Merge rules: list of (token_a, token_b) pairs in order they were learned
        self.merges: list[tuple[int, int]] = []
        # Vocabulary: token_id -> bytes
        self.vocab: dict[int, bytes] = {}
        # Inverse vocabulary: bytes -> token_id
        self.inverse_vocab: dict[bytes, int] = {}

        self._build_base_vocab()

    def _build_base_vocab(self):
        """Initialize vocab with special tokens + 256 byte tokens."""
        self.vocab = {}
        self.inverse_vocab = {}

        # Special tokens (stored as their string encoding)
        for token_str, token_id in SPECIAL_TOKENS.items():
            token_bytes = token_str.encode("utf-8")
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

        # Byte tokens: each single byte gets its own token
        for b in range(256):
            token_id = b + NUM_SPECIAL_TOKENS
            token_bytes = bytes([b])
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

    def train(self, text: str, verbose: bool = True):
        """
        Train BPE on a text corpus.

        Optimized algorithm using frequency-weighted unique words:
        1. Pre-tokenize text using GPT-style regex (word-level splitting)
        2. Count unique words and their frequencies
        3. Work only on unique words (~500K) instead of all occurrences (~100M)
        4. Weight pair counts by word frequency

        This reduces memory from O(N) to O(unique_words) and gives 100x+ speedup.
        """
        import time
        t0 = time.time()

        # Step 1: Pre-tokenize and count unique words
        if verbose:
            print("Pre-tokenizing...")
        chunks = re.findall(GPT_PRETOKENIZE_PATTERN, text)
        if verbose:
            print(f"  {len(chunks):,} chunks from {len(text):,} chars")

        # Step 2: Count unique words → frequency
        # Instead of storing 104M token lists, store ~500K unique words with counts
        if verbose:
            print("Counting unique words...")
        word_freqs = Counter(chunks)
        if verbose:
            print(f"  {len(word_freqs):,} unique words (reduced from {len(chunks):,})")

        # Step 3: Convert unique words to byte-level token tuples
        # Key: tuple of token ids, Value: frequency
        word_table: dict[tuple[int, ...], int] = {}
        for word_str, freq in word_freqs.items():
            byte_seq = word_str.encode("utf-8")
            token_tuple = tuple(b + NUM_SPECIAL_TOKENS for b in byte_seq)
            # Multiple words can map to same byte sequence — add frequencies
            word_table[token_tuple] = word_table.get(token_tuple, 0) + freq

        del chunks, word_freqs  # Free memory

        if verbose:
            elapsed = time.time() - t0
            print(f"  Preprocessing done in {elapsed:.1f}s")
            print(f"  Working set: {len(word_table):,} unique byte sequences")
            print(f"  Starting {self.num_merges:,} merges...")

        # Step 4: Iterative merging on the compact word table
        next_id = 256 + NUM_SPECIAL_TOKENS  # = 260, first merge token

        for i in range(self.num_merges):
            # Count pairs weighted by word frequency
            pair_counts: dict[tuple[int, int], int] = {}
            for word, freq in word_table.items():
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                if verbose:
                    print(f"  No more pairs at merge {i}. Stopping.")
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                if verbose:
                    print(f"  Best pair count < 2 at merge {i}. Stopping.")
                break

            # Record merge
            self.merges.append(best_pair)

            # Create new token
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[next_id] = new_bytes
            self.inverse_vocab[new_bytes] = next_id

            # Apply merge to word table: replace (a, b) with new_id in all words
            a, b = best_pair
            new_word_table: dict[tuple[int, ...], int] = {}
            for word, freq in word_table.items():
                new_word = self._merge_pair_in_word(word, a, b, next_id)
                new_word_table[new_word] = new_word_table.get(new_word, 0) + freq
            word_table = new_word_table

            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (self.num_merges - i - 1) / rate if rate > 0 else 0
                pct = (i + 1) / self.num_merges * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"  [{bar}] {pct:5.1f}% | Merge {i + 1:>5d}/{self.num_merges} | "
                      f"{self.vocab[best_pair[0]]!r}+{self.vocab[best_pair[1]]!r} "
                      f"(count={best_count:,}) | "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left | "
                      f"{len(word_table):,} unique words")

            next_id += 1

        elapsed = time.time() - t0
        if verbose:
            print(f"Training complete in {elapsed:.1f}s. "
                  f"Vocabulary size: {len(self.vocab)}")

    @staticmethod
    def _merge_pair_in_word(word: tuple, a: int, b: int, new_id: int) -> tuple:
        """Replace all (a, b) pairs with new_id in a word tuple."""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                result.append(new_id)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token ids.

        Process:
        1. Pre-tokenize with GPT regex
        2. Convert each chunk to byte-level tokens
        3. Apply learned merges in order (greedy left-to-right)
        4. Optionally prepend <|bos|> and append <|eos|>
        """
        chunks = re.findall(GPT_PRETOKENIZE_PATTERN, text)
        all_ids = []

        if add_special_tokens:
            all_ids.append(SPECIAL_TOKENS["<|bos|>"])

        for chunk in chunks:
            byte_seq = chunk.encode("utf-8")
            ids = [b + NUM_SPECIAL_TOKENS for b in byte_seq]

            # Apply each merge rule in order
            for pair in self.merges:
                new_id = self.inverse_vocab.get(
                    self.vocab[pair[0]] + self.vocab[pair[1]]
                )
                if new_id is None:
                    continue
                ids = self._apply_single_merge(ids, pair, new_id)

            all_ids.extend(ids)

        if add_special_tokens:
            all_ids.append(SPECIAL_TOKENS["<|eos|>"])

        return all_ids

    def _apply_single_merge(self, ids: list[int], pair: tuple[int, int],
                            new_id: int) -> list[int]:
        """Apply a single merge rule to a list of token ids."""
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token ids back to text.

        Each token id maps to its byte sequence; concatenate all bytes
        and decode as UTF-8 (with error replacement for invalid sequences).
        """
        byte_chunks = []
        special_ids = set(SPECIAL_TOKENS.values())

        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                # Unknown token — should not happen in byte-level BPE
                byte_chunks.append(b"?")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": SPECIAL_TOKENS,
            # Store merges as list of [a, b] pairs
            "merges": [[a, b] for a, b in self.merges],
            # Store vocab as {id: hex-encoded bytes}
            "vocab": {
                str(k): v.hex() for k, v in self.vocab.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab = {
            int(k): bytes.fromhex(v) for k, v in data["vocab"].items()
        }
        self.inverse_vocab = {v: int(k) for k, v in data["vocab"].items()}
        print(f"Tokenizer loaded from {path} (vocab_size={len(self.vocab)})")

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<|eos|>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<|unk|>"]


# ─────────────────────────────────────────────────────────
#  Quick demo
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train a tiny tokenizer on sample text
    sample_text = (
        "The transformer architecture revolutionized natural language processing. "
        "Self-attention allows the model to attend to all positions simultaneously. "
        "The transformer uses multi-head attention for richer representations. "
        "Training transformers requires careful learning rate scheduling. "
    ) * 100  # Repeat to get enough frequency signal

    tokenizer = BPETokenizer(vocab_size=300)  # Small for demo
    tokenizer.train(sample_text, verbose=True)

    # Encode / Decode roundtrip
    test = "The transformer model processes tokens."
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal:  {test!r}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded!r}")
    print(f"Roundtrip: {'PASS' if decoded == test else 'FAIL'}")
    print(f"Compression: {len(test)} chars -> {len(encoded)} tokens "
          f"({len(encoded)/len(test):.2f} tokens/char)")
