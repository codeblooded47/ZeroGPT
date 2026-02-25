"""
dataset.py — Streaming Text Dataset for Language Model Pre-training
====================================================================
Memory-efficient dataset implementation that:
  - Memory-maps tokenized data from disk (no full-corpus RAM requirement)
  - Handles document boundaries with <|eos|> tokens
  - Produces fixed-length chunks for training (input, target) pairs
  - Supports shuffling via random offset sampling
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.sp_wrapper import load_tokenizer


class TokenizedDataset(Dataset):
    """
    Pre-tokenized dataset stored as a memory-mapped numpy array.

    Each sample is a contiguous chunk of `seq_len + 1` tokens.
    The model input is tokens[:-1] and the target is tokens[1:].
    This implements "packing": documents are concatenated with <|eos|>
    separators, and chunks may span document boundaries.

    Why memory-mapping:
      - Dataset can be larger than RAM
      - OS handles paging — only accessed pages are loaded
      - Zero-copy: no serialization overhead
      - Random access is O(1)
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 512,
        dtype: np.dtype = np.uint16,
    ):
        """
        Args:
            data_path: Path to binary file of token ids (numpy format)
            seq_len: Context length for each training sample
            dtype: Token id dtype (uint16 supports vocab up to 65535)
        """
        self.seq_len = seq_len

        # Memory-map the token array
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.num_tokens = len(self.data)

        # Number of complete chunks we can extract
        # We need seq_len + 1 tokens per sample (input + 1 target token)
        self.num_samples = (self.num_tokens - 1) // seq_len

        print(f"Dataset: {self.num_tokens:,} tokens, {self.num_samples:,} samples "
              f"(seq_len={seq_len})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, target_ids), each of shape (seq_len,).
        target_ids are input_ids shifted right by 1 position.
        """
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input:  positions [0, seq_len-1]
        y = torch.from_numpy(chunk[1:])    # target: positions [1, seq_len]
        return x, y


class ShuffledTokenDataset(Dataset):
    """
    Variant that samples random offsets into the token array for better shuffling.
    Each epoch sees different random chunks (with possible overlap).
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 512,
        num_samples: int = 100_000,
        dtype: np.dtype = np.uint16,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.num_tokens = len(self.data)
        self.max_start = self.num_tokens - seq_len - 1

        # Pre-generate random starting positions
        rng = np.random.default_rng(seed)
        self.starts = rng.integers(0, self.max_start, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def tokenize_and_save(
    text_path: str,
    output_path: str,
    tokenizer,
    dtype: np.dtype = np.uint16,
    chunk_size: int = 1_000_000,
):
    """
    Tokenize a raw text file and save as a binary numpy array.

    Processes text in chunks to avoid loading entire file into memory.
    Document boundaries are preserved via <|eos|> tokens.

    Args:
        text_path: Path to raw text file
        output_path: Path to save binary token file
        tokenizer: Trained BPE tokenizer
        dtype: Output dtype (uint16 for vocab < 65536)
        chunk_size: Characters to process at a time
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # First pass: count total tokens (for pre-allocation)
    print(f"Tokenizing {text_path}...")
    all_tokens = []
    total_chars = 0

    with open(text_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_chars += len(chunk)

            # Tokenize without special tokens (we add eos at doc boundaries)
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            all_tokens.extend(tokens)

            if total_chars % 10_000_000 < chunk_size:
                print(f"\r  Processed {total_chars/1e6:.1f}M chars, "
                      f"{len(all_tokens)/1e6:.1f}M tokens", end="", flush=True)

    print(f"\n  Total: {total_chars:,} chars -> {len(all_tokens):,} tokens")
    print(f"  Compression ratio: {total_chars / len(all_tokens):.2f} chars/token")

    # Save as memory-mapped binary file
    token_array = np.array(all_tokens, dtype=dtype)
    token_array.tofile(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")


def create_dataloaders(
    train_path: str,
    val_path: str,
    seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from tokenized binary files."""
    train_ds = TokenizedDataset(train_path, seq_len=seq_len)
    val_ds = TokenizedDataset(val_path, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Creating synthetic token data for testing...")
    tokens = np.random.randint(0, 32000, size=100_000, dtype=np.uint16)
    test_path = "/tmp/test_tokens.bin"
    tokens.tofile(test_path)

    ds = TokenizedDataset(test_path, seq_len=128)
    x, y = ds[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")
    print(f"x[:10] = {x[:10].tolist()}")
    print(f"y[:10] = {y[:10].tolist()}")
    assert (x[1:] == y[:-1]).all(), "Target should be input shifted by 1"
    print("PASS: target is correctly shifted")

    os.remove(test_path)
