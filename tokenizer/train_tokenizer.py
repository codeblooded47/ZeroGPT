"""
train_tokenizer.py — Script to train BPE tokenizer on a dataset
================================================================
Usage:
    python tokenizer/train_tokenizer.py --input data/raw/corpus.txt --vocab-size 32000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw text file for training")
    parser.add_argument("--vocab-size", type=int, default=32_000,
                        help="Target vocabulary size (default: 32000)")
    parser.add_argument("--output", type=str, default="tokenizer/bpe_tokenizer.json",
                        help="Output path for trained tokenizer")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Max characters to read from input (for testing)")
    args = parser.parse_args()

    print(f"Loading training text from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        if args.max_chars:
            text = f.read(args.max_chars)
        else:
            text = f.read()
    print(f"  {len(text):,} characters loaded")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    print(f"\nTraining BPE tokenizer (target vocab_size={args.vocab_size})...")
    t0 = time.time()
    tokenizer.train(text, verbose=True)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    tokenizer.save(args.output)

    # Quick validation
    print("\n--- Validation ---")
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "def train(model, data):",
    ]
    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        status = "PASS" if decoded == s else "FAIL"
        print(f"  [{status}] {s!r} -> {len(ids)} tokens")
        if decoded != s:
            print(f"         Got: {decoded!r}")


if __name__ == "__main__":
    main()
