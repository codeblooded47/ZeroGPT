"""
sp_wrapper.py — SentencePiece Tokenizer Wrapper
==================================================
Wraps Google's SentencePiece tokenizer with the same interface as our
custom BPETokenizer, so all downstream code (dataset, training, inference)
works without changes.

Usage:
    from tokenizer.sp_wrapper import SentencePieceTokenizer
    tok = SentencePieceTokenizer("tokenizer/sp_bpe.model")
    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

import sentencepiece as spm
from typing import Optional


class SentencePieceTokenizer:
    """
    Wrapper around SentencePiece that matches BPETokenizer's interface.

    SentencePiece special token IDs (defaults):
        <unk>  = 0
        <s>    = 1  (BOS)
        </s>   = 2  (EOS)

    We map these to match our convention:
        pad_id = -1 (sentencepiece doesn't have pad by default, we use a custom one)
        unk_id = 0
        bos_id = 1
        eos_id = 2
    """

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path
        self._vocab_size = self.sp.GetPieceSize()
        print(f"SentencePiece tokenizer loaded: {model_path} "
              f"(vocab_size={self._vocab_size})")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        pad = self.sp.PieceToId("<pad>")
        return pad if pad != self.sp.unk_id() else 0

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ids = self.sp.Encode(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if skip_special_tokens:
            special = {self.bos_id, self.eos_id, self.unk_id, self.pad_id}
            ids = [i for i in ids if i not in special]
        return self.sp.Decode(ids)

    def save(self, path: str):
        """No-op — SentencePiece model is already saved as .model file."""
        print(f"SentencePiece model is at: {self.model_path}")

    def load(self, path: str):
        """Load a SentencePiece .model file."""
        self.sp.Load(path)
        self.model_path = path
        self._vocab_size = self.sp.GetPieceSize()
        print(f"SentencePiece tokenizer loaded: {path} "
              f"(vocab_size={self._vocab_size})")


def load_tokenizer(path: str):
    """
    Auto-detect and load the right tokenizer.

    - .model file → SentencePiece
    - .json file  → our custom BPETokenizer
    """
    if path.endswith(".model"):
        return SentencePieceTokenizer(path)
    elif path.endswith(".json"):
        from tokenizer.bpe import BPETokenizer
        tok = BPETokenizer()
        tok.load(path)
        return tok
    else:
        raise ValueError(f"Unknown tokenizer format: {path}. Use .model or .json")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer/sp_bpe.model"
    tok = load_tokenizer(path)

    tests = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "def train(model, data):",
    ]

    print(f"\nTokenizer: {path}")
    print(f"Vocab size: {tok.vocab_size}")
    print(f"BOS={tok.bos_id}, EOS={tok.eos_id}, PAD={tok.pad_id}\n")

    for text in tests:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        status = "PASS" if decoded.strip() == text else "FAIL"
        print(f"  [{status}] {text!r}")
        print(f"         -> {len(ids)} tokens: {ids[:10]}{'...' if len(ids) > 10 else ''}")
        if decoded.strip() != text:
            print(f"         decoded: {decoded!r}")
