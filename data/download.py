"""
download.py — Dataset download and preparation helpers
=======================================================
Provides download URLs and utilities for common pre-training corpora.
"""

import os
import urllib.request
import zipfile
from typing import Optional, Iterator


DATASET_URLS = {
    "wikitext-103": {
        "description": "WikiText-103: Long-range dependency benchmark (103M tokens)",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "fallback_url": "https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz",
        "size": "~500MB",
        "note": "Use: load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')",
    },
    "openwebtext": {
        "description": "OpenWebText: Open-source replication of WebText (Reddit-filtered)",
        "url": "https://huggingface.co/datasets/Skylion007/openwebtext",
        "size": "~12GB compressed",
        "note": "Use: load_dataset('Skylion007/openwebtext')",
    },
    "the-pile-subset": {
        "description": "The Pile: 800GB diverse text (deduplicated subset)",
        "url": "https://huggingface.co/datasets/monology/pile-uncopyrighted",
        "size": "~200GB subset",
        "note": "Use: load_dataset('monology/pile-uncopyrighted', streaming=True)",
    },
    "github-code": {
        "description": "GitHub Code: Permissively-licensed source code",
        "url": "https://huggingface.co/datasets/codeparrot/github-code",
        "size": "~1TB",
        "note": "Use with streaming and language filter",
    },
}

DOC_SEPARATOR = "\n<|endofdoc|>\n"


def _progress_hook(count, block_size, total_size):
    """Progress callback for urllib downloads."""
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Downloading: {percent}%", end="", flush=True)


def download_wikitext103(output_dir: str = "data/raw") -> str:
    """
    Download WikiText-103 via HuggingFace datasets.
    Returns path to the saved training text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "wiki.train.raw")
    valid_file = os.path.join(output_dir, "wiki.valid.raw")

    if os.path.exists(train_file):
        print(f"Found cached {train_file}")
        return train_file

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` package to download WikiText-103:\n"
            "  pip install datasets"
        )

    print("Downloading WikiText-103 from HuggingFace...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    # Save train split
    print(f"Saving training data to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for example in ds["train"]:
            f.write(example["text"])
    size_mb = os.path.getsize(train_file) / 1e6
    print(f"  Train: {size_mb:.1f} MB")

    # Save validation split
    print(f"Saving validation data to {valid_file}...")
    with open(valid_file, "w", encoding="utf-8") as f:
        for example in ds["validation"]:
            f.write(example["text"])
    size_mb = os.path.getsize(valid_file) / 1e6
    print(f"  Valid: {size_mb:.1f} MB")

    return train_file


def load_openwebtext_streaming(max_examples: Optional[int] = None) -> Iterator[str]:
    """Load OpenWebText via HuggingFace datasets with streaming."""
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for i, example in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        yield example["text"]


def prepare_text_file(texts: Iterator[str], output_path: str) -> int:
    """
    Write an iterator of text documents to a single file with document separators.
    Returns the number of documents written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            text = text.strip()
            if not text:
                continue
            if count > 0:
                f.write(DOC_SEPARATOR)
            f.write(text)
            count += 1
            if count % 10_000 == 0:
                print(f"\r  Wrote {count:,} documents", end="", flush=True)
    print(f"\n  Total: {count:,} documents -> {output_path}")
    return count


def clean_text(text: str) -> str:
    """
    Basic text cleaning for pre-training data.

    Applied filters:
    1. Strip leading/trailing whitespace
    2. Normalize unicode whitespace
    3. Remove documents shorter than 50 chars (low quality)
    4. Remove documents that are mostly non-alphanumeric (boilerplate)
    """
    import unicodedata

    text = text.strip()
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple newlines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    # Filter: too short
    if len(text) < 50:
        return ""
    # Filter: low alphanumeric ratio (likely boilerplate/navigation)
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and alnum / len(text) < 0.3:
        return ""
    return text


if __name__ == "__main__":
    print("Available datasets:")
    for name, info in DATASET_URLS.items():
        print(f"  {name}: {info['description']}")
        print(f"    URL:  {info['url']}")
        print(f"    Size: {info['size']}")
        if "note" in info:
            print(f"    Note: {info['note']}")
        print()
