"""
Prepare FineWeb-Edu dataset for training.

Downloads and tokenizes FineWeb-Edu samples, saving as train.bin and val.bin
in uint16 format for memory-mapped loading during training.

Based on nanoGPT's data preparation pipeline.

Usage:
    python prepare.py [--num_proc 8] [--shard_size 100000000]
"""

import os
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def tokenize(doc, tokenizer):
    """Tokenize a single document."""
    tokens = [tokenizer.eot_token]  # Start with end-of-text token
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def write_datafile(filename, tokens_np):
    """Write tokens to a binary file."""
    np.save(filename, tokens_np)


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument(
        "--num_proc", type=int, default=max(1, mp.cpu_count() // 2),
        help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--shard_size", type=int, default=10**8,
        help="Size of each shard in tokens (default: 100M)"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_subset", type=str, default="sample-10BT",
        help="Dataset subset/config (default: sample-10BT for ~10B tokens)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.01,
        help="Fraction of data to use for validation"
    )
    args = parser.parse_args()

    # Initialize tokenizer (GPT-2)
    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_subset})")
    dataset = load_dataset(
        args.dataset_name,
        name=args.dataset_subset,
        split="train",
        trust_remote_code=True,
    )

    # Tokenize with multiprocessing
    print(f"Tokenizing with {args.num_proc} processes...")
    tokenize_fn = partial(tokenize, tokenizer=tokenizer)

    all_tokens = []
    with mp.Pool(args.num_proc) as pool:
        for tokens in tqdm(
            pool.imap(tokenize_fn, dataset, chunksize=100),
            total=len(dataset),
            desc="Tokenizing"
        ):
            all_tokens.append(tokens)

    # Concatenate all tokens
    print("Concatenating tokens...")
    all_tokens = np.concatenate(all_tokens)
    total_tokens = len(all_tokens)
    print(f"Total tokens: {total_tokens:,}")

    # Split into train/val
    val_size = int(total_tokens * args.val_ratio)
    train_size = total_tokens - val_size

    print(f"Train tokens: {train_size:,}")
    print(f"Val tokens: {val_size:,}")

    # Shuffle and split
    print("Shuffling...")
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(total_tokens)

    # Actually, for language modeling we don't shuffle - we keep sequences contiguous
    # Just split at the boundary
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]

    # Save as memmap-compatible binary files
    output_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    print(f"Writing {train_path}...")
    train_tokens.astype(np.uint16).tofile(train_path)

    print(f"Writing {val_path}...")
    val_tokens.astype(np.uint16).tofile(val_path)

    print("Done!")
    print(f"  train.bin: {os.path.getsize(train_path) / 1e9:.2f} GB")
    print(f"  val.bin: {os.path.getsize(val_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
