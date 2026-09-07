#!/usr/bin/env python3
"""Measure how residual-quantized ID positions organize a candidate index.

The script has two modes:
1. checkpoint-only: reports codebook usage entropy and codeword energy;
2. full: additionally consumes candidate codes/embeddings and reports prefix
   contraction, branching, empirical entropy, and reconstruction residuals.

It never treats a suffix-order permutation of an unchanged predictor as a
valid intervention: such a permutation changes positional semantics and the
Trie simultaneously.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-codes", type=Path)
    parser.add_argument("--candidate-embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-reconstruction-samples",
        type=int,
        default=100_000,
        help="Deterministic prefix sample used only for reconstruction metrics.",
    )
    return parser.parse_args()


def entropy_stats(counts: np.ndarray) -> tuple[float, float, float]:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan"), float("nan"), 0.0
    probabilities = counts / counts.sum()
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
    normalized = float(entropy_bits / np.log2(max(2, len(probabilities))))
    effective = float(2.0**entropy_bits)
    return entropy_bits, normalized, effective


def load_checkpoint(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint.get("model", checkpoint)
    codebooks: list[np.ndarray] = []
    usage: list[np.ndarray] = []
    # Layer 0 is the three-entry modality codebook. Semantic positions 1--8
    # correspond to residual_rq layers 1--8.
    for level in range(1, 9):
        prefix = f"quantizer.residual_rq.layers.{level}._codebook."
        embed_key = prefix + "embed"
        count_key = prefix + "cluster_size"
        if embed_key not in state or count_key not in state:
            raise KeyError(f"Missing RQ level {level} in {path}")
        codebooks.append(
            state[embed_key].detach().float().squeeze(0).numpy()
        )
        usage.append(
            state[count_key].detach().float().squeeze(0).numpy()
        )
    return codebooks, usage


def expected_bucket_size(inverse: np.ndarray, counts: np.ndarray) -> float:
    # Expected number of candidates sharing the prefix of a uniformly drawn
    # candidate: sum_b |b|^2 / N.
    del inverse
    return float(np.square(counts.astype(np.float64)).sum() / counts.sum())


def prefix_statistics(codes: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    semantic = np.asarray(codes[:, 1:9], dtype=np.int64)
    n = len(semantic)
    previous_unique = 1
    for level in range(1, 9):
        prefixes = np.ascontiguousarray(semantic[:, :level])
        _, inverse, counts = np.unique(
            prefixes, axis=0, return_inverse=True, return_counts=True
        )
        unique_prefixes = len(counts)
        token_counts = np.bincount(semantic[:, level - 1], minlength=4096)
        entropy_bits, normalized_entropy, effective_codes = entropy_stats(
            token_counts
        )
        rows.append(
            {
                "position": level,
                "candidate_count": n,
                "unique_prefixes": unique_prefixes,
                "mean_branching": unique_prefixes / previous_unique,
                "mean_remaining_candidates": expected_bucket_size(
                    inverse, counts
                ),
                "median_remaining_candidates": float(np.median(counts)),
                "p95_remaining_candidates": float(
                    np.percentile(counts, 95)
                ),
                "empirical_token_entropy_bits": entropy_bits,
                "empirical_token_entropy_normalized": normalized_entropy,
                "empirical_effective_codes": effective_codes,
            }
        )
        previous_unique = unique_prefixes
    return rows


def reconstruction_statistics(
    codes: np.ndarray,
    embeddings: np.ndarray,
    codebooks: list[np.ndarray],
    max_samples: int,
) -> list[dict[str, float]]:
    sample_count = min(len(codes), max_samples)
    semantic = np.asarray(codes[:sample_count, 1:9], dtype=np.int64)
    targets = np.asarray(embeddings[:sample_count], dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != codebooks[0].shape[1]:
        raise ValueError(
            f"Embedding shape {targets.shape} is incompatible with "
            f"codebook dimension {codebooks[0].shape[1]}"
        )
    reconstruction = np.zeros_like(targets)
    previous_mse = np.square(targets).sum(axis=1)
    rows: list[dict[str, float]] = []
    for level, codebook in enumerate(codebooks, 1):
        reconstruction += codebook[semantic[:, level - 1]]
        residual = np.square(targets - reconstruction).sum(axis=1)
        gain = previous_mse - residual
        rows.append(
            {
                "position": level,
                "reconstruction_samples": sample_count,
                "mean_residual_energy": float(residual.mean()),
                "median_residual_energy": float(np.median(residual)),
                "mean_residual_energy_reduction": float(gain.mean()),
                "median_residual_energy_reduction": float(np.median(gain)),
            }
        )
        previous_mse = residual
    return rows


def main() -> int:
    args = parse_args()
    codebooks, usage = load_checkpoint(args.checkpoint)
    rows: list[dict[str, float]] = []
    for level, (codebook, counts) in enumerate(zip(codebooks, usage), 1):
        entropy_bits, normalized_entropy, effective_codes = entropy_stats(
            counts
        )
        probability = counts.astype(np.float64)
        probability /= probability.sum()
        squared_norm = np.square(codebook.astype(np.float64)).sum(axis=1)
        rows.append(
            {
                "position": level,
                "codebook_entropy_bits": entropy_bits,
                "codebook_entropy_normalized": normalized_entropy,
                "codebook_effective_codes": effective_codes,
                "mean_codeword_energy": float(squared_norm.mean()),
                "usage_weighted_codeword_energy": float(
                    (probability * squared_norm).sum()
                ),
            }
        )

    if args.candidate_codes:
        codes = np.load(
            args.candidate_codes, mmap_mode="r", allow_pickle=False
        )
        structural = prefix_statistics(codes)
        for row, extra in zip(rows, structural):
            row.update(extra)
        if args.candidate_embeddings:
            embeddings = np.load(
                args.candidate_embeddings, mmap_mode="r", allow_pickle=False
            )
            reconstruction = reconstruction_statistics(
                codes,
                embeddings,
                codebooks,
                args.max_reconstruction_samples,
            )
            for row, extra in zip(rows, reconstruction):
                row.update(extra)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
