#!/usr/bin/env python3
"""Analyze relevant-prefix survival and query-level frontier cost."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.preprocessing.utils import unhash_did, unhash_qid  # noqa: E402


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for line in handle:
            qid, _, did, relevance, *_ = line.split()
            if float(relevance) > 0:
                result[qid].add(did)
    return result


def code_to_token_prefix(code: np.ndarray, depth: int) -> tuple[int, ...]:
    """Map compact RQ indices to this checkpoint's WordLevel token IDs."""
    values = [int(value) for value in np.asarray(code).reshape(-1)[:depth]]
    offsets = [3] + [6 + 4096 * (level - 1) for level in range(1, depth)]
    return tuple(value + offset for value, offset in zip(values, offsets))


def finite_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--candidate-codes", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--chain-per-query", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--switch-depth", type=int, required=True)
    parser.add_argument("--warmup-queries", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = np.load(args.profile)
    chain = np.load(args.chain_per_query)
    candidate_codes = np.load(args.candidate_codes, mmap_mode="r")
    candidate_ids = np.load(args.candidate_ids, mmap_mode="r")
    relevant = load_qrels(args.qrels)
    did_to_code = {
        unhash_did(int(did)): np.asarray(code)
        for did, code in zip(candidate_ids, candidate_codes)
    }

    profile_qids = [unhash_qid(int(qid)) for qid in profile["query_id"]]
    chain_qids = [str(qid) for qid in chain["qid"]]
    if profile_qids != chain_qids:
        raise ValueError("query profile and chain evidence are not aligned")

    frontiers = profile["frontier_prefixes"]
    survival = np.zeros(len(profile_qids), dtype=np.uint8)
    for index, qid in enumerate(profile_qids):
        relevant_prefixes = {
            code_to_token_prefix(did_to_code[did], args.switch_depth)
            for did in relevant.get(qid, set())
            if did in did_to_code
        }
        frontier = {
            tuple(int(value) for value in prefix)
            for prefix in frontiers[index]
        }
        survival[index] = int(bool(relevant_prefixes.intersection(frontier)))

    idr50 = chain["idr50"].astype(np.uint8)
    if np.any((idr50 == 1) & (survival == 0)):
        raise ValueError("IDR@50 success found outside the saved relevant frontier")
    conditional_success = (
        float(idr50[survival == 1].mean()) if np.any(survival == 1) else float("nan")
    )

    leaves = profile["exposed_leaves"].astype(np.float64)
    latency = profile["amortized_generation_latency_ms"].astype(np.float64)
    start = min(max(args.warmup_queries, 0), len(leaves))
    leaf_latency_rho, leaf_latency_p = finite_spearman(
        leaves[start:], latency[start:]
    )
    batch_leaf_means = []
    batch_latencies = []
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(start, len(leaves), batch_size):
        batch_stop = min(batch_start + batch_size, len(leaves))
        batch_leaf_means.append(float(leaves[batch_start:batch_stop].mean()))
        batch_latencies.append(
            float(latency[batch_start:batch_stop].mean() * (batch_stop - batch_start))
        )
    batch_leaf_latency_rho, batch_leaf_latency_p = finite_spearman(
        np.asarray(batch_leaf_means), np.asarray(batch_latencies)
    )
    leaf_success_rho, leaf_success_p = finite_spearman(
        leaves, idr50.astype(np.float64)
    )

    row = {
        "dataset": args.dataset,
        "variant": args.variant,
        "switch_depth": args.switch_depth,
        "queries": len(profile_qids),
        "live_prefixes_mean": float(profile["live_prefixes"].mean()),
        "frontier_survival": float(survival.mean()),
        "idr50": float(idr50.mean()),
        "conditional_idr50_given_survival": conditional_success,
        "leaves_mean": float(leaves.mean()),
        "leaves_median": float(np.median(leaves)),
        "leaves_p90": float(np.percentile(leaves, 90)),
        "leaves_p95": float(np.percentile(leaves, 95)),
        "spearman_leaves_latency": leaf_latency_rho,
        "spearman_leaves_latency_p": leaf_latency_p,
        "spearman_batchmean_leaves_batch_latency": batch_leaf_latency_rho,
        "spearman_batchmean_leaves_batch_latency_p": batch_leaf_latency_p,
        "spearman_leaves_idr50": leaf_success_rho,
        "spearman_leaves_idr50_p": leaf_success_p,
        "warmup_queries_excluded_from_latency_correlation": start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    per_query_path = args.output.with_name(args.output.stem + "_per_query.npz")
    np.savez_compressed(
        per_query_path,
        qid=np.asarray(profile_qids),
        frontier_survival=survival,
        idr50=idr50,
        exposed_leaves=leaves,
        amortized_generation_latency_ms=latency,
    )
    print(args.output.read_text().strip())
    print(per_query_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
