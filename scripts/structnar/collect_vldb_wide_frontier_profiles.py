#!/usr/bin/env python3
"""Summarize fixed-s=3 P2L query profiles for the VLDB supplement.

The latency samples are batch-amortized decoder-side measurements repeated for
the queries in each batch.  They are therefore reported as profiling samples,
not as independent single-query service latencies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = {
    "CIRR-7": Path(
        "docs/results/tip_core_frontier_cirr_b50_20260726_092132_per_query.npz"
    ),
    "NIGHTS-4": Path(
        "docs/results/tip_core_frontier_nights_b50_20260726_092132_per_query.npz"
    ),
    "EDIS-2": Path("docs/results/tip_core_frontier_edis_d3_b50_per_query.npz"),
    "WebQA-1": Path(
        "docs/results/query_profiles/"
        "structnar_webqa_task1_p2l_d3_vldb_wide_frontier_profile_lambda0.npz"
    ),
}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values.astype(np.float64), q))


def summarize(task: str, path: Path, batch_size: int = 8) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing profile for {task}: {path}")
    profile = np.load(path, allow_pickle=False)
    leaves = np.asarray(profile["exposed_leaves"], dtype=np.float64)
    query_latency = np.asarray(
        profile["amortized_generation_latency_ms"], dtype=np.float64
    )
    if leaves.shape != query_latency.shape:
        raise ValueError(
            f"Shape mismatch for {task}: leaves={leaves.shape}, latency={query_latency.shape}"
        )
    # The profiler repeats one amortized batch latency for every query in that
    # batch.  Recover one sample per batch and exclude the first warmup batch,
    # matching the latency-distribution protocol printed by the evaluator.
    latency = query_latency[::batch_size][1:]
    return {
        "task": task,
        "queries": int(leaves.size),
        "latency_batches_after_warmup": int(latency.size),
        "leaves_mean": float(leaves.mean()),
        "leaves_p50": percentile(leaves, 50),
        "leaves_p95": percentile(leaves, 95),
        "leaves_p99": percentile(leaves, 99),
        "leaves_max": float(leaves.max()),
        "latency_mean_ms_q": float(latency.mean()),
        "latency_p50_ms_q": percentile(latency, 50),
        "latency_p95_ms_q": percentile(latency, 95),
        "latency_p99_ms_q": percentile(latency, 99),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/p2l_vldb_wide_frontier_summary.csv"),
    )
    args = parser.parse_args()

    rows = [summarize(task, path) for task, path in DEFAULT_INPUTS.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "{task}: queries={queries} batches={latency_batches_after_warmup} "
            "leaves(mean/p50/p95/p99/max)="
            "{leaves_mean:.2f}/{leaves_p50:.2f}/{leaves_p95:.2f}/"
            "{leaves_p99:.2f}/{leaves_max:.0f} latency_ms_q(mean/p50/p95/p99)="
            "{latency_mean_ms_q:.2f}/{latency_p50_ms_q:.2f}/"
            "{latency_p95_ms_q:.2f}/{latency_p99_ms_q:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
