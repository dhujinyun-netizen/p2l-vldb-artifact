#!/usr/bin/env python3
"""Audit the Eval-12 deterministic streamed top-k campaign.

The input is the campaign JSON produced after applying the fixed tie rule to
both full materialization and streamed running top-k.  The audit intentionally
fails if any task/chunk changes R@10, the selected top-k set, ordered key list,
or score array.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXPECTED_TASKS = {
    "fashion200k_task0",
    "fashion200k_task3",
    "fashioniq_task7",
    "infoseek_task6",
    "infoseek_task8",
    "mscoco_task0",
    "mscoco_task3",
    "oven_task6",
    "oven_task8",
    "visualnews_task0",
    "visualnews_task3",
    "webqa_task2",
}
EXPECTED_CHUNKS = {4096, 16384, 65536}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="audit only one chunk size; 0 audits all available chunks",
    )
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit("campaign JSON contains no rows")

    grouped = defaultdict(list)
    for row in rows:
        chunk = int(row["chunk_size"])
        if args.chunk and chunk != args.chunk:
            continue
        grouped[chunk].append(row)

    if not grouped:
        raise SystemExit("no rows match the requested chunk")

    for chunk, chunk_rows in sorted(grouped.items()):
        tasks = {row["task"] for row in chunk_rows}
        if tasks != EXPECTED_TASKS:
            missing = sorted(EXPECTED_TASKS - tasks)
            extra = sorted(tasks - EXPECTED_TASKS)
            raise SystemExit(
                f"chunk {chunk}: task mismatch; missing={missing}, extra={extra}"
            )

        failures = []
        total_queries = 0
        weighted_generation_seconds = 0.0
        max_buffer = 0
        max_frontier = 0
        for row in chunk_rows:
            task = row["task"]
            total_queries += int(row["queries"])
            weighted_generation_seconds += float(row["generation_seconds"])
            max_buffer = max(max_buffer, int(row["peak_buffer_max"]))
            max_frontier = max(max_frontier, int(row["exposed_max"]))

            checks = {
                "shape_equal": bool(row.get("full_stream_shape_equal")),
                "score_array_equal": bool(row.get("full_stream_score_array_equal")),
                "ordered_identity": bool(row.get("full_stream_exact_ordered_identity")),
                "chunk_identity": bool(row.get("stream_chunk_exact_identity")),
                "zero_set_mismatch": int(row.get("full_stream_set_mismatch_queries", -1)) == 0,
                "zero_tie_permutation": int(row.get("full_stream_tie_permutation_queries", -1)) == 0,
                "recall_equal": abs(
                    float(row.get("recall_at_10", 0.0))
                    - float(row.get("stream_recall_at_10", 1.0))
                ) < 1.0e-12,
            }
            bad = [name for name, ok in checks.items() if not ok]
            if bad:
                failures.append((task, bad))

        if failures:
            details = "; ".join(f"{task}: {','.join(bad)}" for task, bad in failures)
            raise SystemExit(f"chunk {chunk}: exact-streaming audit FAILED: {details}")

        ms_per_query = 1000.0 * weighted_generation_seconds / total_queries
        if max_buffer > chunk:
            raise SystemExit(
                f"chunk {chunk}: observed buffer {max_buffer} exceeds configured chunk"
            )
        print(
            f"chunk={chunk}: PASS | tasks=12 queries={total_queries:,} "
            f"generation={ms_per_query:.3f} ms/q max_buffer={max_buffer:,} "
            f"max_frontier={max_frontier:,} ordered_identity=100%"
        )

    if not args.chunk and set(grouped) != EXPECTED_CHUNKS:
        print(
            "WARNING: campaign does not contain exactly the canonical chunk set "
            f"{sorted(EXPECTED_CHUNKS)}; observed {sorted(grouped)}"
        )


if __name__ == "__main__":
    main()
