#!/usr/bin/env python3
"""Collect the VLDB UNION frontier/runtime matrix from immutable run logs.

The collector deliberately fails on incomplete logs instead of emitting a
partially populated paper table.  Sequential rows do not expose a P2L leaf
frontier, so their leaf-statistic fields are left empty by design.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_LABELS = {
    "nights_task4": "NIGHTS-4",
    "edis_task2": "EDIS-2",
    "webqa_task1": "WebQA-1",
}
DECODER_LABELS = {
    "p2l": "P2L+PCAA",
    "sequential": "Sequential+PCAA",
}
FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


def last_float(pattern: str, text: str, *, required: bool = True) -> float | str:
    values = re.findall(pattern, text)
    if not values:
        if required:
            raise RuntimeError(f"missing required log pattern: {pattern}")
        return ""
    value = values[-1]
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def collect_log(path: Path, task: str, decoder: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Retriever: Mean Recall@10:" not in text:
        raise RuntimeError(f"incomplete evaluation log: {path}")

    generation_seconds = last_float(
        rf"Generation Metrics: seconds={FLOAT}", text
    )
    examples = int(
        last_float(rf"Generation Metrics:.*?examples=([0-9]+)", text)
    )
    retrieval_seconds = last_float(
        rf"Retrieval Metrics: seconds={FLOAT}", text
    )
    row: dict[str, object] = {
        "task": TASK_LABELS[task],
        "decoder": DECODER_LABELS[decoder],
        "examples": examples,
        "r1": last_float(rf"Retriever: Mean Recall@1: {FLOAT}", text),
        "r5": last_float(rf"Retriever: Mean Recall@5: {FLOAT}", text),
        "r10": last_float(rf"Retriever: Mean Recall@10: {FLOAT}", text),
        "generation_seconds": generation_seconds,
        "generation_ms_per_query": 1000.0 * float(generation_seconds) / examples,
        "retrieval_seconds": retrieval_seconds,
        "manifest_d2r_seconds": float(generation_seconds) + float(retrieval_seconds),
        "throughput_qps": last_float(
            rf"Generation Metrics:.*?examples_per_second={FLOAT}", text
        ),
        "p50_ms_per_query": last_float(
            rf"Generation Latency Distribution:.*?normalized_query_p50_ms={FLOAT}",
            text,
        ),
        "p95_ms_per_query": last_float(
            rf"Generation Latency Distribution:.*?normalized_query_p95_ms={FLOAT}",
            text,
        ),
        "p99_ms_per_query": last_float(
            rf"Generation Latency Distribution:.*?normalized_query_p99_ms={FLOAT}",
            text,
        ),
        "peak_allocated_mib": last_float(
            rf"Generation CUDA Memory: peak_allocated_mib={FLOAT}", text
        ),
        "leaves_mean": "",
        "leaves_p50": "",
        "leaves_p95": "",
        "leaves_p99": "",
        "leaves_max": "",
        "source_log": str(path.relative_to(ROOT)),
    }
    if decoder == "p2l":
        for field, token in (
            ("leaves_mean", "candidates_mean"),
            ("leaves_p50", "candidates_p50"),
            ("leaves_p95", "candidates_p95"),
            ("leaves_p99", "candidates_p99"),
            ("leaves_max", "candidates_max"),
        ):
            row[field] = last_float(
                rf"TCIS Query Enumeration:.*?{token}={FLOAT}", text
            )
    return row


def latest_run() -> Path:
    runs = sorted((ROOT / "docs/results").glob("vldb_union_frontier_20*"))
    if not runs:
        raise FileNotFoundError("no VLDB UNION frontier run directory found")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/results/p2l_vldb_union_frontier_runtime.csv",
    )
    args = parser.parse_args()
    run_dir = (args.run_dir or latest_run()).resolve()
    progress_path = run_dir / "progress.tsv"
    progress = list(csv.DictReader(progress_path.open(newline=""), delimiter="\t"))

    completed: dict[tuple[str, str], dict[str, str]] = {}
    for entry in progress:
        if entry["status"] == "DONE":
            completed[(entry["task"], entry["decoder"])] = entry

    expected = [
        (task, decoder)
        for task in TASK_LABELS
        for decoder in ("p2l", "sequential")
    ]
    missing = [key for key in expected if key not in completed]
    if missing:
        formatted = ", ".join(f"{task}/{decoder}" for task, decoder in missing)
        raise RuntimeError(f"run is incomplete; missing DONE entries: {formatted}")

    rows = [
        collect_log(Path(completed[key]["log"]), *key)
        for key in expected
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT=PASS_VLDB_UNION_FRONTIER_COLLECTION rows={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
