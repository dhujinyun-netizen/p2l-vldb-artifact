#!/usr/bin/env python3
"""Extract matched all-task LOCAL metrics from two official evaluator logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


START = re.compile(
    r"Retriever: Retrieving for query:([^|]+)\| split:([^|]+)\| from cand_pool:(.+)$"
)
METRIC = re.compile(r"Retriever: Mean (Recall@\d+): ([0-9.]+)")
GENERATION_TIME = re.compile(
    r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)"
)
RETRIEVAL_TIME = re.compile(r"Retrieval Metrics: seconds=([0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(path: Path, decoder: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    pending_generation: dict[str, object] | None = None
    source_hash = sha256(path)
    for raw in path.read_text(errors="replace").splitlines():
        match = GENERATION_TIME.search(raw)
        if match:
            seconds = float(match.group(1))
            examples = int(match.group(2))
            pending_generation = {
                "generation_seconds": seconds,
                "generation_examples": examples,
                "generation_ms_per_query": 1000.0 * seconds / examples,
            }
            continue
        match = START.search(raw)
        if match:
            current = {
                "decoder": decoder,
                "task": match.group(1).strip(),
                "pool": match.group(3).strip(),
                "source_log": str(path),
                "source_log_sha256": source_hash,
            }
            if pending_generation is None:
                raise ValueError(f"missing generation timing before task {match.group(1).strip()}")
            current.update(pending_generation)
            pending_generation = None
            continue
        match = METRIC.search(raw)
        if match and current is not None:
            current[match.group(1)] = float(match.group(2))
        match = RETRIEVAL_TIME.search(raw)
        if match and current is not None:
            current["retrieval_seconds"] = float(match.group(1))
            rows.append(current)
            current = None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2l-log", type=Path, required=True)
    parser.add_argument("--sequential-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = parse(args.p2l_log, "P2L") + parse(args.sequential_log, "Sequential")
    if not rows:
        raise SystemExit("No complete task rows found in the supplied logs")
    fields = [
        "decoder",
        "task",
        "pool",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
        "generation_seconds",
        "generation_examples",
        "generation_ms_per_query",
        "retrieval_seconds",
        "source_log",
        "source_log_sha256",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    for decoder in ("P2L", "Sequential"):
        subset = [r for r in rows if r["decoder"] == decoder]
        r10 = [float(r["Recall@10"]) for r in subset if "Recall@10" in r]
        if r10:
            print(f"{decoder} LOCAL R@10 macro={sum(r10)/len(r10):.4f} tasks={len(r10)}")
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
