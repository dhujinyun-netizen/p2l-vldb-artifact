#!/usr/bin/env python3
"""No-data audit for the current Independent-Suffix P2L artifact snapshot."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/results"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(value: float, expected: float, tol: float = 0.011) -> None:
    if not math.isclose(value, expected, abs_tol=tol, rel_tol=0.0):
        raise AssertionError(f"expected {expected}, observed {value}")


def main() -> int:
    policy = rows("all16_policy_comparison.csv")
    grouped = {}
    for name in ("Sequential", "Interacting-Suffix P2L", "Independent-Suffix P2L"):
        records = [row for row in policy if row["policy"] == name]
        grouped[name] = {row["task"]: 100.0 * float(row["R@10"])
                         for row in records}
    if any(len(values) != 16 for values in grouped.values()):
        raise AssertionError("expected 16 task rows for every policy")
    if len({frozenset(values) for values in grouped.values()}) != 1:
        raise AssertionError("policy task sets differ")
    macros = {name: sum(values.values()) / 16 for name, values in grouped.items()}
    close(macros["Sequential"], 39.36)
    close(macros["Interacting-Suffix P2L"], 42.83)
    close(macros["Independent-Suffix P2L"], 43.64)
    gains = [grouped["Independent-Suffix P2L"][task] - grouped["Sequential"][task]
             for task in grouped["Sequential"]]
    if sum(gain > 0 for gain in gains) != 16:
        raise AssertionError("expected 16 positive Independent-vs-Sequential gains")

    sweep = rows("validation_depth_sweep.csv")
    tasks = {r["task"] for r in sweep}
    if len(sweep) != 16 or len(tasks) != 4:
        raise AssertionError("validation sweep must contain four depths for four tasks")
    for task in tasks:
        task_rows = [r for r in sweep if r["task"] == task]
        if {int(r["depth"]) for r in task_rows} != {2, 3, 4, 5}:
            raise AssertionError(f"incomplete depth grid: {task}")
        best = max(task_rows, key=lambda r: float(r["R@10"]))
        if int(best["depth"]) != 3:
            raise AssertionError(f"validation R@10 does not peak at s=3: {task}")

    width = rows("candidate_width_sweep.csv")
    expected_tasks = {"CIRR-7", "NIGHTS-4", "EDIS-2", "WebQA-1"}
    if len(width) != 16 or {row["task"] for row in width} != expected_tasks:
        raise AssertionError("candidate-width sweep must contain 16 task-width rows")
    width_gains = []
    width_speedups = []
    for task in expected_tasks:
        task_rows = [row for row in width if row["task"] == task]
        if {int(row["width"]) for row in task_rows} != {10, 20, 50, 100}:
            raise AssertionError(f"incomplete width grid: {task}")
        for row in task_rows:
            seq_r10 = float(row["seq_r10"])
            p2l_r10 = float(row["p2l_r10"])
            seq_ms = float(row["seq_generation_ms"])
            p2l_ms = float(row["p2l_generation_ms"])
            if not p2l_r10 > seq_r10:
                raise AssertionError(f"P2L R@10 not higher: {task} width={row['width']}")
            if not p2l_ms < seq_ms:
                raise AssertionError(f"P2L generation not faster: {task} width={row['width']}")
            width_gains.append(p2l_r10 - seq_r10)
            width_speedups.append(seq_ms / p2l_ms)
    close(min(width_gains), 0.71)
    close(max(width_gains), 14.34)
    close(min(width_speedups), 1.51, tol=0.015)
    close(max(width_speedups), 2.44, tol=0.015)

    print("CURRENT_RELEASE_AUDIT=PASS")
    print(f"tasks=16 sequential_macro={macros['Sequential']:.2f} "
          f"independent_macro={macros['Independent-Suffix P2L']:.2f} "
          f"positive={sum(gain > 0 for gain in gains)}")
    print("validation_tasks=4 depths=2,3,4,5 best_common_depth=3")
    print("candidate_width_tasks=4 widths=10,20,50,100 matched_dominance=16/16 "
          f"gain_range={min(width_gains):.2f}-{max(width_gains):.2f}pp "
          f"speedup_range={min(width_speedups):.2f}-{max(width_speedups):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
