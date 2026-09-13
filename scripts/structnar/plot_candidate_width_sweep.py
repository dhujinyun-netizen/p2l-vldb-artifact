#!/usr/bin/env python3
"""Recreate the manuscript candidate-width quality/generation-latency sweep."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "docs/results/candidate_width_sweep.csv"
OUT_PATH = ROOT / "docs/results/candidate_width_sweep.png"
TASKS = ["CIRR-7", "NIGHTS-4", "EDIS-2", "WebQA-1"]
WIDTHS = [10, 20, 50, 100]


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    rows = load_rows()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.9))
    for ax, task in zip(axes.ravel(), TASKS):
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: int(row["width"]),
        )
        if [int(row["width"]) for row in task_rows] != WIDTHS:
            raise AssertionError(f"unexpected width grid for {task}")

        seq_x = [float(row["seq_generation_ms"]) for row in task_rows]
        seq_y = [float(row["seq_r10"]) for row in task_rows]
        p2l_x = [float(row["p2l_generation_ms"]) for row in task_rows]
        p2l_y = [float(row["p2l_r10"]) for row in task_rows]

        ax.plot(seq_x, seq_y, marker="o", label="Sequential")
        ax.plot(p2l_x, p2l_y, marker="s", label="P2L")
        for row, x, y in zip(task_rows, seq_x, seq_y):
            ax.annotate(row["width"], (x, y), xytext=(2, -11), textcoords="offset points", fontsize=7)
        for row, x, y in zip(task_rows, p2l_x, p2l_y):
            ax.annotate(row["width"], (x, y), xytext=(-5, 5), textcoords="offset points", fontsize=7)
        ax.set_title(task)
        ax.set_xlabel("Generation latency (ms/query)")
        ax.set_ylabel("R@10 (%)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
