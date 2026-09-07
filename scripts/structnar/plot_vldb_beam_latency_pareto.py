#!/usr/bin/env python3
"""Plot the measured beam-width accuracy--latency frontier for PVLDB.

The script reads the archived four-task sweep and never hard-codes measured
values.  Latency is decoder-side generation time divided by the fixed query
manifest size for each task.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs/results/structnar_tip_beam_pareto.csv"
OUTPUT = ROOT / "VLDB/vldb2027/figures/beam_latency_pareto.pdf"

TASKS = [
    ("cirr_task7", "(a) CIRR-7"),
    ("nights_task4", "(b) NIGHTS-4"),
    ("edis_task2", "(c) EDIS-2"),
    ("webqa_task1", "(d) WebQA-1"),
]
STYLES = {
    "sequential": ("Sequential+PCAA", "#D55E00", "s", "--"),
    "p2l": ("P2L+PCAA", "#0072B2", "o", "-"),
}


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latency_ms(row: dict[str, str]) -> float:
    return 1000.0 * float(row["generation_seconds"]) / int(row["examples"])


def assert_p2l_dominance(rows: list[dict[str, str]]) -> None:
    """Verify the caption claim against every plotted Sequential point."""
    for task, _ in TASKS:
        seq = [row for row in rows if row["dataset"] == task and row["method"] == "sequential"]
        p2l = [row for row in rows if row["dataset"] == task and row["method"] == "p2l"]
        for row in seq:
            dominated = any(
                latency_ms(candidate) < latency_ms(row)
                and float(candidate["recall_at_10"]) > float(row["recall_at_10"])
                for candidate in p2l
            )
            if not dominated:
                raise AssertionError(
                    f"Caption claim fails for {task} Sequential B={row['beam']}"
                )


def main() -> None:
    rows = load_rows()
    assert_p2l_dominance(rows)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.55), constrained_layout=True)
    axes = axes.ravel()
    for ax, (task, panel_label) in zip(axes, TASKS):
        for method in ("sequential", "p2l"):
            label, color, marker, line = STYLES[method]
            subset = sorted(
                (row for row in rows if row["dataset"] == task and row["method"] == method),
                key=lambda row: int(row["beam"]),
            )
            latency = [latency_ms(row) for row in subset]
            recall = [100.0 * float(row["recall_at_10"]) for row in subset]
            ax.plot(
                latency,
                recall,
                label=label,
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.55,
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
            )
            for point_index, (x, y, row) in enumerate(zip(latency, recall, subset)):
                offset = (3, 4) if method == "p2l" else (3, -9)
                if task == "nights_task4" and method == "p2l" and point_index == 2:
                    offset = (3, -11)
                ax.annotate(
                    row["beam"],
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    color=color,
                    fontsize=7.0,
                )

        ax.text(
            0.03,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
        )
        ax.set_xlabel("Decoder latency (ms/query)")
        ax.set_ylabel("Recall@10 (%)")
        ax.grid(True, color="#D8DDE3", linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.12, y=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=2.2,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
