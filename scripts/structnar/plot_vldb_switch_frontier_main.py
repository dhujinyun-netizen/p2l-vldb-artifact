#!/usr/bin/env python3
"""Plot the prefix-to-leaf switch trade-off for the PVLDB main paper.

The script reads retained prefix-length traces and shows accuracy, decoder
latency, and exposed-leaf cardinality in one compact three-panel figure.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_tip_operating_regime import (
    BLACK,
    CATEGORICAL,
    DATASETS,
    DATASET_KEYS,
    DEPTH_COLORS,
    load_depth_rows,
    marker_area,
    validate_data,
)
from plot_vldb_switch_column import load_sequential


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "VLDB/vldb2027/figures/switch_frontier_main.pdf"
QUERY_COUNTS = {"CIRR": 4170, "NIGHTS": 2120, "EDIS": 3241}
DISPLAY_NAMES = {"CIRR": "CIRR-7", "NIGHTS": "NIGHTS-4", "EDIS": "EDIS-2"}


def main() -> None:
    grouped = load_depth_rows()
    sequential = load_sequential()
    validate_data(grouped)
    displayed = {
        dataset: [
            row
            for row in rows
            if not (dataset == "EDIS" and int(row["depth"]) == 2)
        ]
        for dataset, rows in grouped.items()
    }
    leaf_logs = [
        math.log10(row["leaves"])
        for rows in displayed.values()
        for row in rows
    ]
    min_log, max_log = min(leaf_logs), max(leaf_logs)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)

    for panel_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        rows = displayed[dataset]
        count = QUERY_COUNTS[dataset]
        x_values = [1000.0 * row["seconds"] / count for row in rows]
        y_values = [row["r10"] for row in rows]
        ax.plot(x_values, y_values, color="#8697A6", linewidth=0.9, zorder=1)

        for x_value, row in zip(x_values, rows):
            depth = int(row["depth"])
            selected = depth == 3
            ax.scatter(
                x_value,
                row["r10"],
                s=marker_area(row["leaves"], min_log, max_log),
                color=DEPTH_COLORS[depth],
                edgecolor=BLACK if selected else "white",
                linewidth=1.1 if selected else 0.65,
                zorder=3,
            )
            label = f"$s={depth}$\n{row['leaves']:,.0f}"
            offset = (4, 4)
            if dataset == "CIRR" and depth == 2:
                offset = (4, -18)
            elif dataset == "NIGHTS" and depth == 5:
                offset = (4, -18)
            elif dataset == "EDIS" and depth == 3:
                offset = (4, -18)
            ax.annotate(
                label,
                (x_value, row["r10"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.2,
                fontweight="bold" if selected else "normal",
                color=BLACK,
                linespacing=0.9,
            )

        seq = sequential[DATASET_KEYS[dataset]]
        seq_x = 1000.0 * seq["seconds"] / count
        ax.scatter(
            seq_x,
            seq["r10"],
            marker="*",
            s=62,
            color=CATEGORICAL[5],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            "Seq.",
            (seq_x, seq["r10"]),
            xytext=(-3, -12),
            textcoords="offset points",
            ha="right",
            fontsize=6.3,
            color=CATEGORICAL[5],
        )

        combined_x = x_values + [seq_x]
        combined_y = y_values + [seq["r10"]]
        x_span = max(combined_x) - min(combined_x)
        y_span = max(combined_y) - min(combined_y)
        ax.set_xlim(min(combined_x) - 0.10 * x_span, max(combined_x) + 0.12 * x_span)
        ax.set_ylim(
            min(combined_y) - max(1.25, 0.22 * y_span),
            max(combined_y) + max(1.25, 0.18 * y_span),
        )
        ax.set_xlabel("Decoder latency (ms/query)")
        if panel_index == 0:
            ax.set_ylabel("Recall@10 (%)")
        ax.text(
            0.03,
            0.96,
            f"({chr(ord('a') + panel_index)}) {DISPLAY_NAMES[dataset]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.grid(True, color="#D8DDE3", linewidth=0.5, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        -0.02,
        "Labels report prefix length and mean exposed leaves; marker area scales with log leaf count.",
        ha="center",
        va="top",
        fontsize=7.1,
        color=BLACK,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
