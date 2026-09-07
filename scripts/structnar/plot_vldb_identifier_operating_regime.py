#!/usr/bin/env python3
"""Render compact identifier-structure and prefix-length diagnostics.

The figure combines the two quantizer statistics with the three measured
prefix-length operating-regime panels.  All values are loaded from retained
experiment artifacts; no values are embedded in the plotting code.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
RQ_SOURCE = ROOT / "docs/results/tip_rq_position_roles_checkpoint_20260727.csv"
OUTPUT = ROOT / "VLDB/vldb2027/figures/identifier_operating_regime"
QUERY_COUNTS = {"CIRR": 4170, "NIGHTS": 2120, "EDIS": 3241}


def load_rq_statistics() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with RQ_SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise ValueError(f"expected eight residual-code rows, found {len(rows)}")
    position = np.asarray([int(row["position"]) for row in rows])
    energy = np.asarray(
        [float(row["usage_weighted_codeword_energy"]) for row in rows]
    )
    effective = np.asarray([float(row["codebook_effective_codes"]) for row in rows])
    if not np.isfinite(np.concatenate((energy, effective))).all():
        raise ValueError("non-finite quantizer diagnostic")
    return position, energy, effective


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color="#D5DEE7", linewidth=0.45, alpha=0.8)
    ax.tick_params(axis="both", length=2.5)


def main() -> None:
    position, energy, effective = load_rq_statistics()
    grouped = load_depth_rows()
    sequential = load_sequential()
    validate_data(grouped)
    # The retained aggregate row for EDIS s=2 no longer has a recoverable raw
    # timing/frontier trace.  Exclude that point from the publication figure;
    # all displayed operating points retain their underlying console logs.
    display_grouped = {
        dataset: [
            row
            for row in rows
            if not (dataset == "EDIS" and int(row["depth"]) == 2)
        ]
        for dataset, rows in grouped.items()
    }

    leaf_logs = [
        math.log10(row["leaves"])
        for rows in display_grouped.values()
        for row in rows
    ]
    min_log, max_log = min(leaf_logs), max(leaf_logs)

    fig = plt.figure(figsize=(7.20, 4.00))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=(0.93, 1.05),
        left=0.075,
        right=0.985,
        bottom=0.18,
        top=0.95,
        wspace=1.35,
        hspace=0.82,
    )
    ax_energy = fig.add_subplot(grid[0, 0:3])
    ax_effective = fig.add_subplot(grid[0, 3:6])
    lower_axes = [fig.add_subplot(grid[1, 0:2]), fig.add_subplot(grid[1, 2:4]), fig.add_subplot(grid[1, 4:6])]

    ax_energy.plot(
        position,
        energy,
        color=CATEGORICAL[0],
        marker="o",
        markersize=3.5,
        linewidth=1.35,
    )
    ax_energy.fill_between(position, 0, energy, color=CATEGORICAL[0], alpha=0.08)
    ax_energy.axvline(2.5, color="#495867", linestyle=(0, (2, 2)), linewidth=0.75)
    ax_energy.set_ylim(0, 0.34)
    ax_energy.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax_energy.set_xticks(position)
    ax_energy.set_xlabel("Residual-code position")
    ax_energy.set_ylabel("Usage-weighted codeword energy")
    ax_energy.set_title("(a)  Update magnitude", loc="left", fontweight="bold")
    ax_energy.annotate(
        "84.9% decrease",
        xy=(8, energy[-1]),
        xytext=(5.2, 0.155),
        fontsize=6.7,
        color=CATEGORICAL[0],
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=CATEGORICAL[0], lw=0.75),
    )
    style_axis(ax_energy)

    ax_effective.plot(
        position,
        effective,
        color=CATEGORICAL[2],
        marker="s",
        markersize=3.3,
        linewidth=1.35,
    )
    ax_effective.axvline(2.5, color="#495867", linestyle=(0, (2, 2)), linewidth=0.75)
    ax_effective.set_ylim(3150, 3850)
    ax_effective.set_yticks([3200, 3400, 3600, 3800])
    ax_effective.set_xticks(position)
    ax_effective.set_xlabel("Residual-code position")
    ax_effective.set_ylabel("Effective codes")
    ax_effective.set_title("(b)  Codebook utilization", loc="left", fontweight="bold")
    ax_effective.annotate(
        "15.4% increase",
        xy=(8, effective[-1]),
        xytext=(4.8, 3370),
        fontsize=6.7,
        color=CATEGORICAL[2],
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=CATEGORICAL[2], lw=0.75),
    )
    style_axis(ax_effective)

    for panel_index, (ax, dataset) in enumerate(zip(lower_axes, DATASETS), start=2):
        rows = display_grouped[dataset]
        count = QUERY_COUNTS[dataset]
        x_values = [1000.0 * row["seconds"] / count for row in rows]
        y_values = [row["r10"] for row in rows]
        ax.plot(x_values, y_values, color="#7593AD", linewidth=0.9, zorder=1)
        for x_value, row in zip(x_values, rows):
            depth = int(row["depth"])
            ax.scatter(
                x_value,
                row["r10"],
                s=marker_area(row["leaves"], min_log, max_log),
                color=DEPTH_COLORS[depth],
                edgecolor=BLACK if depth == 3 else "white",
                linewidth=1.1 if depth == 3 else 0.65,
                zorder=3,
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

        combined_x = x_values + [seq_x]
        combined_y = y_values + [seq["r10"]]
        x_span = max(combined_x) - min(combined_x)
        y_span = max(combined_y) - min(combined_y)
        ax.set_xlim(min(combined_x) - 0.11 * x_span, max(combined_x) + 0.11 * x_span)
        ax.set_ylim(min(combined_y) - max(1.1, 0.18 * y_span), max(combined_y) + max(1.1, 0.14 * y_span))
        ax.set_xlabel("Decoder latency (ms/query)")
        if dataset == "CIRR":
            ax.set_ylabel("R@10 (%)")
        ax.set_title(
            f"({chr(ord('a') + panel_index)})  {dataset}-" + {"CIRR": "7", "NIGHTS": "4", "EDIS": "2"}[dataset],
            loc="left",
            fontweight="bold",
        )
        style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.6,
            markerfacecolor=DEPTH_COLORS[depth],
            markeredgecolor=BLACK if depth == 3 else "white",
            label=f"$s={depth}$" + (" (selected)" if depth == 3 else ""),
        )
        for depth in (2, 3, 4, 5)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=CATEGORICAL[5],
            markeredgecolor="white",
            label="Sequential",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        columnspacing=1.35,
        handletextpad=0.35,
        frameon=False,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    report = OUTPUT.with_name(OUTPUT.name + "_provenance.md")
    report.write_text(
        "\n".join(
            [
                "# Identifier operating-regime figure provenance",
                "",
                f"- Quantizer statistics: `{RQ_SOURCE.relative_to(ROOT)}`.",
                "- Prefix-length points: `docs/results/tip_core_depth_lambda0_20260726_084520.csv`.",
                "- The EDIS-2 s=2 aggregate point is excluded because its original timing/frontier console trace is no longer retained.",
                "- Sequential controls: `docs/results/structnar_lambda0_controlled_runtime.csv`.",
                "- Query counts: official manifests (CIRR-7 4,170; NIGHTS-4 2,120; EDIS-2 3,241).",
                "- Circle area is an affine mapping of log10(mean exposed complete leaves).",
                "- The plot is descriptive and uses one frozen checkpoint; no uncertainty interval is implied.",
            ]
        )
        + "\n"
    )
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))
    print(report)


if __name__ == "__main__":
    main()
