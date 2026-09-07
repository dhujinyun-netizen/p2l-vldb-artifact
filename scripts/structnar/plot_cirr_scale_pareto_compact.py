#!/usr/bin/env python3
"""Create the single-column CIRR scale summary used in the VLDB main paper.

Both panels use candidate-pool size on the horizontal axis so the scale
trajectory is explicit rather than inferred from endpoint annotations.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCALE = ROOT / "docs/results/structnar_cirr_end_to_end_scaling.csv"
OUT = ROOT / "VLDB/vldb2027/figures/cirr_scale_pareto_compact.pdf"
SIZES = [21_551, 100_000, 500_000, 1_000_000, 5_609_079]
COLORS = {"sequential": "#D55E00", "p2l": "#0072B2"}


def main() -> None:
    semantic = pd.read_csv(SCALE)
    observed = {
        (str(row.decoder).lower(), int(row.candidate_count))
        for row in semantic.itertuples()
    }
    expected = {(decoder, size) for decoder in ("sequential", "p2l") for size in SIZES}
    if observed != expected:
        raise SystemExit("incomplete CIRR scale matrix")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 7.4,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.28, 2.75), sharex=True, layout="constrained")
    specs = (
        ("sequential", "Sequential+PCAA", "s", "--"),
        ("p2l", "P2L+PCAA", "o", "-"),
    )
    for decoder, label, marker, linestyle in specs:
        rows = semantic[semantic["decoder"].str.lower() == decoder].sort_values(
            "candidate_count"
        )
        axes[0].plot(
            rows["candidate_count"],
            rows["online_decode_to_rerank_ms_per_query"],
            color=COLORS[decoder],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=3.6,
            label=label,
        )
        axes[1].plot(
            rows["candidate_count"],
            100.0 * rows["recall_at_10"],
            color=COLORS[decoder],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=3.6,
        )

    tick_labels = ["21.5K", "0.1M", "0.5M", "1M", "5.6M"]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SIZES, tick_labels)
        ax.grid(True, color="#9BA8B0", alpha=0.58, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Online D2R (ms/query)")
    axes[1].set_ylabel("R@10 (%)")
    axes[1].set_xlabel("Candidate-pool size (log scale)")
    axes[0].text(0.02, 0.92, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.02, 0.92, "(b)", transform=axes[1].transAxes, fontweight="bold")
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        handlelength=1.7,
        columnspacing=0.9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    print(OUT)


if __name__ == "__main__":
    main()
