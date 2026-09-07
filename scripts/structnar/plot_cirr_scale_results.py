#!/usr/bin/env python3
"""Plot the audited CIRR candidate-pool scaling evidence for VLDB."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCALE = ROOT / "docs/results/structnar_cirr_end_to_end_scaling.csv"
ANN = ROOT / "docs/results/structnar_flatip_gpu_scaling.csv"
OUT = ROOT / "VLDB/vldb2027/figures/cirr_scale_curve.pdf"
EXPECTED_SIZES = [21_551, 100_000, 500_000, 1_000_000, 5_609_079]
LABELS = {
    21_551: "21.5K",
    100_000: "100K",
    500_000: "500K",
    1_000_000: "1M",
    5_609_079: "5.6M",
}
COLORS = {"p2l": "#0072B2", "sequential": "#D55E00", "flatip": "#009E73"}


def require_complete(data: pd.DataFrame) -> None:
    observed = {
        (str(row.decoder).lower(), int(row.candidate_count))
        for row in data.itertuples()
    }
    expected = {
        (decoder, size)
        for decoder in ("p2l", "sequential")
        for size in EXPECTED_SIZES
    }
    if observed != expected:
        raise SystemExit(
            "scale CSV is not the complete 10-cell matrix: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )


def main() -> None:
    data = pd.read_csv(SCALE)
    require_complete(data)
    ann = pd.read_csv(ANN) if ANN.exists() else None
    if ann is not None and sorted(ann["candidate_count"].astype(int).tolist()) != EXPECTED_SIZES:
        raise SystemExit("FlatIP CSV does not contain the expected five candidate pools")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.45), sharex=True, layout="constrained"
    )
    ax_latency, ax_recall, ax_frontier, ax_memory = axes.ravel()

    styles = {
        "sequential": {"marker": "s", "linestyle": "--"},
        "p2l": {"marker": "o", "linestyle": "-"},
    }
    method_handles = []
    for decoder, name in (("sequential", "Sequential+PCAA"), ("p2l", "P2L+PCAA")):
        subset = data[data["decoder"].str.lower() == decoder].sort_values("candidate_count")
        x = subset["candidate_count"].astype(int)
        common = {
            **styles[decoder],
            "linewidth": 1.5,
            "markersize": 4,
            "color": COLORS[decoder],
            "label": name,
        }
        handle = ax_latency.plot(
            x, subset["online_decode_to_rerank_ms_per_query"], **common
        )[0]
        method_handles.append(handle)
        ax_recall.plot(x, 100.0 * subset["recall_at_10"], **common)
        ax_memory.plot(x, subset["peak_allocated_mib"] / 1024.0, **common)

    p2l = data[data["decoder"].str.lower() == "p2l"].sort_values("candidate_count")
    frontier_styles = (
        ("leaves_mean", "Mean", "o", "-"),
        ("leaves_p95", "P95", "s", "--"),
        ("leaves_p99", "P99", "^", ":"),
    )
    for column, name, marker, linestyle in frontier_styles:
        values = pd.to_numeric(p2l[column], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"missing P2L frontier statistic: {column}")
        ax_frontier.plot(
            p2l["candidate_count"].astype(int),
            values,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=3.8,
            color=COLORS["p2l"],
            label=name,
        )

    if ann is not None:
        ann = ann.sort_values("candidate_count")
        x = ann["candidate_count"].astype(int)
        common = {
            "marker": "D",
            "linestyle": "-.",
            "linewidth": 1.35,
            "markersize": 3.5,
            "color": COLORS["flatip"],
            "label": "Exact FlatIP (GPU)",
        }
        handle = ax_latency.plot(x, ann["search_ms_per_query"], **common)[0]
        method_handles.append(handle)
        ax_recall.plot(x, ann["recall_at_10"], **common)
        ax_memory.plot(x, ann["peak_allocated_mib"] / 1024.0, **common)

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks(EXPECTED_SIZES)
        ax.set_xticklabels([LABELS[size] for size in EXPECTED_SIZES])
        ax.set_xlabel("Candidate-pool size")
        ax.grid(True, color="#A7B1B8", alpha=0.62, linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)

    ax_latency.set_yscale("log")
    ax_frontier.set_yscale("log")
    ax_latency.set_ylabel("Online latency (ms/query, log)")
    ax_latency.set_title("(a) Item-retrieval latency")
    ax_recall.set_ylabel("R@10 (%)")
    ax_recall.set_ylim(0.0, 100.0)
    ax_recall.set_title("(b) Retrieval accuracy")
    ax_frontier.set_ylabel("Exposed leaves/query (log)")
    ax_frontier.set_title("(c) P2L frontier distribution")
    ax_memory.set_ylabel("Peak allocated GPU memory (GiB)")
    ax_memory.set_ylim(bottom=0.0)
    ax_memory.set_title("(d) Device-memory footprint")
    ax_frontier.legend(frameon=False, loc="best", ncol=3)
    fig.legend(
        handles=method_handles,
        labels=[handle.get_label() for handle in method_handles],
        loc="outside upper center",
        ncol=len(method_handles),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.2,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches=None, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
