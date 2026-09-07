#!/usr/bin/env python3
"""Validate and summarize the all-16 matched LOCAL decoder intervention.

The runner writes one row per task and decoder.  This post-processing step is
deliberately independent of the evaluator: it refuses incomplete or duplicate
rows, computes paired task-level deltas, and emits a compact CSV/JSON summary,
a publication-ready delta plot, and a LaTeX fragment for the supplement.  The
historical all-task timer includes first-use Trie validation and any required
rebuild, serialization, and loading,
so it is labeled as manifest-stage wall time rather than online decoder time.
No paper claim is changed automatically; the generated artifacts are reviewed
before inclusion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs/results/structnar_matched_all16_local_lambda0.csv"
BOOTSTRAP = ROOT / "docs/results/structnar_matched_all16_bootstrap.csv"
SUMMARY = ROOT / "docs/results/structnar_matched_all16_local_lambda0_summary.json"
FIGURE = ROOT / "VLDB/vldb2027/figures/matched_all16_local_lambda0.pdf"
FRAGMENT = ROOT / "VLDB/vldb2027/supplementary/generated/matched_all16_local_lambda0.tex"

DISPLAY_NAMES = {
    "visualnews_task0": "VisualNews-0",
    "mscoco_task0": "MSCOCO-0",
    "fashion200k_task0": "Fashion200K-0",
    "webqa_task1": "WebQA-1",
    "edis_task2": "EDIS-2",
    "webqa_task2": "WebQA-2",
    "visualnews_task3": "VisualNews-3",
    "mscoco_task3": "MSCOCO-3",
    "fashion200k_task3": "Fashion200K-3",
    "nights_task4": "NIGHTS-4",
    "oven_task6": "OVEN-6",
    "infoseek_task6": "InfoSeek-6",
    "fashioniq_task7": "FashionIQ-7",
    "cirr_task7": "CIRR-7",
    "oven_task8": "OVEN-8",
    "infoseek_task8": "InfoSeek-8",
}


def as_percent(value: str) -> float:
    number = float(value)
    return number * 100.0 if abs(number) <= 1.0 else number


def main() -> int:
    if not INPUT.exists():
        raise SystemExit(f"missing completed result file: {INPUT}")
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not BOOTSTRAP.exists():
        raise SystemExit(f"missing paired-bootstrap result file: {BOOTSTRAP}")
    with BOOTSTRAP.open(newline="") as handle:
        bootstrap_rows = list(csv.DictReader(handle))
    bootstrap_by_scope = {str(row["scope"]): row for row in bootstrap_rows}
    if len(bootstrap_by_scope) != len(bootstrap_rows):
        raise SystemExit("duplicate scopes in paired-bootstrap result file")
    expected_decoders = {"P2L", "Sequential"}
    decoders = {str(row.get("decoder", "")) for row in rows}
    if decoders != expected_decoders:
        raise SystemExit(f"expected decoders {expected_decoders}, found {sorted(decoders)}")
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["decoder"]))
        if key in by_key:
            raise SystemExit(f"duplicate task/decoder row: {key}")
        by_key[key] = row
    tasks = sorted({task for task, _ in by_key})
    if len(tasks) != 16 or len(by_key) != 32:
        raise SystemExit(f"expected 16 tasks and 32 rows, found {len(tasks)} tasks and {len(by_key)} rows")
    unknown_tasks = set(tasks) - set(DISPLAY_NAMES)
    if unknown_tasks:
        raise SystemExit(f"unrecognized M-BEIR task names: {sorted(unknown_tasks)}")
    missing_bootstrap = set(tasks) - set(bootstrap_by_scope)
    if missing_bootstrap or "task_macro" not in bootstrap_by_scope:
        raise SystemExit(
            "incomplete paired-bootstrap results: "
            f"missing tasks={sorted(missing_bootstrap)}, "
            f"task_macro={'task_macro' in bootstrap_by_scope}"
        )
    pairs = []
    for task in tasks:
        seq = by_key[(task, "Sequential")]
        p2l = by_key[(task, "P2L")]
        if not seq.get("Recall@10") or not p2l.get("Recall@10"):
            raise SystemExit(f"missing Recall@10 for {task}")
        seq_r10 = as_percent(seq["Recall@10"])
        p2l_r10 = as_percent(p2l["Recall@10"])
        seq_ms = float(seq["generation_ms_per_query"])
        p2l_ms = float(p2l["generation_ms_per_query"])
        bootstrap = bootstrap_by_scope[task]
        if seq_ms <= 0 or p2l_ms <= 0:
            raise SystemExit(f"non-positive generation time for {task}")
        pairs.append(
            {
                "task": task,
                "display_task": DISPLAY_NAMES.get(task, task),
                "sequential_r10": seq_r10,
                "p2l_r10": p2l_r10,
                "delta_r10": p2l_r10 - seq_r10,
                "sequential_generation_ms_per_query": seq_ms,
                "p2l_generation_ms_per_query": p2l_ms,
                "generation_speedup": seq_ms / p2l_ms,
                "ci95_low": float(bootstrap["ci95_low_points"]),
                "ci95_high": float(bootstrap["ci95_high_points"]),
                "p_two_sided": float(bootstrap["p_two_sided"]),
                "sequential_source_log": seq.get("source_log", ""),
                "sequential_source_log_sha256": seq.get("source_log_sha256", ""),
                "p2l_source_log": p2l.get("source_log", ""),
                "p2l_source_log_sha256": p2l.get("source_log_sha256", ""),
            }
        )
    mean_seq = sum(row["sequential_r10"] for row in pairs) / len(pairs)
    mean_p2l = sum(row["p2l_r10"] for row in pairs) / len(pairs)
    macro_bootstrap = bootstrap_by_scope["task_macro"]
    seq_seconds = sum(float(by_key[(task, "Sequential")]["generation_seconds"]) for task in tasks)
    p2l_seconds = sum(float(by_key[(task, "P2L")]["generation_seconds"]) for task in tasks)
    seq_examples = sum(int(by_key[(task, "Sequential")]["generation_examples"]) for task in tasks)
    p2l_examples = sum(int(by_key[(task, "P2L")]["generation_examples"]) for task in tasks)
    if seq_examples != p2l_examples:
        raise SystemExit(
            f"decoder manifests differ: Sequential={seq_examples}, P2L={p2l_examples}"
        )
    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "task_count": len(tasks),
        "row_count": len(rows),
        "decoders": sorted(expected_decoders),
        "lambda": 0.0,
        "macro_r10": {
            "sequential": mean_seq,
            "p2l": mean_p2l,
            "delta": mean_p2l - mean_seq,
        },
        "positive_delta_tasks": sum(row["delta_r10"] > 0 for row in pairs),
        "per_task_ci_excludes_zero": sum(
            row["ci95_low"] > 0 or row["ci95_high"] < 0 for row in pairs
        ),
        "paired_bootstrap": {
            "samples": int(macro_bootstrap["bootstrap_samples"]),
            "macro_ci95_low": float(macro_bootstrap["ci95_low_points"]),
            "macro_ci95_high": float(macro_bootstrap["ci95_high_points"]),
            "macro_p_two_sided": float(macro_bootstrap["p_two_sided"]),
        },
        "faster_tasks": sum(row["generation_speedup"] > 1 for row in pairs),
        "delta_r10_range": {
            "min": min(row["delta_r10"] for row in pairs),
            "max": max(row["delta_r10"] for row in pairs),
        },
        "generation_speedup_range": {
            "min": min(row["generation_speedup"] for row in pairs),
            "max": max(row["generation_speedup"] for row in pairs),
        },
        "nonpositive_delta_tasks": [
            row["display_task"] for row in pairs if row["delta_r10"] <= 0
        ],
        "nonfaster_tasks": [
            row["display_task"] for row in pairs if row["generation_speedup"] <= 1
        ],
        "generation": {
            "timing_scope": (
                "recorded evaluator-stage manifest wall time; includes "
                "first-use Trie validation and any required rebuild, "
                "serialization, and loading"
            ),
            "queries": seq_examples,
            "sequential_seconds": seq_seconds,
            "p2l_seconds": p2l_seconds,
            "reduction_percent": 100.0 * (1.0 - p2l_seconds / seq_seconds),
            "aggregate_speedup": seq_seconds / p2l_seconds,
        },
        "pairs": pairs,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    order = sorted(pairs, key=lambda row: row["delta_r10"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax_delta = plt.subplots(figsize=(3.35, 4.15))
    labels = [row["display_task"] for row in order]
    values = [row["delta_r10"] for row in order]
    lower_errors = [
        row["delta_r10"] - row["ci95_low"] for row in order
    ]
    upper_errors = [
        row["ci95_high"] - row["delta_r10"] for row in order
    ]
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in values]
    y = list(range(len(order)))
    ax_delta.barh(y, values, color=colors, height=0.62)
    ax_delta.errorbar(
        values,
        y,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor="#20252A",
        elinewidth=0.75,
        capsize=1.8,
        capthick=0.75,
        zorder=4,
    )
    ax_delta.axvline(0, color="#222222", linewidth=0.8)
    ax_delta.set_yticks(y, labels)
    ax_delta.set_xlabel("$\\Delta$R@10 (points)")
    ax_delta.grid(axis="x", color="#B7C4CE", alpha=0.48, linewidth=0.55)
    for yi, value, high in zip(y, values, (row["ci95_high"] for row in order)):
        ax_delta.text(
            max(value, high) + (0.12 if value >= 0 else -0.12),
            yi,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=6.6,
        )

    ax_delta.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.35)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)

    FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by prepare_vldb_matched_all16_artifacts.py; include only after audit.",
        "\\begin{widetable}",
        "\\centering",
        "\\scriptsize",
        "\\caption{Matched LOCAL decoder results on all 16 M-BEIR task configurations at $\\lambda=0$. "
        "Accuracy uses identical query inputs, checkpoint, and realized index. Confidence intervals use "
        "10,000 paired query-bootstrap resamples. Manifest speedup includes first-use Trie validation and, "
        "when stale, rebuild, serialization, and loading; it is not used as the warm online-latency headline.}",
        "\\label{tab:supp-matched-all16}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Task & Seq. R@10 & P2L R@10 & $\\Delta$R@10 & 95\\% CI & Manifest speedup \\\\",
        "\\midrule",
    ]
    for row in order:
        lines.append(
            f"{row['display_task']} & {row['sequential_r10']:.2f} & {row['p2l_r10']:.2f} & "
            f"{row['delta_r10']:+.2f} & [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] & "
            f"{row['generation_speedup']:.2f}$\\times$ \\\\"
        )
    lines += [
        "\\midrule",
        f"Task macro / aggregate & {mean_seq:.2f} & {mean_p2l:.2f} & "
        f"{mean_p2l - mean_seq:+.2f} & "
        f"[{float(macro_bootstrap['ci95_low_points']):.2f}, "
        f"{float(macro_bootstrap['ci95_high_points']):.2f}] & "
        f"{seq_seconds / p2l_seconds:.2f}$\\times$ \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{widetable}",
    ]
    FRAGMENT.write_text("\n".join(lines) + "\n")
    print(f"validated {len(rows)} rows; macro R@10 {mean_seq:.2f} -> {mean_p2l:.2f}")
    print(SUMMARY)
    print(FIGURE)
    print(FRAGMENT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
