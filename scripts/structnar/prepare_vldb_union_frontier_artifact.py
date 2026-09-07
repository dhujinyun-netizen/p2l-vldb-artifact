#!/usr/bin/env python3
"""Create audited PVLDB artifacts from the UNION frontier/runtime CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs/results/p2l_vldb_union_frontier_runtime.csv"
SUMMARY = ROOT / "docs/results/p2l_vldb_union_frontier_runtime_summary.json"
TABLE = ROOT / "VLDB/vldb2027/supplementary/generated/union_frontier_runtime.tex"
TASKS = ("NIGHTS-4", "EDIS-2", "WebQA-1")


def load() -> dict[tuple[str, str], dict[str, str]]:
    rows = list(csv.DictReader(INPUT.open(newline="", encoding="utf-8")))
    keyed = {(row["task"], row["decoder"]): row for row in rows}
    expected = {
        (task, decoder)
        for task in TASKS
        for decoder in ("Sequential+PCAA", "P2L+PCAA")
    }
    if set(keyed) != expected:
        raise AssertionError(f"unexpected UNION matrix keys: {sorted(keyed)}")
    return keyed


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    rows = load()
    summaries: list[dict[str, float | str]] = []
    for task in TASKS:
        seq = rows[(task, "Sequential+PCAA")]
        p2l = rows[(task, "P2L+PCAA")]
        summaries.append(
            {
                "task": task,
                "r10_delta_points": 100.0 * (f(p2l, "r10") - f(seq, "r10")),
                "generation_reduction_percent": 100.0
                * (1.0 - f(p2l, "generation_seconds") / f(seq, "generation_seconds")),
                "p95_reduction_percent": 100.0
                * (1.0 - f(p2l, "p95_ms_per_query") / f(seq, "p95_ms_per_query")),
                "manifest_d2r_reduction_percent": 100.0
                * (1.0 - f(p2l, "manifest_d2r_seconds") / f(seq, "manifest_d2r_seconds")),
                "p2l_leaves_mean": f(p2l, "leaves_mean"),
                "p2l_leaves_p95": f(p2l, "leaves_p95"),
                "p2l_leaves_p99": f(p2l, "leaves_p99"),
                "p2l_leaves_max": f(p2l, "leaves_max"),
            }
        )

    SUMMARY.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    lines = [
        "\\begin{widetable}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\caption{Matched runtime and retrieval behavior on the 5.61M-item UNION index. Gen. is synchronized decoder-side latency; Manifest D2R additionally includes fixed-manifest bucket expansion, reranking, and lookup-cache preparation. P50/P95/P99 are amortized decoder-side latencies, and Peak is allocated device memory. Frontier statistics apply to P2L and count complete stored leaves below retained prefixes.}",
        "\\label{tab:supp-union-frontier-runtime}",
        "\\begin{tabular}{llrrrrrrrrrr}",
        "\\toprule",
        "Task & Decoder & R@10 & Gen. (ms/q) & D2R (s) & P50 & P95 & P99 & Peak (GiB) & Leaves mean & Leaves P95 & Leaves P99 \\\\",
        "\\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        for decoder in ("Sequential+PCAA", "P2L+PCAA"):
            row = rows[(task, decoder)]
            leaf_values = []
            for key in ("leaves_mean", "leaves_p95", "leaves_p99"):
                leaf_values.append(
                    "--" if not row[key] else f"{float(row[key]):,.0f}"
                )
            lines.append(
                f"{task} & {decoder} & {100.0*f(row, 'r10'):.2f} & "
                f"{f(row, 'generation_ms_per_query'):.2f} & "
                f"{f(row, 'manifest_d2r_seconds'):.2f} & "
                f"{f(row, 'p50_ms_per_query'):.2f} & {f(row, 'p95_ms_per_query'):.2f} & "
                f"{f(row, 'p99_ms_per_query'):.2f} & "
                f"{f(row, 'peak_allocated_mib') / 1024.0:.2f} & "
                + " & ".join(leaf_values)
                + " \\\\"
            )
        if task_index != len(TASKS) - 1:
            lines.append("\\addlinespace[1.5pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{widetable}", ""])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print("RESULT=PASS_VLDB_UNION_FRONTIER_ARTIFACT")
    print(f"summary={SUMMARY}")
    print(f"table={TABLE}")


if __name__ == "__main__":
    main()
