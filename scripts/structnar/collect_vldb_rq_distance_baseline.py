#!/usr/bin/env python3
"""Select and collect the traditional RQ-distance scoring baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = (
    0, 1, 2, 5, 10, 20, 40, 80, 160,
    320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920,
)
VAL_TASKS = ("cirr_task7", "nights_task4")
TEST_TASKS = ("cirr_task7", "nights_task4", "edis_task2", "webqa_task1")
LATEX = ROOT / "VLDB/vldb2027/supplementary/generated/rq_distance_baseline.tex"
MAIN_ROW = ROOT / "VLDB/vldb2027/supplementary/generated/rq_distance_main_row.tex"

DISPLAY_TASKS = {
    "cirr_task7": "CIRR-7",
    "nights_task4": "NIGHTS-4",
    "edis_task2": "EDIS-2",
    "webqa_task1": "WebQA-1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cell(marker: Path) -> dict[str, object]:
    fields = dict(
        line.split("=", 1) for line in marker.read_text().splitlines() if "=" in line
    )
    log = Path(fields["log"])
    if sha256(log) != fields["sha256"]:
        raise SystemExit(f"hash mismatch: {marker}")
    text = log.read_text(errors="replace")

    def last(pattern: str) -> str:
        matches = re.findall(pattern, text)
        if not matches:
            raise SystemExit(f"missing metric in {log}: {pattern}")
        return matches[-1]

    generation = re.findall(
        r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
    )
    if not generation:
        raise SystemExit(f"missing generation timing: {log}")
    seconds, examples = generation[-1]
    row: dict[str, object] = {
        "split": fields["split"],
        "task": fields["task"],
        "weight": int(fields["weight"]),
        "recall_at_1": float(last(r"Retriever: Mean Recall@1: ([0-9.]+)")),
        "recall_at_5": float(last(r"Retriever: Mean Recall@5: ([0-9.]+)")),
        "recall_at_10": float(last(r"Retriever: Mean Recall@10: ([0-9.]+)")),
        "generation_seconds": float(seconds),
        "queries": int(examples),
        "generation_ms_per_query": 1000.0 * float(seconds) / int(examples),
        "source_log": str(log.relative_to(ROOT)),
        "source_log_sha256": fields["sha256"],
        "source_config": str(Path(fields["config"]).relative_to(ROOT)),
        "source_config_sha256": fields["config_sha256"],
    }
    return row


def validation_rows(state: Path) -> list[dict[str, object]]:
    rows = []
    for weight in WEIGHTS:
        for task in VAL_TASKS:
            marker = state / f"val_{task}_w{weight}.ok"
            if not marker.is_file():
                raise SystemExit(f"missing validation marker: {marker}")
            rows.append(read_cell(marker))
    return rows


def select_weight(rows: list[dict[str, object]]) -> tuple[int, dict[int, float]]:
    means = {
        weight: sum(
            float(row["recall_at_10"])
            for row in rows
            if int(row["weight"]) == weight
        )
        / len(VAL_TASKS)
        for weight in WEIGHTS
    }
    selected = min(WEIGHTS, key=lambda weight: (-means[weight], weight))
    return selected, means


def format_decimal_percent(value: float) -> str:
    """Format percentages with explicit round-half-up semantics."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def format_decimal_mean(left: float, right: float) -> str:
    mean = (Decimal(str(left)) + Decimal(str(right))) / Decimal("2")
    return str(mean.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    val = validation_rows(args.state)
    selected, means = select_weight(val)
    if args.select_only:
        print(selected)
        return 0

    test = []
    for task in TEST_TASKS:
        marker = args.state / f"test_{task}_w{selected}.ok"
        if not marker.is_file():
            raise SystemExit(f"missing test marker: {marker}")
        test.append(read_cell(marker))
    rows = val + test
    out = ROOT / "docs/results/structnar_vldb_rq_distance_baseline.csv"
    summary = ROOT / "docs/results/structnar_vldb_rq_distance_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.write_text(
        json.dumps(
            {
                "score": "raw_rq_residual_reconstruction_gain",
                "identity": "sum of suffix gains equals terminal squared-error reduction",
                "validation_tasks": list(VAL_TASKS),
                "weights": list(WEIGHTS),
                "validation_mean_r10": means,
                "selected_weight": selected,
                "test": test,
            },
            indent=2,
        )
        + "\n"
    )
    by_val = {
        (int(row["weight"]), str(row["task"])): row
        for row in val
    }
    latex = [
        "% Generated by collect_vldb_rq_distance_baseline.py from hash-verified runs.",
        "\\begin{widetable}",
        "\\centering",
        "\\scriptsize",
        "\\caption{RQ suffix reconstruction-gain control. The raw suffix gain telescopes to the reduction in squared reconstruction error from the retained prefix to the complete identifier. The fusion weight is selected by mean validation R@10 on CIRR-7 and NIGHTS-4, then fixed on all four test tasks.}",
        "\\label{tab:supp-rq-distance}",
        "\\begin{minipage}[t]{0.46\\textwidth}",
        "\\centering",
        "\\textbf{(a) Validation sweep}\\\\[3pt]",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "$\\lambda$ & CIRR-7 & NIGHTS-4 & Mean \\\\",
        "\\midrule",
    ]
    for weight in WEIGHTS:
        cirr = 100.0 * float(by_val[(weight, "cirr_task7")]["recall_at_10"])
        nights = 100.0 * float(by_val[(weight, "nights_task4")]["recall_at_10"])
        mark = "\\textbf{" if weight == selected else ""
        end = "}" if weight == selected else ""
        latex.append(
            f"{mark}{weight}{end} & {mark}{format_decimal_percent(cirr)}{end} & "
            f"{mark}{format_decimal_percent(nights)}{end} & "
            f"{mark}{format_decimal_mean(cirr, nights)}{end} \\\\"
        )
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.50\\textwidth}",
        "\\centering",
        f"\\textbf{{(b) Test results at $\\lambda={selected}$}}\\\\[3pt]",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Task & R@1 & R@5 & R@10 & Gen. (ms/q) \\\\",
        "\\midrule",
    ]
    for row in test:
        latex.append(
            f"{DISPLAY_TASKS[str(row['task'])]} & "
            f"{100.0 * float(row['recall_at_1']):.2f} & "
            f"{100.0 * float(row['recall_at_5']):.2f} & "
            f"{100.0 * float(row['recall_at_10']):.2f} & "
            f"{float(row['generation_ms_per_query']):.2f} \\\\"
        )
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}",
        "\\end{widetable}",
    ]
    LATEX.parent.mkdir(parents=True, exist_ok=True)
    LATEX.write_text("\n".join(latex) + "\n")
    test_by_task = {str(row["task"]): row for row in test}
    MAIN_ROW.write_text(
        "% Generated from hash-verified RQ-distance test runs.\n"
        + "RQ reconstruction gain & "
        + str(selected)
        + " & "
        + " & ".join(
            f"{100.0 * float(test_by_task[task]['recall_at_10']):.2f}"
            for task in TEST_TASKS
        )
        + " \\\\\n"
    )
    print(out)
    print(summary)
    print(LATEX)
    print(MAIN_ROW)
    print(f"SELECTED_RQ_DISTANCE_WEIGHT={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
