#!/usr/bin/env python3
"""Collect the matched raw-cosine score control at prefix width B=50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = (0, 5, 10, 15, 20, 30, 40)
VAL_TASKS = ("cirr_task7", "nights_task4")
TEST_TASKS = ("cirr_task7", "nights_task4", "edis_task2", "webqa_task1")
DISPLAY = {
    "cirr_task7": "CIRR-7",
    "nights_task4": "NIGHTS-4",
    "edis_task2": "EDIS-2",
    "webqa_task1": "WebQA-1",
}
GENERATED = ROOT / "VLDB/vldb2027/supplementary/generated"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cell(marker: Path) -> dict[str, object]:
    fields = dict(
        line.split("=", 1) for line in marker.read_text().splitlines() if "=" in line
    )
    log = Path(fields["log"])
    config = Path(fields["config"])
    if sha256(log) != fields["sha256"] or sha256(config) != fields["config_sha256"]:
        raise SystemExit(f"hash mismatch: {marker}")
    text = log.read_text(errors="replace")

    def last(pattern: str) -> str:
        values = re.findall(pattern, text)
        if not values:
            raise SystemExit(f"missing metric in {log}: {pattern}")
        return values[-1]

    timing = re.findall(
        r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
    )
    if not timing:
        raise SystemExit(f"missing generation timing: {log}")
    seconds, queries = timing[-1]
    return {
        "split": fields["split"],
        "task": fields["task"],
        "weight": int(fields["weight"]),
        "recall_at_1": float(last(r"Retriever: Mean Recall@1: ([0-9.]+)")),
        "recall_at_5": float(last(r"Retriever: Mean Recall@5: ([0-9.]+)")),
        "recall_at_10": float(last(r"Retriever: Mean Recall@10: ([0-9.]+)")),
        "generation_seconds": float(seconds),
        "queries": int(queries),
        "generation_ms_per_query": 1000.0 * float(seconds) / int(queries),
        "source_log": str(log.relative_to(ROOT)),
        "source_log_sha256": fields["sha256"],
        "source_config": str(config.relative_to(ROOT)),
        "source_config_sha256": fields["config_sha256"],
    }


def collect_validation(state: Path) -> list[dict[str, object]]:
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


def verify_zero_weight_frontier(rows: list[dict[str, object]]) -> None:
    """The B=50 raw-cosine zero point must match the B=50 RQ zero point."""
    rq_path = ROOT / "docs/results/structnar_vldb_rq_distance_baseline.csv"
    rq_rows = list(csv.DictReader(rq_path.open(newline="")))
    raw_zero = {
        str(row["task"]): float(row["recall_at_10"])
        for row in rows
        if int(row["weight"]) == 0
    }
    rq_zero = {
        row["task"]: float(row["recall_at_10"])
        for row in rq_rows
        if row["split"] == "val" and int(row["weight"]) == 0
    }
    if raw_zero != {task: rq_zero[task] for task in VAL_TASKS}:
        raise SystemExit("raw-cosine and RQ zero-weight frontiers do not match")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    validation = collect_validation(args.state)
    selected, means = select_weight(validation)
    if args.select_only:
        print(selected)
        return 0

    test = []
    for task in TEST_TASKS:
        marker = args.state / f"test_{task}_w{selected}.ok"
        if not marker.is_file():
            raise SystemExit(f"missing test marker: {marker}")
        test.append(read_cell(marker))
    verify_zero_weight_frontier(validation)

    rows = validation + test
    result_dir = ROOT / "docs/results"
    csv_path = result_dir / "structnar_vldb_raw_cosine_b50.csv"
    json_path = result_dir / "structnar_vldb_raw_cosine_b50.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "score": "raw_cosine",
                "prefix_width": 50,
                "validation_tasks": list(VAL_TASKS),
                "weights": list(WEIGHTS),
                "validation_mean_r10": means,
                "selected_weight": selected,
                "zero_weight_matches_rq_control": True,
                "test": test,
            },
            indent=2,
        )
        + "\n"
    )

    by_val = {
        (int(row["weight"]), str(row["task"])): row for row in validation
    }
    latex = [
        "% Generated from hash-verified raw-cosine B=50 runs.",
        "\\begin{widetable}",
        "\\centering",
        "\\small",
        "\\caption{Matched raw-cosine validation sweep at prefix width $B=50$. The selected weight is fixed for all four test tasks.}",
        "\\label{tab:supp-raw-cosine-b50}",
        "\\begin{minipage}[t]{0.42\\textwidth}",
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
        open_bold = "\\textbf{" if weight == selected else ""
        close_bold = "}" if weight == selected else ""
        latex.append(
            f"{open_bold}{weight}{close_bold} & "
            f"{open_bold}{format_decimal_percent(cirr)}{close_bold} & "
            f"{open_bold}{format_decimal_percent(nights)}{close_bold} & "
            f"{open_bold}{format_decimal_mean(cirr, nights)}{close_bold} \\\\"
        )
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.54\\textwidth}",
        "\\centering",
        f"\\textbf{{(b) Test results at $\\lambda={selected}$}}\\\\[3pt]",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Task & R@1 & R@5 & R@10 & Gen. (ms/q) \\\\",
        "\\midrule",
    ]
    by_test = {str(row["task"]): row for row in test}
    for task in TEST_TASKS:
        row = by_test[task]
        latex.append(
            f"{DISPLAY[task]} & {100.0 * float(row['recall_at_1']):.2f} & "
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
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "raw_cosine_b50.tex").write_text("\n".join(latex) + "\n")
    (GENERATED / "raw_cosine_b50_main_row.tex").write_text(
        "% Generated from hash-verified raw-cosine B=50 test runs.\n"
        + f"Raw cosine & {selected} & "
        + " & ".join(
            f"{100.0 * float(by_test[task]['recall_at_10']):.2f}"
            for task in TEST_TASKS
        )
        + " \\\\\n"
    )
    print(csv_path)
    print(json_path)
    print(f"SELECTED_RAW_COSINE_WEIGHT={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
