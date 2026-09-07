#!/usr/bin/env python3
"""Portable no-checkpoint audit for the P2L reproduction package.

This audit verifies the package file manifest and recomputes the numerical
headline claims from the lightweight CSV/TSV evidence bundled with a release.
The deeper deterministic audit additionally checks raw logs, generated beams,
the checkpoint, and candidate artifacts when those external assets exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/results"
PAPER = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
MANIFEST = ROOT / "RELEASE_SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 0.011) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(f"expected {expected}, observed {actual}")


def audit_manifest() -> int:
    if not MANIFEST.is_file():
        return 0
    checked = 0
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"manifest target missing: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"manifest hash mismatch: {relative}")
        checked += 1
    return checked


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def audit_matched_all16() -> dict:
    rows = read_csv(RESULTS / "structnar_matched_all16_local_lambda0.csv")
    grouped = {
        decoder: [row for row in rows if row["decoder"] == decoder]
        for decoder in ("Sequential", "P2L")
    }
    if any(len(records) != 16 for records in grouped.values()):
        raise AssertionError("matched all-16 evidence is incomplete")
    sequential = {
        row["task"]: float(row["Recall@10"]) for row in grouped["Sequential"]
    }
    p2l = {row["task"]: float(row["Recall@10"]) for row in grouped["P2L"]}
    if set(sequential) != set(p2l):
        raise AssertionError("matched task sets differ")
    seq_macro = 100.0 * sum(sequential.values()) / len(sequential)
    p2l_macro = 100.0 * sum(p2l.values()) / len(p2l)
    positive = sum(p2l[task] > sequential[task] for task in sequential)
    close(seq_macro, 39.36)
    close(p2l_macro, 42.83)
    if positive != 16:
        raise AssertionError(f"expected 16 positive matched tasks, found {positive}")
    return {
        "tasks": 16,
        "positive": positive,
        "sequential_macro_r10": seq_macro,
        "p2l_macro_r10": p2l_macro,
    }


def audit_scale() -> dict:
    rows = read_csv(RESULTS / "structnar_cirr_end_to_end_scaling.csv")
    by_decoder = {
        decoder: {int(row["candidate_count"]): row for row in rows if row["decoder"] == decoder}
        for decoder in ("sequential", "p2l")
    }
    if set(by_decoder["sequential"]) != set(by_decoder["p2l"]):
        raise AssertionError("scale candidate grids differ")
    sizes = sorted(by_decoder["p2l"])
    if sizes != [21_551, 100_000, 500_000, 1_000_000, 5_609_079]:
        raise AssertionError(f"unexpected scale grid: {sizes}")
    generation_reductions = []
    online_reductions = []
    for size in sizes:
        seq = by_decoder["sequential"][size]
        p2l = by_decoder["p2l"][size]
        generation_reductions.append(
            100.0
            * (1.0 - float(p2l["generation_seconds"]) / float(seq["generation_seconds"]))
        )
        online_reductions.append(
            100.0
            * (
                1.0
                - float(p2l["online_decode_to_rerank_ms_per_query"])
                / float(seq["online_decode_to_rerank_ms_per_query"])
            )
        )
    close(min(generation_reductions), 51.5, tolerance=0.06)
    close(max(generation_reductions), 53.8, tolerance=0.06)
    close(min(online_reductions), 50.5, tolerance=0.06)
    close(max(online_reductions), 53.0, tolerance=0.06)
    return {
        "sizes": sizes,
        "generation_reduction_percent": [min(generation_reductions), max(generation_reductions)],
        "online_d2r_reduction_percent": [min(online_reductions), max(online_reductions)],
    }


def audit_complete_union() -> dict:
    fashion = {
        "fashion200k_task0",
        "fashion200k_task3",
        "fashioniq_task7",
    }
    genius_rows = [
        row
        for row in read_tsv(RESULTS / "genius_all32.tsv")
        if row["scope"] == "union"
    ]
    p2l_rows = read_tsv(
        RESULTS / "structnar_p2l_d3_rqc_cosine_w20_all32.tsv"
    )
    if len(genius_rows) != 16:
        raise AssertionError(
            f"expected 16 reproduced GENIUS UNION tasks, found {len(genius_rows)}"
        )
    genius_by_task = {}
    for row in genius_rows:
        task = row["dataset"]
        metric = "Recall@10" if task in fashion else "Recall@5"
        genius_by_task[task] = float(row[metric].rstrip("%"))

    p2l_by_task_metric = {
        (row["Dataset"], row["Metric"]): 100.0 * float(row["Union"])
        for row in p2l_rows
    }
    p2l_by_task = {}
    for task in genius_by_task:
        metric = "Recall@10" if task in fashion else "Recall@5"
        key = (task, metric)
        if key not in p2l_by_task_metric:
            raise AssertionError(f"missing P2L UNION evidence for {task} {metric}")
        p2l_by_task[task] = p2l_by_task_metric[key]

    genius = sum(genius_by_task.values()) / len(genius_by_task)
    p2l = sum(p2l_by_task.values()) / len(p2l_by_task)
    close(genius, 33.92)
    close(p2l, 39.20)
    positive = sum(
        p2l_by_task[task] > genius_by_task[task] for task in genius_by_task
    )
    if positive != 16:
        raise AssertionError(f"expected 16 positive UNION tasks, found {positive}")
    return {"genius_r_macro": genius, "p2l_r_macro": p2l, "positive": positive}


def audit_flatip() -> dict:
    rows = read_csv(RESULTS / "structnar_flatip_gpu_scaling.csv")
    largest = max(rows, key=lambda row: int(row["candidate_count"]))
    if int(largest["candidate_count"]) != 5_609_079:
        raise AssertionError("missing 5.61M FlatIP evidence")
    close(float(largest["recall_at_10"]), 56.04)
    close(float(largest["search_ms_per_query"]), 1.54)
    return {
        "candidate_count": int(largest["candidate_count"]),
        "recall_at_10": float(largest["recall_at_10"]),
        "search_ms_per_query": float(largest["search_ms_per_query"]),
    }


def audit_union_frontier_runtime() -> dict:
    rows = read_csv(RESULTS / "p2l_vldb_union_frontier_runtime.csv")
    expected = {
        (task, decoder)
        for task in ("NIGHTS-4", "EDIS-2", "WebQA-1")
        for decoder in ("Sequential+PCAA", "P2L+PCAA")
    }
    keyed = {(row["task"], row["decoder"]): row for row in rows}
    if len(rows) != 6 or set(keyed) != expected:
        raise AssertionError("UNION frontier/runtime evidence is incomplete")
    expected_claims = {
        "NIGHTS-4": (1.51, 51.3, 48.0, 499),
        "EDIS-2": (16.69, 28.6, 21.0, 11_097),
        "WebQA-1": (19.64, 31.1, 23.9, 10_339),
    }
    report = {}
    for task, (expected_delta, expected_reduction, expected_p95, expected_leaves) in expected_claims.items():
        seq = keyed[(task, "Sequential+PCAA")]
        p2l = keyed[(task, "P2L+PCAA")]
        delta = 100.0 * (float(p2l["r10"]) - float(seq["r10"]))
        reduction = 100.0 * (
            1.0 - float(p2l["generation_seconds"]) / float(seq["generation_seconds"])
        )
        p95_reduction = 100.0 * (
            1.0 - float(p2l["p95_ms_per_query"]) / float(seq["p95_ms_per_query"])
        )
        leaves = round(float(p2l["leaves_mean"]))
        close(delta, expected_delta)
        close(reduction, expected_reduction, tolerance=0.051)
        close(p95_reduction, expected_p95, tolerance=0.051)
        if leaves != expected_leaves:
            raise AssertionError(f"unexpected {task} mean frontier: {leaves}")
        report[task] = {
            "r10_delta_points": delta,
            "generation_reduction_percent": reduction,
            "p95_reduction_percent": p95_reduction,
            "p2l_leaves_mean": leaves,
        }
    return {"cells": len(rows), "tasks": report}


def audit_paper_strings() -> None:
    text = " ".join(PAPER.read_text().split())
    required = (
        "39.36 to 42.83",
        "51.5\\%--53.8\\%",
        "50.5\\%--53.0\\%",
        "33.92 to 39.20",
        "It remains approximate over the full index because prefix search can discard a relevant key before the switch",
        "11,097 and 10,339 leaves/query",
        "falls by 28.6\\% and 31.1\\%",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise AssertionError(f"paper headline fragments missing: {missing}")


def main() -> int:
    report = {
        "verdict": "PASS",
        "manifest_files_checked": audit_manifest(),
        "matched_all16": audit_matched_all16(),
        "cirr_scale": audit_scale(),
        "complete_union": audit_complete_union(),
        "flatip_5_61m": audit_flatip(),
        "union_frontier_runtime": audit_union_frontier_runtime(),
    }
    audit_paper_strings()
    optional = RESULTS / "p2l_gpu_flat_trie_baseline.json"
    if optional.is_file():
        payload = json.loads(optional.read_text())
        if payload.get("verdict") != "PASS":
            raise AssertionError("packaged GPU-flat baseline did not pass")
        equivalence = payload.get("equivalence", {})
        comparisons = equivalence.get("comparisons", [])
        if (
            equivalence.get("verdict") != "PASS"
            or len(comparisons) != 4
            or not all(record.get("exact") for record in comparisons)
            or any(float(record.get("max_abs_difference", -1)) != 0.0 for record in comparisons)
        ):
            raise AssertionError("packaged GPU-flat full-manifest equivalence is invalid")
        records = {record["label"]: record for record in payload.get("records", [])}
        if set(records) != {
            "Canonical Sequential+PCAA",
            "GPU-flat Sequential+PCAA",
            "P2L+PCAA",
        }:
            raise AssertionError("packaged GPU-flat timing records are incomplete")
        reduction = 100.0 * (
            1.0
            - float(records["P2L+PCAA"]["generation_ms_per_query"])
            / float(records["GPU-flat Sequential+PCAA"]["generation_ms_per_query"])
        )
        close(reduction, 53.4, tolerance=0.01)
        report["gpu_flat_trie_baseline"] = {
            **payload["comparison"],
            "full_manifest_arrays_exact": len(comparisons),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
