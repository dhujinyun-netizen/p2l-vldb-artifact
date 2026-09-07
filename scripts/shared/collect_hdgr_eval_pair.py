#!/usr/bin/env python3
"""Collect GPT-HDGR no-rerank and rerank M-BEIR eval TSVs into one record."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATASET_ORDER = {
    "visualnews_task0": 1,
    "mscoco_task0": 2,
    "fashion200k_task0": 3,
    "webqa_task1": 4,
    "edis_task2": 5,
    "webqa_task2": 6,
    "visualnews_task3": 7,
    "mscoco_task3": 8,
    "fashion200k_task3": 9,
    "nights_task4": 10,
    "oven_task6": 11,
    "infoseek_task6": 12,
    "fashioniq_task7": 13,
    "cirr_task7": 14,
    "oven_task8": 15,
    "infoseek_task8": 16,
}
METRIC_ORDER = {
    "Recall@1": 1,
    "Recall@5": 2,
    "Recall@10": 3,
    "Recall@20": 4,
    "Recall@50": 5,
}

Key = Tuple[str, str, str, str, str, str]


def parse_float(x: str):
    if x is None or x == "" or x == "N/A":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x):
    if x is None:
        return ""
    return f"{x:.4f}"


def safe_name(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in x)


def read_tsv(path: Path) -> Tuple[Dict[Key, float], Dict[Key, dict]]:
    values: Dict[Key, float] = {}
    meta: Dict[Key, dict] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"TaskID", "Task", "Dataset", "Split", "Metric", "CandPool", "Value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            key: Key = (
                row["TaskID"],
                row["Task"],
                row["Dataset"],
                row["Split"],
                row["Metric"],
                row["CandPool"],
            )
            val = parse_float(row.get("Value", ""))
            if val is None:
                continue
            values[key] = val
            meta[key] = row
    return values, meta


def sort_key(key: Key):
    task_id, _task, dataset, split, metric, cand = key
    try:
        tid = int(task_id)
    except Exception:
        tid = 999
    return (
        tid,
        DATASET_ORDER.get(dataset.lower(), 99),
        1 if split.lower() == "val" else 2 if split.lower() == "test" else 99,
        METRIC_ORDER.get(metric, 99),
        cand,
    )


def write_combined(out_path: Path, keys: Iterable[Key], no_vals: Dict[Key, float], rr_vals: Dict[Key, float]):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "TaskID",
            "Task",
            "Dataset",
            "Split",
            "Metric",
            "CandPool",
            "NoRerank",
            "Rerank",
            "Delta_Rerank_minus_NoRerank",
        ])
        for key in keys:
            no = no_vals.get(key)
            rr = rr_vals.get(key)
            delta = None if no is None or rr is None else rr - no
            writer.writerow(list(key) + [fmt(no), fmt(rr), fmt(delta)])


def compute_macro(keys: Iterable[Key], no_vals: Dict[Key, float], rr_vals: Dict[Key, float]):
    by_metric = defaultdict(lambda: {"no": [], "rr": []})
    by_dataset = defaultdict(lambda: {"no": [], "rr": []})
    for key in keys:
        _tid, _task, dataset, _split, metric, _cand = key
        no = no_vals.get(key)
        rr = rr_vals.get(key)
        if no is None or rr is None:
            continue
        by_metric[metric]["no"].append(no)
        by_metric[metric]["rr"].append(rr)
        by_dataset[dataset]["no"].append(no)
        by_dataset[dataset]["rr"].append(rr)

    metric_rows = []
    for metric in sorted(by_metric, key=lambda m: METRIC_ORDER.get(m, 99)):
        no_list = by_metric[metric]["no"]
        rr_list = by_metric[metric]["rr"]
        if not no_list:
            continue
        no_avg = sum(no_list) / len(no_list)
        rr_avg = sum(rr_list) / len(rr_list)
        metric_rows.append((metric, len(no_list), no_avg, rr_avg, rr_avg - no_avg))

    dataset_rows = []
    for dataset in sorted(by_dataset, key=lambda d: DATASET_ORDER.get(d.lower(), 99)):
        no_list = by_dataset[dataset]["no"]
        rr_list = by_dataset[dataset]["rr"]
        if not no_list:
            continue
        no_avg = sum(no_list) / len(no_list)
        rr_avg = sum(rr_list) / len(rr_list)
        dataset_rows.append((dataset, len(no_list), no_avg, rr_avg, rr_avg - no_avg))
    return metric_rows, dataset_rows


def write_macro(path: Path, rows: List[Tuple[str, int, float, float, float]], key_name: str):
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([key_name, "N", "NoRerankAvg", "RerankAvg", "Delta"])
        for name, n, no, rr, delta in rows:
            writer.writerow([name, n, fmt(no), fmt(rr), fmt(delta)])


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_per_dataset(out_dir: Path, keys: List[Key], no_vals: Dict[Key, float], rr_vals: Dict[Key, float]):
    per_dir = out_dir / "per_dataset"
    per_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for key in keys:
        grouped[key[2]].append(key)

    index_rows = []
    for dataset in sorted(grouped, key=lambda d: DATASET_ORDER.get(d.lower(), 99)):
        ds_keys = sorted(grouped[dataset], key=sort_key)
        tsv_path = per_dir / f"{safe_name(dataset)}.tsv"
        md_path = per_dir / f"{safe_name(dataset)}.md"

        with tsv_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "TaskID", "Task", "Dataset", "Split", "Metric", "CandPool",
                "NoRerank", "Rerank", "Delta_Rerank_minus_NoRerank",
            ])
            for key in ds_keys:
                no = no_vals.get(key)
                rr = rr_vals.get(key)
                delta = None if no is None or rr is None else rr - no
                writer.writerow(list(key) + [fmt(no), fmt(rr), fmt(delta)])

        no_list, rr_list = [], []
        md_rows = []
        for key in ds_keys:
            task_id, task, dataset_name, split, metric, cand = key
            no = no_vals.get(key)
            rr = rr_vals.get(key)
            delta = None if no is None or rr is None else rr - no
            if no is not None and rr is not None:
                no_list.append(no)
                rr_list.append(rr)
            md_rows.append([task_id, metric, cand, fmt(no), fmt(rr), fmt(delta)])

        no_avg = sum(no_list) / len(no_list) if no_list else None
        rr_avg = sum(rr_list) / len(rr_list) if rr_list else None
        avg_delta = None if no_avg is None or rr_avg is None else rr_avg - no_avg
        index_rows.append((dataset, len(ds_keys), no_avg, rr_avg, avg_delta, str(tsv_path), str(md_path)))

        md = []
        md.append(f"# {dataset} eval record")
        md.append("")
        md.append(f"- Metrics: {len(ds_keys)}")
        md.append(f"- Macro no-rerank: {fmt(no_avg)}")
        md.append(f"- Macro rerank: {fmt(rr_avg)}")
        md.append(f"- Delta: {fmt(avg_delta)}")
        md.append("")
        md.append(md_table(["TaskID", "Metric", "CandPool", "No-rerank", "Rerank", "Delta"], md_rows))
        md.append("")
        md_path.write_text("\n".join(md))

    with (out_dir / "per_dataset_index.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Dataset", "N", "NoRerankAvg", "RerankAvg", "Delta", "TSV", "MD"])
        for dataset, n, no_avg, rr_avg, delta, tsv_path, md_path in index_rows:
            writer.writerow([dataset, n, fmt(no_avg), fmt(rr_avg), fmt(delta), tsv_path, md_path])
    return index_rows


def write_summary(path: Path, args, keys: List[Key], no_vals: Dict[Key, float], rr_vals: Dict[Key, float], metric_rows, dataset_rows, per_dataset_rows):
    detail_rows = []
    for key in keys:
        task_id, task, dataset, split, metric, cand = key
        no = no_vals.get(key)
        rr = rr_vals.get(key)
        delta = None if no is None or rr is None else rr - no
        detail_rows.append([task_id, dataset, metric, cand, fmt(no), fmt(rr), fmt(delta)])

    metric_md_rows = [[m, str(n), fmt(no), fmt(rr), fmt(d)] for m, n, no, rr, d in metric_rows]
    dataset_md_rows = [[dset, str(n), fmt(no), fmt(rr), fmt(d)] for dset, n, no, rr, d in dataset_rows]

    text = []
    text.append(f"# GPT-HDGR local M-BEIR eval record")
    text.append("")
    text.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    text.append(f"- Run ID: `{args.run_id}`")
    text.append(f"- Eval tag: `{args.eval_tag}`")
    text.append(f"- Checkpoint: `{args.ckpt}`")
    text.append(f"- RQ codebook: `{args.rq_path}`")
    text.append(f"- Num beams: `{args.num_beams}`")
    text.append(f"- Eval batch size: `{args.eval_batch_size}`")
    text.append(f"- No-rerank TSV: `{args.norerank_tsv}`")
    text.append(f"- Rerank TSV: `{args.rerank_tsv}`")
    text.append("")
    text.append("## Macro average by metric")
    text.append("")
    text.append(md_table(["Metric", "N", "No-rerank", "Rerank", "Delta"], metric_md_rows))
    text.append("")
    text.append("## Macro average by dataset/task slot")
    text.append("")
    text.append(md_table(["Dataset", "N metrics", "No-rerank", "Rerank", "Delta"], dataset_md_rows))
    text.append("")
    text.append("## Per-dataset record files")
    text.append("")
    per_rows = [[dset, str(n), fmt(no), fmt(rr), fmt(delta), f"`per_dataset/{safe_name(dset)}.tsv`", f"`per_dataset/{safe_name(dset)}.md`"] for dset, n, no, rr, delta, _tsv, _md in per_dataset_rows]
    text.append(md_table(["Dataset", "N", "No-rerank", "Rerank", "Delta", "TSV", "MD"], per_rows))
    text.append("")
    text.append("## Full per-metric comparison")
    text.append("")
    text.append(md_table(["TaskID", "Dataset", "Metric", "CandPool", "No-rerank", "Rerank", "Delta"], detail_rows))
    text.append("")
    path.write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norerank-tsv", required=True, type=Path)
    parser.add_argument("--rerank-tsv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-tag", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rq-path", required=True)
    parser.add_argument("--num-beams", required=True)
    parser.add_argument("--eval-batch-size", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    no_vals, no_meta = read_tsv(args.norerank_tsv)
    rr_vals, rr_meta = read_tsv(args.rerank_tsv)
    keys = sorted(set(no_vals) | set(rr_vals), key=sort_key)

    combined_path = args.out_dir / "combined_norerank_vs_rerank.tsv"
    write_combined(combined_path, keys, no_vals, rr_vals)

    metric_rows, dataset_rows = compute_macro(keys, no_vals, rr_vals)
    write_macro(args.out_dir / "macro_by_metric.tsv", metric_rows, "Metric")
    write_macro(args.out_dir / "macro_by_dataset.tsv", dataset_rows, "Dataset")
    per_dataset_rows = write_per_dataset(args.out_dir, keys, no_vals, rr_vals)
    write_summary(args.out_dir / "SUMMARY.md", args, keys, no_vals, rr_vals, metric_rows, dataset_rows, per_dataset_rows)

    print(f"[collect] wrote {combined_path}")
    print(f"[collect] wrote {args.out_dir / 'SUMMARY.md'}")
    print(f"[collect] wrote per-dataset records in {args.out_dir / 'per_dataset'}")
    print(f"[collect] rows={len(keys)}")


if __name__ == "__main__":
    main()
