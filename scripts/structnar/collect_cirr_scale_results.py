#!/usr/bin/env python3
"""Collect the fixed-query CIRR candidate-pool scaling experiment."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
OUT = ROOT / "docs/results/structnar_cirr_end_to_end_scaling.csv"
LATEX = ROOT / "VLDB/vldb2027/supplementary/generated/cirr_scale_matrix.tex"
SIZES = (21_551, 100_000, 500_000, 1_000_000, 5_609_079)
ACTIVE_RUN = ROOT / "docs/results/structnar_vldb_active_run_id.txt"


def last_float(pattern: str, text: str) -> float:
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"missing pattern: {pattern}")
    return float(matches[-1])


rows = []
if not ACTIVE_RUN.exists():
    raise SystemExit(f"missing active-run witness: {ACTIVE_RUN}")
run_id = ACTIVE_RUN.read_text().strip()
if not run_id:
    raise SystemExit(f"empty active-run witness: {ACTIVE_RUN}")
state_dir = ROOT / "docs/results/vldb_scale" / run_id
expected = {
    (decoder, size)
    for decoder in ("p2l", "sequential")
    for size in SIZES
}
for decoder in ("p2l", "sequential"):
    for size in SIZES:
        log = LOG_DIR / f"structnar_cirr_scale_w20_{decoder}_{size:07d}.log"
        marker = state_dir / f"{decoder}_{size:07d}.ok"
        if not marker.exists():
            continue
        marker_fields = {}
        for line in marker.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                marker_fields[key] = value
        if not log.exists():
            continue
        if Path(marker_fields.get("log", "")) != log:
            raise SystemExit(f"marker/log mismatch for {decoder}, {size}: {marker}")
        observed_hash = hashlib.sha256(log.read_bytes()).hexdigest()
        if marker_fields.get("sha256") != observed_hash:
            raise SystemExit(f"log hash mismatch for {decoder}, {size}: {log}")
        content = log.read_text(errors="replace")
        if not re.search(
            r"Candidate Index Setup: seconds=[0-9.]+ "
            r"excluded_from_generation=true",
            content,
        ):
            raise SystemExit(
                "scale log lacks an index-setup timing-boundary witness: "
                f"{log}"
            )
        if re.search(
            r"candidate tree.*stale; rebuilding|Saved candidate tree from",
            content,
        ):
            raise SystemExit(
                "offline Trie construction contaminates online generation time: "
                f"{log}"
            )
        if "Retriever: Mean Recall@10:" not in content:
            continue
        row = {
            "decoder": decoder,
            "candidate_count": size,
            "queries": int(
                last_float(r"Generation Metrics:.*?examples=([0-9]+)", content)
            ),
            "generation_seconds": last_float(
                r"Generation Metrics: seconds=([0-9.]+)", content
            ),
            "generation_p50_ms_per_query": last_float(
                r"Generation Latency Distribution:.*?normalized_query_p50_ms=([0-9.]+)",
                content,
            ),
            "generation_p95_ms_per_query": last_float(
                r"Generation Latency Distribution:.*?normalized_query_p95_ms=([0-9.]+)",
                content,
            ),
            "generation_p99_ms_per_query": last_float(
                r"Generation Latency Distribution:.*?normalized_query_p99_ms=([0-9.]+)",
                content,
            ),
            "retrieval_seconds": last_float(
                r"Retrieval Metrics: seconds=([0-9.]+)", content
            ),
            "retrieval_p50_ms_per_query": last_float(
                r"Retrieval Latency Distribution:.*?p50_ms=([0-9.]+)", content
            ),
            "retrieval_p95_ms_per_query": last_float(
                r"Retrieval Latency Distribution:.*?p95_ms=([0-9.]+)", content
            ),
            "retrieval_p99_ms_per_query": last_float(
                r"Retrieval Latency Distribution:.*?p99_ms=([0-9.]+)", content
            ),
            "retrieval_mean_ms_per_query": last_float(
                r"Retrieval Latency Distribution:.*?mean_ms=([0-9.]+)", content
            ),
            "recall_at_1": last_float(r"Retriever: Mean Recall@1: ([0-9.]+)", content),
            "recall_at_5": last_float(r"Retriever: Mean Recall@5: ([0-9.]+)", content),
            "recall_at_10": last_float(
                r"Retriever: Mean Recall@10: ([0-9.]+)", content
            ),
            "gt_in_beam_at_1": last_float(r"beam_gt_recall@1=([0-9.]+)", content),
            "gt_in_beam_at_5": last_float(r"beam_gt_recall@5=([0-9.]+)", content),
            "gt_in_beam_at_10": last_float(r"beam_gt_recall@10=([0-9.]+)", content),
            "gt_in_beam_at_50": last_float(r"beam_gt_recall@50=([0-9.]+)", content),
            "unique_codes": int(
                last_float(r"unique_codes=([0-9]+)", content)
            ),
            "collision_item_rate": last_float(
                r"collision_item_rate=([0-9.]+)", content
            ),
            "peak_allocated_mib": last_float(
                r"peak_allocated_mib=([0-9.]+)", content
            ),
            "source_log": str(log.relative_to(ROOT)),
            "source_log_sha256": observed_hash,
        }
        row["generation_ms_per_query"] = (
            1000.0 * float(row["generation_seconds"]) / int(row["queries"])
        )
        # Retrieval Metrics includes one-time candidate lookup-cache preparation,
        # whereas Retrieval Latency Distribution measures only per-query lookup
        # and reranking after that preparation. Keep both scopes explicit.
        row["manifest_decode_to_rerank_seconds"] = float(
            row["generation_seconds"]
        ) + float(row["retrieval_seconds"])
        row["manifest_decode_to_rerank_ms_per_query"] = (
            1000.0
            * float(row["manifest_decode_to_rerank_seconds"])
            / int(row["queries"])
        )
        row["online_decode_to_rerank_ms_per_query"] = (
            float(row["generation_ms_per_query"])
            + float(row["retrieval_mean_ms_per_query"])
        )
        budget = re.findall(
            r"TCIS Query Enumeration:.*?candidates_mean=([0-9.]+).*?candidates_p95=([0-9.]+).*?candidates_p99=([0-9.]+)",
            content,
        )
        if budget:
            row["leaves_mean"], row["leaves_p95"], row["leaves_p99"] = map(
                float, budget[-1]
            )
        else:
            row["leaves_mean"] = row["leaves_p95"] = row["leaves_p99"] = ""
        rows.append(row)

observed = {(str(row["decoder"]), int(row["candidate_count"])) for row in rows}
missing = sorted(expected - observed)
if missing:
    raise SystemExit(f"incomplete CIRR scale matrix; missing {missing}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

LATEX.parent.mkdir(parents=True, exist_ok=True)
latex = [
    "% Generated by collect_cirr_scale_results.py from the hash-verified matrix.",
    "\\begin{widetable}",
    "\\centering",
    "\\scriptsize",
    "\\setlength{\\tabcolsep}{3pt}",
    "\\caption{Measured CIRR-7 candidate-pool scaling at $\\lambda=20$. Gen. is decoder-side latency and Online D2R adds warm per-query bucket lookup and reranking. Manifest D2R is the full fixed-manifest elapsed time, including one-time candidate lookup-cache preparation. Frontier statistics are defined only for P2L.}",
    "\\label{tab:supp-cirr-scale-matrix}",
    "\\begin{tabular}{rlrrrrrrrr}",
    "\\toprule",
    "Candidates & Decoder & R@10 & Gen. (ms/q) & Online D2R (ms/q) & Manifest D2R (s) & Leaves mean & Leaves P95 & Leaves P99 & Peak (GiB) \\\\",
    "\\midrule",
]
for row in sorted(rows, key=lambda value: (int(value["candidate_count"]), str(value["decoder"]))):
    decoder_name = {"p2l": "P2L", "sequential": "Sequential"}[str(row["decoder"])]
    frontier = [row["leaves_mean"], row["leaves_p95"], row["leaves_p99"]]
    shown = ["--" if value == "" else f"{float(value):,.0f}" for value in frontier]
    latex.append(
        f"{int(row['candidate_count']):,} & {decoder_name} & "
        f"{100.0 * float(row['recall_at_10']):.2f} & "
        f"{float(row['generation_ms_per_query']):.2f} & "
        f"{float(row['online_decode_to_rerank_ms_per_query']):.2f} & "
        f"{float(row['manifest_decode_to_rerank_seconds']):.2f} & "
        f"{shown[0]} & {shown[1]} & {shown[2]} & "
        f"{float(row['peak_allocated_mib']) / 1024.0:.2f} \\\\"
    )
latex += ["\\bottomrule", "\\end{tabular}", "\\end{widetable}"]
LATEX.write_text("\n".join(latex) + "\n")
print(f"wrote {len(rows)} rows to {OUT}")
print(LATEX)
