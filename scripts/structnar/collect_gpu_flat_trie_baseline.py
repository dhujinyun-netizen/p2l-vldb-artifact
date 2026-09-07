#!/usr/bin/env python3
"""Collect the faithful GPU-flat Sequential baseline after its exactness audit.

The collector refuses to emit paper-facing artifacts unless the full-manifest
equivalence audit passed and the runner recorded a clean terminal state.  It
joins the optimized run with the canonical Sequential+PCAA and P2L+PCAA logs
used by the clean CIRR-7 LOCAL scaling study.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "profile_output/p2l_gpu_trie_baseline"
STATE = PROFILE_DIR / "state.txt"
EQUIVALENCE = PROFILE_DIR / "equivalence.json"
GPU_LOG = PROFILE_DIR / "cirr_local_full.log"
CANONICAL_LOG = ROOT / "logs/structnar_cirr_scale_w20_sequential_0021551.log"
P2L_LOG = ROOT / "logs/structnar_cirr_scale_w20_p2l_0021551.log"
JSON_OUT = ROOT / "docs/results/p2l_gpu_flat_trie_baseline.json"
CSV_OUT = ROOT / "docs/results/p2l_gpu_flat_trie_baseline.csv"
TEX_OUT = (
    ROOT
    / "VLDB/vldb2027/supplementary/generated/gpu_flat_trie_baseline.tex"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_match(pattern: str, text: str, path: Path) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise RuntimeError(f"Missing required metric in {path}: {pattern}")
    return matches[-1]


def parse_log(path: Path, label: str) -> dict:
    text = path.read_text(errors="replace").replace("\r", "\n")
    generation = final_match(
        r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+) "
        r"examples_per_second=([0-9.]+)",
        text,
        path,
    )
    memory = final_match(
        r"Generation CUDA Memory: peak_allocated_mib=([0-9.]+) "
        r"peak_reserved_mib=([0-9.]+)",
        text,
        path,
    )
    latency = final_match(
        r"Generation Latency Distribution: batches=([0-9]+) "
        r"warmup_batches_excluded=([0-9]+).*?"
        r"normalized_query_p50_ms=([0-9.]+) "
        r"normalized_query_p95_ms=([0-9.]+) "
        r"normalized_query_p99_ms=([0-9.]+)",
        text,
        path,
    )
    stage_matches = list(
        re.finditer(
            r"Generation Stage Breakdown: blocks=([0-9]+) "
            r"model_total_seconds=([0-9.]+) "
            r"constraint_expand_total_seconds=([0-9.]+) "
            r"model_share=([0-9.]+) constraint_expand_share=([0-9.]+)",
            text,
            flags=re.MULTILINE,
        )
    )
    # The GPU-flat path deliberately profiles the end-to-end decoder timer;
    # the older component profiler is not instrumented inside its fused
    # expansion routine.  Component fields are therefore optional and never
    # used in the paper-facing comparison table.
    stage = stage_matches[-1] if stage_matches else None
    recalls = {
        cutoff: float(
            final_match(
                rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text, path
            ).group(1)
        )
        for cutoff in (1, 5, 10)
    }
    seconds = float(generation.group(1))
    examples = int(generation.group(2))
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "generation_seconds": seconds,
        "examples": examples,
        "generation_ms_per_query": 1000.0 * seconds / examples,
        "examples_per_second": float(generation.group(3)),
        "peak_allocated_mib": float(memory.group(1)),
        "peak_reserved_mib": float(memory.group(2)),
        "latency_batches": int(latency.group(1)),
        "warmup_batches_excluded": int(latency.group(2)),
        "p50_ms_per_query": float(latency.group(3)),
        "p95_ms_per_query": float(latency.group(4)),
        "p99_ms_per_query": float(latency.group(5)),
        "blocks": int(stage.group(1)) if stage else None,
        "model_total_seconds": float(stage.group(2)) if stage else None,
        "constraint_expand_total_seconds": float(stage.group(3)) if stage else None,
        "model_share": float(stage.group(4)) if stage else None,
        "constraint_expand_share": float(stage.group(5)) if stage else None,
        "recall_at_1": recalls[1],
        "recall_at_5": recalls[5],
        "recall_at_10": recalls[10],
    }


def write_csv(records: list[dict]) -> None:
    fields = [
        "label",
        "generation_seconds",
        "examples",
        "generation_ms_per_query",
        "p50_ms_per_query",
        "p95_ms_per_query",
        "p99_ms_per_query",
        "peak_allocated_mib",
        "model_total_seconds",
        "constraint_expand_total_seconds",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "path",
        "sha256",
    ]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fields})


def write_tex(records: list[dict]) -> None:
    by_label = {record["label"]: record for record in records}
    labels = [
        ("Canonical Sequential+PCAA", "Canonical Sequential+PCAA"),
        ("GPU-flat Sequential+PCAA", "GPU-flat Sequential+PCAA"),
        ("P2L+PCAA", r"P2L+PCAA ($s=3$)"),
    ]
    rows = []
    for key, shown in labels:
        record = by_label[key]
        rows.append(
            f"{shown} & {100 * record['recall_at_10']:.2f} & "
            f"{record['generation_ms_per_query']:.2f} & "
            f"{record['p95_ms_per_query']:.2f} & "
            f"{record['peak_allocated_mib'] / 1024.0:.2f} \\\\"
        )
    table = "\n".join(
        [
            "% Generated by collect_gpu_flat_trie_baseline.py after full exactness audit.",
            r"\begin{widetable}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\caption{Faithful optimized level-wise Trie baseline on the 4,170-query CIRR-7 LOCAL manifest. The GPU-flat engine replaces Python Trie traversal with level-wise CSR tensors and batched GPU gathers without changing the Sequential+PCAA scoring rule. Its saved top-50 identifiers, scores, query embeddings, and query IDs are byte-identical to the canonical Sequential+PCAA run. P2L changes the suffix scoring policy and is included as the submitted operating point. Generation and P95 are decoder-side milliseconds per query; peak is allocated GPU memory.}",
            r"\label{tab:supp-gpu-flat-trie}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Decoder & R@10 & Gen. & P95 & Peak (GiB) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{widetable}",
            "",
        ]
    )
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(table)


def main() -> int:
    state = STATE.read_text().strip()
    if not state.startswith("status=complete_exact "):
        raise RuntimeError(f"Runner has no clean terminal state: {state}")
    equivalence = json.loads(EQUIVALENCE.read_text())
    if equivalence.get("verdict") != "PASS":
        raise RuntimeError("Full-manifest equivalence audit did not pass")
    if not all(record.get("exact") for record in equivalence["comparisons"]):
        raise RuntimeError("At least one saved full-manifest array is not exact")

    records = [
        parse_log(CANONICAL_LOG, "Canonical Sequential+PCAA"),
        parse_log(GPU_LOG, "GPU-flat Sequential+PCAA"),
        parse_log(P2L_LOG, "P2L+PCAA"),
    ]
    if len({record["examples"] for record in records}) != 1:
        raise RuntimeError("Compared logs do not use the same query count")
    canonical, gpu_flat, p2l = records
    for cutoff in (1, 5, 10):
        key = f"recall_at_{cutoff}"
        if canonical[key] != gpu_flat[key]:
            raise RuntimeError(f"GPU-flat changed {key}")

    comparison = {
        "gpu_flat_speedup_over_canonical": (
            canonical["generation_ms_per_query"]
            / gpu_flat["generation_ms_per_query"]
        ),
        "p2l_speedup_over_gpu_flat": (
            gpu_flat["generation_ms_per_query"] / p2l["generation_ms_per_query"]
        ),
        "p2l_reduction_vs_gpu_flat_percent": 100.0
        * (
            1.0
            - p2l["generation_ms_per_query"]
            / gpu_flat["generation_ms_per_query"]
        ),
    }
    payload = {
        "verdict": "PASS",
        "scope": "CIRR-7 LOCAL, 4,170 queries, B=K=50, lambda=20",
        # Do not copy the full state line here: it contains this JSON's hash
        # and would therefore become stale whenever the collector is rerun.
        "runner_status": "complete_exact",
        "runner_state_path": str(STATE.relative_to(ROOT)),
        "equivalence_path": str(EQUIVALENCE.relative_to(ROOT)),
        "equivalence_sha256": sha256(EQUIVALENCE),
        "equivalence": equivalence,
        "records": records,
        "comparison": comparison,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(records)
    write_tex(records)
    print(json.dumps(comparison, indent=2))
    print(f"json={JSON_OUT.relative_to(ROOT)}")
    print(f"csv={CSV_OUT.relative_to(ROOT)}")
    print(f"tex={TEX_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
