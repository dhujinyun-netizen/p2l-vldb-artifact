#!/usr/bin/env python3
"""Profile an exhaustive continuous-vector reference on the CIRR scale pools.

The retained reference uses independently sampled query prompts from the same
frozen encoder configuration as StructNAR, FP16 execution, batch size 8, and
GPU 0. Its query IDs and candidate embeddings match the scale pools, but its
realized query embeddings differ from the semantic-ID run. Candidate transfer
is reported as offline index-load time and excluded from synchronized search
latency, matching the paper's online timing boundary. The explicit FP16 cast
keeps the 5.61M pool within the memory budget of the designated GPU while
leaving room for the query and top-k workspaces.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "gen_code/STRUCTNAR/Large/Instruct"
QREL = ROOT / "mbeir_data/qrels/test/mbeir_cirr_task7_test_qrels.txt"
OUT = ROOT / "docs/results/structnar_flatip_gpu_scaling.csv"
LATEX = ROOT / "VLDB/vldb2027/supplementary/generated/flatip_gpu_scaling.tex"
QUERY_ID_BASE = 500_000
CANDIDATE_ID_BASE = 10_000_000
BATCH_SIZE = 8
TOPK = 50

SOURCES = (
    (
        21_551,
        BASE / "structnar_cirr_scale_0021551/test/mbeir_cirr_task7_test_embeddings.npy",
        BASE / "structnar_cirr_scale_0021551/test/mbeir_cirr_task7_test_ids.npy",
        BASE / "structnar_cirr_scale_0021551/cand_pool/mbeir_cirrscale_0021551_cand_pool_embeddings.npy",
        BASE / "structnar_cirr_scale_0021551/cand_pool/mbeir_cirrscale_0021551_cand_pool_ids.npy",
    ),
    (
        100_000,
        BASE / "structnar_cirr_scale_0100000/test/mbeir_cirr_task7_test_embeddings.npy",
        BASE / "structnar_cirr_scale_0100000/test/mbeir_cirr_task7_test_ids.npy",
        BASE / "structnar_cirr_scale_0100000/cand_pool/mbeir_cirrscale_0100000_cand_pool_embeddings.npy",
        BASE / "structnar_cirr_scale_0100000/cand_pool/mbeir_cirrscale_0100000_cand_pool_ids.npy",
    ),
    (
        500_000,
        BASE / "structnar_cirr_scale_0500000/test/mbeir_cirr_task7_test_embeddings.npy",
        BASE / "structnar_cirr_scale_0500000/test/mbeir_cirr_task7_test_ids.npy",
        BASE / "structnar_cirr_scale_0500000/cand_pool/mbeir_cirrscale_0500000_cand_pool_embeddings.npy",
        BASE / "structnar_cirr_scale_0500000/cand_pool/mbeir_cirrscale_0500000_cand_pool_ids.npy",
    ),
    (
        1_000_000,
        BASE / "structnar_cirr_scale_1000000/test/mbeir_cirr_task7_test_embeddings.npy",
        BASE / "structnar_cirr_scale_1000000/test/mbeir_cirr_task7_test_ids.npy",
        BASE / "structnar_cirr_scale_1000000/cand_pool/mbeir_cirrscale_1000000_cand_pool_embeddings.npy",
        BASE / "structnar_cirr_scale_1000000/cand_pool/mbeir_cirrscale_1000000_cand_pool_ids.npy",
    ),
    (
        5_609_079,
        BASE / "structnar_cirr_scale_5609079/test/mbeir_cirr_task7_test_embeddings.npy",
        BASE / "structnar_cirr_scale_5609079/test/mbeir_cirr_task7_test_ids.npy",
        BASE / "structnar_cirr_scale_5609079/cand_pool/mbeir_union_cand_pool_embeddings.npy",
        BASE / "structnar_cirr_scale_5609079/cand_pool/mbeir_union_cand_pool_ids.npy",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matched-p2l-union",
        action="store_true",
        help=(
            "profile only the 5.61M UNION collection using the exact query "
            "embeddings consumed by the final matched P2L run"
        ),
    )
    return parser.parse_args()


def unhash_qid(value: int) -> str:
    return f"{int(value) // QUERY_ID_BASE}:{int(value) % QUERY_ID_BASE}"


def unhash_did(value: int) -> str:
    return f"{int(value) // CANDIDATE_ID_BASE}:{int(value) % CANDIDATE_ID_BASE}"


def load_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with QREL.open() as handle:
        for line in handle:
            qid, _, did, relevance, _ = line.split()
            if int(relevance) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def evaluate(rows: np.ndarray, query_ids: np.ndarray, candidate_ids: np.ndarray, qrels):
    hits = {1: 0, 5: 0, 10: 0, 50: 0}
    evaluated = 0
    for query_id, result_rows in zip(query_ids, rows):
        relevant = qrels.get(unhash_qid(int(query_id)))
        if not relevant:
            continue
        retrieved = [unhash_did(int(candidate_ids[row])) for row in result_rows]
        evaluated += 1
        for cutoff in hits:
            if relevant.intersection(retrieved[:cutoff]):
                hits[cutoff] += 1
    return evaluated, {key: 100.0 * value / evaluated for key, value in hits.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2023)
    device = torch.device("cuda:0")
    # Initialize the CUDA context before resetting allocator statistics.
    # In the evaluated PyTorch 2.7/CUDA 12.8 build, calling the reset API
    # before the first context initialization raises ``Invalid device``.
    torch.cuda.init()
    qrels = load_qrels()
    output_rows = []
    sources = SOURCES
    output_path = OUT
    latex_path = LATEX
    caption = (
        "Exact FlatIP scaling on GPU 0 as an independently sampled "
        "continuous-search reference. Query IDs and candidate pools match the "
        "semantic-ID scale study, but sampled query prompts and realized query "
        "embeddings differ. Search latency excludes the one-time index "
        "transfer; memory is peak allocated device memory."
    )
    if args.matched_p2l_union:
        experiment = BASE / "structnar_cirr_scale_w20_p2l_5609079"
        sources = (
            (
                5_609_079,
                experiment / "test/mbeir_cirr_task7_test_embeddings.npy",
                experiment / "test/mbeir_cirr_task7_test_ids.npy",
                experiment / "cand_pool/mbeir_union_cand_pool_embeddings.npy",
                experiment / "cand_pool/mbeir_union_cand_pool_ids.npy",
            ),
        )
        output_path = ROOT / "docs/results/structnar_flatip_gpu_matched_cirr_union.csv"
        latex_path = (
            ROOT
            / "VLDB/vldb2027/supplementary/generated/flatip_gpu_matched_cirr_union.tex"
        )
        caption = (
            "Input-matched exact FlatIP on GPU 0 for CIRR-7 over the 5.61M "
            "UNION collection. The query and candidate embedding files are "
            "exactly those used by the final P2L scale run. Search latency "
            "excludes one-time index transfer; memory is peak allocated device "
            "memory."
        )

    for expected_count, query_path, query_id_path, candidate_path, candidate_id_path in sources:
        for path in (query_path, query_id_path, candidate_path, candidate_id_path):
            if not path.exists():
                raise FileNotFoundError(path)
        query_array = np.load(query_path, mmap_mode="r")
        query_ids = np.load(query_id_path, mmap_mode="r")
        candidate_array = np.load(candidate_path, mmap_mode="r")
        candidate_ids = np.load(candidate_id_path, mmap_mode="r")
        if len(candidate_array) != expected_count:
            raise ValueError(f"Expected {expected_count} candidates, found {len(candidate_array)}")

        torch.cuda.empty_cache()
        # PyTorch 2.7 in the evaluated ``genius2`` environment accepts a
        # CUDA device index here, but rejects a ``torch.device`` instance.
        # Keep the explicit index aligned with ``CUDA_VISIBLE_DEVICES=0``.
        torch.cuda.reset_peak_memory_stats(0)
        load_started = time.perf_counter()
        candidate_tensor = torch.as_tensor(
            np.asarray(candidate_array), device=device, dtype=torch.float16
        )
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started
        candidate_transpose = candidate_tensor.transpose(0, 1)

        warmup = torch.as_tensor(
            np.asarray(query_array[:BATCH_SIZE]), device=device, dtype=torch.float16
        )
        torch.topk(warmup @ candidate_transpose, k=TOPK, dim=1)
        torch.cuda.synchronize(device)
        del warmup

        result_rows = []
        batch_ms = []
        search_started = time.perf_counter()
        for start in range(0, len(query_array), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(query_array))
            batch_started = time.perf_counter()
            query_tensor = torch.as_tensor(
                np.asarray(query_array[start:stop]),
                device=device,
                dtype=torch.float16,
            )
            _, indices = torch.topk(query_tensor @ candidate_transpose, k=TOPK, dim=1)
            torch.cuda.synchronize(device)
            result_batch = indices.cpu().numpy()
            batch_ms.append(
                (time.perf_counter() - batch_started) * 1000.0 / (stop - start)
            )
            result_rows.append(result_batch)
            del query_tensor, indices
        search_seconds = time.perf_counter() - search_started
        result_rows_array = np.concatenate(result_rows, axis=0)
        evaluated, recall = evaluate(result_rows_array, query_ids, candidate_ids, qrels)
        per_query_batch_ms = np.asarray(batch_ms, dtype=np.float64)
        row = {
            "candidate_count": expected_count,
            "queries": len(query_array),
            "evaluated_queries": evaluated,
            "batch_size": BATCH_SIZE,
            "precision": "fp16",
            "index_load_seconds": load_seconds,
            "search_seconds": search_seconds,
            "search_ms_per_query": 1000.0 * search_seconds / len(query_array),
            "p50_ms_per_query": float(np.percentile(per_query_batch_ms, 50)),
            "p95_ms_per_query": float(np.percentile(per_query_batch_ms, 95)),
            "p99_ms_per_query": float(np.percentile(per_query_batch_ms, 99)),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0),
            "recall_at_1": recall[1],
            "recall_at_5": recall[5],
            "recall_at_10": recall[10],
            "recall_at_50": recall[50],
            "query_embedding_file": str(query_path.relative_to(ROOT)),
            "query_id_file": str(query_id_path.relative_to(ROOT)),
            "candidate_embedding_file": str(candidate_path.relative_to(ROOT)),
            "candidate_id_file": str(candidate_id_path.relative_to(ROOT)),
        }
        output_rows.append(row)
        print(row, flush=True)

        del candidate_tensor, candidate_transpose, result_rows_array, result_rows
        gc.collect()
        torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_rows = [
        "% Generated by profile_flatip_gpu_scaling.py after the complete GPU run.",
        "\\begin{widetable}",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        (
            "\\label{tab:supp-flatip-gpu-matched}"
            if args.matched_p2l_union
            else "\\label{tab:supp-flatip-gpu-scale}"
        ),
        "\\begin{tabular}{rrrrrrrr}",
        "\\toprule",
        "Candidates & Search (ms/q) & P50 & P95 & Peak (GiB) & R@1 & R@5 & R@10 \\\\",
        "\\midrule",
    ]
    for row in output_rows:
        count = f"{int(row['candidate_count']):,}"
        latex_rows.append(
            f"{count} & {row['search_ms_per_query']:.3f} & "
            f"{row['p50_ms_per_query']:.3f} & {row['p95_ms_per_query']:.3f} & "
            f"{row['peak_allocated_mib'] / 1024.0:.2f} & "
            f"{row['recall_at_1']:.2f} & {row['recall_at_5']:.2f} & "
            f"{row['recall_at_10']:.2f} \\\\"
        )
    latex_rows += ["\\bottomrule", "\\end{tabular}", "\\end{widetable}"]
    latex_path.write_text("\n".join(latex_rows) + "\n")
    print(f"wrote {len(output_rows)} rows to {output_path}")
    print(latex_path)


if __name__ == "__main__":
    main()
