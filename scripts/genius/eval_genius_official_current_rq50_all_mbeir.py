#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap
from omegaconf import OmegaConf


DATASETS = [
    "visualnews_task0",
    "mscoco_task0",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "mscoco_task3",
    "fashion200k_task3",
    "nights_task4",
    "oven_task6",
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8",
]

LOCAL_CANDIDATES = [
    "visualnews_task0",
    "mscoco_task0_test",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "mscoco_task3_test",
    "fashion200k_task3",
    "nights_task4",
    "oven_task6",
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8",
]

# Names accepted by cand_pools_name_to_gen_code before exceptional-pool rewriting.
CANDIDATE_GENERATION_NAMES = [
    "visualnews_task0",
    "mscoco_task0",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "mscoco_task3",
    "fashion200k_task3",
    "nights_task4",
    "oven_task6",
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8",
]

METRICS = [
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@10, Recall@20, Recall@50",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@10, Recall@20, Recall@50",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@10, Recall@20, Recall@50",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
    "Recall@1, Recall@5, Recall@10",
]

QUERY_RE = re.compile(
    r"Retriever: Retrieving for query:([^\s|]+)\s*\|\s*split:([^\s|]+)\s*\|\s*from cand_pool:([^\s]+)"
)
METRIC_RE = re.compile(r"Retriever: Mean (Recall@\d+):\s*([0-9.]+)")
RESULT_RE = re.compile(r"Retriever: Results saved to (.+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original GENIUS evaluator on all 16 M-BEIR test datasets "
                    "against both local and union candidate pools."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mbeir-data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=50)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--rebuild-local-candidates", action="store_true")
    parser.add_argument("--rebuild-union", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS,
        help="Optional dataset subset for isolated timing runs.",
    )
    parser.add_argument("--deterministic-eval-sampling", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def cache_stem_path(cache_dir: Path, stem: str, kind: str) -> Path:
    return cache_dir / f"mbeir_{stem}_cand_pool_{kind}.npy"


def expected_candidate_paths(cache_dir: Path, stem: str) -> list[Path]:
    return [
        cache_stem_path(cache_dir, stem, "codes"),
        cache_stem_path(cache_dir, stem, "embeddings"),
        cache_stem_path(cache_dir, stem, "ids"),
    ]


def local_cache_complete(cache_dir: Path, stem: str) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in expected_candidate_paths(cache_dir, stem))


def preflight(args: argparse.Namespace, cache_dir: Path) -> None:
    repo = args.repo_root
    data = args.mbeir_data_dir

    require_file(
        repo / "checkpoint/rq_clip_large/Large/Instruct/InBatch/rq_clip_large_epoch_50.pth",
        "RQ50 checkpoint",
    )
    require_file(
        repo / "checkpoint/genius/"
               "genius_t5small_latest.pth",
        "GENIUS checkpoint",
    )
    require_file(repo / "checkpoint/CLIP_SF/clip_sf_large.pth", "CLIP-SF checkpoint")
    require_file(
        repo / "src/common/mbeir_generative_retriever.py",
        "GENIUS evaluator",
    )
    require_file(
        repo / "configs/genius/eval.yaml",
        "official GENIUS evaluation config",
    )

    for dataset, local_pool in zip(DATASETS, LOCAL_CANDIDATES):
        require_file(
            data / f"query/test/mbeir_{dataset}_test.jsonl",
            f"{dataset} test query",
        )
        require_file(
            data / f"qrels/test/mbeir_{dataset}_test_qrels.txt",
            f"{dataset} qrels",
        )
        require_file(
            data / f"cand_pool/local/mbeir_{local_pool}_cand_pool.jsonl",
            f"{local_pool} local candidate pool",
        )

        if args.rebuild_local_candidates or not local_cache_complete(cache_dir, local_pool):
            require_file(
                repo / f"extracted_embed/CLIP_SF/cand/cand_pool_{local_pool}_IT_dict.pt",
                f"{local_pool} extracted candidate features",
            )

    if not args.local_only:
        require_file(
            data / "cand_pool/global/mbeir_union_test_cand_pool.jsonl",
            "union test candidate pool",
        )

    print("[PASS] all 16 query/qrel/local-pool layouts found")
    print("[PASS] trained RQ50 and GENIUS checkpoints found")
    print("[PASS] required extracted features found for missing local caches")


def apply_common_overrides(cfg, args: argparse.Namespace):
    cfg.codebook_config.quantizer_path = (
        "checkpoint/rq_clip_large/Large/Instruct/InBatch/rq_clip_large_epoch_50.pth"
    )
    cfg.dataloader_config.batch_size = args.batch_size
    cfg.dataloader_config.num_workers = args.num_workers
    cfg.data_config.deterministic_eval_sampling = bool(args.deterministic_eval_sampling)

    cfg.experiment.exp_name = "InBatch_Official_CurrentRQ50"
    cfg.model.ckpt_config.ckpt_dir = (
        "checkpoint/genius"
    )
    cfg.model.ckpt_config.ckpt_name = "genius_t5small_latest.pth"
    cfg.model.name = "T5GenerativeRetriever"
    cfg.model.short_name = "GENIUS_t5small"
    cfg.model.size = "Large"
    cfg.model.trie_type = "trie_cpp"

    cfg.retrieval_config.gen_code_dir_name = "gen_code"
    cfg.retrieval_config.index_dir_name = "index"
    cfg.retrieval_config.qrel_dir_name = "qrels"
    cfg.retrieval_config.num_beams = args.num_beams
    cfg.retrieval_config.rerank = True
    cfg.retrieval_config.use_fp16 = True
    cfg.retrieval_config.write_to_tsv = True

    cfg.retrieval_config.train_datasets_config.datasets_name = None
    cfg.retrieval_config.train_datasets_config.correspond_cand_pools_name = None
    cfg.retrieval_config.train_datasets_config.enable_gen_code = False
    cfg.retrieval_config.train_datasets_config.enable_retrieve = False

    cfg.retrieval_config.val_datasets_config.datasets_name = None
    cfg.retrieval_config.val_datasets_config.correspond_cand_pools_name = None
    cfg.retrieval_config.val_datasets_config.correspond_qrels_name = None
    cfg.retrieval_config.val_datasets_config.correspond_metrics_name = None
    cfg.retrieval_config.val_datasets_config.enable_gen_code = False
    cfg.retrieval_config.val_datasets_config.enable_retrieve = False

    cfg.dist_config.dist_url = "env://"
    return cfg


def make_local_config(base_cfg, args: argparse.Namespace, cache_dir: Path):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    cfg = apply_common_overrides(cfg, args)

    missing_generation_names = []
    for logical_name, cache_stem in zip(CANDIDATE_GENERATION_NAMES, LOCAL_CANDIDATES):
        if args.rebuild_local_candidates or not local_cache_complete(cache_dir, cache_stem):
            missing_generation_names.append(logical_name)

    cfg.retrieval_config.cand_pools_config.cand_pools_name_to_gen_code = missing_generation_names
    cfg.retrieval_config.cand_pools_config.enable_gen_code = bool(missing_generation_names)
    cfg.retrieval_config.cand_pools_config.gen_code_union_pool = False
    cfg.retrieval_config.cand_pools_config.is_extracted = True

    test = cfg.retrieval_config.test_datasets_config
    test.datasets_name = list(DATASETS)
    test.correspond_cand_pools_name = list(LOCAL_CANDIDATES)
    test.correspond_qrels_name = list(DATASETS)
    test.correspond_metrics_name = list(METRICS)
    test.enable_gen_code = True
    test.enable_retrieve = True

    cfg.retrieval_config.results_dir_name = (
        f"retrieval_results/genius_official_current_rq50/all_mbeir/"
        f"local_rerank_beam{args.num_beams}"
    )
    return cfg, missing_generation_names


def make_union_config(base_cfg, args: argparse.Namespace):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    cfg = apply_common_overrides(cfg, args)

    cfg.retrieval_config.cand_pools_config.cand_pools_name_to_gen_code = []
    cfg.retrieval_config.cand_pools_config.enable_gen_code = False
    cfg.retrieval_config.cand_pools_config.gen_code_union_pool = False
    cfg.retrieval_config.cand_pools_config.is_extracted = True

    test = cfg.retrieval_config.test_datasets_config
    test.datasets_name = list(DATASETS)
    test.correspond_cand_pools_name = ["UNION"] * len(DATASETS)
    test.correspond_qrels_name = list(DATASETS)
    test.correspond_metrics_name = list(METRICS)
    test.enable_gen_code = True
    test.enable_retrieve = True

    cfg.retrieval_config.results_dir_name = (
        f"retrieval_results/genius_official_current_rq50/all_mbeir/"
        f"union_rerank_beam{args.num_beams}"
    )
    return cfg


def stream_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    print("[RUN]", " ".join(command))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def run_eval(config_path: Path, args: argparse.Namespace, log_path: Path) -> int:
    repo = args.repo_root
    common_dir = repo / "src/common"
    evaluator = common_dir / "mbeir_generative_retriever.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        evaluator.name,
        "--config_path",
        str(config_path),
        "--genir_dir",
        str(repo),
        "--mbeir_data_dir",
        str(args.mbeir_data_dir),
    ]
    return stream_process(command, common_dir, env, log_path)


def source_signature(paths: Iterable[Path]) -> list[dict]:
    sig = []
    for path in paths:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        sig.append({
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })
    return sig


def union_cache_valid(cache_dir: Path, local_paths_by_kind: dict[str, list[Path]]) -> bool:
    union_paths = {
        kind: cache_dir / f"mbeir_union_cand_pool_{kind}.npy"
        for kind in ("codes", "embeddings", "ids")
    }
    manifest_path = cache_dir / "mbeir_union_cand_pool_stream_manifest.json"
    if not all(path.is_file() and path.stat().st_size > 0 for path in union_paths.values()):
        return False
    if not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text())
        for kind, paths in local_paths_by_kind.items():
            if manifest["sources"][kind] != source_signature(paths):
                return False
        total = sum(np.load(path, mmap_mode="r").shape[0] for path in local_paths_by_kind["codes"])
        for path in union_paths.values():
            if np.load(path, mmap_mode="r").shape[0] != total:
                return False
    except Exception:
        return False
    return True


def stream_concat_npy(source_paths: list[Path], destination: Path, chunk_rows: int = 65536) -> dict:
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in source_paths]
    if not arrays:
        raise ValueError("No source arrays supplied")

    first = arrays[0]
    tail_shape = first.shape[1:]
    dtype = first.dtype
    total_rows = 0
    for path, arr in zip(source_paths, arrays):
        if arr.shape[1:] != tail_shape:
            raise RuntimeError(
                f"Shape mismatch for {path}: expected tail={tail_shape}, got {arr.shape[1:]}"
            )
        if arr.dtype != dtype:
            raise RuntimeError(
                f"Dtype mismatch for {path}: expected {dtype}, got {arr.dtype}"
            )
        total_rows += arr.shape[0]

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    out = open_memmap(tmp, mode="w+", dtype=dtype, shape=(total_rows, *tail_shape))

    offset = 0
    for path, arr in zip(source_paths, arrays):
        rows = arr.shape[0]
        print(f"[UNION] copy {path.name}: rows={rows:,}")
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            out[offset + start: offset + stop] = arr[start:stop]
        offset += rows
    out.flush()
    del out
    os.replace(tmp, destination)
    return {
        "path": str(destination),
        "shape": [total_rows, *tail_shape],
        "dtype": str(dtype),
    }


def build_union_cache(cache_dir: Path, rebuild: bool) -> None:
    local_paths_by_kind = {
        kind: [cache_stem_path(cache_dir, stem, kind) for stem in LOCAL_CANDIDATES]
        for kind in ("codes", "embeddings", "ids")
    }
    for paths in local_paths_by_kind.values():
        for path in paths:
            require_file(path, "local candidate cache")

    if not rebuild and union_cache_valid(cache_dir, local_paths_by_kind):
        print("[UNION] valid streaming union cache already exists; reuse")
        return

    print("[UNION] building memory-safe union cache from 16 local pools")
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for kind in ("codes", "embeddings", "ids"):
        destination = cache_dir / f"mbeir_union_cand_pool_{kind}.npy"
        outputs[kind] = stream_concat_npy(local_paths_by_kind[kind], destination)

    code_rows = outputs["codes"]["shape"][0]
    if outputs["embeddings"]["shape"][0] != code_rows or outputs["ids"]["shape"][0] != code_rows:
        raise RuntimeError("Union codes/embeddings/ids row counts differ")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            kind: source_signature(paths)
            for kind, paths in local_paths_by_kind.items()
        },
        "outputs": outputs,
    }
    manifest_path = cache_dir / "mbeir_union_cand_pool_stream_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    trie_path = cache_dir / "mbeir_union_cand_pool_trie.pkl"
    if trie_path.exists():
        trie_path.unlink()
        print(f"[UNION] removed stale trie: {trie_path}")

    print(f"[UNION] complete: rows={code_rows:,}")
    print(f"[UNION] manifest={manifest_path}")


def parse_log(log_path: Path, scope: str) -> list[dict]:
    rows = []
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = QUERY_RE.search(line)
        if match:
            if current and current["metrics"]:
                rows.append(current)
            current = {
                "scope": scope,
                "dataset": match.group(1).lower(),
                "split": match.group(2).lower(),
                "cand_pool": match.group(3).lower(),
                "metrics": {},
                "log": str(log_path),
            }
            continue
        match = METRIC_RE.search(line)
        if match and current is not None:
            current["metrics"][match.group(1)] = float(match.group(2))
    if current and current["metrics"]:
        rows.append(current)
    return rows


def write_summary(rows: list[dict], run_dir: Path) -> None:
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), 0 if row["scope"] == "local" else 1))
    metrics = ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@50"]

    tsv_path = run_dir / "all_32_results.tsv"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset", "scope", "cand_pool", *metrics, "log"])
        for row in rows:
            writer.writerow([
                row["dataset"],
                row["scope"],
                row["cand_pool"],
                *[
                    f"{100 * row['metrics'][metric]:.2f}%"
                    if metric in row["metrics"] else ""
                    for metric in metrics
                ],
                row["log"],
            ])

    md_path = run_dir / "all_32_results.md"
    lines = [
        "# GENIUS Original — All M-BEIR Encoded Tests",
        "",
        "16 datasets evaluated against local candidate pools and UNION/global pool.",
        "",
        "| Dataset | Pool | R@1 | R@5 | R@10 | R@20 | R@50 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"{100 * row['metrics'][metric]:.2f}%"
            if metric in row["metrics"] else "—"
            for metric in metrics
        ]
        lines.append(
            f"| {row['dataset']} | {row['scope']} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |"
        )
    if not rows:
        lines.extend(["", "No completed retrieval results found."])
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[SUMMARY] tsv={tsv_path}")
    print(f"[SUMMARY] markdown={md_path}")
    print(md_path.read_text())


def main() -> int:
    global DATASETS, LOCAL_CANDIDATES, CANDIDATE_GENERATION_NAMES, METRICS
    args = parse_args()
    if args.datasets:
        selected = set(args.datasets)
        rows = [
            row for row in zip(DATASETS, LOCAL_CANDIDATES, CANDIDATE_GENERATION_NAMES, METRICS)
            if row[0] in selected
        ]
        DATASETS = [row[0] for row in rows]
        LOCAL_CANDIDATES = [row[1] for row in rows]
        CANDIDATE_GENERATION_NAMES = [row[2] for row in rows]
        METRICS = [row[3] for row in rows]
    args.repo_root = args.repo_root.resolve()
    args.mbeir_data_dir = args.mbeir_data_dir.resolve()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.repo_root
        / "logs/genius_official_current_rq50"
        / f"all_mbeir_encoded_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    official_config_path = (
        args.repo_root
        / "configs/genius/eval.yaml"
    )
    base_cfg = OmegaConf.load(official_config_path)

    cache_dir = (
        args.repo_root
        / "gen_code/GENIUS_t5small/Large/Instruct/InBatch_Official_CurrentRQ50/cand_pool"
    )

    preflight(args, cache_dir)

    local_cfg, missing = make_local_config(base_cfg, args, cache_dir)
    local_cfg_path = run_dir / "local_all16.yaml"
    OmegaConf.save(local_cfg, local_cfg_path)

    union_cfg = make_union_config(base_cfg, args)
    union_cfg_path = run_dir / "union_all16.yaml"
    OmegaConf.save(union_cfg, union_cfg_path)

    print(f"[CONFIG] local={local_cfg_path}")
    print(f"[CONFIG] union={union_cfg_path}")
    print(f"[CACHE] candidate={cache_dir}")
    print(f"[LOCAL] missing candidate pools={missing if missing else 'none'}")

    if args.dry_run:
        print("[DRY-RUN] preflight and config generation passed")
        return 0

    local_log = run_dir / "local_all16.log"
    status = run_eval(local_cfg_path, args, local_log)
    if status != 0:
        print(f"[FAIL] local phase exited with status={status}", file=sys.stderr)
        write_summary(parse_log(local_log, "local"), run_dir)
        return status

    rows = parse_log(local_log, "local")

    if args.local_only:
        write_summary(rows, run_dir)
        print("[PASS] all 16 local-pool tests completed")
        return 0

    build_union_cache(cache_dir, rebuild=args.rebuild_union)

    union_log = run_dir / "union_all16.log"
    status = run_eval(union_cfg_path, args, union_log)
    rows.extend(parse_log(union_log, "union"))
    write_summary(rows, run_dir)

    if status != 0:
        print(f"[FAIL] union phase exited with status={status}", file=sys.stderr)
        return status

    local_count = sum(row["scope"] == "local" for row in rows)
    union_count = sum(row["scope"] == "union" for row in rows)
    if local_count != 16 or union_count != 16:
        print(
            f"[FAIL] expected 16 local + 16 union results, got "
            f"{local_count} local + {union_count} union",
            file=sys.stderr,
        )
        return 2

    print("[PASS] all 32 official M-BEIR encoded test mappings completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
