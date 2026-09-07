#!/usr/bin/env python3
"""Audit the experimental GPU-flat sequential decoder against canonical output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    ROOT
    / "gen_code/STRUCTNAR/Large/Instruct/structnar_cirr_scale_w20_sequential_0021551/test"
)
GPU_FLAT = (
    ROOT
    / "gen_code/STRUCTNAR/Large/Instruct/structnar_vldb_gpu_flat_seq_cirr_smoke/test"
)
OUTPUT = ROOT / "profile_output/p2l_gpu_trie_baseline/equivalence.json"
CANONICAL_CONFIG = (
    ROOT
    / "configs/structnar/generated/structnar_vldb_canonical_seq_cirr_smoke.yaml"
)
GPU_FLAT_CONFIG = (
    ROOT
    / "configs/structnar/generated/structnar_vldb_gpu_flat_seq_cirr_smoke.yaml"
)
ALLOWED_CONFIG_DIFFERENCES = {
    "experiment.exp_name",
    "model.memory_efficient_tree_decode",
    "model.gpu_flat_tree_decode",
    "retrieval_config.results_dir_name",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(name: str) -> dict:
    canonical_path = CANONICAL / name
    gpu_flat_path = GPU_FLAT / name
    left = np.load(canonical_path, allow_pickle=True)
    right = np.load(gpu_flat_path, allow_pickle=True)
    same_shape = left.shape == right.shape
    exact = same_shape and np.array_equal(left, right)
    numeric = np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype, np.number
    )
    max_abs = None
    if numeric and same_shape and left.size:
        max_abs = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
    return {
        "name": name,
        "canonical_shape": list(left.shape),
        "gpu_flat_shape": list(right.shape),
        "canonical_dtype": str(left.dtype),
        "gpu_flat_dtype": str(right.dtype),
        "exact": bool(exact),
        "max_abs_difference": max_abs,
        "canonical_sha256": sha256(canonical_path),
        "gpu_flat_sha256": sha256(gpu_flat_path),
    }


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(flatten_mapping(child, child_key))
    return flattened


def compare_configs() -> dict:
    with CANONICAL_CONFIG.open() as stream:
        canonical = yaml.safe_load(stream)
    with GPU_FLAT_CONFIG.open() as stream:
        gpu_flat = yaml.safe_load(stream)
    left = flatten_mapping(canonical)
    right = flatten_mapping(gpu_flat)
    different = sorted(
        key
        for key in set(left) | set(right)
        if left.get(key, "<missing>") != right.get(key, "<missing>")
    )
    unexpected = sorted(set(different) - ALLOWED_CONFIG_DIFFERENCES)
    required_flags = {
        "model.memory_efficient_tree_decode": True,
        "model.gpu_flat_tree_decode": True,
    }
    flags_ok = all(right.get(key) is value for key, value in required_flags.items())
    return {
        "canonical_path": str(CANONICAL_CONFIG.relative_to(ROOT)),
        "gpu_flat_path": str(GPU_FLAT_CONFIG.relative_to(ROOT)),
        "canonical_sha256": sha256(CANONICAL_CONFIG),
        "gpu_flat_sha256": sha256(GPU_FLAT_CONFIG),
        "different_keys": different,
        "allowed_different_keys": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "unexpected_different_keys": unexpected,
        "required_gpu_flags": required_flags,
        "passed": not unexpected and flags_ok,
    }


def main() -> int:
    names = [
        "mbeir_cirr_task7_test_codes.npy",
        "mbeir_cirr_task7_test_beam_scores.npy",
        "mbeir_cirr_task7_test_embeddings.npy",
        "mbeir_cirr_task7_test_ids.npy",
    ]
    comparisons = [compare(name) for name in names]
    config_comparison = compare_configs()
    passed = all(record["exact"] for record in comparisons) and config_comparison[
        "passed"
    ]
    payload = {
        "verdict": "PASS" if passed else "FAIL",
        "scope": "CIRR-7 full manifest, sequential+PCAA, B=K=50, lambda=20",
        "config_comparison": config_comparison,
        "comparisons": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
