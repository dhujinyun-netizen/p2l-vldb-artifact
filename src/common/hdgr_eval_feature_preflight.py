"""Preflight checks for HDGR generator evaluation.

The generator eval path can consume pre-extracted CLIP-SF candidate embeddings
(`extracted_embed/CLIP_SF/cand/*_IT_dict.pt`).  This script checks that the
files requested by configs/generator/eval.yaml exist before launching the more
expensive retrieval process.  Exit code 2 means: missing files that can usually
be created by running `scripts/feature_extraction/extract_cand.sh`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

EXCEPTIONAL_CAND_POOLS = {"mscoco_task0", "mscoco_task3", "flickr30k_task0", "flickr30k_task3"}


def _normalize_cand_pool_name(name: str) -> str:
    name = str(name).lower()
    if name in EXCEPTIONAL_CAND_POOLS:
        name = name + "_test"
    return name


def _iter_requested_extracted_files(config, genir_dir: Path) -> Iterable[Path]:
    data_config = config.data_config
    retrieval_config = config.retrieval_config
    if not bool(data_config.get("is_extracted", False)):
        return []

    cand_cfg = retrieval_config.get("cand_pools_config", None)
    if not cand_cfg or not bool(cand_cfg.get("enable_gen_code", False)):
        return []

    extracted_dir = genir_dir / str(data_config.extracted_dir)
    requested = []
    for cand_pool_name in cand_cfg.get("cand_pools_name_to_gen_code", []) or []:
        normalized = _normalize_cand_pool_name(cand_pool_name)
        requested.append(extracted_dir / f"cand_pool_{normalized}_IT_dict.pt")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--genir_dir", required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    genir_dir = Path(args.genir_dir).expanduser().resolve()
    expected_files = list(_iter_requested_extracted_files(config, genir_dir))
    missing = [p for p in expected_files if not p.exists()]

    if not expected_files:
        print("[eval preflight] No extracted candidate embedding files are required by this eval config.")
        return 0

    if not missing:
        print("[eval preflight] All extracted candidate embedding files exist:")
        for p in expected_files:
            print(f"  OK: {p}")
        return 0

    print("[eval preflight] Missing extracted candidate embedding files:")
    for p in missing:
        print(f"  MISSING: {p}")
    print("\nCreate them with:")
    print("  bash scripts/feature_extraction/extract_cand.sh")
    print("\nThe eval wrapper can do this automatically when AUTO_EXTRACT_CAND=1.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
