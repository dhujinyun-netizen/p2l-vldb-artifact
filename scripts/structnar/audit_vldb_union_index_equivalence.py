#!/usr/bin/env python3
"""Verify that matched UNION policies load the same realized semantic IDs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/structnar/generated/vldb_union_frontier_20260904_214403"
OUTPUT = ROOT / "docs/results/p2l_vldb_union_index_equivalence.json"
TASKS = ("nights_task4", "edis_task2", "webqa_task1")
POLICIES = ("sequential", "p2l")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name(config: Path) -> str:
    match = re.search(r"^\s*exp_name:\s*(\S+)\s*$", config.read_text(), re.MULTILINE)
    if not match:
        raise AssertionError(f"missing exp_name in {config}")
    return match.group(1)


def main() -> None:
    rows = []
    cached_hashes: dict[Path, str] = {}
    for task in TASKS:
        for policy in POLICIES:
            candidates = sorted(CONFIG_DIR.glob(f"*_{task}_{policy}_*_union.yaml"))
            if len(candidates) != 1:
                raise AssertionError(
                    f"expected one UNION config for {task}/{policy}, got {candidates}"
                )
            config = candidates[0]
            exp_name = experiment_name(config)
            link = (
                ROOT
                / "gen_code/STRUCTNAR/Large/Instruct"
                / exp_name
                / "cand_pool/mbeir_union_cand_pool_codes.npy"
            )
            if not link.is_symlink():
                raise AssertionError(f"expected a candidate-code symlink: {link}")
            resolved = link.resolve(strict=True)
            if resolved not in cached_hashes:
                cached_hashes[resolved] = sha256(resolved)
            array = np.load(resolved, mmap_mode="r", allow_pickle=False)
            stat = resolved.stat()
            rows.append(
                {
                    "task": task,
                    "policy": policy,
                    "config": str(config.relative_to(ROOT)),
                    "experiment": exp_name,
                    "candidate_code_link": str(link.relative_to(ROOT)),
                    "symlink_target": os.readlink(link),
                    "resolved_path": str(resolved),
                    "size_bytes": stat.st_size,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "sha256": cached_hashes[resolved],
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )

    hashes = {row["sha256"] for row in rows}
    shapes = {tuple(row["shape"]) for row in rows}
    dtypes = {row["dtype"] for row in rows}
    resolved_paths = {row["resolved_path"] for row in rows}
    if len(hashes) != 1 or shapes != {(5_609_079, 9)} or len(dtypes) != 1:
        raise AssertionError(
            f"UNION index mismatch: hashes={hashes}, shapes={shapes}, dtypes={dtypes}"
        )

    report = {
        "status": "PASS",
        "claim": (
            "All six cross-workload matched runs use byte-identical realized "
            "UNION semantic-ID assignments."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration_directory": str(CONFIG_DIR.relative_to(ROOT)),
        "rows": rows,
        "summary": {
            "run_count": len(rows),
            "unique_resolved_paths": len(resolved_paths),
            "unique_sha256": len(hashes),
            "candidate_count": rows[0]["shape"][0],
            "identifier_length": rows[0]["shape"][1],
            "dtype": rows[0]["dtype"],
            "sha256": rows[0]["sha256"],
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("RESULT=PASS_VLDB_UNION_INDEX_EQUIVALENCE")
    print(f"output={OUTPUT}")
    print(f"sha256={rows[0]['sha256']}")


if __name__ == "__main__":
    main()
