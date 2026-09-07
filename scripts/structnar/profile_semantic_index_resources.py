#!/usr/bin/env python3
"""Profile the nested semantic-ID indexes used by the CIRR scale study."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GEN_ROOT = ROOT / "gen_code/STRUCTNAR/Large/Instruct"
OUT = ROOT / "docs/results/structnar_semantic_index_resources.csv"
SIZES = (21_551, 100_000, 500_000, 1_000_000, 5_609_079)


def paths_for(size: int) -> dict[str, Path]:
    exp = GEN_ROOT / f"structnar_cirr_scale_{size:07d}" / "cand_pool"
    if size == 5_609_079:
        stem = exp / "mbeir_union_cand_pool"
    else:
        stem = exp / f"mbeir_cirrscale_{size:07d}_cand_pool"
    return {
        key: Path(f"{stem}_{suffix}")
        for key, suffix in {
            "codes": "codes.npy",
            "ids": "ids.npy",
            "embeddings": "embeddings.npy",
            "trie": "trie.pkl",
        }.items()
    }


def prefix_counts(codes: np.ndarray) -> list[int]:
    # Sort once lexicographically. Adjacent rows then expose the number of
    # distinct prefixes at every depth without materializing Python tuples.
    keys = tuple(codes[:, level] for level in reversed(range(codes.shape[1])))
    order = np.lexsort(keys)
    sorted_codes = np.asarray(codes[order])
    counts: list[int] = []
    for depth in range(1, sorted_codes.shape[1] + 1):
        changed = np.any(sorted_codes[1:, :depth] != sorted_codes[:-1, :depth], axis=1)
        counts.append(1 + int(np.count_nonzero(changed)))
    return counts


def mib(path: Path) -> float:
    return path.stat().st_size / (2**20)


def main() -> int:
    rows = []
    for size in SIZES:
        paths = paths_for(size)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing index artifacts: " + ", ".join(missing))
        codes = np.load(paths["codes"], mmap_mode="r")
        started = time.perf_counter()
        counts = prefix_counts(codes)
        analysis_seconds = time.perf_counter() - started
        total_nodes = 1 + sum(counts)
        internal_nodes = total_nodes - counts[-1]
        rows.append(
            {
                "candidate_count": int(codes.shape[0]),
                "identifier_length": int(codes.shape[1]),
                "unique_identifiers": counts[-1],
                **{f"prefixes_d{depth}": value for depth, value in enumerate(counts, 1)},
                "trie_nodes_including_root": total_nodes,
                "trie_internal_nodes": internal_nodes,
                "mean_internal_fanout": (total_nodes - 1) / max(internal_nodes, 1),
                "codes_mib": mib(paths["codes"]),
                "item_ids_mib": mib(paths["ids"]),
                "candidate_embeddings_mib": mib(paths["embeddings"]),
                "serialized_trie_mib": mib(paths["trie"]),
                "trie_bytes_per_candidate": paths["trie"].stat().st_size / codes.shape[0],
                "offline_analysis_seconds": analysis_seconds,
            }
        )
        print(
            f"size={size:,} unique={counts[-1]:,} nodes={total_nodes:,} "
            f"trie={mib(paths['trie']):.1f} MiB analysis={analysis_seconds:.1f}s",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
