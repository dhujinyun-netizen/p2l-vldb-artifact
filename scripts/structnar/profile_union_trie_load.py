#!/usr/bin/env python3
"""Profile isolated deserialization of a serialized semantic-ID Trie.

The structured record produced here is the raw evidence for the offline index
load statement in the P2L supplement.  Neural model loading, query features,
reranker embeddings, and online retrieval are intentionally outside scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    trie_path = args.trie.resolve()
    if not trie_path.is_file():
        raise FileNotFoundError(trie_path)

    artifact_sha256 = sha256(trie_path)
    started = time.perf_counter()
    with trie_path.open("rb") as handle:
        payload = pickle.load(handle)
    load_seconds = time.perf_counter() - started

    tree = payload.get("tree") if isinstance(payload, dict) else None
    codes = payload.get("cand_codes") if isinstance(payload, dict) else None
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record = {
        "profile": "isolated_semantic_id_trie_deserialization",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "input": {
            "path": str(trie_path),
            "sha256": artifact_sha256,
            "serialized_bytes": trie_path.stat().st_size,
            "serialized_mib": trie_path.stat().st_size / (1024.0 * 1024.0),
        },
        "measurement": {
            "pickle_load_seconds": load_seconds,
            "peak_process_rss_kib": peak_rss_kib,
            "peak_process_rss_mib": peak_rss_kib / 1024.0,
        },
        "loaded_payload": {
            "top_level_type": type(payload).__name__,
            "top_level_keys": sorted(payload) if isinstance(payload, dict) else [],
            "tree_entries": len(tree) if tree is not None else None,
            "candidate_code_shape": list(codes.shape) if hasattr(codes, "shape") else None,
            "candidate_code_dtype": str(codes.dtype) if hasattr(codes, "dtype") else None,
        },
        "scope": {
            "included": [
                "Python pickle deserialization",
                "nested Candidate Trie",
                "candidate-code tensors serialized in the Trie payload",
            ],
            "excluded": [
                "neural checkpoint and model loading",
                "query features",
                "separately stored reranker embeddings",
                "online query execution",
            ],
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
