#!/usr/bin/env python3
"""Reassemble P2L inference shards into the normal checkpoint format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state = {}
    epoch = None
    source_sha = None
    config = None
    source_metadata = None
    for entry in manifest["shards"]:
        path = args.shard_dir / entry["name"]
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard.get("format") != "p2l-inference-shard-v1":
            raise SystemExit(f"unexpected shard format: {path}")
        epoch = shard["epoch"] if epoch is None else epoch
        source_sha = shard["source_sha256"] if source_sha is None else source_sha
        if shard["source_sha256"] != source_sha:
            raise SystemExit("shards refer to different source checkpoints")
        overlap = set(state).intersection(shard["state"])
        if overlap:
            raise SystemExit(f"duplicate tensor keys: {sorted(overlap)[:3]}")
        state.update(shard["state"])
        if shard.get("config") is not None:
            config = shard["config"]
        if shard.get("metadata") is not None:
            source_metadata = shard["metadata"]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state,
        "config": config,
        "epoch": int(epoch),
        "format": "p2l-inference-only-v1",
        "metadata": source_metadata or {
            "assembled_from": str(args.manifest),
            "source_checkpoint_sha256": source_sha,
        },
    }
    torch.save(payload, output, _use_new_zipfile_serialization=True)
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_checkpoint_sha256": source_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
