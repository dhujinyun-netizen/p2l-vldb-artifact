#!/usr/bin/env python3
"""Split an inference-only P2L checkpoint into deterministic upload shards."""
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


def tensor_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if torch.is_tensor(value) else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mib", type=int, default=120)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit("source must be an inference checkpoint with a model key")
    state = payload["model"]
    target = args.target_mib * 1024 * 1024
    groups: list[dict[str, object]] = []
    current: dict[str, object] = {}
    current_bytes = 0
    for key, value in state.items():
        size = tensor_bytes(value)
        if current and current_bytes + size > target:
            groups.append({"state": current, "bytes": current_bytes})
            current = {}
            current_bytes = 0
        current[key] = value
        current_bytes += size
    if current:
        groups.append({"state": current, "bytes": current_bytes})

    source_sha = sha256(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "p2l-inference-shards-v1",
        "source": str(args.source),
        "source_sha256": source_sha,
        "epoch": int(payload.get("epoch", 0)),
        "num_shards": len(groups),
        "target_mib": args.target_mib,
        "shards": [],
    }
    for index, group in enumerate(groups):
        name = f"p2l_evaluated_epoch99_shard-{index:03d}-of-{len(groups):03d}.pth"
        path = output_dir / name
        shard = {
            "format": "p2l-inference-shard-v1",
            "shard_index": index,
            "num_shards": len(groups),
            "epoch": int(payload.get("epoch", 0)),
            "source_sha256": source_sha,
            "config": payload.get("config") if index == 0 else None,
            "metadata": payload.get("metadata") if index == 0 else None,
            "state": group["state"],
        }
        torch.save(shard, path, _use_new_zipfile_serialization=True)
        manifest["shards"].append({
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tensor_bytes": group["bytes"],
        })
    manifest_path = output_dir / "p2l_evaluated_epoch99_shards.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
