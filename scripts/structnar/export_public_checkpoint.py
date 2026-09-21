#!/usr/bin/env python3
"""Create an inference-only release checkpoint for the evaluated P2L model.

The training checkpoint contains optimizer, scheduler, and scaler state that is
not needed for reproduction.  This exporter keeps the complete ``model`` state
(including the trained quantizer and GPT-HDGR generator), the epoch/config
metadata, and removes training-only state.  The resulting file can be loaded by
the existing evaluation code because it preserves the top-level ``model`` key.
"""
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantizer-output",
        type=Path,
        help="also export the trained RQ quantizer in the format expected by eval configs",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source checkpoint not found: {source}")
    if output == source:
        raise SystemExit("refusing to overwrite source checkpoint")

    source_sha = sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise SystemExit("source checkpoint must be a mapping with a top-level model key")

    # Keep the original model state exactly; it includes the trained RQ codebook
    # and all generator weights required by the P2L evaluation configuration.
    config = payload.get("config")
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(config, resolve=False) if config is not None else None
    except Exception:
        # Config metadata is informative only; evaluation receives its YAML via
        # the normal --config/CONFIG_PATH path.
        config = None

    public_payload = {
        "model": payload["model"],
        "config": config,
        "epoch": int(payload.get("epoch", 0)),
        "format": "p2l-inference-only-v1",
        "metadata": {
            "source_checkpoint": str(args.source),
            "source_sha256": source_sha,
            "removed_keys": ["optimizer", "scheduler", "scaler"],
            "kept_model_state_includes": ["GPT-HDGR generator", "trained RQ quantizer"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public_payload, output, _use_new_zipfile_serialization=True)
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "source_sha256": source_sha,
        "epoch": public_payload["epoch"],
        "format": public_payload["format"],
    }, indent=2))
    if args.quantizer_output:
        quantizer_output = args.quantizer_output.expanduser().resolve()
        quantizer_state = {
            key.removeprefix("quantizer."): value
            for key, value in payload["model"].items()
            if key.startswith("quantizer.")
        }
        if not quantizer_state:
            raise SystemExit("source model has no quantizer.* state")
        quantizer_payload = {
            "model": quantizer_state,
            "epoch": int(payload.get("epoch", 0)),
            "format": "p2l-rq-quantizer-inference-v1",
            "metadata": {
                "source_checkpoint": str(args.source),
                "source_sha256": source_sha,
                "state_prefix_removed": "quantizer.",
            },
        }
        quantizer_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantizer_payload, quantizer_output, _use_new_zipfile_serialization=True)
        print(json.dumps({
            "quantizer_output": str(quantizer_output),
            "bytes": quantizer_output.stat().st_size,
            "sha256": sha256(quantizer_output),
            "format": quantizer_payload["format"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
