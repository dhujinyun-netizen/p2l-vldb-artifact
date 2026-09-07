#!/usr/bin/env python3
"""Inspect whether a checkpoint contains native GPT-HDGR generator weights.

A native checkpoint must contain the HDGR denoiser (`id_generator.hdgr`) and
must not contain a full SoundStorm model (`soundstorm_full_model`).  The script
does not infer the training loss from a directory name; use the exact training
YAML/log for that.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
import sys

import torch

STATE_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "module",
    "net",
    "network",
    "retriever",
)


def tensor_mapping(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    tensor_count = sum(torch.is_tensor(v) for v in obj.values())
    return tensor_count >= max(1, min(8, len(obj) // 4))


def find_state_dict(obj: Any, depth: int = 0) -> Mapping[str, Any] | None:
    if depth > 4:
        return None
    if tensor_mapping(obj):
        return obj
    if isinstance(obj, Mapping):
        for key in STATE_KEYS:
            if key in obj:
                found = find_state_dict(obj[key], depth + 1)
                if found is not None:
                    return found
        for value in obj.values():
            if isinstance(value, Mapping):
                found = find_state_dict(value, depth + 1)
                if found is not None:
                    return found
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--show", type=int, default=20, help="number of matching keys to print")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    try:
        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(args.checkpoint, map_location="cpu")
    except Exception as exc:
        print(f"ERROR: failed to load checkpoint: {exc}", file=sys.stderr)
        return 3

    state = find_state_dict(payload)
    if state is None:
        print("ERROR: could not locate a tensor state_dict in checkpoint", file=sys.stderr)
        if isinstance(payload, Mapping):
            print(f"top_level_keys={list(payload)[:40]}")
        return 4

    keys = [str(k) for k, v in state.items() if torch.is_tensor(v)]
    lower = [k.lower() for k in keys]
    full_keys = [k for k, lk in zip(keys, lower) if "soundstorm_full_model" in lk]
    hdgr_keys = [
        k for k, lk in zip(keys, lower)
        if "id_generator.hdgr." in lk
        or lk.startswith("id_generator.hdgr.")
        or ".id_generator.hdgr." in lk
    ]
    id_generator_keys = [k for k, lk in zip(keys, lower) if "id_generator." in lk]

    prefixes = Counter(k.split(".", 1)[0] for k in keys)
    print(f"checkpoint={args.checkpoint}")
    print(f"tensor_keys={len(keys)}")
    print(f"top_prefixes={dict(prefixes.most_common(12))}")
    print(f"native_hdgr_key_count={len(hdgr_keys)}")
    print(f"soundstorm_full_key_count={len(full_keys)}")
    print(f"id_generator_key_count={len(id_generator_keys)}")
    if hdgr_keys:
        print("sample_native_hdgr_keys:")
        for key in hdgr_keys[: args.show]:
            print(f"  {key}")
    if full_keys:
        print("sample_soundstorm_full_keys:")
        for key in full_keys[: args.show]:
            print(f"  {key}")

    if full_keys:
        print(
            "RESULT=REJECT_SOUNDSTORM_FULL: checkpoint contains a full SoundStorm model; "
            "do not use it for native-HDGR inference.",
            file=sys.stderr,
        )
        return 10
    if not hdgr_keys:
        print(
            "RESULT=REJECT_NO_HDGR: checkpoint does not expose id_generator.hdgr weights.",
            file=sys.stderr,
        )
        return 11

    print("RESULT=PASS_NATIVE_HDGR_CHECKPOINT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
