#!/usr/bin/env python3
"""Preflight checks for GENIUS Stage-2 baseline with current RQ50.

Strict mode reproduces the official configuration exactly.
Low-RAM mode permits only two system-level deviations that do not alter the
model, objective, optimizer, batch size, number of steps, or training data:
  * dataloader_config.num_workers = 0
  * evaluator.enable_eval = false
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


EXPECTED = {
    "codebook_config.codebook_level": 8,
    "codebook_config.codebook_vocab": 4096,
    "dataloader_config.num_workers": 8,
    "dataloader_config.train_batch_size": 256,
    "dataloader_config.valid_batch_size": 128,
    "evaluator.enable_eval": True,
    "evaluator.eval_freq": 5,
    "evaluator.eval_start": 40,
    "hyperparameter_config.alpha": 2,
    "model.name": "GENIUS_t5small",
    "model.short_name": "GENIUS_t5small",
    "model.size": "Large",
    "seed": 2023,
    "trainer_config.gradient_accumulation_steps": 1,
    "trainer_config.learning_rate": 1.0e-4,
    "trainer_config.num_train_epochs": 31,
    "trainer_config.warmup_steps": 0,
}

LOW_RAM_OVERRIDES = {
    "dataloader_config.num_workers": 0,
    "evaluator.enable_eval": False,
}


def select(cfg, key: str):
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise KeyError(f"missing config field: {key}")
    return value


def check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    print(f"[PASS] {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--mbeir-data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help=(
            "Allow only num_workers=0 and enable_eval=false. All model, data, "
            "batch, optimizer, and epoch settings remain strictly checked."
        ),
    )
    args = parser.parse_args()

    root = Path(args.repo_root).expanduser().resolve()
    mbeir = Path(args.mbeir_data_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    check_file(config_path, "training config")

    expected = dict(EXPECTED)
    if args.low_ram:
        expected.update(LOW_RAM_OVERRIDES)
        print("[INFO] preflight mode=LOW_RAM_OFFICIAL")
        print("[INFO] allowed deviations: num_workers=0, enable_eval=false")
    else:
        print("[INFO] preflight mode=STRICT_OFFICIAL")

    cfg = OmegaConf.load(config_path)
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = select(cfg, key)
        if actual != expected_value:
            failures.append(
                f"{key}: expected {expected_value!r}, got {actual!r}"
            )
        else:
            label = "low-RAM setting" if key in LOW_RAM_OVERRIDES and args.low_ram else "official setting"
            print(f"[PASS] {label} {key}={actual!r}")
    if failures:
        print("[FAIL] config mismatch:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    repo_files = {
        "trained RQ50 checkpoint": root / select(cfg, "codebook_config.quantizer_path"),
        "CLIP-SF checkpoint": root
        / select(cfg, "model.pretrained_config.pretrained_dir")
        / select(cfg, "model.pretrained_config.pretrained_name"),
        "query extracted features": root / select(cfg, "codebook_config.query_path"),
        "pool extracted features": root / select(cfg, "codebook_config.pool_path"),
        "query instructions": mbeir / select(cfg, "data_config.query_instruct_path"),
        "union train query JSONL": mbeir / select(cfg, "data_config.train_query_data_path"),
        "union train candidate JSONL": mbeir / select(cfg, "data_config.train_cand_pool_path"),
        "union val query JSONL": mbeir / select(cfg, "data_config.val_query_data_path"),
        "union val candidate JSONL": mbeir / select(cfg, "data_config.val_cand_pool_path"),
    }
    for label, path in repo_files.items():
        check_file(path, label)

    rq_path = repo_files["trained RQ50 checkpoint"]
    ckpt = torch.load(rq_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"RQ checkpoint must be a dict containing 'model': {rq_path}")
    model_state = ckpt["model"]
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError("RQ checkpoint['model'] is empty or not a state dict")
    rq_like = [
        key
        for key in model_state
        if "residual_rq" in key or "modality" in key or "codebook" in key
    ]
    if not rq_like:
        raise RuntimeError("checkpoint does not look like a GENIUS RQ state dict")
    print(
        f"[PASS] RQ checkpoint format: tensors={len(model_state)} "
        f"rq_related_keys={len(rq_like)}"
    )

    world_size = int(os.environ.get("NPROC", "1"))
    local_batch = int(select(cfg, "dataloader_config.train_batch_size"))
    accum = int(select(cfg, "trainer_config.gradient_accumulation_steps"))
    print(
        f"[INFO] effective global batch = {local_batch} x {world_size} x {accum} "
        f"= {local_batch * world_size * accum}"
    )
    if args.low_ram:
        print(
            "[INFO] Low-RAM mode changes only data-loader/evaluator residency; "
            "the model, objective, data, batch size, optimizer, and 31-epoch "
            "training schedule remain official."
        )
    else:
        print(
            "[INFO] official config has eval_start=40 and num_train_epochs=31; "
            "the validation loader is built, but scheduled validation is not reached."
        )
    print("RESULT=PASS_GENIUS_OFFICIAL_CURRENT_RQ50_PREFLIGHT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
