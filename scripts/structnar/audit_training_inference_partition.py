#!/usr/bin/env python3
"""Fail-fast audit for the checkpoint training/inference partition claim."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = ROOT / "configs/structnar/train.yaml"
EVAL_CONFIG = ROOT / "configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml"
TRAIN_LOG = ROOT / "logs/train_code_tied_hdgr_official_mbeir_20260715_044040.log"
MAIN_TEX = ROOT / "VLDB/vldb2027/structnar_vldb.tex"


def require(text: str, token: str, source: Path) -> None:
    if token not in text:
        raise AssertionError(f"missing {token!r} in {source}")


def main() -> None:
    train = TRAIN_CONFIG.read_text(encoding="utf-8")
    evaluate = EVAL_CONFIG.read_text(encoding="utf-8")
    log = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
    paper = MAIN_TEX.read_text(encoding="utf-8")

    require(train, "gpt_hdgr_hierarchy_aware_blocks: false", TRAIN_CONFIG)
    require(train, "gpt_hdgr_block_size: 1", TRAIN_CONFIG)
    require(log, "hierarchy_on=False block_training=False block_decoding=False", TRAIN_LOG)
    require(evaluate, "gpt_hdgr_hierarchy_aware_blocks: true", EVAL_CONFIG)
    require(evaluate, "hierarchy_block_sizes:\n  - 1\n  - 1\n  - 1\n  - 6", EVAL_CONFIG)
    require(paper, "trained with nine singleton attention blocks", MAIN_TEX)
    require(paper, "inference-only $[1,1,1,6]$ partition", MAIN_TEX)
    require(paper, "rather than a\nsemantics-preserving rewrite", MAIN_TEX)

    print("RESULT=PASS_TRAINING_INFERENCE_PARTITION_AUDIT")
    print(f"training_config={TRAIN_CONFIG.relative_to(ROOT)}")
    print(f"training_log={TRAIN_LOG.relative_to(ROOT)}")
    print(f"evaluation_config={EVAL_CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
