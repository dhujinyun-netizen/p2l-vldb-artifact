#!/usr/bin/env python
"""Lightweight official M-BEIR layout/config checker.

This intentionally avoids loading the full UNION candidate pool into memory.
It checks that configured query/candidate/qrels paths exist and reports the exact
32 official test mappings that will be evaluated.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from omegaconf import OmegaConf

EXCEPTIONAL = {"mscoco_task0", "mscoco_task3", "flickr30k_task0", "flickr30k_task3"}

def norm_cand(name: str) -> str:
    name = str(name).lower()
    return name + "_test" if name in EXCEPTIONAL else name

def as_list(x):
    return [] if x is None else list(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", required=True)
    p.add_argument("--mbeir-data-dir", required=True)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--check-extracted", action="store_true")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config_path)
    data = cfg.data_config
    retrieval = cfg.retrieval_config
    mbeir = Path(args.mbeir_data_dir)
    repo = Path(args.repo_root) if args.repo_root else Path.cwd()
    errors = []

    for d in ["query", "cand_pool", "qrels", "instructions", "mbeir_images"]:
        if not (mbeir / d).exists():
            errors.append(f"missing directory: {mbeir/d}")
    if not (mbeir / str(data.query_instruct_path)).exists():
        errors.append(f"missing query instructions: {mbeir/str(data.query_instruct_path)}")

    for split in ["train", "val", "test"]:
        scfg = getattr(retrieval, f"{split}_datasets_config", None)
        if not scfg or not bool(getattr(scfg, "enable_retrieve", False)):
            continue
        dsets = as_list(getattr(scfg, "datasets_name", None))
        cands = as_list(getattr(scfg, "correspond_cand_pools_name", None))
        qrels = as_list(getattr(scfg, "correspond_qrels_name", None))
        mets = as_list(getattr(scfg, "correspond_metrics_name", None))
        if not (len(dsets) == len(cands) == len(qrels) == len(mets)):
            errors.append(f"{split}: config list length mismatch")
            continue
        qdir = mbeir / str(getattr(data, f"{split}_dir_name"))
        cdir = mbeir / str(data.cand_pool_dir_name)
        udir = mbeir / str(data.union_cand_pool_dir_name)
        rdir = mbeir / str(retrieval.qrel_dir_name) / split
        print(f"[layout] split={split} n_mappings={len(dsets)}")
        for i, (ds, cp, qr, met) in enumerate(zip(dsets, cands, qrels, mets), 1):
            ds = str(ds).lower(); cp_raw = str(cp).lower(); cp_norm = norm_cand(cp_raw); qr = str(qr).lower()
            qpath = qdir / f"mbeir_{ds}_{split}.jsonl"
            cpath = (udir / f"mbeir_union_{split}_cand_pool.jsonl") if cp_norm == "union" else (cdir / f"mbeir_{cp_norm}_cand_pool.jsonl")
            rpath = rdir / f"mbeir_{qr}_{split}_qrels.txt"
            for kind, path in [("query", qpath), ("candidate", cpath), ("qrels", rpath)]:
                if not path.exists():
                    errors.append(f"{split}/{i}/{ds}: missing {kind}: {path}")
            print(f"[layout] {i:02d}: dataset={ds} cand={cp_raw}->{cp_norm} qrel={qr} metric={met}")

    if args.check_extracted:
        edir = repo / str(data.extracted_dir)
        expected = [norm_cand(x) for x in as_list(retrieval.cand_pools_config.cand_pools_name_to_gen_code)]
        if bool(getattr(retrieval.cand_pools_config, "gen_code_union_pool", False)):
            expected.append("mbeir_union_cand_pool")
        for name in expected:
            if name == "mbeir_union_cand_pool":
                pt = edir / "mbeir_union_cand_pool_IT_dict.pt"
            else:
                pt = edir / f"cand_pool_{name}_IT_dict.pt"
            if not pt.exists():
                errors.append(f"missing extracted candidate feature: {pt}")

    if errors:
        print("[layout] FAILED")
        for e in errors:
            print("[layout] ERROR:", e)
        raise SystemExit(2)
    print("[layout] OK")

if __name__ == "__main__":
    main()
