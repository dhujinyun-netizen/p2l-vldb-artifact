#!/usr/bin/env python3
"""Create a final StructNAR config for a fixed CIRR nested candidate pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--decoder", choices=("p2l", "sequential"), required=True)
    parser.add_argument("--weight", type=float, default=20.0)
    parser.add_argument("--union", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.base.read_text())
    cfg["experiment"]["exp_name"] = args.exp_name
    cfg["dataloader_config"]["batch_size"] = 8
    cfg["data_config"]["auto_align_candidate_pool"] = False
    cfg["data_config"]["deterministic_eval_sampling"] = True
    model = cfg["model"]
    model["use_tcis"] = True
    model["use_rrg"] = True
    model["rrg_weight"] = float(args.weight)
    model["rqc_score_mode"] = "cosine"
    model["rqc_apply_from_level"] = 3
    model["rqc_prefix_start_level"] = 0
    model["tcis_max_leaves"] = 0
    if args.decoder == "sequential":
        model["hierarchy_block_sizes"] = [1] * 9
        model["hierarchy_block_names"] = [f"level_{i}" for i in range(9)]
    else:
        model["hierarchy_block_sizes"] = [1, 1, 1, 6]
        model["hierarchy_block_names"] = [
            "prefix_0",
            "prefix_1",
            "prefix_2",
            "complete_id_selection",
        ]
    retrieval = cfg["retrieval_config"]
    retrieval["num_beams"] = 50
    retrieval["results_dir_name"] = (
        f"retrieval_results/structnar_scale_w{int(args.weight)}/{args.decoder}/cirr_task7"
    )
    # Never reuse semantic-ID beams from an older score or decoder policy.
    retrieval["load_saved_beam_scores"] = False
    retrieval["save_beam_scores"] = True
    # The full UNION codes already exist.  Setting gen_code_union_pool would
    # rebuild them by concatenating every local pool; for this query-only scale
    # experiment we instead reuse the validated cache directly.
    retrieval["cand_pools_config"]["cand_pools_name_to_gen_code"] = (
        [] if args.union else [args.pool]
    )
    retrieval["cand_pools_config"]["gen_code_union_pool"] = False
    test = retrieval["test_datasets_config"]
    test["datasets_name"] = ["cirr_task7"]
    test["correspond_cand_pools_name"] = ["UNION" if args.union else args.pool]
    test["correspond_qrels_name"] = ["cirr_task7"]
    test["correspond_metrics_name"] = ["Recall@1, Recall@5, Recall@10"]
    test["enable_gen_code"] = True
    test["enable_retrieve"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
