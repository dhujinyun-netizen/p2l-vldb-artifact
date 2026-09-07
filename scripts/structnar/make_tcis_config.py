#!/usr/bin/env python3
"""Build an evaluation config for Trie-Constrained Complete-ID Selection."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("configs/structnar/eval.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switch-depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=50)
    parser.add_argument("--tcis-max-leaves", type=int, default=0)
    parser.add_argument(
        "--frontier-leaf-budget",
        type=int,
        default=0,
        help=(
            "Enable the query-level frontier planner when positive. A query "
            "switches once the unique live-prefix descendants fit this budget."
        ),
    )
    parser.add_argument(
        "--frontier-min-prefix",
        type=int,
        default=2,
        help="Minimum prefix length (including y0) before a planned switch.",
    )
    parser.add_argument(
        "--frontier-max-prefix",
        type=int,
        default=5,
        help="Maximum prefix length (including y0) before a forced switch.",
    )
    parser.add_argument(
        "--block-refinement-steps",
        type=int,
        default=1,
        help="Masked refinement passes per decoding block; values >1 provide a suffix-dependency-aware control.",
    )
    parser.add_argument("--tcis-selected-rrg", action="store_true")
    parser.add_argument("--tcis-batched-cfg", action="store_true")
    parser.add_argument("--tcis-compact-head", action="store_true")
    parser.add_argument("--tcis-dynamic-intermediate-beams", action="store_true")
    parser.add_argument(
        "--intermediate-beam-multiplier",
        type=int,
        default=1,
        help="Keep multiplier*num_beams hypotheses before the final block, then return num_beams.",
    )
    parser.add_argument(
        "--prefix-beams",
        type=int,
        default=0,
        help="Independent prefix-frontier width; 0 keeps the historical num_beams width.",
    )
    parser.add_argument("--tcis-prefix-kv-cache", action="store_true")
    parser.add_argument("--tcis-active-token-pruning", action="store_true")
    parser.add_argument(
        "--tcis-isolate-suffix-states",
        action="store_true",
        help=(
            "Keep complete-leaf enumeration and one-round suffix scoring, but "
            "use singleton attention blocks for all suffix positions. This is "
            "an ablation of latent suffix-state interaction, not a sequential "
            "decoder."
        ),
    )
    parser.add_argument("--tcis-force-legal-token-pruning", action="store_true")
    parser.add_argument("--deterministic-eval-sampling", action="store_true")
    parser.add_argument(
        "--block-score-mode", choices=("sum", "mean", "last", "prefix"), default="sum",
        help="How legal suffix token log-probabilities are reduced for beam ranking.",
    )
    parser.add_argument("--dataset", default="cirr_task7")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--rrg-weight", type=float, default=20.0)
    parser.add_argument(
        "--rrg-normalization",
        choices=("full_codebook", "none"),
        default="full_codebook",
        help="Z-standardize compatibility over the complete level codebook or keep raw scores.",
    )
    parser.add_argument(
        "--rqc-score-mode", choices=("gain", "alignment", "cosine"), default="cosine"
    )
    parser.add_argument("--rqc-apply-from-level", type=int, default=3)
    parser.add_argument("--rqc-prefix-start-level", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--rqc-level-weights",
        default="",
        help="Optional comma-separated weight for every ID level.",
    )
    parser.add_argument("--exp-name", default="structnar_cirr_tcis_d3")
    parser.add_argument("--ckpt-dir", default="checkpoint/code_tied")
    parser.add_argument("--ckpt-name", default="gpt_hdgr_latest.pth")
    args = parser.parse_args()
    rqc_level_weights = None
    if args.rqc_level_weights:
        rqc_level_weights = [
            float(value) for value in str(args.rqc_level_weights).split(",")
        ]

    cfg = yaml.safe_load(args.base.read_text())
    model = cfg["model"]
    length = int(cfg["codebook_config"]["codebook_level"]) + 1
    start = int(args.switch_depth)
    if start < 1 or start > length:
        raise ValueError(f"switch-depth must be in [1,{length}], got {start}")

    # Prefix positions remain single-token constrained expansions.  Every token
    # after `start` is completed in one block, but candidates for that block are
    # enumerated exclusively from the current Trie node.
    # `start == length` is the matched sequential control: nine legal
    # single-token expansions through exactly the same block decoder.
    sizes = [1] * start
    names = [f"prefix_{i}" for i in range(start)]
    if start < length:
        sizes.append(length - start)
        names.append("complete_id_selection")
    model.update(
        {
            "name": "GPTHDGRBackboneRetriever",
            "generator_type": "gpt_hdgr",
            "hdgr_code_tied_output_head": True,
            "ckpt_config": {
                "ckpt_dir": str(args.ckpt_dir),
                "ckpt_name": str(args.ckpt_name),
            },
            "gpt_hdgr_hierarchy_aware_blocks": True,
            "hierarchy_aware_blocks": True,
            "hierarchy_block_sizes": sizes,
            "hierarchy_block_names": names,
            "use_tcis": True,
            "gpt_hdgr_block_diffusion_steps": max(1, int(args.block_refinement_steps)),
            "gpt_hdgr_block_score_normalization": str(args.block_score_mode),
            "block_trie_max_candidates": 0,
            "deduplicate_block_beams": True,
            "tcis_max_leaves": int(args.tcis_max_leaves),
            "tcis_frontier_leaf_budget": max(0, int(args.frontier_leaf_budget)),
            "tcis_frontier_min_prefix": max(1, int(args.frontier_min_prefix)),
            "tcis_frontier_max_prefix": max(1, int(args.frontier_max_prefix)),
            "tcis_selected_rrg": bool(args.tcis_selected_rrg),
            "tcis_batched_cfg": bool(args.tcis_batched_cfg),
            "tcis_compact_head": bool(args.tcis_compact_head),
            "tcis_dynamic_intermediate_beams": bool(
                args.tcis_dynamic_intermediate_beams
            ),
            "tcis_intermediate_beam_multiplier": max(
                1, int(args.intermediate_beam_multiplier)
            ),
            "tcis_prefix_beams": max(0, int(args.prefix_beams)),
            "tcis_prefix_kv_cache": bool(args.tcis_prefix_kv_cache),
            "tcis_active_token_pruning": bool(args.tcis_active_token_pruning),
            "tcis_isolate_suffix_states": bool(args.tcis_isolate_suffix_states),
            "tcis_force_legal_token_pruning": bool(
                args.tcis_force_legal_token_pruning
            ),
            "use_rrg": abs(float(args.rrg_weight)) > 0.0,
            "rrg_weight": float(args.rrg_weight),
            "rrg_normalize": args.rrg_normalization == "full_codebook",
            "rqc_score_mode": str(args.rqc_score_mode),
            "rqc_apply_from_level": int(args.rqc_apply_from_level),
            "rqc_prefix_start_level": int(args.rqc_prefix_start_level),
            "rqc_level_weights": rqc_level_weights,
        }
    )

    cfg["experiment"]["exp_name"] = args.exp_name
    cfg["dataloader_config"]["batch_size"] = int(args.batch_size)
    cfg["data_config"]["deterministic_eval_sampling"] = bool(args.deterministic_eval_sampling)
    ret = cfg["retrieval_config"]
    ret["num_beams"] = int(args.num_beams)
    ret["rerank"] = True
    ret["use_hybrid_score"] = False
    ret["save_beam_scores"] = True
    ret["load_saved_beam_scores"] = True
    dataset = str(args.dataset)
    ret["results_dir_name"] = f"retrieval_results/structnar_tcis/{dataset}"

    test = ret["test_datasets_config"]
    datasets = list(test["datasets_name"])
    local_indices = [i for i, name in enumerate(datasets[:16]) if name == dataset]
    if len(local_indices) != 1:
        raise ValueError(f"Expected one local mapping for {dataset}, got {local_indices}")
    i = local_indices[0]
    for key in (
        "datasets_name",
        "correspond_cand_pools_name",
        "correspond_qrels_name",
        "correspond_metrics_name",
    ):
        test[key] = [copy.deepcopy(test[key][i])]
    test["enable_gen_code"] = True
    test["enable_retrieve"] = True
    if args.split == "val":
        ret["val_datasets_config"] = copy.deepcopy(test)
        test["datasets_name"] = None
        test["correspond_cand_pools_name"] = None
        test["correspond_qrels_name"] = None
        test["correspond_metrics_name"] = None
        test["enable_gen_code"] = False
        test["enable_retrieve"] = False
    ret["cand_pools_config"]["enable_gen_code"] = True
    ret["cand_pools_config"]["gen_code_union_pool"] = False
    selected = ret["val_datasets_config"] if args.split == "val" else test
    candidate_name = str(selected["correspond_cand_pools_name"][0])
    ret["cand_pools_config"]["cand_pools_name_to_gen_code"] = [candidate_name]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[TCIS] wrote {args.output}; spans={sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
