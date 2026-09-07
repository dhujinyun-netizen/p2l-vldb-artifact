#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
import yaml


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--local-out', required=True)
    p.add_argument('--union-out', required=True)
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--ckpt-name', required=True)
    p.add_argument('--exp-name', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-beams', type=int, default=50)
    p.add_argument('--rrg-weight', type=float, default=0.0)
    p.add_argument('--rrg-normalize', action='store_true')
    p.add_argument('--rqc-score-mode', choices=['gain', 'alignment', 'cosine'], default='cosine')
    p.add_argument('--rqc-apply-from-level', type=int, default=3)
    p.add_argument('--rqc-prefix-start-level', type=int, default=0)
    p.add_argument('--switch-depth', type=int, default=3)
    p.add_argument('--tcis-batched-cfg', action='store_true')
    p.add_argument('--tcis-selected-rrg', action='store_true')
    p.add_argument('--tcis-compact-head', action='store_true')
    p.add_argument('--tcis-dynamic-intermediate-beams', action='store_true')
    p.add_argument(
        '--deterministic-eval-sampling',
        action='store_true',
        help='Select the evaluation prompt and positive deterministically from qid.',
    )
    p.add_argument('--results-root', default='retrieval_results/structnar_all32')
    p.add_argument(
        '--datasets', nargs='*', default=None,
        help='Optional dataset names to retain in each phase (for low-cost validation).',
    )
    return p.parse_args()


def build(base: dict, args, phase: str):
    import copy
    cfg=copy.deepcopy(base)
    model=cfg['model']
    model['name']='GPTHDGRBackboneRetriever'
    model['generator_type']='gpt_hdgr'
    model['training_objective']=str(model.get('training_objective','masked_ce'))
    model['hdgr_code_tied_output_head']=True
    model['ckpt_config']={'ckpt_dir':args.ckpt_dir,'ckpt_name':args.ckpt_name}
    # Tree-Risk is a training objective; no asset/loss is needed during inference.
    model['use_tree_risk']=False
    model['use_rrg']=abs(args.rrg_weight)>0
    model['rrg_weight']=float(args.rrg_weight)
    model['rrg_normalize']=bool(args.rrg_normalize)
    model['rrg_skip_modality']=True
    model['rrg_eps']=1e-6
    model['rqc_score_mode']=args.rqc_score_mode
    model['rqc_apply_from_level']=int(args.rqc_apply_from_level)
    model['rqc_prefix_start_level']=int(args.rqc_prefix_start_level)
    model['rqc_level_weights']=None
    length=int(cfg['codebook_config']['codebook_level'])+1
    start=int(args.switch_depth)
    if start < 1 or start > length:
        raise SystemExit(f'--switch-depth must be in [1,{length}], got {start}')
    model['gpt_hdgr_hierarchy_aware_blocks']=True
    model['hierarchy_aware_blocks']=True
    model['hierarchy_block_sizes']=[1] * start
    model['hierarchy_block_names']=[f'prefix_{i}' for i in range(start)]
    if start < length:
        model['hierarchy_block_sizes'].append(length-start)
        model['hierarchy_block_names'].append('complete_id_selection')
    model['gpt_hdgr_block_diffusion_steps']=1
    model['gpt_hdgr_block_score_normalization']='sum'
    model['block_trie_max_candidates']=0
    model['deduplicate_block_beams']=True
    model['tcis_max_leaves']=0
    # Full RRG is the exact reference implementation. The selected-row shortcut
    # can perturb low-ranked beams and is therefore excluded from official runs.
    model['tcis_selected_rrg']=bool(args.tcis_selected_rrg)
    model['tcis_batched_cfg']=bool(args.tcis_batched_cfg)
    model['tcis_compact_head']=bool(args.tcis_compact_head)
    model['tcis_dynamic_intermediate_beams']=bool(args.tcis_dynamic_intermediate_beams)
    model['use_tcis']=True
    for k in list(model):
        if str(k).startswith('soundstorm_'):
            model.pop(k)

    cfg.setdefault('dataloader_config',{})['batch_size']=args.batch_size
    cfg.setdefault('data_config',{})['deterministic_eval_sampling'] = bool(
        args.deterministic_eval_sampling
    )
    cfg.setdefault('experiment',{})['exp_name']=args.exp_name
    cfg['experiment'].setdefault('path_suffix','${model.short_name}/${model.size}/${experiment.instruct_status}/${experiment.exp_name}/')

    ret=cfg['retrieval_config']
    ret['num_beams']=args.num_beams
    ret['rerank']=True
    ret['save_beam_scores']=True
    ret['load_saved_beam_scores']=True
    ret['use_hybrid_score']=False
    ret['write_to_tsv']=True
    ret['results_dir_name']=f"{args.results_root}/{phase}"

    test=ret['test_datasets_config']
    # The official file lists 16 local mappings followed by 16 UNION mappings.
    sl=slice(0,16) if phase=='local' else slice(16,32)
    for key in ['datasets_name','correspond_cand_pools_name','correspond_qrels_name','correspond_metrics_name']:
        values=list(test[key])
        if len(values)<32:
            raise SystemExit(f'Expected 32 official mappings in {key}, got {len(values)}')
        test[key]=values[sl]
    if args.datasets:
        keep_names=set(args.datasets)
        keep=[i for i,name in enumerate(test['datasets_name']) if name in keep_names]
        missing=keep_names.difference(test['datasets_name'])
        if missing:
            raise SystemExit(f'Datasets not present in {phase} mappings: {sorted(missing)}')
        for key in ['datasets_name','correspond_cand_pools_name','correspond_qrels_name','correspond_metrics_name']:
            test[key]=[test[key][i] for i in keep]
    test['enable_gen_code']=True
    test['enable_retrieve']=True

    cpc=ret['cand_pools_config']
    if phase=='local':
        # Candidate files may be linked from prior RQ50 runs; missing ones are generated.
        cpc['enable_gen_code']=True
        cpc['gen_code_union_pool']=False
        if args.datasets:
            # Avoid scanning/link-checking unrelated candidate pools during a
            # selected-dataset speed probe. Exceptional names such as the
            # MSCOCO *_test pool are taken from the already filtered mapping.
            selected_pools=[]
            for name in test['correspond_cand_pools_name']:
                if name not in selected_pools:
                    selected_pools.append(name)
            cpc['cand_pools_name_to_gen_code']=selected_pools
    else:
        # UNION files are built by the streaming cache assembler before this phase.
        cpc['enable_gen_code']=False
        cpc['gen_code_union_pool']=False
    return cfg


def main():
    args=parse_args()
    if not math.isfinite(args.rrg_weight):
        raise SystemExit('pro-weight must be finite')
    base=yaml.safe_load(Path(args.base).read_text())
    for phase,out in [('local',args.local_out),('union',args.union_out)]:
        cfg=build(base,args,phase)
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
        print(f'[all32] wrote {phase} config: {out}')

if __name__=='__main__':
    main()
