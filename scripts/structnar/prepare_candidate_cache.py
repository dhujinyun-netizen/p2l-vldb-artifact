#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap

POOLS=[
'visualnews_task0','mscoco_task0_test','fashion200k_task0','webqa_task1',
'edis_task2','webqa_task2','visualnews_task3','mscoco_task3_test',
'fashion200k_task3','nights_task4','oven_task6','infoseek_task6',
'fashioniq_task7','cirr_task7','oven_task8','infoseek_task8']


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',required=True)
    p.add_argument('--exp-name',required=True)
    p.add_argument('--quantizer-name',default='rq_clip_large_epoch_50.pth')
    p.add_argument('--model-short-name',default='GPT_HDGR')
    p.add_argument('--link-existing',action='store_true')
    p.add_argument('--build-union',action='store_true')
    return p.parse_args()


def manifest_ok(path: Path, quantizer_name: str)->bool:
    if not path.exists(): return True
    try:
        txt=path.read_text(errors='ignore')
        return quantizer_name in txt or 'quantizer' not in txt.lower()
    except Exception:
        return False


def find_source(root: Path, target_dir: Path, pool: str, quantizer_name: str):
    stem=f'mbeir_{pool}_cand_pool'
    for code in sorted(root.glob(f'*/cand_pool/{stem}_codes.npy'), key=lambda p:p.stat().st_mtime, reverse=True):
        d=code.parent
        if d.resolve()==target_dir.resolve(): continue
        emb=d/f'{stem}_embeddings.npy'; ids=d/f'{stem}_ids.npy'; man=d/f'{stem}_manifest.json'
        if emb.exists() and ids.exists() and manifest_ok(man,quantizer_name):
            return d
    return None


def link_file(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink(): return
    dst.symlink_to(src.resolve())


def load_mmap(path: Path):
    try: return np.load(path,mmap_mode='r',allow_pickle=False)
    except ValueError: return np.load(path,allow_pickle=True)


def build_union(cand: Path):
    kinds=['codes','embeddings','ids']
    sources={k:[] for k in kinds}
    for pool in POOLS:
        stem=f'mbeir_{pool}_cand_pool'
        for k in kinds:
            p=cand/f'{stem}_{k}.npy'
            if not p.exists(): raise SystemExit(f'Missing local candidate cache required for UNION: {p}')
            sources[k].append(p)
    # UNION candidate data is independent of the generator checkpoint and RRG
    # weight.  Reuse a complete existing cache instead of rewriting tens of GB
    # on every all-32 evaluation.
    expected_total = sum(len(load_mmap(p)) for p in sources['codes'])
    existing = {k: cand / f'mbeir_union_cand_pool_{k}.npy' for k in kinds}
    if all(p.exists() for p in existing.values()):
        try:
            union_arrays = {k: load_mmap(p) for k, p in existing.items()}
            lengths = {k: len(a) for k, a in union_arrays.items()}
            if all(n == expected_total for n in lengths.values()):
                print(
                    '[all32-cache] reusing complete UNION cache: '
                    f'rows={expected_total:,} files=' + ','.join(str(p.name) for p in existing.values())
                )
                return
            print(f'[all32-cache] existing UNION cache has wrong lengths {lengths}; rebuilding')
        except Exception as exc:
            print(f'[all32-cache] existing UNION cache validation failed ({exc}); rebuilding')

    for kind in kinds:
        arrays=[load_mmap(p) for p in sources[kind]]
        total=sum(len(a) for a in arrays)
        tail=arrays[0].shape[1:]
        if any(a.shape[1:]!=tail for a in arrays):
            raise SystemExit(f'Shape mismatch while building UNION {kind}')
        dtype=np.result_type(*[a.dtype for a in arrays])
        out=cand/f'mbeir_union_cand_pool_{kind}.npy'
        tmp=out.with_suffix('.npy.tmp')
        if tmp.exists(): tmp.unlink()
        mm=open_memmap(tmp,mode='w+',dtype=dtype,shape=(total,*tail))
        pos=0
        for p,a in zip(sources[kind],arrays):
            n=len(a); mm[pos:pos+n]=a; pos+=n
            print(f'[all32-cache] {kind}: copied {n:,} rows from {p.name}')
        mm.flush(); del mm
        os.replace(tmp,out)
        print(f'[all32-cache] built {out} rows={total:,} shape={(total,*tail)} dtype={dtype}')


def main():
    a=args(); repo=Path(a.repo_root).resolve()
    root=repo/'gen_code/GPT_HDGR/Large/Instruct'
    cand=repo/f'gen_code/{a.model_short_name}/Large/Instruct'/a.exp_name/'cand_pool'
    cand.mkdir(parents=True,exist_ok=True)
    if a.link_existing:
        linked=missing=0
        for pool in POOLS:
            src=find_source(root,cand,pool,a.quantizer_name)
            if src is None:
                print(f'[all32-cache] no reusable cache found for {pool}; evaluation will generate it')
                missing+=1; continue
            stem=f'mbeir_{pool}_cand_pool'
            for suffix in ['codes.npy','embeddings.npy','ids.npy','manifest.json','trie.pkl']:
                s=src/f'{stem}_{suffix}'
                if s.exists(): link_file(s,cand/s.name)
            print(f'[all32-cache] linked {pool} from {src.parent.name}')
            linked+=1
        print(f'[all32-cache] linked_pools={linked} missing_pools={missing}')
        # The union is quantizer-dependent but generator-independent. Link a
        # complete existing union cache so StructNAR does not rewrite ~10 GB.
        union_stem='mbeir_union_cand_pool'
        required=['codes.npy','embeddings.npy','ids.npy','manifest.json','trie.pkl']
        for code in sorted(root.glob(f'*/cand_pool/{union_stem}_codes.npy'), key=lambda p:p.stat().st_mtime, reverse=True):
            source_dir=code.parent
            if all((source_dir/f'{union_stem}_{suffix}').exists() for suffix in required):
                for suffix in required:
                    link_file(source_dir/f'{union_stem}_{suffix}', cand/f'{union_stem}_{suffix}')
                print(f'[all32-cache] linked complete UNION from {source_dir.parent.name}')
                break
    if a.build_union:
        build_union(cand)
    print('RESULT=PASS_ALL32_CANDIDATE_CACHE_PREP')

if __name__=='__main__': main()
