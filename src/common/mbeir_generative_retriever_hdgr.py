import os
import argparse
import json
import hashlib
import time
from omegaconf import OmegaConf
import tqdm
import gc
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.cuda.amp import autocast
import transformers
from datetime import datetime
import torch.nn.functional as F

import dist_utils
from dist_utils import ContiguousDistributedSampler
from hdgr_utils import build_model_from_config
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    MBEIRDictCandDataset,
    Mode,
)
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

from data.preprocessing.utils import (
    load_jsonl_as_list,
    save_list_as_jsonl,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_dataset_stats,
    unhash_did,
    unhash_qid,
    get_mbeir_task_name,
)
import sys
import csv

sys.setrecursionlimit(10000)


def _ids_batch_to_list(ids):
    """Normalize batched IDs from datasets/quantizer into a Python list.

    Some HDGR paths return numeric tensor IDs from pre-extracted dictionaries, while
    raw MBEIR candidate-pool fallback returns hashed string IDs such as "10:151".
    Retrieval/eval code should preserve both forms instead of assuming tensors.
    """
    if ids is None:
        return []
    if isinstance(ids, torch.Tensor):
        flat = ids.detach().cpu().view(-1).tolist()
        return flat
    if isinstance(ids, np.ndarray):
        return ids.reshape(-1).tolist()
    if isinstance(ids, (list, tuple)):
        out = []
        for item in ids:
            if isinstance(item, torch.Tensor):
                out.extend(item.detach().cpu().view(-1).tolist())
            elif isinstance(item, np.ndarray):
                out.extend(item.reshape(-1).tolist())
            elif isinstance(item, (list, tuple)):
                out.extend(_ids_batch_to_list(item))
            else:
                out.append(item)
        return out
    return [ids]


def _tensor_batch_to_2d(value, *, name, id_count=None, expected_width=None, device=None):
    """Normalize batched code/embedding tensors to [N, D].

    Raw MBEIR fallback and pre-extracted dictionary paths may return slightly
    different shapes, especially for a final batch of size 1.  This helper
    prevents mixing [D] and [N, D] tensors before torch.cat().
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if device is not None:
        value = value.to(device)
    if value.dim() == 0:
        value = value.view(1, 1)
    elif value.dim() == 1:
        n = int(value.numel())
        if id_count == 1:
            value = value.view(1, -1)
        elif expected_width is not None and expected_width > 0 and n == int(id_count or 0) * expected_width:
            value = value.view(int(id_count), expected_width)
        elif expected_width is not None and expected_width > 0 and n == expected_width:
            value = value.view(1, expected_width)
        elif id_count is not None and id_count > 0 and n == id_count:
            value = value.view(id_count, 1)
        else:
            # Semantic-ID code vectors are usually returned as [D] only when
            # batch size is 1, so [1, D] is the safest fallback.
            value = value.view(1, -1)
    elif value.dim() > 2:
        value = value.view(value.size(0), -1)

    if id_count is not None and value.size(0) != id_count:
        raise RuntimeError(
            f"{name} batch size mismatch after normalization: tensor shape={tuple(value.shape)}, "
            f"id_count={id_count}. This usually means the dataset collator or quantizer returned "
            "an unexpected code/id shape."
        )
    return value.contiguous()


def _code_batch_to_tensor(value, *, name, id_count=None, expected_width=None, preserve_beams=False, device=None):
    """Normalize semantic-ID code tensors without destroying beam structure.

    Candidate codes are normally [B, L]. Query generation can return [B, K, L].
    Earlier revisions flattened [B, K, L] into [B, K*L], which made retrieval
    iterate over scalar numpy.int64 values.  This helper keeps query beams as a
    3D tensor while still accepting final-batch singleton shapes such as [L].
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if device is not None:
        value = value.to(device)
    if value.dim() == 0:
        value = value.view(1, 1, 1) if preserve_beams else value.view(1, 1)
    elif value.dim() == 1:
        n = int(value.numel())
        if preserve_beams:
            if id_count is not None and id_count > 1 and expected_width and n == id_count * expected_width:
                value = value.view(int(id_count), 1, int(expected_width))
            elif id_count is not None and id_count > 1 and n == id_count:
                value = value.view(int(id_count), 1, 1)
            else:
                value = value.view(1, 1, -1)
        else:
            if id_count == 1:
                value = value.view(1, -1)
            elif expected_width is not None and expected_width > 0 and id_count and n == id_count * expected_width:
                value = value.view(int(id_count), int(expected_width))
            elif id_count is not None and id_count > 0 and n == id_count:
                value = value.view(int(id_count), 1)
            else:
                value = value.view(1, -1)
    elif value.dim() == 2:
        if preserve_beams:
            if id_count is not None and value.size(0) == id_count:
                value = value.unsqueeze(1)  # [B, L] -> [B, 1, L]
            elif id_count == 1:
                value = value.unsqueeze(0)  # [K, L] -> [1, K, L]
            else:
                # Last-resort interpretation: batch-major single beam.
                value = value.unsqueeze(1)
    elif value.dim() > 3:
        value = value.view(value.size(0), value.size(1), -1) if preserve_beams else value.view(value.size(0), -1)

    if id_count is not None and value.size(0) != id_count:
        raise RuntimeError(
            f"{name} batch size mismatch after normalization: tensor shape={tuple(value.shape)}, "
            f"id_count={id_count}. preserve_beams={preserve_beams}."
        )
    return value.contiguous()


def _beam_score_batch_to_tensor(value, *, name, id_count=None, expected_beams=None, device=None):
    """Normalize generated sequence scores to ``[batch, num_beams]``."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if device is not None:
        value = value.to(device)
    value = value.float()

    if value.dim() == 0:
        value = value.view(1, 1)
    elif value.dim() == 1:
        if id_count == 1:
            value = value.view(1, -1)
        elif id_count is not None and expected_beams and value.numel() == int(id_count) * int(expected_beams):
            value = value.view(int(id_count), int(expected_beams))
        elif id_count is not None and value.numel() == int(id_count):
            value = value.view(int(id_count), 1)
        else:
            value = value.view(1, -1)
    elif value.dim() > 2:
        value = value.view(value.size(0), -1)

    if id_count is not None and value.size(0) != int(id_count):
        raise RuntimeError(
            f"{name} batch size mismatch after normalization: shape={tuple(value.shape)}, "
            f"id_count={id_count}."
        )
    if expected_beams is not None and value.size(1) != int(expected_beams):
        raise RuntimeError(
            f"{name} beam-count mismatch: shape={tuple(value.shape)}, expected_beams={expected_beams}."
        )
    return value.contiguous()


def _normalize_candidate_codes(codes):
    """Return candidate semantic IDs as [num_candidates, code_length]."""
    arr = np.asarray(codes)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.int64, copy=False)


def _normalize_query_codes_for_retrieval(query_codes, candidate_code_width=None):
    """Return generated query beams as [num_queries, num_beams, code_length]."""
    arr = np.asarray(query_codes)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1, 1)
    elif arr.ndim == 1:
        width = int(candidate_code_width or arr.shape[0])
        if candidate_code_width and arr.size % width == 0:
            arr = arr.reshape(1, arr.size // width, width)
        else:
            arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        if candidate_code_width and arr.shape[1] != candidate_code_width and arr.shape[1] % candidate_code_width == 0:
            arr = arr.reshape(arr.shape[0], arr.shape[1] // candidate_code_width, candidate_code_width)
        else:
            arr = arr[:, None, :]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    return arr.astype(np.int64, copy=False)


def _normalize_query_beam_scores_for_retrieval(query_beam_scores, num_queries, num_beams):
    """Return optional beam scores as ``[num_queries, num_beams]``."""
    if query_beam_scores is None:
        return None
    arr = np.asarray(query_beam_scores, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if int(num_queries) == 1:
            arr = arr.reshape(1, -1)
        elif arr.size == int(num_queries) * int(num_beams):
            arr = arr.reshape(int(num_queries), int(num_beams))
        elif arr.size == int(num_queries) and int(num_beams) == 1:
            arr = arr.reshape(int(num_queries), 1)
        else:
            raise ValueError(
                f"Cannot reshape beam scores of shape {arr.shape} to "
                f"[{num_queries}, {num_beams}]."
            )
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[0] != int(num_queries):
        raise ValueError(
            f"Beam-score query count mismatch: scores={arr.shape}, num_queries={num_queries}."
        )
    if arr.shape[1] != int(num_beams):
        raise ValueError(
            f"Beam-score width mismatch: scores={arr.shape}, num_beams={num_beams}. "
            "Delete stale generated-code caches and regenerate query beams."
        )
    return arr.astype(np.float32, copy=False)


def _code_to_tuple(code):
    arr = np.asarray(code)
    if arr.ndim == 0:
        return (int(arr.item()),)
    return tuple(int(x) for x in arr.reshape(-1).tolist())


def _iter_code_beams(query_code):
    arr = np.asarray(query_code)
    if arr.ndim == 0:
        yield arr.reshape(1)
    elif arr.ndim == 1:
        yield arr
    elif arr.ndim == 2:
        for row in arr:
            yield row
    else:
        flat = arr.reshape(arr.shape[0], -1)
        for row in flat:
            yield row

EXCEPTIONAL_CAND_POOLS = ['mscoco_task0', 'mscoco_task3', 'flickr30k_task0', 'flickr30k_task3']


def _normalize_cand_pool_name(cand_pool_name):
    cand_pool_name = str(cand_pool_name).lower()
    if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
        cand_pool_name = cand_pool_name + '_test'
    return cand_pool_name


def _expected_extracted_cand_path(genir_dir, data_config, split_name, cand_pool_name):
    extracted_file_name = f"{split_name}_{cand_pool_name}_IT_dict.pt"
    return os.path.join(genir_dir, data_config.extracted_dir, extracted_file_name)


def _missing_extracted_cand_message(missing_path, cand_pool_name, cand_pool_data_path):
    return (
        "Missing extracted candidate embedding file for HDGR eval.\n"
        f"  expected: {missing_path}\n"
        f"  cand_pool: {cand_pool_name}\n"
        f"  cand_jsonl: {cand_pool_data_path}\n"
        "Run candidate feature extraction before eval, for example:\n"
        "  bash scripts/feature_extraction/extract_cand.sh\n"
        "or use the eval wrapper with AUTO_EXTRACT_CAND=1 so it is generated automatically.\n"
        "To bypass the extracted-feature path for debugging, set:\n"
        "  data_config.auto_fallback_to_raw_if_missing_extracted=true\n"
    )



def _resolve_mbeir_jsonl_path(mbeir_data_dir, rel_or_abs_path):
    """Resolve a MBEIR jsonl path across official and project-local layouts."""
    path = rel_or_abs_path if os.path.isabs(str(rel_or_abs_path)) else os.path.join(mbeir_data_dir, str(rel_or_abs_path))
    if os.path.exists(path):
        return path
    basename = os.path.basename(str(rel_or_abs_path))
    matches = []
    for root, _, files in os.walk(mbeir_data_dir):
        if basename in files:
            matches.append(os.path.join(root, basename))
    if matches:
        matches = sorted(matches, key=lambda x: ("/src_data/" not in x.replace(os.sep, "/"), x))
        return matches[0]
    return path


def _read_jsonl_entries(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(__import__("json").loads(line))
    return entries


def _positive_dids_from_query_jsonl(query_jsonl_path):
    positives = []
    sample_qids = []
    for entry in _read_jsonl_entries(query_jsonl_path):
        qid = entry.get("qid")
        if qid is not None and len(sample_qids) < 5:
            sample_qids.append(str(qid))
        for did in entry.get("pos_cand_list", []) or []:
            positives.append(str(did))
    return set(positives), sample_qids


def _candidate_dids_from_pool_jsonl(cand_pool_jsonl_path):
    dids = []
    for entry in _read_jsonl_entries(cand_pool_jsonl_path):
        did = entry.get("did")
        if did is not None:
            dids.append(str(did))
    return set(dids)


def _candidate_pool_search_roots(mbeir_data_dir, preferred_rel_paths):
    roots = []
    for rel_path in preferred_rel_paths:
        if not rel_path:
            continue
        abs_path = rel_path if os.path.isabs(str(rel_path)) else os.path.join(mbeir_data_dir, str(rel_path))
        root = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)
        if root and os.path.exists(root):
            roots.append(root)
    roots.append(mbeir_data_dir)
    deduped = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if root not in seen:
            seen.add(root)
            deduped.append(root)
    return deduped


def _find_candidate_pool_files(mbeir_data_dir, preferred_rel_paths=()):
    files = []
    seen = set()
    for root in _candidate_pool_search_roots(mbeir_data_dir, preferred_rel_paths):
        for cur_root, _, names in os.walk(root):
            for name in names:
                if name.endswith("_cand_pool.jsonl") and name.startswith("mbeir_"):
                    path = os.path.join(cur_root, name)
                    if path not in seen:
                        seen.add(path)
                        files.append(path)
    return sorted(files)


def _auto_align_candidate_pool_path(mbeir_data_dir, query_data_path, cand_pool_data_path, *, enabled=True, min_overlap=1):
    """Return a candidate-pool path whose dids overlap the query positives.

    Some local MBEIR copies contain task-specific files whose names look aligned
    but whose positive dids are absent from the configured pool.  For retrieval
    this makes Recall@k impossible.  When enabled, scan available candidate-pool
    jsonl files and switch to the one with the largest positive-did overlap.
    The caller may still save generated candidate codes under the configured
    pool name so downstream paths remain stable.
    """
    resolved_query_path = _resolve_mbeir_jsonl_path(mbeir_data_dir, query_data_path)
    resolved_configured_pool_path = _resolve_mbeir_jsonl_path(mbeir_data_dir, cand_pool_data_path)
    if not os.path.exists(resolved_query_path) or not os.path.exists(resolved_configured_pool_path):
        return cand_pool_data_path, {"aligned": False, "reason": "path_missing"}

    positive_dids, sample_qids = _positive_dids_from_query_jsonl(resolved_query_path)
    configured_dids = _candidate_dids_from_pool_jsonl(resolved_configured_pool_path)
    configured_overlap = len(positive_dids & configured_dids)
    stats = {
        "aligned": False,
        "query_path": resolved_query_path,
        "configured_pool_path": resolved_configured_pool_path,
        "num_positive_dids": len(positive_dids),
        "configured_num_dids": len(configured_dids),
        "configured_overlap": configured_overlap,
        "sample_qids": sample_qids,
        "top_candidates": [],
    }
    if configured_overlap >= min_overlap or not enabled:
        stats["aligned"] = True
        stats["selected_pool_path"] = resolved_configured_pool_path
        stats["selected_overlap"] = configured_overlap
        return cand_pool_data_path, stats

    # Prefer files in the same configured split directory, then scan the full MBEIR root.
    candidate_files = _find_candidate_pool_files(
        mbeir_data_dir,
        preferred_rel_paths=(cand_pool_data_path, os.path.dirname(str(cand_pool_data_path)), os.path.dirname(str(query_data_path))),
    )
    scored = []
    query_basename = os.path.basename(str(query_data_path)).lower()
    for pool_path in candidate_files:
        try:
            dids = _candidate_dids_from_pool_jsonl(pool_path)
        except Exception:
            continue
        overlap = len(positive_dids & dids)
        # Mild tie-breaker for files with similar dataset/split names.
        basename = os.path.basename(pool_path).lower()
        name_bonus = 0
        for tok in query_basename.replace(".jsonl", "").split("_"):
            if tok and tok in basename:
                name_bonus += 1
        scored.append((overlap, name_bonus, len(dids), pool_path))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    stats["top_candidates"] = [
        {"overlap": int(overlap), "num_dids": int(num_dids), "path": path}
        for overlap, _, num_dids, path in scored[:10]
    ]

    if scored and scored[0][0] >= min_overlap:
        selected = scored[0][3]
        selected_rel = os.path.relpath(selected, mbeir_data_dir)
        stats["aligned"] = True
        stats["selected_pool_path"] = selected
        stats["selected_overlap"] = int(scored[0][0])
        stats["selected_num_dids"] = int(scored[0][2])
        return selected_rel, stats

    return cand_pool_data_path, stats


def _print_candidate_alignment_stats(stats):
    if not stats or "configured_overlap" not in stats:
        return
    print(
        "[HDGR eval] Candidate-pool overlap check: "
        f"positive_dids={stats.get('num_positive_dids')} "
        f"configured_pool_dids={stats.get('configured_num_dids')} "
        f"configured_overlap={stats.get('configured_overlap')}"
    )
    if stats.get("sample_qids"):
        print(f"[HDGR eval] Sample query ids: {stats.get('sample_qids')}")
    if stats.get("selected_pool_path") and stats.get("selected_pool_path") != stats.get("configured_pool_path"):
        print(
            "[HDGR eval] Auto-selected aligned candidate pool: "
            f"{stats.get('selected_pool_path')} "
            f"(overlap={stats.get('selected_overlap')}, dids={stats.get('selected_num_dids')})"
        )
    if stats.get("top_candidates"):
        print("[HDGR eval] Candidate-pool overlap top matches:")
        for item in stats.get("top_candidates")[:5]:
            print(f"  overlap={item['overlap']} dids={item['num_dids']} path={item['path']}")



def _resolve_under_root(root_dir, maybe_path):
    if maybe_path is None:
        return None
    maybe_path = str(maybe_path)
    if not maybe_path:
        return None
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(root_dir, maybe_path)


def _nested_get(obj, dotted_key, default=None):
    cur = obj
    for part in str(dotted_key).split('.'):
        if cur is None:
            return default
        try:
            cur = getattr(cur, part)
        except Exception:
            try:
                cur = cur[part]
            except Exception:
                return default
    return cur


def _file_signature(path, *, include_hash=False):
    if path is None:
        return {"exists": False, "path": None}
    abs_path = os.path.abspath(str(path))
    if not os.path.exists(abs_path):
        return {"exists": False, "path": abs_path}
    stat = os.stat(abs_path)
    out = {
        "exists": True,
        "path": abs_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        h = hashlib.sha1()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out["sha1"] = h.hexdigest()
    return out


def _candidate_output_paths(genir_dir, gen_code_dir_name, expt_dir_name, output_cand_pool_name):
    cand_save_dir = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool")
    prefix = os.path.join(cand_save_dir, f"mbeir_{output_cand_pool_name}_cand_pool")
    return {
        "cand_save_dir": cand_save_dir,
        "code_path": prefix + "_codes.npy",
        "emb_path": prefix + "_embeddings.npy",
        "id_path": prefix + "_ids.npy",
        "trie_path": prefix + "_trie.pkl",
        "manifest_path": prefix + "_manifest.json",
    }


def _candidate_cache_signature(*, config, mbeir_data_dir, genir_dir, output_cand_pool_name, cand_pool_data_path,
                               source_kind="auto", extracted_path=None):
    """Build a stable signature for reusable candidate semantic-ID cache."""
    resolved_pool = _resolve_mbeir_jsonl_path(mbeir_data_dir, cand_pool_data_path)
    try:
        candidate_dids = _candidate_dids_from_pool_jsonl(resolved_pool) if os.path.exists(resolved_pool) else set()
    except Exception:
        candidate_dids = set()
    quantizer_path = _resolve_under_root(genir_dir, _nested_get(config, "codebook_config.quantizer_path"))
    clip_sf_dir = _resolve_under_root(genir_dir, _nested_get(config, "model.pretrained_config.pretrained_dir"))
    clip_sf_name = _nested_get(config, "model.pretrained_config.pretrained_name")
    clip_sf_path = os.path.join(clip_sf_dir, clip_sf_name) if clip_sf_dir and clip_sf_name else None
    include_hash = bool(getattr(getattr(config, "data_config", object()), "cache_hash_files", False))
    return {
        "signature_version": 2,
        "role": "candidate_semantic_id_cache",
        "output_cand_pool_name": str(output_cand_pool_name),
        "selected_cand_pool_path": os.path.abspath(str(resolved_pool)),
        "selected_cand_pool_file": _file_signature(resolved_pool, include_hash=include_hash),
        "selected_cand_pool_num_dids": int(len(candidate_dids)),
        "source_kind": str(source_kind),
        "extracted_feature_file": _file_signature(extracted_path, include_hash=False) if extracted_path else {"exists": False, "path": None},
        "quantizer_file": _file_signature(quantizer_path, include_hash=False),
        "clip_sf_file": _file_signature(clip_sf_path, include_hash=False),
        "codebook_level": int(_nested_get(config, "codebook_config.codebook_level", -1)),
        "codebook_vocab": int(_nested_get(config, "codebook_config.codebook_vocab", -1)),
        "generator_type": str(_nested_get(config, "model.generator_type", "")),
        "hierarchy_block_sizes": list(_nested_get(config, "model.hierarchy_block_sizes", []) or []),
        "modality_index": bool(_nested_get(config, "model.modality_index", True)),
    }


def _json_equal(a, b):
    # Candidate semantic IDs are produced by the fixed RQ quantizer and do not
    # depend on how the query decoder groups positions into blocks.  Older
    # manifests included this query-only field, which caused needless recoding
    # when comparing sequential, fixed-suffix, and adaptive decoders.
    if isinstance(a, dict) and isinstance(b, dict):
        a = {k: v for k, v in a.items() if k != "hierarchy_block_sizes"}
        b = {k: v for k, v in b.items() if k != "hierarchy_block_sizes"}
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


def _save_candidate_cache_manifest(paths, signature):
    os.makedirs(paths["cand_save_dir"], exist_ok=True)
    manifest = {
        "manifest_version": 1,
        "created_at_unix": time.time(),
        "signature": signature,
        "files": {
            "codes": os.path.abspath(paths["code_path"]),
            "embeddings": os.path.abspath(paths["emb_path"]),
            "ids": os.path.abspath(paths["id_path"]),
            "trie": os.path.abspath(paths["trie_path"]),
        },
    }
    with open(paths["manifest_path"], "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)




def _candidate_cache_payload_matches(paths, signature, *, require_embeddings=True):
    """Validate cached candidate arrays against the selected candidate pool.

    A manifest can be stale after rebuilding a local pool or switching between
    extracted/raw sources.  Always verify that code/id/embedding row counts match
    the currently selected candidate-pool DID count before reusing the cache.
    """
    expected = int(signature.get("selected_cand_pool_num_dids", -1))
    if expected < 0:
        expected = None
    try:
        ids = np.load(paths["id_path"], allow_pickle=True)
        num_ids = int(np.asarray(ids, dtype=object).reshape(-1).shape[0])
    except Exception as exc:
        return False, f"could not read candidate ids: {exc}"
    try:
        codes = np.load(paths["code_path"], mmap_mode="r", allow_pickle=False)
        num_code_rows = int(codes.shape[0]) if codes.ndim > 0 else 1
    except Exception as exc:
        return False, f"could not read candidate codes: {exc}"
    if expected is not None and num_ids != expected:
        return False, f"cached id rows {num_ids} != selected pool did count {expected}"
    if expected is not None and num_code_rows != expected:
        return False, f"cached code rows {num_code_rows} != selected pool did count {expected}"
    if num_code_rows != num_ids:
        return False, f"cached code rows {num_code_rows} != id rows {num_ids}"
    if require_embeddings:
        try:
            embs = np.load(paths["emb_path"], mmap_mode="r", allow_pickle=False)
            num_emb_rows = int(embs.shape[0]) if embs.ndim > 0 else 1
        except Exception as exc:
            return False, f"could not read candidate embeddings: {exc}"
        if num_emb_rows != num_ids:
            return False, f"cached embedding rows {num_emb_rows} != id rows {num_ids}"
    return True, "payload row counts match"


def _candidate_pool_expected_count(mbeir_data_dir, cand_pool_data_path):
    resolved_pool = _resolve_mbeir_jsonl_path(mbeir_data_dir, cand_pool_data_path)
    return int(len(_candidate_dids_from_pool_jsonl(resolved_pool))), resolved_pool

def _try_adopt_legacy_candidate_cache(paths, signature, *, require_embeddings=True):
    """Attach a manifest to an existing pre-v22 cache if it matches the selected pool size."""
    if not (os.path.exists(paths["code_path"]) and os.path.exists(paths["id_path"])):
        return False, "missing code/id files"
    if require_embeddings and not os.path.exists(paths["emb_path"]):
        return False, "missing embedding file"
    try:
        ids = np.load(paths["id_path"], allow_pickle=True)
        num_ids = int(np.asarray(ids, dtype=object).reshape(-1).shape[0])
    except Exception as exc:
        return False, f"could not read ids: {exc}"
    expected = int(signature.get("selected_cand_pool_num_dids", -1))
    if expected >= 0 and num_ids != expected:
        return False, f"legacy id count {num_ids} != selected pool did count {expected}"
    try:
        codes = np.load(paths["code_path"], mmap_mode="r", allow_pickle=False)
        if codes.shape[0] != num_ids:
            return False, f"code rows {codes.shape[0]} != id count {num_ids}"
    except Exception as exc:
        return False, f"could not read codes: {exc}"
    _save_candidate_cache_manifest(paths, signature)
    return True, "adopted legacy cache"


def _candidate_cache_is_valid(paths, signature, *, require_embeddings=True, allow_legacy_adopt=True):
    required = [paths["code_path"], paths["id_path"]]
    if require_embeddings:
        required.append(paths["emb_path"])
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return False, "missing files: " + ", ".join(missing)

    if os.path.exists(paths["manifest_path"]):
        try:
            with open(paths["manifest_path"], "r") as f:
                manifest = json.load(f)
            if _json_equal(manifest.get("signature"), signature):
                payload_ok, payload_reason = _candidate_cache_payload_matches(
                    paths, signature, require_embeddings=require_embeddings
                )
                if payload_ok:
                    return True, "manifest match; " + payload_reason
                return False, "manifest match but payload mismatch: " + payload_reason
            return False, "manifest signature mismatch"
        except Exception as exc:
            return False, f"could not read manifest: {exc}"

    if allow_legacy_adopt:
        ok, reason = _try_adopt_legacy_candidate_cache(paths, signature, require_embeddings=require_embeddings)
        if ok:
            return True, reason
        return False, "legacy cache not adoptable: " + reason
    return False, "manifest missing"



def _cand_pool_name_from_jsonl_path(cand_pool_jsonl_path):
    name = os.path.basename(str(cand_pool_jsonl_path))
    if name.startswith("mbeir_"):
        name = name[len("mbeir_"):]
    if name.endswith("_cand_pool.jsonl"):
        name = name[: -len("_cand_pool.jsonl")]
    elif name.endswith(".jsonl"):
        name = name[:-len(".jsonl")]
    return name


def _ensure_candidate_codes_for_pool(
        *,
        model,
        config,
        mbeir_data_dir,
        genir_dir,
        gen_code_dir_name,
        expt_dir_name,
        output_cand_pool_name,
        cand_pool_data_path,
        img_preprocess_fn,
        clip_tokenizer,
        image_size,
        force=False,
):
    """Generate candidate codes for a selected pool when eval auto-aligns pools.

    The files are intentionally saved under ``output_cand_pool_name`` so the
    rest of the legacy retrieval pipeline can keep using configured names while
    the actual candidate-pool jsonl is corrected by overlap diagnostics.
    """
    paths = _candidate_output_paths(genir_dir, gen_code_dir_name, expt_dir_name, output_cand_pool_name)
    cand_save_dir = paths["cand_save_dir"]
    code_path = paths["code_path"]
    emb_path = paths["emb_path"]
    id_path = paths["id_path"]

    selected_name = _cand_pool_name_from_jsonl_path(cand_pool_data_path)
    extracted_path = _expected_extracted_cand_path(
        genir_dir=genir_dir,
        data_config=config.data_config,
        split_name="cand_pool",
        cand_pool_name=selected_name,
    )
    will_use_extracted = bool(getattr(config.data_config, "is_extracted", False) and os.path.exists(extracted_path))
    signature = _candidate_cache_signature(
        config=config,
        mbeir_data_dir=mbeir_data_dir,
        genir_dir=genir_dir,
        output_cand_pool_name=output_cand_pool_name,
        cand_pool_data_path=cand_pool_data_path,
        source_kind="extracted" if will_use_extracted else "raw",
        extracted_path=extracted_path if will_use_extracted else None,
    )
    cache_ok, cache_reason = _candidate_cache_is_valid(
        paths,
        signature,
        require_embeddings=bool(config.retrieval_config.rerank),
        allow_legacy_adopt=True,
    )
    if (not force) and cache_ok:
        if dist_utils.is_main_process():
            print(
                f"[HDGR eval] Reusing cached candidate semantic-ID codes for {output_cand_pool_name} "
                f"({cache_reason})."
            )
        if dist.is_initialized():
            dist.barrier()
        return

    if dist_utils.is_main_process():
        action = "Regenerating" if force else "Generating"
        print(
            f"[HDGR eval] {action} candidate semantic-ID codes for aligned pool: {cand_pool_data_path}\n"
            f"[HDGR eval] Saving under configured pool name: {output_cand_pool_name}\n"
            f"[HDGR eval] Candidate cache miss reason: {cache_reason}"
        )

    actual_is_extracted = False
    expected_pool_count, resolved_pool_path = _candidate_pool_expected_count(mbeir_data_dir, cand_pool_data_path)
    if getattr(config.data_config, "is_extracted", False) and os.path.exists(extracted_path):
        dataset = MBEIRDictCandDataset(
            mbeir_data_dir=mbeir_data_dir,
            cand_pool_path=cand_pool_data_path,
            pool_dict_dir=extracted_path,
            print_config=dist_utils.is_main_process(),
        )
        extracted_count = len(dataset)
        if expected_pool_count > 0 and extracted_count != expected_pool_count:
            if dist_utils.is_main_process():
                print(
                    f"[HDGR eval] Extracted candidate cache row-count mismatch for {output_cand_pool_name}: "
                    f"usable_extracted_rows={extracted_count}, selected_pool_dids={expected_pool_count}, "
                    f"extracted_path={extracted_path}. Falling back to raw candidate-pool encoding."
                )
            dataset = MBEIRCandidatePoolDataset(
                mbeir_data_dir=mbeir_data_dir,
                cand_pool_data_path=cand_pool_data_path,
                img_preprocess_fn=img_preprocess_fn,
                print_config=dist_utils.is_main_process(),
            )
            collator = MBEIRCandidatePoolCollator(tokenizer=clip_tokenizer, image_size=image_size)
            actual_is_extracted = False
        else:
            if dist_utils.is_main_process():
                print(f"[HDGR eval] Using extracted candidate embeddings: {extracted_path}")
            collator = None
            actual_is_extracted = True
    else:
        if dist_utils.is_main_process():
            if getattr(config.data_config, "is_extracted", False):
                print(
                    f"[HDGR eval] Extracted embeddings not found for selected pool at {extracted_path}; "
                    "falling back to raw candidate-pool encoding."
                )
            print(f"[HDGR eval] Raw candidate-pool jsonl: {cand_pool_data_path}")
        dataset = MBEIRCandidatePoolDataset(
            mbeir_data_dir=mbeir_data_dir,
            cand_pool_data_path=cand_pool_data_path,
            img_preprocess_fn=img_preprocess_fn,
            print_config=dist_utils.is_main_process(),
        )
        collator = MBEIRCandidatePoolCollator(tokenizer=clip_tokenizer, image_size=image_size)
        actual_is_extracted = False

    sampler = ContiguousDistributedSampler(
        dataset,
        num_replicas=dist_utils.get_world_size(),
        rank=dist_utils.get_rank(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.dataloader_config.batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )
    if dist.is_initialized():
        dist.barrier()
    codes, embeddings, id_list = generate_codes_for_dataset(
        model=model,
        data_loader=data_loader,
        device=config.dist_config.gpu_id,
        use_fp16=config.retrieval_config.use_fp16,
        is_extracted=actual_is_extracted,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(cand_save_dir, exist_ok=True)
        np.save(code_path, codes)
        if config.retrieval_config.rerank:
            np.save(emb_path, embeddings)
        np.save(id_path, id_list)
        _save_candidate_cache_manifest(paths, signature)
        print(f"[HDGR eval] Saved aligned candidate codes to {code_path}")
        if config.retrieval_config.rerank:
            print(f"[HDGR eval] Saved aligned candidate embeddings to {emb_path}")
        print(f"[HDGR eval] Saved aligned candidate ids to {id_path}")
        print(f"[HDGR eval] Saved candidate cache manifest to {paths['manifest_path']}")
    if dist.is_initialized():
        dist.barrier()

# 确保这里没有反斜杠，直接换行
@torch.no_grad()


def generate_codes_for_dataset(
        model,
        data_loader,
        device,
        use_fp16=True,
        is_quantizer=True,
        num_beams=10,
        cand_codes=None,
        is_extracted=False,
        use_embedding=True,
        trie_save_path=None
):
    # Candidate-index construction and deserialization are offline setup, not
    # query execution.  Historically the first model call initialized the
    # Trie after ``generation_started_at`` and therefore charged a potentially
    # multi-minute UNION-index load to the first query batch.  Preload the
    # already prepared index before resetting CUDA statistics and starting the
    # online generation timer.  ``encode_mbeir_batch(init_dataset=True)`` keeps
    # its compatibility call, which becomes a constant-time no-op because the
    # same candidate signature is already resident.
    if not is_quantizer and cand_codes is not None:
        index_setup_started_at = time.perf_counter()
        base_model = model.module if hasattr(model, "module") else model
        if not hasattr(base_model, "distribute_trie"):
            raise AttributeError(
                "Generative retriever does not expose distribute_trie for "
                "candidate-index initialization"
            )
        base_model.distribute_trie(cand_codes, trie_save_path)
        if dist.is_initialized():
            dist.barrier()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "Candidate Index Setup: "
                f"seconds={time.perf_counter() - index_setup_started_at:.6f} "
                "excluded_from_generation=true"
            )

    measure_cuda = (not is_quantizer) and torch.cuda.is_available()
    profile_latency = (
        measure_cuda
        and bool(int(os.environ.get("STRUCTNAR_PROFILE_LATENCY", "0")))
    )
    batch_profile_events = []
    if measure_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    generation_started_at = time.perf_counter()
    codes_tensor = []
    id_list_local = []
    beam_scores_tensor = []
    has_beam_scores = None
    if use_embedding:
        encode_tensor = []

    total_cores = os.cpu_count() or 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = max(1, total_cores // world_size)
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")

    init_dataset = True
    max_eval_batches = max(
        0, int(os.environ.get("STRUCTNAR_MAX_EVAL_BATCHES", "0"))
    )
    for batch_index, batch in enumerate(data_loader):
        if max_eval_batches and batch_index >= max_eval_batches:
            print(
                "[StructNAR smoke] stopping generation after "
                f"{max_eval_batches} batches"
            )
            break
        batch_size_hint = None
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    batch_size_hint = int(value.size(0))
                    break
        batch_start_event = batch_end_event = None
        if profile_latency:
            batch_start_event = torch.cuda.Event(enable_timing=True)
            batch_end_event = torch.cuda.Event(enable_timing=True)
            batch_start_event.record()
        if not is_extracted:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                    for k, v in value.items():
                        batch[key][k] = v.to(device)

        with autocast(enabled=use_fp16):
            if is_quantizer:
                base_model = model.module if hasattr(model, "module") else model
                codes_batched, encode_batched, ids_list_batched = base_model.quantizer(
                    batch,
                    evaluation=True,
                    encode_mbeir_batch=(not is_extracted),
                    code_output=True,
                    encode_output=True
                )
                beam_scores_batched = None
            else:
                model_output = model(
                    batch,
                    cand_codes=cand_codes,
                    num_beams=num_beams,
                    encode_mbeir_batch=True,
                    init_dataset=init_dataset,
                    trie_save_path=trie_save_path
                )
                if not isinstance(model_output, (tuple, list)) or len(model_output) not in {3, 4}:
                    raise RuntimeError(
                        "Generative retriever encode_mbeir_batch must return "
                        "(codes, embeddings, ids) or (codes, embeddings, ids, beam_scores)."
                    )
                codes_batched, encode_batched, ids_list_batched = model_output[:3]
                beam_scores_batched = model_output[3] if len(model_output) == 4 else None
        if profile_latency:
            batch_end_event.record()

        ids_current = _ids_batch_to_list(ids_list_batched)
        if profile_latency:
            batch_profile_events.append(
                (batch_start_event, batch_end_event, len(ids_current) or batch_size_hint or 1)
            )
        if is_quantizer:
            expected_code_width = codes_tensor[0].size(1) if codes_tensor else None
            codes_batched = _code_batch_to_tensor(
                codes_batched,
                name="codes_batched",
                id_count=len(ids_current) if ids_current else None,
                expected_width=expected_code_width,
                preserve_beams=False,
                device=device,
            )
        else:
            expected_code_width = codes_tensor[0].size(-1) if codes_tensor else None
            codes_batched = _code_batch_to_tensor(
                codes_batched,
                name="codes_batched",
                id_count=len(ids_current) if ids_current else None,
                expected_width=expected_code_width,
                preserve_beams=True,
                device=device,
            )
        codes_tensor.append(codes_batched)
        id_list_local.extend(ids_current)

        current_has_scores = beam_scores_batched is not None
        if has_beam_scores is None:
            has_beam_scores = current_has_scores
        elif has_beam_scores != current_has_scores:
            raise RuntimeError("Beam-score availability changed between batches; model output must be consistent.")
        if current_has_scores:
            beam_scores_batched = _beam_score_batch_to_tensor(
                beam_scores_batched,
                name="beam_scores_batched",
                id_count=len(ids_current) if ids_current else None,
                expected_beams=codes_batched.size(1) if codes_batched.dim() == 3 else 1,
                device=device,
            )
            beam_scores_tensor.append(beam_scores_batched)

        if use_embedding:
            expected_embed_width = encode_tensor[0].size(1) if encode_tensor else None
            encode_batched = _tensor_batch_to_2d(
                encode_batched,
                name="encode_batched",
                id_count=len(ids_current) if ids_current else None,
                expected_width=expected_embed_width,
                device=device,
            )
            encode_tensor.append(encode_batched.half())

        if init_dataset:
            init_dataset = False

    if measure_cuda:
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started_at
    if not dist.is_initialized() or dist.get_rank() == 0:
        examples = len(id_list_local)
        throughput = examples / max(generation_seconds, 1.0e-9)
        print(
            "Generation Metrics: "
            f"seconds={generation_seconds:.6f} examples={examples} "
            f"examples_per_second={throughput:.6f}"
        )
        base_model = model.module if hasattr(model, "module") else model
        planner_owner = base_model
        if not hasattr(planner_owner, "_tcis_frontier_stop_depths"):
            nested_generator = getattr(base_model, "id_generator", None)
            if nested_generator is not None and hasattr(
                nested_generator, "_tcis_frontier_stop_depths"
            ):
                planner_owner = nested_generator
        stop_depths = getattr(planner_owner, "_tcis_frontier_stop_depths", [])
        if stop_depths:
            stop_array = np.asarray(stop_depths, dtype=np.int64)
            unique_depths, depth_counts = np.unique(stop_array, return_counts=True)
            histogram = ",".join(
                f"s{int(depth)}:{int(count)}"
                for depth, count in zip(unique_depths, depth_counts)
            )
            print(
                "Frontier Planner Summary: "
                f"queries={stop_array.size} mean_s={stop_array.mean():.6f} "
                f"hist={histogram}"
            )
            delattr(planner_owner, "_tcis_frontier_stop_depths")
        if measure_cuda:
            print(
                "Generation CUDA Memory: "
                f"peak_allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.3f} "
                f"peak_reserved_mib={torch.cuda.max_memory_reserved(device) / 2**20:.3f}"
            )
        if profile_latency and batch_profile_events:
            warmup_batches = max(
                0, int(os.environ.get("STRUCTNAR_LATENCY_WARMUP_BATCHES", "0"))
            )
            measured_events = batch_profile_events[warmup_batches:]
            if not measured_events:
                measured_events = batch_profile_events
            batch_ms = np.asarray(
                [start.elapsed_time(end) for start, end, _ in measured_events],
                dtype=np.float64,
            )
            per_query_ms = np.asarray(
                [
                    start.elapsed_time(end) / max(int(batch_size), 1)
                    for start, end, batch_size in measured_events
                ],
                dtype=np.float64,
            )
            print(
                "Generation Latency Distribution: "
                f"batches={batch_ms.size} "
                f"warmup_batches_excluded={min(warmup_batches, len(batch_profile_events))} "
                f"batch_p50_ms={np.percentile(batch_ms, 50):.6f} "
                f"batch_p95_ms={np.percentile(batch_ms, 95):.6f} "
                f"batch_p99_ms={np.percentile(batch_ms, 99):.6f} "
                f"normalized_query_p50_ms={np.percentile(per_query_ms, 50):.6f} "
                f"normalized_query_p95_ms={np.percentile(per_query_ms, 95):.6f} "
                f"normalized_query_p99_ms={np.percentile(per_query_ms, 99):.6f}"
            )
            base_model = model.module if hasattr(model, "module") else model
            full_stage_profile_path = os.environ.get(
                "STRUCTNAR_FULL_STAGE_PROFILE_PATH", ""
            ).strip()
            if full_stage_profile_path:
                all_generation_ms = []
                for start, end, batch_size in batch_profile_events:
                    normalized_ms = start.elapsed_time(end) / max(int(batch_size), 1)
                    all_generation_ms.extend([normalized_ms] * int(batch_size))

                def _event_segments(event_rows, segment_count):
                    outputs = [[] for _ in range(segment_count)]
                    for events, batch_size in event_rows:
                        for segment in range(segment_count):
                            normalized_ms = (
                                events[segment].elapsed_time(events[segment + 1])
                                / max(int(batch_size), 1)
                            )
                            outputs[segment].extend([normalized_ms] * int(batch_size))
                    return outputs

                query_input_rows = getattr(
                    base_model, "_structnar_query_input_events", []
                )
                query_input_segments = _event_segments(query_input_rows, 1)
                inference_rows = getattr(
                    base_model, "_structnar_full_stage_events", []
                )
                inference_segments = _event_segments(inference_rows, 4)
                profile_size = min(
                    len(id_list_local),
                    len(all_generation_ms),
                    len(query_input_segments[0]) if query_input_segments else 0,
                    len(inference_segments[0]) if inference_segments else 0,
                )
                if profile_size != len(id_list_local):
                    raise RuntimeError(
                        "Full-stage profiler lost query rows: "
                        f"ids={len(id_list_local)} generation={len(all_generation_ms)} "
                        f"query_input={len(query_input_segments[0]) if query_input_segments else 0} "
                        f"inference={len(inference_segments[0]) if inference_segments else 0}"
                    )
                warmup_queries = sum(
                    int(batch_size)
                    for _, _, batch_size in batch_profile_events[:warmup_batches]
                )
                os.makedirs(os.path.dirname(full_stage_profile_path), exist_ok=True)
                np.savez_compressed(
                    full_stage_profile_path,
                    query_id=np.asarray(id_list_local[:profile_size]),
                    warmup_queries=np.asarray([warmup_queries], dtype=np.int64),
                    generation_ms=np.asarray(all_generation_ms[:profile_size], dtype=np.float64),
                    query_input_ms=np.asarray(query_input_segments[0][:profile_size], dtype=np.float64),
                    rq_pipeline_ms=np.asarray(inference_segments[0][:profile_size], dtype=np.float64),
                    query_projector_ms=np.asarray(inference_segments[1][:profile_size], dtype=np.float64),
                    semantic_search_ms=np.asarray(inference_segments[2][:profile_size], dtype=np.float64),
                    output_finalize_ms=np.asarray(inference_segments[3][:profile_size], dtype=np.float64),
                )
                measured_slice = slice(warmup_queries, profile_size)
                print(
                    "Full Generation Stage Breakdown: "
                    f"queries={profile_size - warmup_queries} "
                    f"query_input_mean_ms={np.mean(query_input_segments[0][measured_slice]):.6f} "
                    f"rq_pipeline_mean_ms={np.mean(inference_segments[0][measured_slice]):.6f} "
                    f"query_projector_mean_ms={np.mean(inference_segments[1][measured_slice]):.6f} "
                    f"semantic_search_mean_ms={np.mean(inference_segments[2][measured_slice]):.6f} "
                    f"output_finalize_mean_ms={np.mean(inference_segments[3][measured_slice]):.6f} "
                    f"profile={full_stage_profile_path}"
                )
                delattr(base_model, "_structnar_query_input_events")
                delattr(base_model, "_structnar_full_stage_events")
            tcis_owner = getattr(base_model, "id_generator", None) or base_model
            event_owner = (
                tcis_owner
                if hasattr(tcis_owner, "_tcis_profile_event_rows")
                else base_model
            )
            event_rows = getattr(event_owner, "_tcis_profile_event_rows", [])
            if event_rows:
                model_ms = np.asarray(
                    [row[4].elapsed_time(row[5]) for row in event_rows],
                    dtype=np.float64,
                )
                expand_ms = np.asarray(
                    [row[5].elapsed_time(row[6]) for row in event_rows],
                    dtype=np.float64,
                )
                print(
                    "Generation Stage Breakdown: "
                    f"blocks={len(event_rows)} "
                    f"model_total_seconds={model_ms.sum() / 1000.0:.6f} "
                    f"constraint_expand_total_seconds={expand_ms.sum() / 1000.0:.6f} "
                    f"model_share={model_ms.sum() / max(model_ms.sum() + expand_ms.sum(), 1e-9):.6f} "
                    f"constraint_expand_share={expand_ms.sum() / max(model_ms.sum() + expand_ms.sum(), 1e-9):.6f}"
                )
                delattr(event_owner, "_tcis_profile_event_rows")
            # TCIS accounting is produced by the identifier generator.  Some
            # model wrappers expose it as ``id_generator`` rather than storing
            # the rows on the outer retrieval module, so use the same resolved
            # owner as the CUDA event profile above.
            budget_owner = (
                tcis_owner
                if hasattr(tcis_owner, "_tcis_query_budget_rows")
                else base_model
            )
            query_budget_rows = getattr(budget_owner, "_tcis_query_budget_rows", [])
            if query_budget_rows:
                parents = np.asarray([row[0] for row in query_budget_rows], dtype=np.float64)
                candidates = np.asarray([row[1] for row in query_budget_rows], dtype=np.float64)
                enumeration = np.asarray([row[2] for row in query_budget_rows], dtype=np.float64)
                cache = getattr(budget_owner, "block_transition_cache", {})
                cache_bytes = sum(
                    value.numel() * value.element_size()
                    for value in cache.values()
                    if isinstance(value, torch.Tensor)
                )
                print(
                    "TCIS Query Enumeration: "
                    f"queries={candidates.size} "
                    f"parents_mean={parents.mean():.6f} "
                    f"candidates_mean={candidates.mean():.6f} "
                    f"candidates_p50={np.percentile(candidates, 50):.6f} "
                    f"candidates_p90={np.percentile(candidates, 90):.6f} "
                    f"candidates_p95={np.percentile(candidates, 95):.6f} "
                    f"candidates_p99={np.percentile(candidates, 99):.6f} "
                    f"candidates_max={candidates.max():.0f} "
                    f"enumeration_cpu_seconds={enumeration.sum():.6f} "
                    f"cache_entries={len(cache)} cache_mib={cache_bytes / 2**20:.6f}"
                )
                query_profile_path = os.environ.get(
                    "STRUCTNAR_QUERY_PROFILE_PATH", ""
                ).strip()
                if query_profile_path:
                    frontier_batches = getattr(
                        budget_owner, "_tcis_frontier_prefix_batches", []
                    )
                    frontier_prefixes = (
                        np.concatenate(frontier_batches, axis=0)
                        if frontier_batches
                        else np.empty((0, 0, 0), dtype=np.int64)
                    )
                    expanded_latency_ms = []
                    for start_event, end_event, batch_size in batch_profile_events:
                        per_item_ms = (
                            start_event.elapsed_time(end_event)
                            / max(int(batch_size), 1)
                        )
                        expanded_latency_ms.extend(
                            [per_item_ms] * int(batch_size)
                        )
                    profile_size = min(
                        len(id_list_local),
                        len(query_budget_rows),
                        len(expanded_latency_ms),
                    )
                    os.makedirs(os.path.dirname(query_profile_path), exist_ok=True)
                    np.savez_compressed(
                        query_profile_path,
                        query_id=np.asarray(id_list_local[:profile_size]),
                        live_prefixes=parents[:profile_size],
                        exposed_leaves=candidates[:profile_size],
                        enumeration_cpu_seconds=enumeration[:profile_size],
                        frontier_prefixes=frontier_prefixes[:profile_size],
                        amortized_generation_latency_ms=np.asarray(
                            expanded_latency_ms[:profile_size], dtype=np.float64
                        ),
                    )
                    print(
                        "TCIS Query Profile: "
                        f"path={query_profile_path} queries={profile_size}"
                    )
                    if hasattr(budget_owner, "_tcis_frontier_prefix_batches"):
                        delattr(budget_owner, "_tcis_frontier_prefix_batches")
                delattr(budget_owner, "_tcis_query_budget_rows")
            component_owner = (
                tcis_owner
                if hasattr(tcis_owner, "_tcis_component_event_rows")
                else base_model
            )
            component_rows = getattr(component_owner, "_tcis_component_event_rows", [])
            if component_rows:
                gather_ms = np.asarray(
                    [row[0].elapsed_time(row[1]) for row in component_rows], dtype=np.float64
                )
                rqc_ms = np.asarray(
                    [row[1].elapsed_time(row[2]) for row in component_rows], dtype=np.float64
                )
                topk_ms = np.asarray(
                    [row[2].elapsed_time(row[3]) for row in component_rows], dtype=np.float64
                )
                print(
                    "TCIS Component Breakdown: "
                    f"queries={len(component_rows)} "
                    f"gather_total_seconds={gather_ms.sum() / 1000.0:.6f} "
                    f"rqc_total_seconds={rqc_ms.sum() / 1000.0:.6f} "
                    f"topk_total_seconds={topk_ms.sum() / 1000.0:.6f}"
                )
                delattr(component_owner, "_tcis_component_event_rows")

    codes_tensor = torch.cat(codes_tensor, dim=0)
    if len(id_list_local) != codes_tensor.size(0):
        raise RuntimeError(
            f"ID/code batch size mismatch: got {len(id_list_local)} ids for {codes_tensor.size(0)} codes. "
            "This usually indicates an unexpected dataset collate format."
        )
    if use_embedding:
        encode_tensor = torch.cat(encode_tensor, dim=0)
    if has_beam_scores:
        beam_scores_tensor = torch.cat(beam_scores_tensor, dim=0)
        if beam_scores_tensor.size(0) != codes_tensor.size(0):
            raise RuntimeError(
                f"Beam-score/code batch mismatch: scores={tuple(beam_scores_tensor.shape)}, "
                f"codes={tuple(codes_tensor.shape)}"
            )

    codes_list = None
    id_list = None
    if use_embedding:
        encodes_list = None
    beam_scores_list = None

    if dist.is_initialized():
        size_tensor = torch.tensor([codes_tensor.size(0)], dtype=torch.long, device=device)

        if dist.get_rank() == 0:
            gathered_codes = [torch.empty_like(codes_tensor) for _ in range(dist.get_world_size())]
            gathered_ids = [list() for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            if use_embedding:
                gathered_encodes = [torch.empty_like(encode_tensor) for _ in range(dist.get_world_size())]
            if has_beam_scores:
                gathered_beam_scores = [torch.empty_like(beam_scores_tensor) for _ in range(dist.get_world_size())]
        else:
            gathered_codes = None
            gathered_encodes = None
            gathered_beam_scores = None
            gathered_ids = None
            sizes = None

        dist.gather(codes_tensor, gather_list=gathered_codes, dst=0)
        if use_embedding:
            dist.gather(encode_tensor, gather_list=gathered_encodes, dst=0)
        if has_beam_scores:
            dist.gather(beam_scores_tensor, gather_list=gathered_beam_scores, dst=0)
        dist.gather_object(id_list_local, object_gather_list=gathered_ids, dst=0)
        dist.gather(size_tensor, gather_list=sizes, dst=0)

        if dist.get_rank() == 0:
            torch.set_num_threads(total_cores)
            for i in range(dist.get_world_size()):
                gathered_codes[i] = gathered_codes[i][: sizes[i][0]]
                if use_embedding:
                    gathered_encodes[i] = gathered_encodes[i][: sizes[i][0]]
                if has_beam_scores:
                    gathered_beam_scores[i] = gathered_beam_scores[i][: sizes[i][0]]

            codes_list = torch.cat(gathered_codes, dim=0).cpu().numpy()
            if use_embedding:
                encodes_list = torch.cat(gathered_encodes, dim=0).cpu().numpy()
            if has_beam_scores:
                beam_scores_list = torch.cat(gathered_beam_scores, dim=0).float().cpu().numpy()
            id_list = [item for sublist in gathered_ids for item in sublist]

            assert len(id_list) == codes_list.shape[0], f"Got {len(id_list)} ids for {codes_list.shape[0]} codes"
            assert len(set(map(str, id_list))) == len(id_list), "Hashed IDs should be unique"
            print(f"Log: Finished processing embeddings and ids on rank 0.")
            print(f"Log: Converted codes_list to cpu numpy array of type {codes_list.dtype}.")
            torch.set_num_threads(initial_threads_per_process)

    else:
        codes_list = codes_tensor.detach().cpu().numpy()
        id_list = id_list_local
        if use_embedding:
            encodes_list = encode_tensor.detach().cpu().numpy()
        if has_beam_scores:
            beam_scores_list = beam_scores_tensor.detach().float().cpu().numpy()

    if use_embedding and has_beam_scores:
        return codes_list, encodes_list, id_list, beam_scores_list
    if use_embedding:
        return codes_list, encodes_list, id_list
    if has_beam_scores:
        return codes_list, id_list, beam_scores_list
    else:
        return codes_list, id_list


@torch.no_grad()
def generate_noembed_codes_for_dataset(
        model,
        data_loader,
        device,
        use_fp16=True,
        is_quantizer=True,
        num_beams=10,
        cand_codes=None,
        is_extracted=False,
        trie_save_path=None
):
    """Generate semantic-ID codes without persisting embeddings.

    When the model is configured with save_beam_scores=true, encode_mbeir_batch
    may return a fourth tensor of beam log-scores.  Preserve that tensor so
    no-rerank runs can still save generator diagnostics while avoiding embedding
    caches.
    """
    return generate_codes_for_dataset(
        model=model,
        data_loader=data_loader,
        device=device,
        use_fp16=use_fp16,
        is_quantizer=is_quantizer,
        num_beams=num_beams,
        cand_codes=cand_codes,
        is_extracted=is_extracted,
        use_embedding=False,
        trie_save_path=trie_save_path,
    )


@torch.no_grad()
def generate_codes_for_config(model, img_preprocess_fn, clip_tokenizer, seq2seq_tokenizer, config):
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    gen_code_dir_name = retrieval_config.gen_code_dir_name
    expt_dir_name = config.experiment.path_suffix

    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    union_cand_pool_dir = data_config.union_cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))
    is_extracted = data_config.is_extracted

    splits = []

    gen_code_cand_pool_config = retrieval_config.cand_pools_config
    if gen_code_cand_pool_config and gen_code_cand_pool_config.enable_gen_code:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = gen_code_cand_pool_config.cand_pools_name_to_gen_code
        splits.append(
            (
                split_name,
                split_dir_name,
                [None] * len(cand_pool_name_list),
                cand_pool_name_list,
            )
        )

    dataset_types = ["train", "val", "test"]
    all_cand_pool_name_list = [
        "visualnews_task0",
        "mscoco_task0_test",
        "fashion200k_task0",
        "webqa_task1",
        "edis_task2",
        "webqa_task2",
        "visualnews_task3",
        "mscoco_task3_test",
        "fashion200k_task3",
        "nights_task4",
        "oven_task6",
        "infoseek_task6",
        "fashioniq_task7",
        "cirr_task7",
        "oven_task8",
        "infoseek_task8"
    ]
    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        gen_code_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if gen_code_dataset_config and gen_code_dataset_config.enable_gen_code:
            dataset_name_list = getattr(gen_code_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(gen_code_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."

    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(
                    f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to gen_code: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to retrieval: {dataset_name_list}")
            print("-" * 30)

    has_query_splits_to_generate = any(item[0] != "cand_pool" for item in splits)

    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            actual_is_extracted = False
            if split_name == "cand_pool":
                cand_pool_name = _normalize_cand_pool_name(cand_pool_name)
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                if bool(getattr(data_config, "auto_align_candidate_pool", True)) and has_query_splits_to_generate:
                    if dist_utils.is_main_process():
                        print(
                            f"[HDGR eval] Deferring candidate-pool code generation for {cand_pool_name} "
                            "until query/candidate overlap alignment is known."
                        )
                    if dist.is_initialized():
                        dist.barrier()
                    continue

                preflight_extracted_path = _expected_extracted_cand_path(
                    genir_dir=genir_dir,
                    data_config=data_config,
                    split_name=split_name,
                    cand_pool_name=cand_pool_name,
                )
                preflight_will_use_extracted = bool(is_extracted and os.path.exists(preflight_extracted_path))
                preflight_paths = _candidate_output_paths(genir_dir, gen_code_dir_name, expt_dir_name, cand_pool_name)
                preflight_signature = _candidate_cache_signature(
                    config=config,
                    mbeir_data_dir=mbeir_data_dir,
                    genir_dir=genir_dir,
                    output_cand_pool_name=cand_pool_name,
                    cand_pool_data_path=cand_pool_data_path,
                    source_kind="extracted" if preflight_will_use_extracted else "raw",
                    extracted_path=preflight_extracted_path if preflight_will_use_extracted else None,
                )
                cache_ok, cache_reason = _candidate_cache_is_valid(
                    preflight_paths,
                    preflight_signature,
                    require_embeddings=bool(retrieval_config.rerank),
                    allow_legacy_adopt=True,
                )
                if cache_ok:
                    if dist_utils.is_main_process():
                        print(
                            f"[HDGR eval] Reusing cached candidate semantic-ID codes for {cand_pool_name} "
                            f"({cache_reason})."
                        )
                    if dist.is_initialized():
                        dist.barrier()
                    continue
                elif dist_utils.is_main_process():
                    print(f"[HDGR eval] Candidate cache miss for {cand_pool_name}: {cache_reason}")

                if is_extracted:
                    cand_pool_extracted_path = _expected_extracted_cand_path(
                        genir_dir=genir_dir,
                        data_config=data_config,
                        split_name=split_name,
                        cand_pool_name=cand_pool_name,
                    )
                    if os.path.exists(cand_pool_extracted_path):
                        print_config = False
                        if dist_utils.is_main_process():
                            print(f"Generate Code Log: Generating codes from {cand_pool_extracted_path}...")
                            print_config = True
                        dataset = MBEIRDictCandDataset(
                            mbeir_data_dir=config.mbeir_data_dir,
                            cand_pool_path=cand_pool_data_path,
                            pool_dict_dir=cand_pool_extracted_path,
                            print_config=print_config,
                        )
                        expected_pool_count, _ = _candidate_pool_expected_count(mbeir_data_dir, cand_pool_data_path)
                        extracted_count = len(dataset)
                        if expected_pool_count > 0 and extracted_count != expected_pool_count:
                            if dist_utils.is_main_process():
                                print(
                                    f"Warning: extracted candidate embedding count mismatch for {cand_pool_name}: "
                                    f"usable_extracted_rows={extracted_count}, selected_pool_dids={expected_pool_count}, "
                                    f"extracted_path={cand_pool_extracted_path}. Falling back to raw candidate-pool encoding."
                                )
                            print_config = dist_utils.is_main_process()
                            if dist_utils.is_main_process():
                                print(f"Generate Code Log: Generating codes for {cand_pool_data_path}...")
                            dataset = MBEIRCandidatePoolDataset(
                                mbeir_data_dir=mbeir_data_dir,
                                cand_pool_data_path=cand_pool_data_path,
                                img_preprocess_fn=img_preprocess_fn,
                                print_config=print_config,
                            )
                            collator = MBEIRCandidatePoolCollator(
                                tokenizer=clip_tokenizer,
                                image_size=image_size,
                            )
                            actual_is_extracted = False
                        else:
                            collator = None
                            actual_is_extracted = True
                    else:
                        msg = _missing_extracted_cand_message(
                            missing_path=cand_pool_extracted_path,
                            cand_pool_name=cand_pool_name,
                            cand_pool_data_path=os.path.join(mbeir_data_dir, cand_pool_data_path),
                        )
                        fallback_to_raw = bool(getattr(data_config, "auto_fallback_to_raw_if_missing_extracted", False))
                        if not fallback_to_raw:
                            raise FileNotFoundError(msg)
                        if dist_utils.is_main_process():
                            print("Warning: " + msg.replace("\n", "\nWarning: "))
                            print("Warning: Falling back to raw candidate-pool encoding. This is slower than pre-extracted eval.")
                        print_config = False
                        if dist_utils.is_main_process():
                            print(f"Generate Code Log: Generating codes for {cand_pool_data_path}...")
                            print_config = True
                        dataset = MBEIRCandidatePoolDataset(
                            mbeir_data_dir=mbeir_data_dir,
                            cand_pool_data_path=cand_pool_data_path,
                            img_preprocess_fn=img_preprocess_fn,
                            print_config=print_config,
                        )
                        collator = MBEIRCandidatePoolCollator(
                            tokenizer=clip_tokenizer,
                            image_size=image_size,
                        )
                        actual_is_extracted = False
                else:
                    print_config = False
                    if dist_utils.is_main_process():
                        print(f"Generate Code Log: Generating codes for {cand_pool_data_path}...")
                        print_config = True
                    dataset = MBEIRCandidatePoolDataset(
                        mbeir_data_dir=mbeir_data_dir,
                        cand_pool_data_path=cand_pool_data_path,
                        img_preprocess_fn=img_preprocess_fn,
                        print_config=print_config,
                    )
                    collator = MBEIRCandidatePoolCollator(
                        tokenizer=clip_tokenizer,
                        image_size=image_size,
                    )
                    actual_is_extracted = False

            else:
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                cand_pool_name = cand_pool_name.lower()
                if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                    cand_pool_name = cand_pool_name + '_test'
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                if cand_pool_name == "union":
                    cand_pool_data_path = os.path.join(union_cand_pool_dir,
                                                       f"mbeir_{cand_pool_name}_{split_name}_cand_pool.jsonl")
                else:
                    cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                configured_cand_pool_data_path = cand_pool_data_path
                auto_align_candidate_pool = bool(getattr(data_config, "auto_align_candidate_pool", True))
                cand_pool_data_path, align_stats = _auto_align_candidate_pool_path(
                    mbeir_data_dir,
                    query_data_path,
                    cand_pool_data_path,
                    enabled=auto_align_candidate_pool,
                )
                if dist_utils.is_main_process():
                    _print_candidate_alignment_stats(align_stats)
                _ensure_candidate_codes_for_pool(
                    model=model,
                    config=config,
                    mbeir_data_dir=mbeir_data_dir,
                    genir_dir=genir_dir,
                    gen_code_dir_name=gen_code_dir_name,
                    expt_dir_name=expt_dir_name,
                    output_cand_pool_name=cand_pool_name,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    clip_tokenizer=clip_tokenizer,
                    image_size=image_size,
                    force=False,
                )

                print_config = False
                if dist_utils.is_main_process():
                    print(f"Generate Code Log: Generating codes for {query_data_path} with {cand_pool_data_path}...")
                    print_config = True
                mode = Mode.EVAL
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=mode,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                    deterministic_eval_sampling=bool(
                        getattr(data_config, "deterministic_eval_sampling", False)
                    ),
                )
                collator = MBEIRMainCollator(
                    tokenizer=clip_tokenizer,
                    image_size=image_size,
                    mode=mode,
                    seq2seq_tokenizer=seq2seq_tokenizer,
                )

            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
                collate_fn=collator,
                drop_last=False,
            )

            if dist.is_initialized():
                dist.barrier()

            cand_trie_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool",
                                          f"mbeir_{cand_pool_name}_cand_pool_trie.pkl")

            if dist_utils.is_main_process():
                if split_name == "cand_pool":
                    print(f"Log: Data loader for {cand_pool_data_path} is set up.")
                    print(f"Log: Generating codes for {cand_pool_data_path}...")
                else:
                    print(
                        f"Log: Data loader for {query_data_path} with candidate pool {cand_pool_data_path} is set up."
                    )
                    print(f"Log: Generating codes for {query_data_path} ...")
                print(f"Inference with half precision: {config.retrieval_config.use_fp16}")

            beam_scores = None
            if split_name == "cand_pool":
                generation_output = generate_codes_for_dataset(
                    model=model,
                    data_loader=data_loader,
                    device=config.dist_config.gpu_id,
                    use_fp16=config.retrieval_config.use_fp16,
                    is_extracted=actual_is_extracted,
                )
                codes, embeddings, id_list = generation_output[:3]
            else:
                # load cand_codes per each dataset
                cand_code_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool",
                                              f"mbeir_{cand_pool_name}_cand_pool_codes.npy")
                if os.path.exists(cand_code_path):
                    cand_codes = np.load(cand_code_path)
                    print(f"Log: Loaded candidate pool codes from {cand_code_path}.")
                else:
                    print(f"Warning: Candidate pool codes not found at {cand_code_path}. Using None for cand_codes.")
                    cand_codes = None

                if retrieval_config.rerank:
                    generation_output = generate_codes_for_dataset(
                        model=model,
                        data_loader=data_loader,
                        device=config.dist_config.gpu_id,
                        use_fp16=config.retrieval_config.use_fp16,
                        is_quantizer=False,
                        cand_codes=cand_codes,
                        num_beams=config.retrieval_config.num_beams,
                        trie_save_path=cand_trie_path
                    )
                    codes, embeddings, id_list = generation_output[:3]
                    if len(generation_output) == 4:
                        beam_scores = generation_output[3]
                else:
                    generation_output = generate_noembed_codes_for_dataset(
                        model=model,
                        data_loader=data_loader,
                        device=config.dist_config.gpu_id,
                        use_fp16=config.retrieval_config.use_fp16,
                        is_quantizer=False,
                        cand_codes=cand_codes,
                        num_beams=config.retrieval_config.num_beams,
                        trie_save_path=cand_trie_path
                    )
                    if len(generation_output) == 3:
                        codes, id_list, beam_scores = generation_output
                    else:
                        codes, id_list = generation_output

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Log: Codes list length: {len(codes)}")
                if retrieval_config.rerank:
                    print(f"Log: Embedding list length: {len(embeddings)}")
                print(f"Log: ID list length: {len(id_list)}")

                mid_name = cand_pool_name if split_name == "cand_pool" else dataset_name
                save_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name)

                code_data_name = f"mbeir_{mid_name}_{split_name}_codes.npy"
                code_path = os.path.join(save_path, code_data_name)
                os.makedirs(os.path.dirname(code_path), exist_ok=True)
                np.save(code_path, codes)
                print(f"Log: Saved codes to {code_path}.")

                if split_name != "cand_pool":
                    beam_score_data_name = f"mbeir_{mid_name}_{split_name}_beam_scores.npy"
                    beam_score_path = os.path.join(save_path, beam_score_data_name)
                    if beam_scores is not None:
                        np.save(beam_score_path, np.asarray(beam_scores, dtype=np.float32))
                        print(f"Log: Saved beam scores to {beam_score_path}.")
                    elif os.path.exists(beam_score_path):
                        # Codes were regenerated without sequence scores.  Never
                        # leave an older checkpoint's scores next to fresh codes.
                        os.remove(beam_score_path)
                        print(f"Log: Removed stale beam-score cache {beam_score_path}.")

                if retrieval_config.rerank:
                    embedding_data_name = f"mbeir_{mid_name}_{split_name}_embeddings.npy"
                    embeddings_path = os.path.join(save_path, embedding_data_name)
                    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
                    np.save(embeddings_path, embeddings)
                    print(f"Log: Saved Embeddings to {embeddings_path}.")

                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                id_path = os.path.join(save_path, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, id_list)
                print(f"Log: Saved ids to {id_path}.")

                if split_name == "cand_pool":
                    manifest_paths = _candidate_output_paths(genir_dir, gen_code_dir_name, expt_dir_name, mid_name)
                    manifest_signature = _candidate_cache_signature(
                        config=config,
                        mbeir_data_dir=mbeir_data_dir,
                        genir_dir=genir_dir,
                        output_cand_pool_name=mid_name,
                        cand_pool_data_path=cand_pool_data_path,
                        source_kind="extracted" if actual_is_extracted else "raw",
                        extracted_path=cand_pool_extracted_path if actual_is_extracted and 'cand_pool_extracted_path' in locals() else None,
                    )
                    _save_candidate_cache_manifest(manifest_paths, manifest_signature)
                    print(f"Log: Saved candidate cache manifest to {manifest_paths['manifest_path']}.")

            if dist.is_initialized():
                dist.barrier()

            del codes
            del id_list
            del data_loader
            del dataset
            del collator
            del sampler
            try:
                del embeddings
            except:
                None
            try:
                del beam_scores
            except:
                None

            gc.collect()
            torch.cuda.empty_cache()

        # Union pool embeddings

        if split_name == "cand_pool" and gen_code_cand_pool_config.gen_code_union_pool:
            # To efficiently generate codes for the union(global) pool,
            # We concat previously saved codes and ids from single(local) pool
            # Instead of embed the union pool directly.
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\nLog: Generating codes for union pool...")

                # Increase number of threads for rank 0
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                total_cores = os.cpu_count() or 1
                initial_threads_per_process = max(1, total_cores // world_size)
                torch.set_num_threads(total_cores)

                all_codes = []
                all_embeddings = []
                all_ids = []
                for cand_pool_name in cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                        cand_pool_name = cand_pool_name + '_test'
                    cand_pool_name = f"mbeir_{cand_pool_name}_{split_name}"
                    code_data_name = f"{cand_pool_name}_codes.npy"
                    embedding_data_name = f"{cand_pool_name}_embeddings.npy"
                    id_data_name = f"{cand_pool_name}_ids.npy"
                    code_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        code_data_name,
                    )
                    embedding_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        embedding_data_name,
                    )
                    id_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        id_data_name,
                    )
                    all_codes.append(np.load(code_path))
                    if retrieval_config.rerank:
                        all_embeddings.append(np.load(embedding_path))
                    all_ids.append(np.load(id_path))
                    print(f"Log: Concatenating codes from {code_path} and ids from {id_path}.")
                    if retrieval_config.rerank:
                        print("All Num Code: {}, All Num Emb: {}, All Num ID: {}".format(
                            len(np.concatenate(all_codes, axis=0)), len(np.concatenate(all_embeddings, axis=0)),
                            len(np.concatenate(all_ids, axis=0))))
                    else:
                        print("All Num Code: {}, All Num ID: {}".format(len(np.concatenate(all_codes, axis=0)),
                                                                        len(np.concatenate(all_ids, axis=0))))

                all_codes = np.concatenate(all_codes, axis=0)
                if retrieval_config.rerank:
                    all_embeddings = np.concatenate(all_embeddings, axis=0)
                all_ids = np.concatenate(all_ids, axis=0)
                assert len(all_codes) == len(all_ids), "Mismatch between codes and IDs length."
                if retrieval_config.rerank:
                    assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
                print(f"Log: all_codes length: {len(all_codes)} and all_ids length: {len(all_ids)}.")

                # Save the codes to .npy
                code_data_name = f"mbeir_union_{split_name}_codes.npy"
                code_path = os.path.join(
                    genir_dir,
                    gen_code_dir_name,
                    expt_dir_name,
                    split_name,
                    code_data_name,
                )
                os.makedirs(os.path.dirname(code_path), exist_ok=True)
                np.save(code_path, all_codes)
                print(f"Log: Saved codes to {code_path}.")

                if retrieval_config.rerank:
                    # Save the embeddings to .npy
                    embeddings_data_name = f"mbeir_union_{split_name}_embeddings.npy"
                    embeddings_path = os.path.join(
                        genir_dir,
                        gen_code_dir_name,
                        expt_dir_name,
                        split_name,
                        embeddings_data_name,
                    )
                    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
                    np.save(embeddings_path, all_embeddings)
                    print(f"Log: Saved embeddings to {embeddings_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_union_{split_name}_ids.npy"
                id_path = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, all_ids)
                print(f"Log: Saved ids to {id_path}.")

                # Delete the codes and IDs to free up memory
                del all_codes
                if retrieval_config.rerank:
                    del all_embeddings
                del all_ids

                # Explicitly call the garbage collector
                gc.collect()

                # Reset number of threads to initial value after conversion
                torch.set_num_threads(initial_threads_per_process)

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the codes and ids.


def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0

    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    relevant_docs_set = set(relevant_docs)

    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid


# Create a hash map for candidate pool codes
def create_hash_map(codes, ids):
    codes = _normalize_candidate_codes(codes)
    hash_map = defaultdict(list)
    for code, id in zip(codes, ids):
        hash_map[_code_to_tuple(code)].append(id)
    return hash_map


def create_id_to_index_map(ids):
    return {id: index for index, id in enumerate(ids)}


def _safe_unhash_qid(qid):
    """Convert stored qid representation to the string form used by qrels."""
    if isinstance(qid, bytes):
        qid = qid.decode("utf-8")
    if isinstance(qid, np.generic):
        qid = qid.item()
    try:
        return unhash_qid(qid)
    except Exception:
        return str(qid)


def _safe_unhash_did(did):
    """Convert stored did representation to the string form used by qrels."""
    if isinstance(did, bytes):
        did = did.decode("utf-8")
    if isinstance(did, np.generic):
        did = did.item()
    try:
        return unhash_did(did)
    except Exception:
        return str(did)


def compute_candidate_code_collision_stats(cand_pool_hash_map, total_candidates=None):
    """Summarize semantic-ID collisions in the candidate pool."""
    bucket_sizes = [len(v) for v in cand_pool_hash_map.values()]
    total = int(total_candidates if total_candidates is not None else sum(bucket_sizes))
    unique_codes = len(bucket_sizes)
    collision_buckets = sum(1 for s in bucket_sizes if s > 1)
    collision_items = sum(max(0, s - 1) for s in bucket_sizes)
    max_bucket = max(bucket_sizes) if bucket_sizes else 0
    return {
        "total_candidates": total,
        "unique_codes": unique_codes,
        "collision_buckets": collision_buckets,
        "collision_items": collision_items,
        "collision_item_rate": collision_items / max(total, 1),
        "max_bucket_size": max_bucket,
    }


def _unique_candidate_ids_from_query_code(query_code, cand_pool_hash_map):
    """Expand generated semantic-ID beams into unique candidate ids in beam order."""
    unique_candidates = []
    for code in _iter_code_beams(query_code):
        code_tuple = _code_to_tuple(code)
        if code_tuple in cand_pool_hash_map:
            unique_candidates.extend(cand_pool_hash_map[code_tuple])
    return list(dict.fromkeys(unique_candidates))


def compute_beam_ground_truth_recall_stats(query_codes, query_ids, qrel, cand_pool_hash_map, ks=(1, 5, 10, 50)):
    """Recall upper bound from generated code beams before embedding reranking.

    This answers: did the generator put the true item anywhere among its legal
    semantic-ID beams?  If this is low, the problem is the generator/semantic-ID
    prediction; if this is high but final Recall@k is low, the reranker or final
    ranking stage is the bottleneck.
    """
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    hits = {k: 0 for k in ks}
    evaluated = 0
    missing_qrels = 0
    avg_unique = 0.0
    first_gt_rank_sum = 0.0
    first_gt_rank_count = 0
    for idx, query_code in enumerate(query_codes):
        qid = _safe_unhash_qid(query_ids[idx]) if idx < len(query_ids) else str(idx)
        relevant = set(qrel.get(qid, []))
        if not relevant:
            missing_qrels += 1
            continue
        candidates = _unique_candidate_ids_from_query_code(query_code, cand_pool_hash_map)
        eval_candidates = [_safe_unhash_did(c) for c in candidates]
        evaluated += 1
        avg_unique += len(eval_candidates)
        first_rank = None
        for rank, cand_did in enumerate(eval_candidates, start=1):
            if cand_did in relevant:
                first_rank = rank
                break
        if first_rank is not None:
            first_gt_rank_sum += first_rank
            first_gt_rank_count += 1
        for k in ks:
            if relevant.intersection(eval_candidates[:k]):
                hits[k] += 1
    denom = max(evaluated, 1)
    out = {f"beam_gt_recall@{k}": hits[k] / denom for k in ks}
    out.update({
        "evaluated_queries": evaluated,
        "missing_qrels": missing_qrels,
        "avg_unique_candidates_per_query": avg_unique / denom,
        "mean_first_gt_rank": (first_gt_rank_sum / first_gt_rank_count) if first_gt_rank_count else None,
    })
    return out


def build_gt_code_oracle_query_codes(query_codes, query_ids, qrel, cand_pool_codes, cand_pool_ids):
    """Replace generated query codes with ground-truth candidate codes for pipeline debugging.

    Enable via retrieval_config.use_gt_code_oracle=true.  This should give a high
    Recall@1 unless semantic-ID collisions or reranking are wrong.  It is not a
    real model result; it only checks the retrieval/evaluation plumbing.
    """
    eval_did_to_index = {}
    for idx, cand_id in enumerate(cand_pool_ids):
        eval_did_to_index[_safe_unhash_did(cand_id)] = idx
    original = _normalize_query_codes_for_retrieval(query_codes, cand_pool_codes.shape[1])
    oracle_codes = []
    used = 0
    missed = 0
    for idx, query_id in enumerate(query_ids):
        qid = _safe_unhash_qid(query_id)
        chosen_index = None
        for rel_did in qrel.get(qid, []):
            if rel_did in eval_did_to_index:
                chosen_index = eval_did_to_index[rel_did]
                break
        if chosen_index is None:
            missed += 1
            oracle_codes.append(original[idx, :1, :])
        else:
            used += 1
            oracle_codes.append(cand_pool_codes[chosen_index][None, :])
    return np.stack(oracle_codes, axis=0).astype(np.int64, copy=False), {"used": used, "missed": missed}


def compute_generated_code_hit_stats(query_codes, cand_pool_hash_map):
    """Summarize whether generated semantic-ID beams exist in the candidate pool.

    Accepts [Q, K, L] query beams, [Q, L] single-beam arrays, and older
    accidentally flattened [Q, K*L] files.  Retrieval is based on exact semantic
    ID lookup, so a 0.0 hit rate implies Recall@k must be 0 regardless of rerank.
    """
    total = 0
    any_hit = 0
    first_beam_hit = 0
    total_hit_beams = 0
    total_beams = 0
    for query_code in query_codes:
        total += 1
        query_has_hit = False
        for beam_idx, code in enumerate(_iter_code_beams(query_code)):
            total_beams += 1
            if _code_to_tuple(code) in cand_pool_hash_map:
                total_hit_beams += 1
                query_has_hit = True
                if beam_idx == 0:
                    first_beam_hit += 1
        if query_has_hit:
            any_hit += 1
    denom = max(total, 1)
    beam_denom = max(total_beams, 1)
    return {
        "num_queries": total,
        "any_code_hit_rate": any_hit / denom,
        "first_beam_hit_rate": first_beam_hit / denom,
        "beam_code_hit_rate": total_hit_beams / beam_denom,
        "avg_hit_beams_per_query": total_hit_beams / denom,
    }


# Function to retrieve indices for a query using hash map
def retrieve_indices_for_query(query_code, cand_pool_hash_map, k):
    unique_candidates = []
    for code in _iter_code_beams(query_code):
        code_tuple = _code_to_tuple(code)
        if code_tuple in cand_pool_hash_map:
            unique_candidates.extend(cand_pool_hash_map[code_tuple])

    # Ensure the order of unique_candidates matches the order in query_code
    # unique_candidates = sorted(set(unique_candidates), key=unique_candidates.index)\

    # If the number of unique candidates exceeds k, trim to k
    if len(unique_candidates) > k:
        unique_candidates = unique_candidates[:k]

    return unique_candidates


def compute_cosine_similarities(query_embedding, candidate_embeddings):
    query_embedding = torch.as_tensor(query_embedding, dtype=torch.float32)
    candidate_embeddings = torch.as_tensor(candidate_embeddings, dtype=torch.float32)
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    return F.linear(query_embedding, candidate_embeddings).squeeze(0)


def _safe_standardize(values):
    """Query-local standardization with a deterministic constant fallback."""
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() <= 1:
        return torch.zeros_like(values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or float(std.item()) < 1e-8:
        return torch.zeros_like(values)
    return (values - mean) / std


def _safe_minmax(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() <= 1:
        return torch.zeros_like(values)
    lo = values.min()
    hi = values.max()
    span = hi - lo
    if not torch.isfinite(span) or float(span.item()) < 1e-8:
        return torch.zeros_like(values)
    return (values - lo) / span


def _candidate_generation_metadata(query_code, cand_pool_hash_map, query_beam_scores=None):
    """Expand beams into unique candidates while retaining generation evidence.

    Each candidate receives the best score and earliest rank among all beams that
    map to it.  When old code caches do not contain explicit sequence scores, the
    monotonic fallback ``-log(1 + beam_rank)`` still lets ``use_hybrid_score``
    exploit the generator ordering instead of silently reverting to cosine-only.
    """
    beam_scores = None
    if query_beam_scores is not None:
        beam_scores = np.asarray(query_beam_scores, dtype=np.float32).reshape(-1)

    candidate_order = []
    candidate_meta = {}
    for beam_idx, code in enumerate(_iter_code_beams(query_code)):
        code_tuple = _code_to_tuple(code)
        candidate_ids = cand_pool_hash_map.get(code_tuple, [])
        if beam_scores is not None and beam_idx < beam_scores.size and np.isfinite(beam_scores[beam_idx]):
            generation_score = float(beam_scores[beam_idx])
        else:
            generation_score = -float(np.log1p(beam_idx))

        for cand_id in candidate_ids:
            if cand_id not in candidate_meta:
                candidate_order.append(cand_id)
                candidate_meta[cand_id] = {
                    "beam_rank": int(beam_idx),
                    "generation_score": generation_score,
                }
            else:
                meta = candidate_meta[cand_id]
                meta["beam_rank"] = min(int(meta["beam_rank"]), int(beam_idx))
                meta["generation_score"] = max(float(meta["generation_score"]), generation_score)
    return candidate_order, candidate_meta


def retrieve_and_rank_for_query(query_code,
                                cand_pool_hash_map,
                                query_embedding,
                                cand_pool_embeddings,
                                cand_pool_ids,
                                id_to_index_map,
                                k,
                                query_beam_scores=None,
                                use_hybrid_score=False,
                                hybrid_score_method="weighted_zscore",
                                hybrid_clip_weight=0.8,
                                hybrid_generator_weight=0.2,
                                hybrid_rrf_k=60.0,
                                stage_profile=None):
    lookup_started_at = time.perf_counter()
    unique_candidates, generation_meta = _candidate_generation_metadata(
        query_code=query_code,
        cand_pool_hash_map=cand_pool_hash_map,
        query_beam_scores=query_beam_scores,
    )

    if len(unique_candidates) == 0:
        if stage_profile is not None:
            stage_profile["lookup_seconds"].append(
                time.perf_counter() - lookup_started_at
            )
            stage_profile["rerank_seconds"].append(0.0)
        return []

    # Convert to PyTorch tensors
    query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    filtered_candidates = [cand_id for cand_id in unique_candidates if cand_id in id_to_index_map]
    candidate_indices = [id_to_index_map[cand_id] for cand_id in filtered_candidates]
    if len(candidate_indices) == 0:
        if stage_profile is not None:
            stage_profile["lookup_seconds"].append(
                time.perf_counter() - lookup_started_at
            )
            stage_profile["rerank_seconds"].append(0.0)
        return []
    lookup_seconds = time.perf_counter() - lookup_started_at
    rerank_started_at = time.perf_counter()
    candidate_embeddings_tensor = torch.tensor(cand_pool_embeddings[candidate_indices], dtype=torch.float32)

    # Calculate cosine similarities
    similarities = compute_cosine_similarities(query_embedding_tensor, candidate_embeddings_tensor)

    method = str(hybrid_score_method or "weighted_zscore").lower()
    clip_weight = max(0.0, float(hybrid_clip_weight))
    generator_weight = max(0.0, float(hybrid_generator_weight))
    weight_sum = clip_weight + generator_weight
    if weight_sum <= 0:
        clip_weight, generator_weight = 1.0, 0.0
    else:
        clip_weight /= weight_sum
        generator_weight /= weight_sum

    generation_scores = torch.tensor(
        [generation_meta[cand_id]["generation_score"] for cand_id in filtered_candidates],
        dtype=torch.float32,
    )
    generation_ranks = torch.tensor(
        [generation_meta[cand_id]["beam_rank"] for cand_id in filtered_candidates],
        dtype=torch.long,
    )

    if not bool(use_hybrid_score) or method in {"cosine", "cosine_only", "clip", "clip_only"}:
        final_scores = similarities.float()
    elif method in {"generation", "generation_only", "beam", "beam_only"}:
        final_scores = generation_scores
    elif method in {"weighted_minmax", "minmax", "linear_minmax"}:
        final_scores = (
            clip_weight * _safe_minmax(similarities)
            + generator_weight * _safe_minmax(generation_scores)
        )
    elif method in {"rrf", "reciprocal_rank_fusion"}:
        rrf_k = max(0.0, float(hybrid_rrf_k))
        cosine_order = torch.argsort(similarities, descending=True)
        cosine_ranks = torch.empty_like(cosine_order)
        cosine_ranks[cosine_order] = torch.arange(cosine_order.numel(), dtype=cosine_order.dtype)
        final_scores = (
            clip_weight / (rrf_k + cosine_ranks.float() + 1.0)
            + generator_weight / (rrf_k + generation_ranks.float() + 1.0)
        )
    elif method in {"weighted_zscore", "zscore", "linear", "hybrid"}:
        final_scores = (
            clip_weight * _safe_standardize(similarities)
            + generator_weight * _safe_standardize(generation_scores)
        )
    else:
        raise ValueError(
            f"Unsupported retrieval_config.hybrid_score_method={hybrid_score_method!r}. "
            "Use weighted_zscore, weighted_minmax, rrf, cosine_only, or generation_only."
        )

    # Python's stable sort gives deterministic tie-breaking by generator rank and
    # then by original candidate order (which is itself beam ordered).
    ranked_positions = sorted(
        range(len(filtered_candidates)),
        key=lambda idx: (
            -float(final_scores[idx].item()),
            int(generation_ranks[idx].item()),
            idx,
        ),
    )
    top_k_indices = ranked_positions[:k]

    # Map back to candidate IDs
    top_k_candidates = [filtered_candidates[i] for i in top_k_indices]

    if stage_profile is not None:
        stage_profile["lookup_seconds"].append(lookup_seconds)
        stage_profile["rerank_seconds"].append(
            time.perf_counter() - rerank_started_at
        )

    return top_k_candidates


def generative_retrieve(config):
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    qrel_dir_name = retrieval_config.qrel_dir_name
    gen_code_dir_name = retrieval_config.gen_code_dir_name
    expt_dir_name = config.experiment.path_suffix

    results_dir_name = retrieval_config.results_dir_name
    exp_results_dir = os.path.join(genir_dir, results_dir_name, expt_dir_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_run_file_dir = os.path.join(exp_results_dir, "run_files")
    os.makedirs(exp_run_file_dir, exist_ok=True)
    exp_tsv_results_dir = os.path.join(exp_results_dir, "final_tsv")
    os.makedirs(exp_tsv_results_dir, exist_ok=True)

    splits = []
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        retrieval_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if retrieval_dataset_config and retrieval_dataset_config.enable_retrieve:
            dataset_name_list = getattr(retrieval_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(retrieval_dataset_config, "correspond_cand_pools_name", None)
            qrel_name_list = getattr(retrieval_dataset_config, "correspond_qrels_name", None)
            metric_names_list = getattr(retrieval_dataset_config, "correspond_metrics_name", None)
            dataset_gen_code_dir = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, split_name)
            splits.append(
                (
                    split_name,
                    dataset_gen_code_dir,
                    dataset_name_list,
                    cand_pool_name_list,
                    qrel_name_list,
                    metric_names_list,
                )
            )
            assert (
                    len(dataset_name_list) == len(cand_pool_name_list) == len(qrel_name_list) == len(metric_names_list)
            ), "Mismatch between datasets and candidate pools and qrels."

    print("-" * 30)
    for (
            split_name,
            dataset_gen_code_dir,
            dataset_name_list,
            cand_pool_name_list,
            qrel_name_list,
            metric_names_list,
    ) in splits:
        print(
            f"Split: {split_name}, Retrieval Datasets: {dataset_name_list}, Candidate Pools: {cand_pool_name_list}, Metric: {metric_names_list})"
        )
        print("-" * 30)

    eval_results = []
    qrel_dir = os.path.join(mbeir_data_dir, qrel_dir_name)
    cand_dir = os.path.join(genir_dir, gen_code_dir_name, expt_dir_name, "cand_pool")
    # UNION evaluation reuses one 5.6M-item candidate pool for many query
    # datasets. Loading it and rebuilding the semantic-ID and DID maps for
    # every task is pure duplicate work, so retain the active pool bundle.
    candidate_pool_cache = {}
    for (
            split,
            dataset_gen_code_dir,
            dataset_name_list,
            cand_pool_name_list,
            qrel_name_list,
            metric_names_list,
    ) in splits:
        for dataset_name, cand_pool_name, qrel_name, metric_names in zip(
                dataset_name_list, cand_pool_name_list, qrel_name_list, metric_names_list
        ):
            retrieval_started_at = time.perf_counter()
            print("\n" + "-" * 30)
            print(f"Retriever: Retrieving for query:{dataset_name} | split:{split} | from cand_pool:{cand_pool_name}")

            dataset_name = dataset_name.lower()
            cand_pool_name = cand_pool_name.lower()
            if cand_pool_name in EXCEPTIONAL_CAND_POOLS:
                cand_pool_name = cand_pool_name + '_test'
            qrel_name = qrel_name.lower()

            qrel_path = os.path.join(qrel_dir, split, f"mbeir_{qrel_name}_{split}_qrels.txt")
            qrel, qid_to_taskid = load_qrel(qrel_path)

            gen_code_query_id_path = os.path.join(dataset_gen_code_dir, f"mbeir_{dataset_name}_{split}_ids.npy")
            query_ids = np.load(gen_code_query_id_path)
            gen_code_query_path = os.path.join(dataset_gen_code_dir, f"mbeir_{dataset_name}_{split}_codes.npy")
            query_codes = np.load(gen_code_query_path)
            query_beam_scores = None
            beam_score_query_path = os.path.join(
                dataset_gen_code_dir,
                f"mbeir_{dataset_name}_{split}_beam_scores.npy",
            )
            use_hybrid_score_requested = bool(getattr(retrieval_config, "use_hybrid_score", False))
            load_saved_beam_scores = bool(getattr(retrieval_config, "load_saved_beam_scores", True))
            require_beam_scores_for_hybrid = bool(
                getattr(retrieval_config, "require_beam_scores_for_hybrid", True)
            )
            if retrieval_config.rerank and load_saved_beam_scores and os.path.exists(beam_score_query_path):
                query_beam_scores = np.load(beam_score_query_path)
            elif use_hybrid_score_requested and require_beam_scores_for_hybrid:
                raise FileNotFoundError(
                    "Hybrid reranking requested but saved beam scores were not found: "
                    f"{beam_score_query_path}. Regenerate query beams with "
                    "retrieval_config.save_beam_scores=true, or explicitly set "
                    "retrieval_config.require_beam_scores_for_hybrid=false to allow "
                    "the beam-rank fallback."
                )
            if retrieval_config.rerank:
                embeddings_query_path = os.path.join(dataset_gen_code_dir,
                                                     f"mbeir_{dataset_name}_{split}_embeddings.npy")
                query_embeddings = np.load(embeddings_query_path)

            cand_pool_code_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_codes.npy")
            if retrieval_config.rerank:
                cand_pool_embeddings_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_embeddings.npy")
            cand_pool_id_path = os.path.join(cand_dir, f"mbeir_{cand_pool_name}_cand_pool_ids.npy")
            candidate_cache_key = (
                os.path.abspath(cand_pool_code_path),
                os.path.abspath(cand_pool_id_path),
                os.path.abspath(cand_pool_embeddings_path) if retrieval_config.rerank else None,
            )
            candidate_bundle = candidate_pool_cache.get(candidate_cache_key)
            if candidate_bundle is None:
                # A one-entry cache is sufficient: LOCAL pools change per
                # task, whereas every UNION task uses the same pool. Avoid
                # retaining all LOCAL embeddings in host memory.
                candidate_pool_cache.clear()
                cand_pool_codes = _normalize_candidate_codes(np.load(cand_pool_code_path))
                if retrieval_config.rerank:
                    cand_pool_embeddings = np.load(cand_pool_embeddings_path)
                else:
                    cand_pool_embeddings = None
                cand_pool_ids = np.load(cand_pool_id_path)
                cand_pool_hash_map = create_hash_map(cand_pool_codes, cand_pool_ids)
                id_to_index_map = create_id_to_index_map(cand_pool_ids)
                collision_stats = compute_candidate_code_collision_stats(
                    cand_pool_hash_map, total_candidates=len(cand_pool_ids)
                )
                candidate_bundle = (
                    cand_pool_codes,
                    cand_pool_embeddings,
                    cand_pool_ids,
                    cand_pool_hash_map,
                    id_to_index_map,
                    collision_stats,
                )
                candidate_pool_cache[candidate_cache_key] = candidate_bundle
                print(f"Retriever: Cached candidate pool indices for {cand_pool_name}.")
            else:
                print(f"Retriever: Reusing cached candidate pool indices for {cand_pool_name}.")
            (
                cand_pool_codes,
                cand_pool_embeddings,
                cand_pool_ids,
                cand_pool_hash_map,
                id_to_index_map,
                collision_stats,
            ) = candidate_bundle
            candidate_code_width = cand_pool_codes.shape[1]
            query_codes = _normalize_query_codes_for_retrieval(query_codes, candidate_code_width)
            query_beam_scores = _normalize_query_beam_scores_for_retrieval(
                query_beam_scores,
                num_queries=query_codes.shape[0],
                num_beams=query_codes.shape[1],
            )
            if dist_utils.is_main_process():
                print(
                    "Retriever Diagnostics: code array shapes | "
                    f"query_codes={query_codes.shape} cand_pool_codes={cand_pool_codes.shape} "
                    f"beam_scores={None if query_beam_scores is None else query_beam_scores.shape}"
                )
            metric_list = [metric.strip() for metric in metric_names.split(",")]
            metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]

            print(
                "Retriever Diagnostics: candidate semantic-ID collisions | "
                f"candidates={collision_stats['total_candidates']} "
                f"unique_codes={collision_stats['unique_codes']} "
                f"collision_items={collision_stats['collision_items']} "
                f"collision_item_rate={collision_stats['collision_item_rate']:.6f} "
                f"collision_buckets={collision_stats['collision_buckets']} "
                f"max_bucket_size={collision_stats['max_bucket_size']}"
            )

            if getattr(retrieval_config, "use_gt_code_oracle", False):
                query_codes, oracle_stats = build_gt_code_oracle_query_codes(
                    query_codes=query_codes,
                    query_ids=query_ids,
                    qrel=qrel,
                    cand_pool_codes=cand_pool_codes,
                    cand_pool_ids=cand_pool_ids,
                )
                print(
                    "Retriever Diagnostics: using GT semantic-ID oracle codes | "
                    f"used={oracle_stats['used']} missed={oracle_stats['missed']}"
                )
                query_beam_scores = None

            hit_stats = compute_generated_code_hit_stats(query_codes, cand_pool_hash_map)
            print(
                "Retriever Diagnostics: generated semantic-ID exact lookup | "
                f"queries={hit_stats['num_queries']} "
                f"any_code_hit_rate={hit_stats['any_code_hit_rate']:.4f} "
                f"first_beam_hit_rate={hit_stats['first_beam_hit_rate']:.4f} "
                f"beam_code_hit_rate={hit_stats['beam_code_hit_rate']:.4f} "
                f"avg_hit_beams_per_query={hit_stats['avg_hit_beams_per_query']:.2f}"
            )

            beam_diag_ks = [1, 5, 10, 50]
            if query_codes.shape[1] >= 100:
                beam_diag_ks.append(100)
            beam_gt_stats = compute_beam_ground_truth_recall_stats(
                query_codes=query_codes,
                query_ids=query_ids,
                qrel=qrel,
                cand_pool_hash_map=cand_pool_hash_map,
                ks=beam_diag_ks,
            )
            beam_recall_text = " ".join(
                f"beam_gt_recall@{diag_k}={beam_gt_stats[f'beam_gt_recall@{diag_k}']:.4f}"
                for diag_k in beam_diag_ks
            )
            print(
                "Retriever Diagnostics: GT inside generated code beams before rerank | "
                f"queries={beam_gt_stats['evaluated_queries']} "
                f"{beam_recall_text} "
                f"avg_unique_candidates={beam_gt_stats['avg_unique_candidates_per_query']:.2f} "
                f"mean_first_gt_rank={beam_gt_stats['mean_first_gt_rank']}"
            )
            if hit_stats['any_code_hit_rate'] == 0.0:
                print(
                    "Retriever Diagnostics: no generated code beam exists in the candidate pool. "
                    "Recall@k will be 0 regardless of reranking. Check that query/candidate codes "
                    "were generated with the same quantizer checkpoint and the same codebook_level/vocab, "
                    "and delete stale gen_code/candidate_tree files before regenerating."
                )

            k = max([int(metric.split("@")[1]) for metric in metric_recall_list])

            use_hybrid_score = bool(getattr(retrieval_config, "use_hybrid_score", False))
            hybrid_score_method = str(getattr(retrieval_config, "hybrid_score_method", "weighted_zscore"))
            hybrid_clip_weight = float(getattr(retrieval_config, "hybrid_clip_weight", 0.8))
            hybrid_generator_weight = float(getattr(retrieval_config, "hybrid_generator_weight", 0.2))
            hybrid_rrf_k = float(getattr(retrieval_config, "hybrid_rrf_k", 60.0))
            if retrieval_config.rerank:
                if query_beam_scores is not None:
                    score_source = "saved beam log-scores"
                    if not use_hybrid_score:
                        score_source += " (loaded, hybrid disabled)"
                else:
                    score_source = "beam-rank fallback"
                print(
                    "Retriever: rerank scoring | "
                    f"hybrid={use_hybrid_score} method={hybrid_score_method} "
                    f"clip_weight={hybrid_clip_weight:.3f} generator_weight={hybrid_generator_weight:.3f} "
                    f"generator_source={score_source}"
                )

            # Retrieve indices for all queries
            retrieved_indices = []
            retrieval_query_seconds = []
            profile_full_stages = bool(
                int(os.environ.get("STRUCTNAR_PROFILE_FULL_STAGES", "0"))
            )
            retrieval_stage_profile = {
                "lookup_seconds": [],
                "rerank_seconds": [],
            }
            for i, query_code in enumerate(query_codes):
                query_retrieval_started_at = time.perf_counter()
                if retrieval_config.rerank:
                    retrieved_indices.append(retrieve_and_rank_for_query(
                        query_code,
                        cand_pool_hash_map,
                        query_embeddings[i],
                        cand_pool_embeddings,
                        cand_pool_ids,
                        id_to_index_map,
                        k,
                        query_beam_scores=None if query_beam_scores is None else query_beam_scores[i],
                        use_hybrid_score=use_hybrid_score,
                        hybrid_score_method=hybrid_score_method,
                        hybrid_clip_weight=hybrid_clip_weight,
                        hybrid_generator_weight=hybrid_generator_weight,
                        hybrid_rrf_k=hybrid_rrf_k,
                        stage_profile=(
                            retrieval_stage_profile if profile_full_stages else None
                        ),
                    ))
                else:
                    retrieved_indices.append(retrieve_indices_for_query(query_code, cand_pool_hash_map, k))
                retrieval_query_seconds.append(time.perf_counter() - query_retrieval_started_at)

            full_stage_profile_path = os.environ.get(
                "STRUCTNAR_FULL_STAGE_PROFILE_PATH", ""
            ).strip()
            if profile_full_stages and full_stage_profile_path:
                if not os.path.isfile(full_stage_profile_path):
                    raise FileNotFoundError(
                        "Generation did not write the required full-stage profile: "
                        f"{full_stage_profile_path}"
                    )
                with np.load(full_stage_profile_path, allow_pickle=False) as source:
                    profile_arrays = {key: source[key] for key in source.files}
                generated_ids = np.asarray(profile_arrays["query_id"])
                retrieval_ids = np.asarray(query_ids)
                if not np.array_equal(generated_ids, retrieval_ids):
                    raise RuntimeError(
                        "Full-stage generation/retrieval query IDs are not aligned"
                    )
                retrieval_ms = np.asarray(
                    retrieval_query_seconds, dtype=np.float64
                ) * 1000.0
                lookup_ms = np.asarray(
                    retrieval_stage_profile["lookup_seconds"], dtype=np.float64
                ) * 1000.0
                rerank_ms = np.asarray(
                    retrieval_stage_profile["rerank_seconds"], dtype=np.float64
                ) * 1000.0
                if not (
                    retrieval_ms.size == lookup_ms.size == rerank_ms.size
                    == generated_ids.size
                ):
                    raise RuntimeError(
                        "Full-stage retrieval profile has inconsistent row counts"
                    )
                e2e_ms = np.asarray(profile_arrays["generation_ms"], dtype=np.float64) + retrieval_ms
                warmup_queries = int(np.asarray(profile_arrays["warmup_queries"]).reshape(-1)[0])
                measured = slice(warmup_queries, e2e_ms.size)
                profile_arrays.update(
                    retrieval_ms=retrieval_ms,
                    bucket_lookup_ms=lookup_ms,
                    embedding_rerank_ms=rerank_ms,
                    online_decode_to_rerank_ms=e2e_ms,
                )
                np.savez_compressed(full_stage_profile_path, **profile_arrays)
                residual_ms = retrieval_ms - lookup_ms - rerank_ms
                print(
                    "Retrieval Stage Breakdown: "
                    f"queries={e2e_ms.size - warmup_queries} "
                    f"bucket_lookup_mean_ms={lookup_ms[measured].mean():.6f} "
                    f"embedding_rerank_mean_ms={rerank_ms[measured].mean():.6f} "
                    f"retrieval_wrapper_mean_ms={residual_ms[measured].mean():.6f}"
                )
                print(
                    "Online Decode-to-Rerank Latency Distribution: "
                    f"queries={e2e_ms.size - warmup_queries} "
                    f"warmup_queries_excluded={warmup_queries} "
                    f"p50_ms={np.percentile(e2e_ms[measured], 50):.6f} "
                    f"p95_ms={np.percentile(e2e_ms[measured], 95):.6f} "
                    f"p99_ms={np.percentile(e2e_ms[measured], 99):.6f} "
                    f"mean_ms={e2e_ms[measured].mean():.6f} "
                    f"profile={full_stage_profile_path}"
                )

            if not os.path.exists(exp_run_file_dir):
                os.makedirs(exp_run_file_dir)
            run_id = f"mbeir_{dataset_name}_single_pool_{split}"
            run_file_name = f"{run_id}_run.txt"
            run_file_path = os.path.join(exp_run_file_dir, run_file_name)
            with open(run_file_path, "w") as run_file:
                for idx, indices in enumerate(retrieved_indices):
                    qid = _safe_unhash_qid(query_ids[idx])
                    task_id = qid_to_taskid[qid]
                    for rank, retrieved_id in enumerate(indices, start=1):
                        run_file.write(f"{qid} Q0 {retrieved_id} {rank} {1.0} {run_id} {task_id}\n")
            print(f"Retriever: Run file saved to {run_file_path}")

            recall_values_by_task = defaultdict(lambda: defaultdict(list))
            for i, retrieved_indices_for_qid in enumerate(retrieved_indices):
                retrieved_indices_for_qid = [_safe_unhash_did(idx) for idx in retrieved_indices_for_qid]
                qid = _safe_unhash_qid(query_ids[i])
                relevant_docs = qrel[qid]
                task_id = qid_to_taskid[qid]

                # Compute Recall@k for each metric
                for metric in metric_recall_list:
                    k = int(metric.split("@")[1])
                    recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                    recall_values_by_task[task_id][metric].append(recall_at_k)

            for task_id, recalls in recall_values_by_task.items():
                task_name = get_mbeir_task_name(int(task_id))
                result = {
                    "TaskID": int(task_id),
                    "Task": task_name,
                    "Dataset": dataset_name,
                    "Split": split,
                    "CandPool": cand_pool_name,
                }
                for metric in metric_recall_list:
                    mean_recall_at_k = round(sum(recalls[metric]) / len(recalls[metric]), 4)
                    result[metric] = mean_recall_at_k
                    print(f"Retriever: Mean {metric}: {mean_recall_at_k}")
                eval_results.append(result)

            retrieval_seconds = time.perf_counter() - retrieval_started_at
            retrieval_qps = len(query_ids) / retrieval_seconds if retrieval_seconds > 0 else float("inf")
            print(
                "Retrieval Metrics: "
                f"seconds={retrieval_seconds:.6f} examples={len(query_ids)} "
                f"examples_per_second={retrieval_qps:.6f}"
            )
            if bool(int(os.environ.get("STRUCTNAR_PROFILE_LATENCY", "0"))) and retrieval_query_seconds:
                query_ms = np.asarray(retrieval_query_seconds, dtype=np.float64) * 1000.0
                print(
                    "Retrieval Latency Distribution: "
                    f"queries={query_ms.size} "
                    f"p50_ms={np.percentile(query_ms, 50):.6f} "
                    f"p95_ms={np.percentile(query_ms, 95):.6f} "
                    f"p99_ms={np.percentile(query_ms, 99):.6f} "
                    f"mean_ms={query_ms.mean():.6f}"
                )

    dataset_order = {
        "visualnews_task0": 1,
        "mscoco_task0": 2,
        "fashion200k_task0": 3,
        "webqa_task1": 4,
        "edis_task2": 5,
        "webqa_task2": 6,
        "visualnews_task3": 7,
        "mscoco_task3": 8,
        "fashion200k_task3": 9,
        "nights_task4": 10,
        "oven_task6": 11,
        "infoseek_task6": 12,
        "fashioniq_task7": 13,
        "cirr_task7": 14,
        "oven_task8": 15,
        "infoseek_task8": 16,
    }
    split_order = {"val": 1, "test": 2}
    cand_pool_order = {"union": 99}
    eval_results_sorted = sorted(
        eval_results,
        key=lambda x: (
            x["TaskID"],
            dataset_order.get(x["Dataset"].lower(), 99),
            split_order.get(x["Split"].lower(), 99),
            cand_pool_order.get(x["CandPool"].lower(), 0),
        ),
    )

    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    available_recall_metrics = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
    ]
    for result in eval_results_sorted:
        key = (result["TaskID"], result["Task"], result["Dataset"], result["Split"])
        for metric in available_recall_metrics:
            grouped_results[key][result["CandPool"]].update({metric: result.get(metric, None)})

    if retrieval_config.write_to_tsv:
        date_time = datetime.now().strftime("%m-%d-%H")
        tsv_file_name = f"eval_results_{date_time}.tsv"
        tsv_file_path = os.path.join(exp_tsv_results_dir, tsv_file_name)
        tsv_data = []
        header = [
            "TaskID",
            "Task",
            "Dataset",
            "Split",
            "Metric",
            "CandPool",
            "Value",
            "UnionPool",
            "UnionValue",
        ]
        tsv_data.append(header)

        for (task_id, task, dataset, split), cand_pools in grouped_results.items():
            union_results = cand_pools.get("union", {})
            for metric in available_recall_metrics:
                for cand_pool, metrics in cand_pools.items():
                    if cand_pool != "union":
                        row = [
                            task_id,
                            task,
                            dataset,
                            split,
                            metric,
                            cand_pool,
                            metrics.get(metric, None),
                        ]
                        if row[-1] is None:
                            continue
                        if union_results:
                            row.extend(["union", union_results.get(metric, "N/A")])
                        else:
                            row.extend(["", ""])
                        tsv_data.append(row)

        with open(tsv_file_path, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            for row in tsv_data:
                writer.writerow(row)

        print(f"Retriever: Results saved to {tsv_file_path}")

    return eval_results_sorted


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_eval_defaults(config):
    """Fill optional eval config fields expected by evaluator code."""
    try:
        if not hasattr(config, "retrieval_config") or config.retrieval_config is None:
            config.retrieval_config = {}
    except Exception:
        config.retrieval_config = {}

    defaults = {
        "use_fp16": True,
        "write_to_tsv": True,
        "rerank": True,
        "use_hybrid_score": False,
        "save_beam_scores": False,
        "load_saved_beam_scores": True,
        "require_beam_scores_for_hybrid": True,
        "hybrid_score_method": "weighted_zscore",
        "hybrid_clip_weight": 0.8,
        "hybrid_generator_weight": 0.2,
        "hybrid_rrf_k": 60,
        "use_gt_code_oracle": False,
        "use_oracle_inference": False,
        "gen_code_dir_name": "gen_code",
        "index_dir_name": "index",
        "qrel_dir_name": "qrels",
        "results_dir_name": "retrieval_results",
    }
    for key, value in defaults.items():
        try:
            getattr(config.retrieval_config, key)
        except Exception:
            setattr(config.retrieval_config, key, value)
    return config



def main(config):
    config = _ensure_eval_defaults(config)
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    model = build_model_from_config(config)

    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode_mbeir_batch' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'get_img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_clip_tokenizer")):
        raise AttributeError("The provided model does not have a 'get_clip_tokenizer' attribute.")
    if not callable(getattr(model, "get_seq2seq_tokenizer")):
        raise AttributeError("The provided model does not have a 'get_seq2seq_tokenizer' attribute.")

    img_preprocess_fn = model.get_img_preprocess_fn()
    clip_tokenizer = model.get_clip_tokenizer()
    seq2seq_tokenizer = model.get_seq2seq_tokenizer()

    model = model.to(config.dist_config.gpu_id)
    if config.dist_config.distributed_mode:
        model = DDP(
            model,
            device_ids=[config.dist_config.gpu_id],
            broadcast_buffers=False,
            find_unused_parameters=False
        )
    model.eval()

    print(f"Models are set up on GPU {config.dist_config.gpu_id}.")

    with torch.inference_mode():
        generate_codes_for_config(
            model=model,
            img_preprocess_fn=img_preprocess_fn,
            clip_tokenizer=clip_tokenizer,
            seq2seq_tokenizer=seq2seq_tokenizer,
            config=config,
        )

        if dist_utils.is_main_process():
            generative_retrieve(config)



def apply_cli_dotlist_overrides(config, unknown_args):
    """Apply Hydra/OmegaConf-style CLI overrides for eval scripts.

    Supports examples like `model.ckpt_config.ckpt_name=...` and the
    historical alias `runtime.quantizer_path=...`.
    """
    if not unknown_args:
        return config
    dotlist = []
    ignored = []
    for item in unknown_args:
        if not isinstance(item, str):
            continue
        if "=" in item and not item.startswith("-"):
            if item.startswith("runtime.quantizer_path="):
                value = item.split("=", 1)[1]
                if not hasattr(config, "codebook_config"):
                    config.codebook_config = {}
                config.codebook_config.quantizer_path = value
                print(f"[CLI override] runtime.quantizer_path -> codebook_config.quantizer_path = {value}")
            else:
                dotlist.append(item)
        else:
            ignored.append(item)
    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
        print(f"[CLI override] Applied dotlist overrides: {dotlist}")
    try:
        runtime_q = config.runtime.quantizer_path
    except Exception:
        runtime_q = None
    if runtime_q:
        if not hasattr(config, "codebook_config"):
            config.codebook_config = {}
        config.codebook_config.quantizer_path = runtime_q
        print(f"[CLI override] Synced runtime.quantizer_path -> codebook_config.quantizer_path = {runtime_q}")
    if ignored:
        print(f"[CLI override] Ignored unknown non-dotlist args: {ignored}")
    return config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Codes for MBEIR")
    parser.add_argument("--genir_dir", type=str, default="/GENIUS")
    parser.add_argument("--mbeir_data_dir", type=str, default="/GENIUS/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--quantizer_path", type=str, default="", help="Optional override for codebook_config.quantizer_path.")
    args, unknown_args = parser.parse_known_args()
    args.config_overrides = unknown_args
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir
    if args.quantizer_path:
        if not hasattr(config, "codebook_config"):
            config.codebook_config = {}
        config.codebook_config.quantizer_path = args.quantizer_path
        print(f"[CLI override] --quantizer_path -> codebook_config.quantizer_path = {args.quantizer_path}")
    config = apply_cli_dotlist_overrides(config, getattr(args, "config_overrides", []))

    args.dist_url = config.dist_config.dist_url
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
