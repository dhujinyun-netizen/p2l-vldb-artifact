"""Tree-aligned risk learning for GPT-HDGR.

The objective is deliberately evaluated on the same prefix-regeneration states
used by constrained HDGR inference:

    [committed prefix | MASK suffix]

Two complementary losses are implemented:

1. Multi-positive branch loss over the *full training candidate Trie*.
2. Inference-exact cumulative prefix-risk loss against hard valid candidate
   prefixes mined from the current in-batch candidate set.

The hard-prefix scorer is fully differentiable with respect to the HDGR model,
but the negative selection itself is stop-gradient.  This avoids attempting to
backpropagate through discrete beam search while still optimizing the exact
per-level scores used by inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


TREE_RISK_ASSET_VERSION = 1


def _as_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def build_concentrate_attention_mask(
    mask_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Local copy of the HDGR additive attention mask helper.

    Keeping this helper here avoids a circular import with ``retriever_gpt``.
    """

    device = mask_tokens.device
    _, length = mask_tokens.shape
    masked = torch.zeros_like(mask_tokens, dtype=torch.float32) - 10000.0
    visible = torch.zeros_like(mask_tokens, dtype=torch.float32)
    attention = torch.where(mask_tokens == int(mask_token_id), masked, visible)
    attention = attention.unsqueeze(1).repeat_interleave(repeats=length, dim=1)
    eye = torch.eye(length, device=device, dtype=torch.bool).unsqueeze(0)
    attention = attention.masked_fill(eye, 0.0)
    padding = (1.0 - valid_mask.unsqueeze(1).float()) * -10000.0
    return torch.clamp(attention + padding, -10000.0, 0.0)


@dataclass
class CompactTrieAsset:
    """Compact immutable Trie and query-positive path supervision."""

    code_length: int
    children_offsets: np.ndarray
    children_tokens: np.ndarray
    children_nodes: np.ndarray
    qid_positive_paths: Mapping[int, np.ndarray]
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "CompactTrieAsset":
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("version", -1))
        if version != TREE_RISK_ASSET_VERSION:
            raise ValueError(
                f"Unsupported Tree-Risk asset version {version}; "
                f"expected {TREE_RISK_ASSET_VERSION}: {path}"
            )

        def _numpy(name: str, dtype: np.dtype) -> np.ndarray:
            value = payload[name]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype)

        raw_paths = payload.get("qid_positive_paths", {})
        qid_positive_paths: dict[int, np.ndarray] = {}
        for qid, paths in raw_paths.items():
            if torch.is_tensor(paths):
                paths = paths.detach().cpu().numpy()
            array = np.asarray(paths, dtype=np.int32)
            if array.ndim == 1:
                array = array[None, :]
            qid_positive_paths[int(qid)] = array

        asset = cls(
            code_length=int(payload["code_length"]),
            children_offsets=_numpy("children_offsets", np.int64),
            children_tokens=_numpy("children_tokens", np.int32),
            children_nodes=_numpy("children_nodes", np.int32),
            qid_positive_paths=qid_positive_paths,
            metadata=payload.get("metadata", {}),
        )
        asset.validate()
        return asset

    def validate(self) -> None:
        if self.code_length <= 0:
            raise ValueError("Tree-Risk asset code_length must be positive")
        if self.children_offsets.ndim != 1:
            raise ValueError("children_offsets must be one-dimensional")
        if self.children_tokens.ndim != 1 or self.children_nodes.ndim != 1:
            raise ValueError("children arrays must be one-dimensional")
        if self.children_tokens.size != self.children_nodes.size:
            raise ValueError("children_tokens and children_nodes size mismatch")
        if self.children_offsets.size < 2:
            raise ValueError("Tree-Risk asset contains no root node")
        if int(self.children_offsets[-1]) != int(self.children_tokens.size):
            raise ValueError("children_offsets terminal value mismatch")

    @property
    def num_nodes(self) -> int:
        return int(self.children_offsets.size - 1)

    def child_slice(self, node_id: int) -> tuple[np.ndarray, np.ndarray]:
        node_id = int(node_id)
        if node_id < 0 or node_id >= self.num_nodes:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
        start = int(self.children_offsets[node_id])
        end = int(self.children_offsets[node_id + 1])
        return self.children_tokens[start:end], self.children_nodes[start:end]

    def find_child_node(self, node_id: int, token_id: int) -> int:
        tokens, nodes = self.child_slice(node_id)
        if tokens.size == 0:
            return -1
        pos = int(np.searchsorted(tokens, int(token_id)))
        if pos >= tokens.size or int(tokens[pos]) != int(token_id):
            return -1
        return int(nodes[pos])

    def node_for_prefix(self, prefix: Sequence[int]) -> int:
        node = 0
        for token_id in prefix:
            node = self.find_child_node(node, int(token_id))
            if node < 0:
                return -1
        return node

    def legal_children(self, prefix: Sequence[int]) -> np.ndarray:
        node = self.node_for_prefix(prefix)
        if node < 0:
            return np.empty((0,), dtype=np.int32)
        tokens, _ = self.child_slice(node)
        return tokens

    def positive_paths(self, qid: int) -> np.ndarray | None:
        return self.qid_positive_paths.get(int(qid))

    def positive_children(self, qid: int, prefix: Sequence[int]) -> np.ndarray:
        paths = self.positive_paths(qid)
        level = len(prefix)
        if paths is None or level >= self.code_length:
            return np.empty((0,), dtype=np.int32)
        if level > 0:
            prefix_array = np.asarray(prefix, dtype=np.int32)
            matches = np.all(paths[:, :level] == prefix_array[None, :], axis=1)
            paths = paths[matches]
        if paths.size == 0:
            return np.empty((0,), dtype=np.int32)
        return np.unique(paths[:, level]).astype(np.int32, copy=False)


class MutableCompactTrieBuilder:
    """Memory-conscious incremental Trie builder used by the asset script."""

    def __init__(self) -> None:
        self.children: list[dict[int, int]] = [dict()]

    def add(self, sequence: Sequence[int]) -> None:
        node = 0
        for value in sequence:
            token = int(value)
            child = self.children[node].get(token)
            if child is None:
                child = len(self.children)
                self.children[node][token] = child
                self.children.append(dict())
            node = child

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = [0]
        flat_tokens: list[int] = []
        flat_nodes: list[int] = []
        for child_map in self.children:
            for token, node in sorted(child_map.items()):
                flat_tokens.append(int(token))
                flat_nodes.append(int(node))
            offsets.append(len(flat_tokens))
        return (
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(flat_tokens, dtype=torch.int32),
            torch.tensor(flat_nodes, dtype=torch.int32),
        )


def save_tree_risk_asset(
    path: str | Path,
    *,
    code_length: int,
    trie_builder: MutableCompactTrieBuilder,
    qid_positive_paths: Mapping[int, Sequence[Sequence[int]]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    offsets, tokens, nodes = trie_builder.finalize()
    packed_paths: dict[int, torch.Tensor] = {}
    for qid, paths in qid_positive_paths.items():
        unique = sorted({tuple(int(v) for v in path) for path in paths})
        if not unique:
            continue
        packed_paths[int(qid)] = torch.tensor(unique, dtype=torch.int32)
    payload = {
        "version": TREE_RISK_ASSET_VERSION,
        "code_length": int(code_length),
        "children_offsets": offsets,
        "children_tokens": tokens,
        "children_nodes": nodes,
        "qid_positive_paths": packed_paths,
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _timestep_for_level(code_length: int, time_step: int, level: int) -> int:
    return max((int(code_length) - int(level)) * int(time_step) // max(int(code_length), 1), 1)


def _prefix_regen_logits(
    owner: Any,
    *,
    target_paths: torch.Tensor,
    levels: torch.Tensor,
    cond_prefix: torch.Tensor,
    query_indices: torch.Tensor,
    use_cfg: bool,
) -> torch.Tensor:
    """Return current-level logits on exact HDGR prefix-regeneration states."""

    device = target_paths.device
    rows, length = target_paths.shape
    xt = target_paths.clone()
    for row, level in enumerate(levels.detach().cpu().tolist()):
        xt[row, int(level):] = int(owner.mask_token_id)
    tokens_for_embed = xt.clone()
    tokens_for_embed[tokens_for_embed == int(owner.mask_token_id)] = 0
    valid_mask = torch.ones_like(xt, dtype=torch.long, device=device)
    attention = build_concentrate_attention_mask(
        xt, valid_mask, int(owner.mask_token_id)
    )
    timesteps = torch.tensor(
        [
            _timestep_for_level(length, int(owner.time_step), int(level))
            for level in levels.detach().cpu().tolist()
        ],
        device=device,
        dtype=torch.long,
    )
    cond_rows = cond_prefix.index_select(0, query_indices)
    out_c = owner.id_generator(
        tokens=tokens_for_embed,
        mask_tokens=xt,
        prefix=cond_rows,
        mask=attention,
        t=timesteps,
        labels=None,
    )
    logits_c = owner._mask_invalid_logits(out_c.logits.float())
    if not use_cfg:
        return logits_c

    uncond_base = owner.null_condition.expand(
        cond_prefix.size(0), -1, -1
    ).to(device=device, dtype=cond_prefix.dtype)
    uncond_rows = uncond_base.index_select(0, query_indices)
    out_u = owner.id_generator(
        tokens=tokens_for_embed,
        mask_tokens=xt,
        prefix=uncond_rows,
        mask=attention,
        t=timesteps,
        labels=None,
    )
    logits_u = owner._mask_invalid_logits(out_u.logits.float())
    guided = logits_u + float(owner.guidance_scale) * (logits_c - logits_u)
    if float(owner.temperature) != 1.0:
        guided = guided / max(float(owner.temperature), 1.0e-6)
    return guided


def _choose_branch_records(
    asset: CompactTrieAsset,
    target_tokens: torch.Tensor,
    h_qid: torch.Tensor,
    *,
    levels_per_sample: int,
    min_level: int,
    max_level: int,
    require_competition: bool,
    max_attempts: int = 8,
    max_samples: int = 0,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    records: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    target_cpu = target_tokens.detach().cpu().numpy().astype(np.int32, copy=False)
    qids = h_qid.detach().cpu().view(-1).tolist()
    max_level = min(int(max_level), int(target_tokens.size(1) - 1))
    min_level = max(0, int(min_level))
    if max_level < min_level:
        return records

    row_order = list(range(len(qids)))
    random.shuffle(row_order)
    if int(max_samples) > 0:
        row_order = row_order[: int(max_samples)]
    for row in row_order:
        path = target_cpu[row]
        qid = qids[row]
        chosen: set[int] = set()
        candidate_levels = list(range(min_level, max_level + 1))
        random.shuffle(candidate_levels)
        for level in candidate_levels[: max(max_attempts, levels_per_sample)]:
            prefix = path[:level]
            legal = asset.legal_children(prefix)
            if legal.size == 0:
                continue
            positive = asset.positive_children(int(qid), prefix)
            if positive.size == 0:
                positive = np.asarray([int(path[level])], dtype=np.int32)
            positive = np.intersect1d(positive, legal, assume_unique=False)
            if positive.size == 0:
                continue
            if require_competition and legal.size <= positive.size:
                continue
            records.append((row, level, legal, positive))
            chosen.add(level)
            if len(chosen) >= int(levels_per_sample):
                break
    return records


def compute_multi_positive_branch_loss(
    owner: Any,
    asset: CompactTrieAsset,
    *,
    target_tokens: torch.Tensor,
    h_qid: torch.Tensor,
    cond_prefix: torch.Tensor,
    levels_per_sample: int,
    min_level: int,
    max_level: int,
    temperature: float,
    use_cfg: bool,
    require_competition: bool,
    max_samples: int = 0,
) -> dict[str, torch.Tensor]:
    device = target_tokens.device
    zero = target_tokens.new_zeros((), dtype=torch.float32)
    records = _choose_branch_records(
        asset,
        target_tokens,
        h_qid,
        levels_per_sample=levels_per_sample,
        min_level=min_level,
        max_level=max_level,
        require_competition=require_competition,
        max_samples=max_samples,
    )
    if not records:
        return {
            "loss": zero,
            "accuracy": zero,
            "positive_mass": zero,
            "count": zero,
        }

    row_indices = torch.tensor([r[0] for r in records], device=device, dtype=torch.long)
    levels = torch.tensor([r[1] for r in records], device=device, dtype=torch.long)
    paths = target_tokens.index_select(0, row_indices)
    logits = _prefix_regen_logits(
        owner,
        target_paths=paths,
        levels=levels,
        cond_prefix=cond_prefix,
        query_indices=row_indices,
        use_cfg=use_cfg,
    )

    losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    temp = max(float(temperature), 1.0e-6)
    for idx, (_, level, legal_np, positive_np) in enumerate(records):
        legal = torch.as_tensor(legal_np, device=device, dtype=torch.long)
        positive = torch.as_tensor(positive_np, device=device, dtype=torch.long)
        scores = logits[idx, int(level)].index_select(0, legal) / temp
        positive_mask = torch.isin(legal, positive)
        if not bool(positive_mask.any().item()):
            continue
        log_denom = torch.logsumexp(scores, dim=0)
        log_numer = torch.logsumexp(scores[positive_mask], dim=0)
        losses.append(log_denom - log_numer)
        probabilities = torch.softmax(scores, dim=0)
        masses.append(probabilities[positive_mask].sum())
        best_token = legal[torch.argmax(scores)]
        accuracies.append(torch.isin(best_token.view(1), positive).float().squeeze(0))

    if not losses:
        return {
            "loss": zero,
            "accuracy": zero,
            "positive_mass": zero,
            "count": zero,
        }
    return {
        "loss": torch.stack(losses).mean(),
        "accuracy": torch.stack(accuracies).mean(),
        "positive_mass": torch.stack(masses).mean(),
        "count": torch.tensor(float(len(losses)), device=device),
    }


def _select_risk_queries(
    h_qid: torch.Tensor,
    max_queries: int,
) -> torch.Tensor:
    qids = h_qid.detach().cpu().view(-1).tolist()
    first_rows: dict[int, int] = {}
    for row, qid in enumerate(qids):
        first_rows.setdefault(int(qid), int(row))
    rows = list(first_rows.values())
    random.shuffle(rows)
    rows = rows[: max(0, int(max_queries))]
    return torch.tensor(rows, device=h_qid.device, dtype=torch.long)


def _hard_negative_rows(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    h_qid: torch.Tensor,
    selected_rows: torch.Tensor,
    k: int,
) -> list[torch.Tensor]:
    """Mine stop-gradient in-batch hard negatives on one compute device.

    ``h_qid`` and the sampled row indices commonly remain on CPU because they
    originate from dataloader metadata, while query/candidate embeddings are
    already on CUDA.  PyTorch requires an ``index_select`` index tensor to live
    on the same device as the indexed tensor, so normalize all mining operands
    to the query-embedding device here.  Returned row ids are still ordinary
    batch row indices and may safely be converted back to Python integers by
    the caller.
    """

    with torch.no_grad():
        device = q_emb.device
        selected_on_device = selected_rows.to(device=device, dtype=torch.long)
        p_emb_on_device = p_emb.to(device=device)
        all_qids = h_qid.view(-1).to(device=device)

        similarity = (
            q_emb.index_select(0, selected_on_device).float()
            @ p_emb_on_device.float().T
        )
        selected_qids = all_qids.index_select(0, selected_on_device)
        same_qid = selected_qids[:, None] == all_qids[None, :]
        similarity = similarity.masked_fill(same_qid, -float("inf"))

        k = min(int(k), max(0, p_emb_on_device.size(0) - 1))
        if k <= 0:
            return [
                torch.empty((0,), device=device, dtype=torch.long)
                for _ in range(selected_on_device.numel())
            ]
        top_rows = torch.topk(
            similarity,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        return [top_rows[i] for i in range(top_rows.size(0))]


def _unique_paths(paths: Iterable[Sequence[int]]) -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for path in paths:
        key = tuple(int(v) for v in path)
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def _score_prefix_candidates(
    owner: Any,
    *,
    cond_prefix: torch.Tensor,
    groups: list[dict[str, Any]],
    use_cfg: bool,
) -> list[torch.Tensor]:
    """Score valid prefixes using the exact sequential regeneration score.

    Each candidate at level ``l`` is scored as the sum of the log-probability
    assigned at regeneration steps ``0..l``.  All rows are packed into one
    vectorized model call (two when CFG is enabled).
    """

    if not groups:
        return []
    device = cond_prefix.device
    length = int(groups[0]["paths"].size(1))

    packed_paths: list[torch.Tensor] = []
    packed_levels: list[int] = []
    packed_query_rows: list[int] = []
    packed_target_tokens: list[int] = []
    packed_candidate_ids: list[int] = []
    candidate_offsets: list[tuple[int, int]] = []
    global_candidate = 0

    for group in groups:
        paths: torch.Tensor = group["paths"]
        level = int(group["level"])
        query_row = int(group["query_row"])
        start = global_candidate
        for candidate in range(paths.size(0)):
            path = paths[candidate]
            for step in range(level + 1):
                packed_paths.append(path)
                packed_levels.append(step)
                packed_query_rows.append(query_row)
                packed_target_tokens.append(int(path[step].item()))
                packed_candidate_ids.append(global_candidate)
            global_candidate += 1
        candidate_offsets.append((start, global_candidate))

    all_paths = torch.stack(packed_paths, dim=0).to(device=device, dtype=torch.long)
    all_levels = torch.tensor(packed_levels, device=device, dtype=torch.long)
    all_query_rows = torch.tensor(packed_query_rows, device=device, dtype=torch.long)
    all_targets = torch.tensor(packed_target_tokens, device=device, dtype=torch.long)
    logits = _prefix_regen_logits(
        owner,
        target_paths=all_paths,
        levels=all_levels,
        cond_prefix=cond_prefix,
        query_indices=all_query_rows,
        use_cfg=use_cfg,
    )
    row_ids = torch.arange(logits.size(0), device=device)
    step_logits = logits[row_ids, all_levels]
    log_probs = F.log_softmax(step_logits, dim=-1)
    token_logp = log_probs[row_ids, all_targets]
    candidate_ids = torch.tensor(packed_candidate_ids, device=device, dtype=torch.long)
    candidate_scores = torch.zeros(global_candidate, device=device, dtype=torch.float32)
    candidate_scores.scatter_add_(0, candidate_ids, token_logp.float())

    return [candidate_scores[start:end] for start, end in candidate_offsets]


def compute_inbatch_prefix_risk_loss(
    owner: Any,
    asset: CompactTrieAsset | None,
    *,
    target_tokens: torch.Tensor,
    h_qid: torch.Tensor,
    cond_prefix: torch.Tensor,
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    max_queries: int,
    hard_negatives: int,
    max_positive_paths: int,
    min_level: int,
    max_level: int,
    margin: float,
    temperature: float,
    use_cfg: bool,
) -> dict[str, torch.Tensor]:
    device = target_tokens.device
    zero = target_tokens.new_zeros((), dtype=torch.float32)
    selected = _select_risk_queries(h_qid, max_queries)
    if selected.numel() == 0:
        return {"loss": zero, "accuracy": zero, "margin": zero, "count": zero}
    negative_rows = _hard_negative_rows(
        q_emb, p_emb, h_qid, selected, hard_negatives
    )

    target_cpu = target_tokens.detach().cpu().numpy().astype(np.int32, copy=False)
    qids_cpu = h_qid.detach().cpu().view(-1).tolist()
    min_level = max(0, int(min_level))
    max_level = min(int(max_level), int(target_tokens.size(1) - 1))
    if max_level < min_level:
        return {"loss": zero, "accuracy": zero, "margin": zero, "count": zero}

    groups: list[dict[str, Any]] = []
    for local_idx, query_row_tensor in enumerate(selected):
        query_row = int(query_row_tensor.item())
        qid = int(qids_cpu[query_row])
        level = random.randint(min_level, max_level)

        positive_candidates: list[Sequence[int]] = [target_cpu[query_row]]
        same_qid_rows = [
            idx for idx, other_qid in enumerate(qids_cpu) if int(other_qid) == qid
        ]
        positive_candidates.extend(target_cpu[idx] for idx in same_qid_rows)
        if asset is not None:
            asset_paths = asset.positive_paths(qid)
            if asset_paths is not None:
                positive_candidates.extend(asset_paths)
        positive_paths = _unique_paths(positive_candidates)[: max(1, int(max_positive_paths))]

        negative_paths = _unique_paths(
            target_cpu[int(row.item())] for row in negative_rows[local_idx]
        )
        positive_prefixes = {tuple(path[: level + 1]) for path in positive_paths}
        negative_paths = [
            path for path in negative_paths
            if tuple(path[: level + 1]) not in positive_prefixes
        ]
        if not negative_paths:
            continue
        all_paths = positive_paths + negative_paths
        groups.append(
            {
                "query_row": query_row,
                "level": level,
                "num_positive": len(positive_paths),
                "paths": torch.tensor(all_paths, device=device, dtype=torch.long),
            }
        )

    if not groups:
        return {"loss": zero, "accuracy": zero, "margin": zero, "count": zero}

    scored_groups = _score_prefix_candidates(
        owner,
        cond_prefix=cond_prefix,
        groups=groups,
        use_cfg=use_cfg,
    )
    losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    temp = max(float(temperature), 1.0e-6)
    for group, scores in zip(groups, scored_groups):
        n_pos = int(group["num_positive"])
        positive = scores[:n_pos]
        negative = scores[n_pos:]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        positive_anchor = torch.logsumexp(positive / temp, dim=0) * temp
        relative = (negative - positive_anchor + float(margin)) / temp
        losses.append(torch.logaddexp(relative.new_zeros(()), torch.logsumexp(relative, dim=0)))
        best_positive = positive.max()
        best_negative = negative.max()
        accuracies.append((best_positive > best_negative).float())
        margins.append(best_positive - best_negative)

    if not losses:
        return {"loss": zero, "accuracy": zero, "margin": zero, "count": zero}
    return {
        "loss": torch.stack(losses).mean(),
        "accuracy": torch.stack(accuracies).mean(),
        "margin": torch.stack(margins).mean(),
        "count": torch.tensor(float(len(losses)), device=device),
    }


def tree_risk_weight_scale(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    step = int(step)
    warmup_steps = max(0, int(warmup_steps))
    ramp_steps = max(0, int(ramp_steps))
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, float(step - warmup_steps + 1) / float(ramp_steps))
