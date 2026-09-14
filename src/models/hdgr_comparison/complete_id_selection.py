"""Chunked scoring utilities for one-pass complete-ID selection."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def deterministic_topk_by_index(
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lexicographic top-k under ``(-score, candidate_index)``.

    ``torch.topk`` does not define a deterministic ordering for equal values.
    P2L streaming repeatedly merges partial top-k sets, so an unstable tie at
    the boundary can make the selected identifier depend on chunk size even
    when every candidate score is identical to full materialization.  This
    helper makes the secondary key explicit: lower global candidate-row index
    wins an exact score tie.

    ``scores`` and ``indices`` may be 1-D or 2-D and must have the same shape.
    The returned tensors preserve that dimensionality and are ordered exactly
    as a full stable sort by descending score and ascending candidate index.
    """
    if scores.shape != indices.shape:
        raise ValueError("scores and indices must have identical shapes")
    if scores.dim() not in (1, 2):
        raise ValueError("scores and indices must be 1-D or 2-D")
    if scores.numel() == 0:
        raise ValueError("scores must contain at least one candidate")

    squeeze = scores.dim() == 1
    if squeeze:
        scores = scores.unsqueeze(0)
        indices = indices.unsqueeze(0)

    keep = min(max(1, int(k)), int(scores.size(1)))

    # First establish the secondary order (candidate index ascending).  The
    # subsequent stable score sort preserves this order inside exact-score
    # ties, giving the lexicographic order (-score, candidate_index).
    by_index = torch.argsort(indices, dim=1, descending=False, stable=True)
    scores_by_index = scores.gather(1, by_index)
    indices_by_index = indices.gather(1, by_index)
    by_score = torch.argsort(
        scores_by_index, dim=1, descending=True, stable=True
    )[:, :keep]
    top_scores = scores_by_index.gather(1, by_score)
    top_indices = indices_by_index.gather(1, by_score)

    if squeeze:
        return top_scores.squeeze(0), top_indices.squeeze(0)
    return top_scores, top_indices


def sample_prefix_neighbor_codes(
    sorted_legal_codes: np.ndarray,
    target_codes: np.ndarray,
    *,
    prefix_depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    candidates_per_depth: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample legal IDs sharing prefixes of different depths with targets."""
    if sorted_legal_codes.ndim != 2 or target_codes.ndim != 2:
        raise ValueError("sorted legal codes and targets must both be 2-D")
    if sorted_legal_codes.shape[1] != target_codes.shape[1]:
        raise ValueError("legal and target code lengths differ")
    rng = rng or np.random.default_rng()
    rows: list[np.ndarray] = []
    total = int(sorted_legal_codes.shape[0])
    length = int(sorted_legal_codes.shape[1])
    for target in target_codes:
        for depth in prefix_depths:
            depth = int(depth)
            if depth < 0 or depth >= length:
                continue
            lo, hi = 0, total
            for level in range(depth):
                column = sorted_legal_codes[lo:hi, level]
                value = int(target[level])
                left = int(np.searchsorted(column, value, side="left"))
                right = int(np.searchsorted(column, value, side="right"))
                lo, hi = lo + left, lo + right
                if lo >= hi:
                    break
            if lo >= hi:
                continue
            count = min(max(1, int(candidates_per_depth)), hi - lo)
            indices = rng.choice(hi - lo, size=count, replace=False) + lo
            sampled = np.asarray(sorted_legal_codes[indices])
            sampled = sampled[np.any(sampled != target[None, :], axis=1)]
            if sampled.size:
                rows.append(sampled)
    if not rows:
        return np.empty((0, length), dtype=sorted_legal_codes.dtype)
    return np.unique(np.concatenate(rows, axis=0), axis=0)


def inbatch_complete_id_ranking_loss(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrast queries against complete legal IDs supplied by the batch.

    Duplicate target IDs are treated as multiple positives instead of false
    negatives. Returns the multi-positive loss and top-1 complete-ID accuracy.
    """
    if logits.dim() != 3 or target_token_ids.dim() != 2:
        raise ValueError("logits must be [B,L,V] and targets must be [B,L]")
    batch, length, vocab = logits.shape
    if target_token_ids.shape != (batch, length):
        raise ValueError("target_token_ids must match logits batch and length")
    if batch < 2:
        zero = logits.sum() * 0.0
        return zero, zero.detach()
    if bool(((target_token_ids < 0) | (target_token_ids >= vocab)).any().item()):
        raise ValueError("target_token_ids contains an out-of-vocabulary token")

    log_probs = F.log_softmax(logits.float(), dim=-1)
    sequence_scores = torch.zeros((batch, batch), device=logits.device)
    for level in range(length):
        sequence_scores += log_probs[:, level, :].index_select(
            1, target_token_ids[:, level]
        )
    sequence_scores = sequence_scores / max(float(temperature), 1.0e-6)
    positive_mask = (target_token_ids[:, None, :] == target_token_ids[None, :, :]).all(-1)
    positive_scores = sequence_scores.masked_fill(~positive_mask, float("-inf"))
    loss = -(torch.logsumexp(positive_scores, dim=1) - torch.logsumexp(sequence_scores, dim=1)).mean()
    prediction = sequence_scores.argmax(dim=1)
    accuracy = positive_mask.gather(1, prediction[:, None]).float().mean()
    return loss, accuracy


def sampled_hard_negative_ranking_loss(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    legal_candidate_token_ids: torch.Tensor,
    *,
    num_hard_negatives: int = 32,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mine high-scoring legal IDs and rank the target above them."""
    batch, length, vocab = logits.shape
    candidates = legal_candidate_token_ids.to(device=logits.device, dtype=torch.long)
    if candidates.dim() != 2 or candidates.size(1) != length:
        raise ValueError("legal candidates must be [N,L]")
    if bool(((candidates < 0) | (candidates >= vocab)).any().item()):
        raise ValueError("legal candidates contain an out-of-vocabulary token")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    candidate_scores = torch.zeros((batch, candidates.size(0)), device=logits.device)
    positive_scores = torch.zeros(batch, device=logits.device)
    for level in range(length):
        candidate_scores += log_probs[:, level, :].index_select(1, candidates[:, level])
        positive_scores += log_probs[:, level, :].gather(
            1, target_token_ids[:, level : level + 1]
        ).squeeze(1)
    is_positive = (target_token_ids[:, None, :] == candidates[None, :, :]).all(-1)
    candidate_scores = candidate_scores.masked_fill(is_positive, float("-inf"))
    keep = min(max(1, int(num_hard_negatives)), candidates.size(0))
    hard_scores = candidate_scores.topk(keep, dim=1).values
    ranking_logits = torch.cat((positive_scores[:, None], hard_scores), dim=1)
    ranking_logits = ranking_logits / max(float(temperature), 1.0e-6)
    labels = torch.zeros(batch, device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(ranking_logits, labels)
    accuracy = (ranking_logits.argmax(dim=1) == 0).float().mean()
    margin = (positive_scores - hard_scores[:, 0]).mean()
    return loss, accuracy, margin


def topk_complete_id_scores(
    log_probs: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k candidate row indices and summed position log-probabilities.

    ``log_probs`` is ``[B,L,V]`` and candidates are legal complete IDs with
    shape ``[N,L]``. Candidate rows are streamed in chunks.  Selection uses the
    deterministic lexicographic rule ``(-score, global_candidate_row)`` so a
    chunked run is identical to full materialization even at exact-score ties.
    """
    if log_probs.dim() != 3 or candidate_token_ids.dim() != 2:
        raise ValueError("log_probs must be [B,L,V] and candidates must be [N,L]")
    batch, length, vocab = log_probs.shape
    if candidate_token_ids.size(1) != length:
        raise ValueError(
            f"candidate length mismatch: expected {length}, got {candidate_token_ids.size(1)}"
        )
    num_candidates = int(candidate_token_ids.size(0))
    keep = min(max(1, int(k)), num_candidates)
    if num_candidates == 0:
        raise ValueError("candidate_token_ids must contain at least one legal ID")
    chunk_size = max(1, int(chunk_size))
    best_scores = torch.empty((batch, 0), device=log_probs.device, dtype=log_probs.dtype)
    best_indices = torch.empty((batch, 0), device=log_probs.device, dtype=torch.long)

    for start in range(0, num_candidates, chunk_size):
        tokens = candidate_token_ids[start : start + chunk_size].to(
            device=log_probs.device, dtype=torch.long, non_blocking=True
        )
        if bool(((tokens < 0) | (tokens >= vocab)).any().item()):
            raise ValueError("candidate_token_ids contains an out-of-vocabulary token")
        scores = torch.zeros(
            (batch, tokens.size(0)), device=log_probs.device, dtype=log_probs.dtype
        )
        for level in range(length):
            scores += log_probs[:, level, :].index_select(1, tokens[:, level])
        indices = torch.arange(start, start + tokens.size(0), device=log_probs.device)
        indices = indices.unsqueeze(0).expand(batch, -1)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_indices = torch.cat((best_indices, indices), dim=1)
        best_scores, best_indices = deterministic_topk_by_index(
            merged_scores, merged_indices, keep
        )

    return best_indices, best_scores
