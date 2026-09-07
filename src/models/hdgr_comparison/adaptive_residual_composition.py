"""Safe counterfactual residual composition for Code-Tied GPT-HDGR.

Semantic IDs produced by ResidualVQ are compositional: every level contributes
one codeword that explains the residual left by the committed prefix.  This
module learns a state-dependent mixture over several residual-explanation
strengths while treating the zero-strength generator as an explicit safety
baseline.

V2 removes the soft-oracle objective that can collapse to a uniform mixture.
It optimizes the exact utility of the effective (mixture-induced) strength,
minimizes regret to the best counterfactual expert, and adds a no-harm penalty
whenever the learned composition underperforms the unmodified generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompositionOutput:
    expert_logits: torch.Tensor
    expert_probabilities: torch.Tensor
    strength: torch.Tensor
    normalized_entropy: torch.Tensor
    generator_margin: torch.Tensor
    residual_margin: torch.Tensor
    generator_residual_agreement: torch.Tensor


def _sanitize_masked_scores(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite scores and a mask that excludes non-finite entries.

    Multiplying +/-inf by a zero mask still produces NaN.  Therefore invalid
    values must be replaced *before* moment/statistic computation.
    """
    finite_mask = mask & torch.isfinite(values)
    safe = torch.nan_to_num(
        values.float(), nan=0.0, posinf=float(clip_value), neginf=-float(clip_value)
    ).clamp(min=-float(clip_value), max=float(clip_value))
    safe = torch.where(finite_mask, safe, torch.zeros_like(safe))
    return safe, finite_mask


def _masked_moments(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(dtype=values.dtype)
    count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    safe = torch.where(mask, values, torch.zeros_like(values))
    mean = safe.sum(dim=-1, keepdim=True) / count
    centered = torch.where(mask, safe - mean, torch.zeros_like(safe))
    var = centered.square().sum(dim=-1, keepdim=True) / count
    return mean, var.clamp_min(1.0e-8).sqrt()


def _masked_top2_margin(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    very_small = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, very_small)
    k = min(2, values.size(-1))
    top = torch.topk(masked, k=k, dim=-1).values
    if k == 1:
        return torch.zeros_like(top[:, 0])
    finite = torch.isfinite(top[:, 1])
    return torch.where(finite, top[:, 0] - top[:, 1], torch.zeros_like(top[:, 0]))


class AdaptiveResidualComposer(nn.Module):
    """Predict a query/prefix/level-dependent residual-composition strength."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_levels: int,
        expert_strengths: Sequence[float],
        hidden_dim: int = 128,
        level_dim: int = 24,
        dropout: float = 0.1,
        init_strength: float = 10.0,
        init_probabilities: Sequence[float] | None = None,
        detach_state_features: bool = True,
    ) -> None:
        super().__init__()
        strengths = torch.as_tensor(list(expert_strengths), dtype=torch.float32)
        if strengths.ndim != 1 or strengths.numel() < 2:
            raise ValueError("expert_strengths must contain at least two values")
        if not torch.isfinite(strengths).all():
            raise ValueError("expert_strengths must be finite")
        if bool((strengths[1:] < strengths[:-1]).any().item()):
            raise ValueError("expert_strengths must be sorted in ascending order")
        if int(num_levels) <= 0:
            raise ValueError("num_levels must be positive")

        self.embedding_dim = int(embedding_dim)
        self.num_levels = int(num_levels)
        self.hidden_dim = int(hidden_dim)
        self.detach_state_features = bool(detach_state_features)
        self.register_buffer("expert_strengths", strengths, persistent=True)

        vector_dim = max(24, self.hidden_dim // 3)
        self.query_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, vector_dim), nn.LayerNorm(vector_dim), nn.GELU()
        )
        self.residual_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, vector_dim), nn.LayerNorm(vector_dim), nn.GELU()
        )
        self.prefix_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, vector_dim), nn.LayerNorm(vector_dim), nn.GELU()
        )
        self.level_embedding = nn.Embedding(self.num_levels, int(level_dim))

        # Scalar features:
        # level fraction, generator entropy, generator margin, residual margin,
        # generator/residual agreement, residual norm, prefix norm,
        # query-prefix cosine, generator std, residual-score std.
        scalar_dim = 10
        self.scalar_projector = nn.Sequential(
            nn.Linear(scalar_dim, vector_dim), nn.LayerNorm(vector_dim), nn.GELU()
        )
        trunk_in = vector_dim * 4 + int(level_dim)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.expert_head = nn.Linear(self.hidden_dim, strengths.numel())
        self.gradient_clip_value = 1.0

        # Start near the validated weak-composition regime while keeping the
        # safety expert and stronger experts reachable.  Unlike the V1 +/-4
        # logits, this prior does not saturate softmax at initialization.
        nn.init.zeros_(self.expert_head.weight)
        if init_probabilities is None:
            closest = int(torch.argmin((strengths - float(init_strength)).abs()).item())
            prior = torch.full_like(strengths, 0.05 / float(max(strengths.numel() - 1, 1)))
            prior[closest] = 0.95
        else:
            prior = torch.as_tensor(list(init_probabilities), dtype=torch.float32)
            if prior.shape != strengths.shape:
                raise ValueError("init_probabilities must match expert_strengths")
            if not torch.isfinite(prior).all() or bool((prior < 0).any().item()):
                raise ValueError("init_probabilities must be finite and non-negative")
            if float(prior.sum().item()) <= 0.0:
                raise ValueError("init_probabilities must have positive mass")
            prior = prior / prior.sum()
        prior_logits = prior.clamp_min(1.0e-6).log()
        prior_logits = prior_logits - prior_logits.mean()
        with torch.no_grad():
            self.expert_head.bias.copy_(prior_logits)

        # The controller is intentionally isolated from rare non-finite or
        # extremely large auxiliary gradients.  This does not clip the native
        # HDGR denoising gradients.
        for parameter in self.parameters():
            parameter.register_hook(self._sanitize_gradient)

    def _sanitize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            gradient, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=-self.gradient_clip_value, max=self.gradient_clip_value)

    def forward(
        self,
        *,
        query_embedding: torch.Tensor,
        prefix_reconstruction: torch.Tensor,
        generator_scores: torch.Tensor,
        residual_scores: torch.Tensor,
        level: int | torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> CompositionOutput:
        if query_embedding.ndim != 2 or prefix_reconstruction.shape != query_embedding.shape:
            raise ValueError("query_embedding and prefix_reconstruction must be aligned [N,D]")
        if generator_scores.ndim != 2 or residual_scores.shape != generator_scores.shape:
            raise ValueError("generator_scores and residual_scores must be aligned [N,K]")
        if generator_scores.size(0) != query_embedding.size(0):
            raise ValueError("state and score batch sizes do not match")

        device = query_embedding.device
        query = query_embedding.float()
        prefix = prefix_reconstruction.float()
        generator = generator_scores.float()
        residual_score = residual_scores.float()
        if valid_mask is None:
            mask = torch.ones_like(generator, dtype=torch.bool, device=device)
        else:
            mask = valid_mask.to(device=device, dtype=torch.bool)
            if mask.shape != generator.shape:
                raise ValueError("valid_mask must match generator_scores")
        # Every row must contain at least one candidate.
        empty = ~mask.any(dim=-1)
        if bool(empty.any().item()):
            mask = mask.clone()
            mask[empty, 0] = True

        if self.detach_state_features:
            query = query.detach()
            prefix = prefix.detach()
            generator = generator.detach()
            residual_score = residual_score.detach()

        query = torch.nan_to_num(query, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        prefix = torch.nan_to_num(prefix, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        generator, generator_mask = _sanitize_masked_scores(
            generator, mask, clip_value=1.0e4
        )
        residual_score, residual_mask = _sanitize_masked_scores(
            residual_score, mask, clip_value=20.0
        )
        mask = generator_mask & residual_mask
        empty = ~mask.any(dim=-1)
        if bool(empty.any().item()):
            mask = mask.clone()
            generator = generator.clone()
            residual_score = residual_score.clone()
            mask[empty, 0] = True
            generator[empty, 0] = 0.0
            residual_score[empty, 0] = 0.0

        query = F.normalize(query.float(), dim=-1)
        prefix = prefix.float().clamp(min=-1.0e4, max=1.0e4)
        residual = torch.nan_to_num(
            query - prefix, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)

        g_mean, g_std = _masked_moments(generator, mask)
        r_mean, r_std = _masked_moments(residual_score, mask)
        g_norm = ((generator - g_mean) / g_std).clamp(min=-20.0, max=20.0)
        r_norm = ((residual_score - r_mean) / r_std).clamp(min=-20.0, max=20.0)

        weight = mask.to(generator.dtype)
        count = weight.sum(dim=-1).clamp_min(1.0)
        agreement = (g_norm * r_norm * weight).sum(dim=-1) / count

        very_small = torch.finfo(generator.dtype).min
        masked_generator = generator.masked_fill(~mask, very_small)
        probabilities = torch.softmax(masked_generator, dim=-1) * weight
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        entropy = -(probabilities.clamp_min(1.0e-12).log() * probabilities).sum(dim=-1)
        entropy = entropy / count.clamp_min(2.0).log()

        generator_margin = _masked_top2_margin(g_norm, mask)
        residual_margin = _masked_top2_margin(r_norm, mask)
        residual_norm = residual.norm(dim=-1) / math.sqrt(max(self.embedding_dim, 1))
        prefix_norm = prefix.norm(dim=-1) / math.sqrt(max(self.embedding_dim, 1))
        query_prefix_cos = F.cosine_similarity(query, prefix, dim=-1, eps=1.0e-6)

        if isinstance(level, int):
            level_ids = torch.full(
                (query.size(0),), int(level), device=device, dtype=torch.long
            )
        else:
            level_ids = level.to(device=device, dtype=torch.long).view(-1)
            if level_ids.numel() == 1 and query.size(0) > 1:
                level_ids = level_ids.expand(query.size(0))
        if level_ids.numel() != query.size(0):
            raise ValueError("level must be scalar or have one entry per row")
        level_ids = level_ids.clamp(min=0, max=self.num_levels - 1)
        level_fraction = level_ids.float() / float(max(self.num_levels - 1, 1))

        scalar = torch.stack(
            [
                level_fraction,
                entropy,
                generator_margin,
                residual_margin,
                agreement,
                residual_norm,
                prefix_norm,
                query_prefix_cos,
                g_std.squeeze(-1),
                r_std.squeeze(-1),
            ],
            dim=-1,
        )
        scalar = torch.nan_to_num(
            scalar, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)
        state = torch.cat(
            [
                self.query_projector(query),
                self.residual_projector(residual),
                self.prefix_projector(prefix),
                self.scalar_projector(scalar),
                self.level_embedding(level_ids),
            ],
            dim=-1,
        )
        expert_logits = self.expert_head(self.trunk(state))
        expert_logits = torch.nan_to_num(
            expert_logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(min=-20.0, max=20.0)
        expert_probabilities = torch.softmax(expert_logits, dim=-1)
        strength = (
            expert_probabilities
            * self.expert_strengths.to(device=device, dtype=expert_probabilities.dtype)
        ).sum(dim=-1)
        return CompositionOutput(
            expert_logits=expert_logits,
            expert_probabilities=expert_probabilities,
            strength=strength,
            normalized_entropy=entropy,
            generator_margin=generator_margin,
            residual_margin=residual_margin,
            generator_residual_agreement=agreement,
        )


def composition_weight_scale(step: int, *, warmup_steps: int, ramp_steps: int) -> float:
    step = int(step)
    warmup_steps = max(0, int(warmup_steps))
    ramp_steps = max(0, int(ramp_steps))
    if step < warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (step - warmup_steps + 1) / float(ramp_steps)))


def _build_attention_mask(mask_tokens: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    batch, length = mask_tokens.shape
    visible = torch.zeros_like(mask_tokens, dtype=torch.float32)
    hidden = visible - 10000.0
    attention = torch.where(mask_tokens == int(mask_token_id), hidden, visible)
    attention = attention.unsqueeze(1).expand(batch, length, length).clone()
    eye = torch.eye(length, device=mask_tokens.device, dtype=torch.bool).unsqueeze(0)
    attention.masked_fill_(eye, 0.0)
    return attention


def _exact_prefix_logits(
    owner: Any,
    *,
    target_paths: torch.Tensor,
    levels: torch.Tensor,
    cond_prefix: torch.Tensor,
    query_indices: torch.Tensor,
    use_cfg: bool,
) -> torch.Tensor:
    """Evaluate the native HDGR denoiser on exact prefix-regeneration states."""
    device = target_paths.device
    rows, length = target_paths.shape
    xt = target_paths.clone()
    for row, level in enumerate(levels.detach().cpu().tolist()):
        xt[row, int(level):] = int(owner.mask_token_id)
    tokens_for_embed = xt.clone()
    tokens_for_embed[tokens_for_embed == int(owner.mask_token_id)] = 0
    attention = _build_attention_mask(xt, int(owner.mask_token_id))
    timesteps = torch.tensor(
        [
            max((length - int(level)) * int(owner.time_step) // max(length, 1), 1)
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
        if float(owner.temperature) != 1.0:
            logits_c = logits_c / max(float(owner.temperature), 1.0e-6)
        return logits_c

    uncond = owner.null_condition.expand(cond_prefix.size(0), -1, -1).to(
        device=device, dtype=cond_prefix.dtype
    )
    out_u = owner.id_generator(
        tokens=tokens_for_embed,
        mask_tokens=xt,
        prefix=uncond.index_select(0, query_indices),
        mask=attention,
        t=timesteps,
        labels=None,
    )
    logits_u = owner._mask_invalid_logits(out_u.logits.float())
    guided = logits_u + float(owner.guidance_scale) * (logits_c - logits_u)
    if float(owner.temperature) != 1.0:
        guided = guided / max(float(owner.temperature), 1.0e-6)
    return guided


def _sample_records(
    owner: Any,
    *,
    target_tokens: torch.Tensor,
    h_qid: torch.Tensor,
    asset: Any | None,
    max_queries: int,
    levels_per_query: int,
    min_level: int,
    max_level: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    target_cpu = target_tokens.detach().cpu().numpy().astype(np.int32, copy=False)
    qids = h_qid.detach().cpu().view(-1).tolist()
    first_rows: dict[int, int] = {}
    for row, qid in enumerate(qids):
        first_rows.setdefault(int(qid), int(row))
    rows = list(first_rows.values())
    random.shuffle(rows)
    if int(max_queries) > 0:
        rows = rows[: int(max_queries)]

    min_level = max(0, int(min_level))
    max_level = min(int(max_level), int(target_tokens.size(1) - 1))
    records: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for row in rows:
        levels = list(range(min_level, max_level + 1))
        random.shuffle(levels)
        for level in levels[: max(1, int(levels_per_query))]:
            path = target_cpu[row]
            if asset is None:
                level_size = int(owner.level_vocab_sizes[level].item())
                legal = owner.level_token_ids[level, :level_size].detach().cpu().numpy().astype(np.int32)
                positive = np.asarray([int(path[level])], dtype=np.int32)
            else:
                prefix = path[:level]
                legal = asset.legal_children(prefix)
                positive = asset.positive_children(int(qids[row]), prefix)
                if positive.size == 0:
                    positive = np.asarray([int(path[level])], dtype=np.int32)
                positive = np.intersect1d(positive, legal, assume_unique=False)
            if legal.size == 0 or positive.size == 0:
                continue
            records.append((row, level, legal, positive))
    return records


def _safe_result_template(zero: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "route_loss": zero,
        "branch_loss": zero,
        "regret_loss": zero,
        "no_harm_loss": zero,
        "clear_ce_loss": zero,
        "route_acc": zero,
        "clear_fraction": zero,
        "positive_gain_fraction": zero,
        "branch_acc": zero,
        "baseline_branch_acc": zero,
        "strength_mean": zero,
        "strength_std": zero,
        "oracle_strength_mean": zero,
        "expert0_probability": zero,
        "policy_entropy": zero,
        "utility_gain": zero,
        "best_utility_gain": zero,
        "utility_gap": zero,
        "expert0_win_rate": zero,
        "expert10_win_rate": zero,
        "expert50_win_rate": zero,
        "expert100_win_rate": zero,
        "count": zero,
        "clear_count": zero,
    }


def safe_counterfactual_policy_objective(
    *,
    expert_logits: torch.Tensor,
    expert_strengths: torch.Tensor,
    base_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    advantage_temperature: float,
    min_utility_gain: float,
    min_winner_gap: float,
    no_harm_weight: float,
    clear_ce_weight: float,
) -> dict[str, torch.Tensor]:
    """Exact safe objective for one legal query-prefix-level state.

    The policy is executed exactly as at inference: probabilities induce one
    effective continuous strength, and that strength is applied to scores before
    positive-set utility is measured.  The zero-strength expert is the safety
    baseline.  All score tensors are expected to be detached from the backbone.
    """
    if expert_logits.ndim != 1:
        raise ValueError("expert_logits must be [M]")
    if expert_strengths.ndim != 1 or expert_strengths.shape != expert_logits.shape:
        raise ValueError("expert_strengths must align with expert_logits")
    if base_scores.ndim != 1 or residual_scores.shape != base_scores.shape:
        raise ValueError("base_scores and residual_scores must be [K]")
    if positive_mask.shape != base_scores.shape or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean [K]")
    if not bool(positive_mask.any().item()):
        raise ValueError("positive_mask must contain at least one positive")

    tau = max(float(advantage_temperature), 1.0e-6)
    policy = torch.softmax(expert_logits, dim=-1)
    strengths = expert_strengths.to(device=policy.device, dtype=policy.dtype)
    effective_strength = (policy * strengths).sum()

    expert_scores = base_scores.unsqueeze(0) + strengths.unsqueeze(1) * residual_scores.unsqueeze(0)
    expert_utility = (
        torch.logsumexp(expert_scores[:, positive_mask], dim=-1)
        - torch.logsumexp(expert_scores, dim=-1)
    )
    expert_utility = torch.nan_to_num(
        expert_utility, nan=-50.0, posinf=0.0, neginf=-50.0
    ).clamp(min=-50.0, max=0.0)

    policy_scores = base_scores + effective_strength * residual_scores
    policy_utility = (
        torch.logsumexp(policy_scores[positive_mask], dim=0)
        - torch.logsumexp(policy_scores, dim=0)
    )
    policy_utility = torch.nan_to_num(
        policy_utility, nan=-50.0, posinf=0.0, neginf=-50.0
    ).clamp(min=-50.0, max=0.0)

    baseline_utility = expert_utility[0].detach()
    best_utility, best_index = torch.max(expert_utility.detach(), dim=0)
    advantage = (policy_utility - baseline_utility) / tau
    best_advantage = (best_utility - baseline_utility) / tau
    regret_loss = F.relu(best_advantage - advantage)
    no_harm_loss = F.relu(-advantage)

    topk = torch.topk(expert_utility.detach(), k=min(2, expert_utility.numel())).values
    winner_gap = topk[0] - topk[1] if topk.numel() > 1 else topk.new_zeros(())
    winner_gain = best_utility - baseline_utility
    clear_positive = (best_index != 0) & (winner_gain >= float(min_utility_gain)) & (
        winner_gap >= float(min_winner_gap)
    )
    clear_baseline = (best_index == 0) & (winner_gap >= float(min_winner_gap))
    clear = clear_positive | clear_baseline
    if bool(clear.item()):
        clear_ce_loss = F.cross_entropy(
            expert_logits.unsqueeze(0), best_index.view(1), reduction="mean"
        )
    else:
        clear_ce_loss = expert_logits.sum() * 0.0

    route_loss = (
        regret_loss
        + float(no_harm_weight) * no_harm_loss
        + float(clear_ce_weight) * clear_ce_loss
    )
    policy_entropy = -(
        policy.clamp_min(1.0e-12).log() * policy
    ).sum() / math.log(max(policy.numel(), 2))
    return {
        "route_loss": route_loss,
        "regret_loss": regret_loss,
        "no_harm_loss": no_harm_loss,
        "clear_ce_loss": clear_ce_loss,
        "policy": policy,
        "effective_strength": effective_strength,
        "expert_utility": expert_utility,
        "policy_utility": policy_utility,
        "best_index": best_index,
        "clear": clear.float(),
        "positive_gain": (winner_gain >= float(min_utility_gain)).float(),
        "winner_gain": winner_gain,
        "winner_gap": winner_gap,
        "policy_entropy": policy_entropy,
        "policy_scores": policy_scores,
    }


def compute_counterfactual_composition_loss(
    owner: Any,
    *,
    target_tokens: torch.Tensor,
    h_qid: torch.Tensor,
    cond_prefix: torch.Tensor,
    query_embedding: torch.Tensor,
    asset: Any | None,
    max_queries: int,
    levels_per_query: int,
    min_level: int,
    max_level: int,
    oracle_temperature: float = 0.5,
    use_cfg: bool,
    normalize_oracle_utility: bool = False,
    advantage_temperature: float = 0.1,
    min_utility_gain: float = 0.02,
    min_winner_gap: float = 0.01,
    no_harm_weight: float = 2.0,
    clear_ce_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Train safe counterfactual residual composition on exact prefix states."""
    del oracle_temperature, normalize_oracle_utility  # retained for config compatibility
    device = target_tokens.device
    zero = query_embedding.new_zeros((), dtype=torch.float32)
    records = _sample_records(
        owner,
        target_tokens=target_tokens,
        h_qid=h_qid,
        asset=asset,
        max_queries=max_queries,
        levels_per_query=levels_per_query,
        min_level=min_level,
        max_level=max_level,
    )
    if not records:
        return _safe_result_template(zero)

    row_indices = torch.tensor([x[0] for x in records], device=device, dtype=torch.long)
    levels = torch.tensor([x[1] for x in records], device=device, dtype=torch.long)
    paths = target_tokens.index_select(0, row_indices)
    exact_logits = _exact_prefix_logits(
        owner,
        target_paths=paths,
        levels=levels,
        cond_prefix=cond_prefix,
        query_indices=row_indices,
        use_cfg=use_cfg,
    )
    q_rows = query_embedding.index_select(0, row_indices)

    values: dict[str, list[torch.Tensor]] = {
        key: [] for key in [
            "route_loss", "branch_loss", "regret_loss", "no_harm_loss",
            "clear_ce_loss", "route_acc", "clear", "positive_gain",
            "branch_acc", "baseline_branch_acc", "strength",
            "oracle_strength", "expert0_probability", "policy_entropy",
            "utility_gain", "best_utility_gain", "utility_gap",
        ]
    }
    win_counts = [0, 0, 0, 0]
    expert_values = owner.residual_composer.expert_strengths.to(device=device)
    token_to_code = owner.token_id_to_code.to(device)

    for idx, (_, level, legal_np, positive_np) in enumerate(records):
        level = int(level)
        legal_tokens = torch.as_tensor(legal_np, device=device, dtype=torch.long)
        positive_tokens = torch.as_tensor(positive_np, device=device, dtype=torch.long)
        positive_mask = torch.isin(legal_tokens, positive_tokens)
        if not bool(positive_mask.any().item()):
            continue
        legal_codes = token_to_code.index_select(0, legal_tokens)
        level_size = int(owner.level_vocab_sizes[level].item())
        valid = (legal_codes >= 0) & (legal_codes < level_size)
        if not bool(valid.all().item()):
            legal_tokens = legal_tokens[valid]
            legal_codes = legal_codes[valid]
            positive_mask = positive_mask[valid]
        if legal_tokens.numel() == 0 or not bool(positive_mask.any().item()):
            continue

        level_token_ids = owner.level_token_ids[level, :level_size].to(device=device, dtype=torch.long)
        full_generator = exact_logits[idx, level].index_select(0, level_token_ids)
        full_residual = owner._rrg_scores_for_rows(
            q_rows[idx : idx + 1], paths[idx : idx + 1], level
        )
        if full_residual is None:
            continue
        full_residual = full_residual[0, :level_size]
        valid_mask = torch.zeros((1, level_size), device=device, dtype=torch.bool)
        valid_mask[0, legal_codes] = True
        prefix_reconstruction = owner._rrg_prefix_reconstruction_from_token_ids(
            paths[idx : idx + 1], level, dtype=torch.float32
        )
        composition = owner.residual_composer(
            query_embedding=q_rows[idx : idx + 1],
            prefix_reconstruction=prefix_reconstruction,
            generator_scores=full_generator.unsqueeze(0),
            residual_scores=full_residual.unsqueeze(0),
            level=level,
            valid_mask=valid_mask,
        )

        base_raw = full_generator.index_select(0, legal_codes)
        residual_raw = full_residual.index_select(0, legal_codes)
        if not bool(torch.isfinite(base_raw).any().item()) or not bool(torch.isfinite(residual_raw).any().item()):
            continue
        base = torch.nan_to_num(
            base_raw.detach().float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4
        ).clamp(min=-1.0e4, max=1.0e4)
        residual = torch.nan_to_num(
            residual_raw.detach().float(), nan=0.0, posinf=8.0, neginf=-8.0
        ).clamp(min=-8.0, max=8.0)
        if not bool(torch.isfinite(composition.expert_logits).all().item()):
            continue

        objective = safe_counterfactual_policy_objective(
            expert_logits=composition.expert_logits[0],
            expert_strengths=expert_values,
            base_scores=base,
            residual_scores=residual,
            positive_mask=positive_mask,
            advantage_temperature=advantage_temperature,
            min_utility_gain=min_utility_gain,
            min_winner_gap=min_winner_gap,
            no_harm_weight=no_harm_weight,
            clear_ce_weight=clear_ce_weight,
        )
        if not bool(torch.isfinite(objective["route_loss"]).item()):
            continue

        policy_scores = objective["policy_scores"]
        baseline_scores = base
        best_index = int(objective["best_index"].item())
        if best_index < 4:
            win_counts[best_index] += 1
        is_clear = bool(objective["clear"].item() > 0.5)
        selected_index = torch.argmax(objective["policy"])

        values["route_loss"].append(objective["route_loss"])
        # Diagnostic only in V2; training config sets branch weight to zero.
        values["branch_loss"].append(-objective["policy_utility"])
        values["regret_loss"].append(objective["regret_loss"])
        values["no_harm_loss"].append(objective["no_harm_loss"])
        values["clear_ce_loss"].append(objective["clear_ce_loss"])
        if is_clear:
            values["route_acc"].append((selected_index == objective["best_index"]).float())
        values["clear"].append(objective["clear"])
        values["positive_gain"].append(objective["positive_gain"])
        values["branch_acc"].append(positive_mask[torch.argmax(policy_scores)].float())
        values["baseline_branch_acc"].append(positive_mask[torch.argmax(baseline_scores)].float())
        values["strength"].append(objective["effective_strength"])
        values["oracle_strength"].append(expert_values[objective["best_index"]])
        values["expert0_probability"].append(objective["policy"][0])
        values["policy_entropy"].append(objective["policy_entropy"])
        values["utility_gain"].append(objective["policy_utility"] - objective["expert_utility"][0].detach())
        values["best_utility_gain"].append(objective["winner_gain"])
        values["utility_gap"].append(objective["winner_gap"])

    if not values["route_loss"]:
        return _safe_result_template(zero)

    def mean(name: str) -> torch.Tensor:
        items = values[name]
        return torch.stack(items).mean() if items else zero

    strength_tensor = torch.stack(values["strength"])
    count = float(len(values["route_loss"]))
    result = {
        "route_loss": mean("route_loss"),
        "branch_loss": mean("branch_loss"),
        "regret_loss": mean("regret_loss"),
        "no_harm_loss": mean("no_harm_loss"),
        "clear_ce_loss": mean("clear_ce_loss"),
        "route_acc": mean("route_acc"),
        "clear_fraction": mean("clear"),
        "positive_gain_fraction": mean("positive_gain"),
        "branch_acc": mean("branch_acc"),
        "baseline_branch_acc": mean("baseline_branch_acc"),
        "strength_mean": strength_tensor.mean(),
        "strength_std": strength_tensor.std(unbiased=False),
        "oracle_strength_mean": mean("oracle_strength"),
        "expert0_probability": mean("expert0_probability"),
        "policy_entropy": mean("policy_entropy"),
        "utility_gain": mean("utility_gain"),
        "best_utility_gain": mean("best_utility_gain"),
        "utility_gap": mean("utility_gap"),
        "expert0_win_rate": zero.new_tensor(win_counts[0] / count),
        "expert10_win_rate": zero.new_tensor(win_counts[1] / count),
        "expert50_win_rate": zero.new_tensor(win_counts[2] / count),
        "expert100_win_rate": zero.new_tensor(win_counts[3] / count),
        "count": zero.new_tensor(count),
        "clear_count": zero.new_tensor(float(len(values["route_acc"]))),
    }
    return {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in result.items()}
