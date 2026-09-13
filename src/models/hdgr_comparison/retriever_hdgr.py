"""
HDGR hierarchical denoising generator.

This module keeps the public ``T5ForGenerativeRetrieval`` interface used by the
legacy training/evaluation code, but the actual ID generator is a hierarchical
block-denoising semantic-ID generator:

  * block-by-block generation over residual-quantizer semantic-ID levels;
  * masked discrete denoising inside each hierarchy block;
  * vectorized training input ``[noisy_codes | clean_codes]``;
  * explicit block attention that lets a noisy block see only its own noisy
    tokens and previous clean blocks;
  * optional candidate-tree constrained block decoding for retrieval.
"""

from __future__ import annotations

# Standard library
import glob
import hashlib
import math
import os
import pickle
import random
import re
import string
from types import SimpleNamespace
from typing import Any, Optional

# Third-party
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
# Local modules
from models.residual_quantization.residual_quantization import RQ
from models.hdgr_comparison.shared_code_head import (
    CodeTiedOutputHead,
    migrate_legacy_untied_state_dict,
)
from models.uniir_clip import utils

IGNORE_INDEX = -100
VERY_NEGATIVE = -1.0e4


# =========================================================
# small helpers
# =========================================================


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read from dict / OmegaConf / namespace without binding this file to OmegaConf."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceil_div(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _as_int_list(value: Any) -> Optional[list[int]]:
    """Best-effort conversion from OmegaConf/list/string to a list of ints."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = value.strip("[]()")
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    try:
        return [int(x) for x in list(value)]
    except Exception:
        return None


def _default_hierarchy_block_sizes(codebook_level: int, modality_index: bool, block_size: int) -> list[int]:
    """
    Default hierarchy-aware partition for semantic IDs.

    If level 0 is a modality token, split as:
        [modality] + [coarse residuals] + [middle residuals] + [fine residuals].

    Important: in this project, config.codebook_config.codebook_level denotes the
    number of RQ/semantic levels. RQ(modality_index=True) prepends one extra
    modality level, so an 8-level RQ code becomes a 9-token generator sequence.
    The common 9-token ID is therefore split as [1, 2, 3, 3], i.e.
        block0: level 0
        block1: levels 1-2
        block2: levels 3-5
        block3: levels 6-8

    If there is no modality token, fall back to uniform block_size chunks.
    """
    L = int(codebook_level)
    if L <= 0:
        raise ValueError("codebook_level must be positive")
    if bool(modality_index) and L >= 4:
        rem = L - 1
        coarse = max(1, rem // 3)
        middle = max(1, (rem - coarse) // 2)
        fine = rem - coarse - middle
        if fine <= 0:
            fine = 1
            if middle > 1:
                middle -= 1
            else:
                coarse = max(1, coarse - 1)
        return [1, coarse, middle, fine]
    bs = max(1, int(block_size))
    sizes = []
    left = L
    while left > 0:
        cur = min(bs, left)
        sizes.append(cur)
        left -= cur
    return sizes


def build_hierarchy_block_spans(
    codebook_level: int,
    model_cfg: Any,
    modality_index: bool,
    block_size: int,
) -> tuple[list[tuple[int, int]], list[str]]:
    """
    Build non-uniform semantic-ID block spans.

    Supported config keys:
      model.hierarchy_aware_blocks: bool, default True
      model.hierarchy_block_sizes: e.g. [1, 2, 3, 3] for modality + 8 RQ levels
      model.hierarchy_block_names: e.g. [modality, coarse, middle, fine]

    If hierarchy_aware_blocks is false, the code reverts to uniform block_size.
    """
    L = int(codebook_level)
    hierarchy_on = bool(cfg_get(model_cfg, "hierarchy_aware_blocks", True))
    configured = _as_int_list(cfg_get(model_cfg, "hierarchy_block_sizes", None))
    if not hierarchy_on:
        configured = None
        sizes = _default_hierarchy_block_sizes(L, False, block_size)
    elif configured is not None:
        sizes = configured
    else:
        sizes = _default_hierarchy_block_sizes(L, modality_index, block_size)

    if any(int(x) <= 0 for x in sizes):
        raise ValueError(f"hierarchy_block_sizes must be positive, got {sizes}")
    size_sum = sum(sizes)
    if size_sum != L:
        # Backward-compatible guard for the common mistake introduced by v9:
        # users may configure sizes for the 8 RQ levels while the actual generator
        # sequence is 9 tokens because RQ(modality_index=True) prepends a modality
        # level. If the configured partition is exactly one level short and starts
        # with the modality block, assign the missing level to the fine block.
        if bool(modality_index) and size_sum == L - 1 and len(sizes) >= 2 and int(sizes[0]) == 1:
            old_sizes = list(sizes)
            sizes = list(sizes)
            sizes[-1] = int(sizes[-1]) + 1
            print(
                "[HDGR generator] Warning: hierarchy_block_sizes "
                f"{old_sizes} sum to {size_sum}, but effective generator code length is {L} "
                "because modality_index=True prepends a modality level. "
                f"Auto-expanded the final/fine block to {sizes}. "
                "Recommended explicit setting: hierarchy_block_sizes: [1, 2, 3, 3] "
                "for modality + 8 RQ levels."
            )
        else:
            raise ValueError(
                f"hierarchy_block_sizes must sum to effective generator code length={L}, "
                f"got {sizes} with sum={size_sum}. "
                "Note: when modality_index=True, the RQ quantizer prepends one modality level, "
                "so config.codebook_config.codebook_level=8 becomes generator length 9. "
                "Use e.g. hierarchy_block_sizes: [1, 2, 3, 3]."
            )

    spans: list[tuple[int, int]] = []
    cur = 0
    for size in sizes:
        spans.append((cur, cur + int(size)))
        cur += int(size)

    names = cfg_get(model_cfg, "hierarchy_block_names", None)
    try:
        if isinstance(names, str):
            raw = names.strip().strip("[]()")
            names = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            names = [str(x) for x in list(names)] if names is not None else []
    except Exception:
        names = []
    default_names = ["modality", "coarse", "middle", "fine"]
    while len(names) < len(spans):
        names.append(default_names[len(names)] if len(names) < len(default_names) else f"block{len(names)}")
    return spans, names[: len(spans)]


def block_ids_from_spans(length: int, block_spans: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    ids = torch.empty(int(length), dtype=torch.long, device=device)
    for block_idx, (start, end) in enumerate(block_spans):
        ids[int(start): int(end)] = int(block_idx)
    return ids


def expand_block_values_by_spans(values: torch.Tensor, block_spans: list[tuple[int, int]], length: int) -> torch.Tensor:
    """Expand [B, n_blocks] block values to [B, L] according to arbitrary spans."""
    if values.dim() == 1:
        values = values[:, None]
    B = values.size(0)
    out = values.new_zeros((B, int(length)))
    for block_idx, (start, end) in enumerate(block_spans):
        out[:, int(start): int(end)] = values[:, int(block_idx): int(block_idx) + 1]
    return out



def _epoch_from_path(path: str) -> int:
    """Best-effort epoch parser used only for auto-discovering checkpoints."""
    m = re.search(r"epoch[_-]?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_quantizer_checkpoint(config: Any, raw_path: str) -> str:
    """
    Resolve the frozen RQ quantizer checkpoint needed by the HDGR generator.

    Generator training cannot start without this model because it supplies both
    the target residual-quantizer codes and the query conditioning embedding.
    This helper keeps the default path behavior, but also auto-discovers common
    rq_clip checkpoints and gives an actionable error message when none exists.
    """
    genir_dir = os.path.abspath(str(cfg_get(config, "genir_dir", os.getcwd())))
    raw_path = str(raw_path or "").strip()

    candidates: list[str] = []
    if raw_path:
        candidates.append(raw_path if os.path.isabs(raw_path) else os.path.join(genir_dir, raw_path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    search_patterns = [
        os.path.join(genir_dir, "checkpoint", "**", "rq_clip*_epoch_*.pth"),
        os.path.join(genir_dir, "checkpoint", "**", "rq_clip*.pth"),
        os.path.join(genir_dir, "checkpoint", "**", "*quantizer*.pth"),
    ]
    discovered: list[str] = []
    for pattern in search_patterns:
        discovered.extend(glob.glob(pattern, recursive=True))
    discovered = sorted(set(discovered), key=lambda p: (_epoch_from_path(p), os.path.getmtime(p) if os.path.exists(p) else 0), reverse=True)

    if discovered:
        chosen = os.path.abspath(discovered[0])
        rel = os.path.relpath(chosen, genir_dir)
        print(
            "[HDGR generator] configured quantizer checkpoint was not found:\n"
            f"  {os.path.abspath(candidates[0]) if candidates else '<empty>'}\n"
            "[HDGR generator] auto-discovered and will use:\n"
            f"  {chosen}\n"
            f"You can make this explicit by setting codebook_config.quantizer_path: {rel}"
        )
        return chosen

    expected = os.path.abspath(candidates[0]) if candidates else "<empty quantizer_path>"
    raise FileNotFoundError(
        "Missing RQ quantizer checkpoint for HDGR block-denoising generator.\n\n"
        f"Expected path:\n  {expected}\n\n"
        "Why it is required:\n"
        "  The HDGR generator learns to generate residual-quantizer code sequences. "
        "During training this wrapper first runs the frozen RQ model to obtain "
        "query conditioning embeddings and target pool codes.\n\n"
        "Fix options:\n"
        "  1) Train the quantizer first, then train the generator:\n"
        f"     cd {genir_dir}\n"
        "     bash scripts/train/train_quantizer.sh\n"
        "     ls checkpoint/rq_clip_large/Large/Instruct/InBatch/rq_clip_large_epoch_*.pth\n\n"
        "  2) If your quantizer checkpoint already exists elsewhere, point the generator to it:\n"
        "     edit configs/generator/train.yaml:\n"
        "       codebook_config:\n"
        "         quantizer_path: checkpoint/.../your_rq_checkpoint.pth\n"
        "     or run train.py with:\n"
        "       --quantizer_path checkpoint/.../your_rq_checkpoint.pth\n\n"
        "  3) If you used a different experiment path_suffix for quantizer training, copy or symlink it to the configured path."
    )


def process_token(target: torch.Tensor, code_length: int, pad_token_id: int = IGNORE_INDEX):
    """
    Input:
        target: [B, L] token ids
    Return:
        tokens: [B, L]
        mask:   [B, L], 1 valid / 0 padding
        gt:     [B, L]
    """
    if target.dim() == 1:
        target = target.unsqueeze(0)

    if target.size(1) != code_length:
        raise ValueError(f"Expected code_length={code_length}, got {target.size(1)}")

    tokens = target.clone().long()
    gt = target.clone().long()
    mask = (target != pad_token_id).long()
    return tokens, mask, gt


def block_ids_for_length(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device)
    return torch.div(pos, int(block_size), rounding_mode="floor")


def expand_block_values(values: torch.Tensor, block_size: int, length: int) -> torch.Tensor:
    """Expand [B, n_blocks] block values to [B, L] token values."""
    if values.dim() == 1:
        values = values[:, None]
    expanded = values.repeat_interleave(int(block_size), dim=1)
    return expanded[:, :length]


def sample_block_timesteps(
    batch_size: int,
    num_blocks: int,
    num_timesteps: int,
    device: torch.device,
    eps_min: float = 1.0e-3,
    eps_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    block-denoising block-wise noising time.

    Returns:
        t_block: [B, n_blocks], integer timesteps in [1, T]
        p_block: [B, n_blocks], mask probabilities in [eps_min, eps_max]
    """
    eps_min = float(max(0.0, min(1.0, eps_min)))
    eps_max = float(max(eps_min, min(1.0, eps_max)))
    p_block = eps_min + torch.rand(batch_size, num_blocks, device=device) * (eps_max - eps_min)
    t_block = torch.clamp(torch.ceil(p_block * int(num_timesteps)).long(), min=1, max=int(num_timesteps))
    return t_block, p_block


def q_xt(
    x0: torch.Tensor,
    p_block: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    block_spans: Optional[list[tuple[int, int]]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward noising process q(x_t | x_0) for masked discrete diffusion.

    ``p_block`` is expanded to token-level probabilities.  In the original
    HDGR setting each block can receive a different mask rate; in the
    BD3-LM-faithful setting the caller passes the same per-sample probability
    for every block, yielding one global p with iid token masking over the
    entire semantic-ID sequence.  At least one valid token is masked in each
    sample so that the denoising loss is well-defined.
    """
    device = x0.device
    B, L = x0.shape
    if block_spans is not None:
        p_token = expand_block_values_by_spans(p_block, block_spans=block_spans, length=L)
    else:
        p_token = expand_block_values(p_block, block_size=block_size, length=L)
    rand = torch.rand(B, L, device=device)
    mask_pos = (rand < p_token) & (valid_mask > 0)

    none_masked = mask_pos.sum(dim=1) == 0
    if none_masked.any():
        rows = torch.nonzero(none_masked, as_tuple=False).squeeze(1)
        for r in rows.tolist():
            valid_idx = torch.nonzero(valid_mask[r] > 0, as_tuple=False).squeeze(1)
            if valid_idx.numel() > 0:
                pick = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=device)]
                mask_pos[r, pick] = True

    xt = x0.clone()
    xt[mask_pos] = mask_token_id
    return xt, mask_pos, p_token


def build_hdgr_attention_mask(
    length: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
    clean_visible_mask: Optional[torch.Tensor] = None,
    block_spans: Optional[list[tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    Build the vectorized HDGR attention mask for ``[noisy | clean]`` input.

    Sequence layout:
        0 .. L-1      : noisy tokens x_t
        L .. 2L-1     : clean context tokens x_0

    Rules:
        * noisy token in block b attends to noisy tokens in block b;
        * noisy token in block b attends to clean tokens in blocks < b;
        * clean token in block b attends to clean tokens in blocks <= b;
        * clean tokens never attend to noisy tokens.
    """
    L = int(length)
    S = 2 * L
    block_size = max(1, int(block_size))

    pos = torch.arange(S, device=device)
    is_noisy = pos < L
    inner_pos = pos % L
    if block_spans is not None:
        block_ids = block_ids_from_spans(L, block_spans=block_spans, device=device)
        block = block_ids[inner_pos]
    else:
        block = torch.div(inner_pos, block_size, rounding_mode="floor")

    q_noisy = is_noisy[:, None]
    k_noisy = is_noisy[None, :]
    q_block = block[:, None]
    k_block = block[None, :]

    noisy_to_same_noisy = q_noisy & k_noisy & (q_block == k_block)
    noisy_to_prev_clean = q_noisy & (~k_noisy) & (k_block < q_block)
    clean_to_clean_causal = (~q_noisy) & (~k_noisy) & (k_block <= q_block)
    allowed = noisy_to_same_noisy | noisy_to_prev_clean | clean_to_clean_causal

    mask = torch.full((S, S), VERY_NEGATIVE, device=device, dtype=torch.float32)
    mask = mask.masked_fill(allowed, 0.0)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

    if valid_mask is not None:
        valid_2 = torch.cat([valid_mask, valid_mask], dim=1).bool()  # [B, 2L]
        key_invalid = ~valid_2[:, None, :]
        mask = mask.masked_fill(key_invalid, VERY_NEGATIVE)

    # During prefix-conditioned inference, the clean-context half contains
    # MASK placeholders after the committed prefix.  Those placeholders are
    # not visible identifier states and must not become keys merely because an
    # isolated suffix is represented as singleton hierarchy blocks.  The noisy
    # half remains valid so that each unresolved position can read its own
    # noisy MASK state.
    if clean_visible_mask is not None:
        if tuple(clean_visible_mask.shape) != (batch_size, L):
            raise ValueError(
                "clean_visible_mask must have shape "
                f"({batch_size}, {L}), got {tuple(clean_visible_mask.shape)}"
            )
        clean_key_invalid = ~clean_visible_mask.to(device=device).bool()
        mask[:, :, L:] = mask[:, :, L:].masked_fill(
            clean_key_invalid[:, None, :], VERY_NEGATIVE
        )

    return mask


def get_layer_nodes_from_tree(tree: Optional[dict], tokens_1d: torch.Tensor, layer_idx: int, mask_token_id: int):
    """Return candidate-tree children allowed at ``layer_idx`` for one partial sequence."""
    try:
        if tree is None:
            return None
        if layer_idx == 0:
            return list(tree.keys())

        current = tree
        for i in range(layer_idx):
            tid = int(tokens_1d[i].item())
            if tid == mask_token_id:
                return None
            current = current.get(tid, None)
            if current is None:
                return None
        return list(current.keys())
    except Exception:
        return None


# =========================================================
# HDGR denoising backbone
# =========================================================


class HDGRSelfAttention(nn.Module):
    """Multi-head self-attention with only the caller-provided HDGR mask.

    This class intentionally does **not** contain GPT-style causal masking.  The
    complete visibility pattern is the additive ``attention_mask`` built by
    ``build_hdgr_attention_mask``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, S, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # attention_mask: [B, S, S], additive; 0 for visible, -1e4 for blocked.
            scores = scores + attention_mask[:, None, :, :].to(dtype=scores.dtype, device=scores.device)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class HDGRTransformerBlock(nn.Module):
    """HDGR denoising block: explicit self-attn mask + query-prefix cross-attn."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = HDGRSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_mlp = nn.LayerNorm(d_model)
        hidden = int(d_model * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def project_prefix_kv(self, prefix: torch.Tensor):
        """Project cross-attention K/V once for reuse across denoising rounds."""
        weight = self.cross_attn.in_proj_weight
        bias = self.cross_attn.in_proj_bias
        d = self.cross_attn.embed_dim
        k = F.linear(prefix, weight[d:2 * d], None if bias is None else bias[d:2 * d])
        v = F.linear(prefix, weight[2 * d:], None if bias is None else bias[2 * d:])
        b, p, _ = k.shape
        h = self.cross_attn.num_heads
        dh = d // h
        k = k.view(b, p, h, dh).transpose(1, 2).contiguous()
        v = v.view(b, p, h, dh).transpose(1, 2).contiguous()
        return k, v

    def _cached_cross_attention(self, query: torch.Tensor, prefix_kv) -> torch.Tensor:
        weight = self.cross_attn.in_proj_weight
        bias = self.cross_attn.in_proj_bias
        d = self.cross_attn.embed_dim
        q = F.linear(query, weight[:d], None if bias is None else bias[:d])
        b, s, _ = q.shape
        h = self.cross_attn.num_heads
        dh = d // h
        q = q.view(b, s, h, dh).transpose(1, 2)
        k, v = prefix_kv
        dropout_p = float(self.cross_attn.dropout) if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return F.linear(out, self.cross_attn.out_proj.weight, self.cross_attn.out_proj.bias)

    def forward(
        self, x: torch.Tensor, prefix: torch.Tensor, attention_mask: torch.Tensor,
        prefix_kv=None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln_self(x), attention_mask=attention_mask)
        cross_query = self.ln_cross(x)
        if prefix_kv is None:
            cross_out, _ = self.cross_attn(
                query=cross_query, key=prefix, value=prefix, need_weights=False,
            )
        else:
            cross_out = self._cached_cross_attention(cross_query, prefix_kv)
        x = x + cross_out
        x = x + self.mlp(self.ln_mlp(x))
        return x


class HDGRBlockDenoisingGenerator(nn.Module):
    """
    HDGR denoising Transformer for RQ code tokens.

    The class name is kept for backward compatibility with earlier patches, but
    the implementation no longer wraps GPT2.  It consumes ``[x_t | x_0_context]``
    and uses only the explicit HDGR block-causal mask supplied by the caller.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        code_length: int,
        num_prefix: int,
        time_step: int = 8,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: Optional[int] = None,
        use_time_embedding: bool = True,
        code_tied_output_head: bool = False,
        tied_checkpoint_merge: str = "average",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.mask_token_id = self.num_classes - 1
        self.d_model = int(d_model)
        self.code_length = int(code_length)
        self.num_prefix = int(num_prefix)
        self.time_step = int(max(1, time_step))
        self.use_time_embedding = bool(use_time_embedding)
        self.code_tied_output_head = bool(code_tied_output_head)
        self.tied_checkpoint_merge = str(tied_checkpoint_merge).lower().strip()
        if self.tied_checkpoint_merge not in {"average", "input", "output"}:
            raise ValueError(
                "tied_checkpoint_merge must be one of: average, input, output; "
                f"got {self.tied_checkpoint_merge!r}"
            )
        self.max_positions = int(max_position_embeddings or max(2 * self.code_length + 8, 32))

        self.token_embed = nn.Embedding(self.num_classes, self.d_model)
        self.input_embed = self.token_embed  # legacy alias used by codebook init
        self.position_embed = nn.Embedding(self.max_positions, self.d_model)
        self.segment_embed = nn.Embedding(2, self.d_model)
        self.token_time_embed = nn.Embedding(self.time_step + 1, self.d_model) if self.use_time_embedding else None
        self.input_ln = nn.LayerNorm(self.d_model)
        self.prefix_ln = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(float(dropout))

        self.blocks = nn.ModuleList([
            HDGRTransformerBlock(
                d_model=self.d_model,
                num_heads=int(num_heads),
                dropout=float(dropout),
            )
            for _ in range(int(num_layers))
        ])
        self.final_ln = nn.LayerNorm(self.d_model)
        if self.code_tied_output_head:
            self.lm_head = CodeTiedOutputHead(self.token_embed, self.vocab_size)
        else:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.code_tied_output_head:
            migrated = migrate_legacy_untied_state_dict(
                state_dict,
                prefix=prefix,
                vocab_size=self.vocab_size,
                strategy=self.tied_checkpoint_merge,
            )
            if migrated:
                print(
                    "[Code-Tied HDGR] Migrated legacy untied checkpoint at "
                    f"{prefix or '<root>'} using merge={self.tied_checkpoint_merge}."
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x0_context: torch.Tensor,             # [B, L]
        xt: torch.Tensor,                     # [B, L]
        prefix: torch.Tensor,                 # [B, P, D]
        t_block: torch.Tensor,                # [B, n_blocks] or [B]
        attention_mask: torch.Tensor,         # [B, 2L, 2L], additive
        block_size: int,
        block_spans: Optional[list[tuple[int, int]]] = None,
        labels: Optional[torch.Tensor] = None,
        project_logits: bool = True,
        prefix_kv_cache=None,
        prefix_is_normalized: bool = False,
        active_span: Optional[tuple[int, int]] = None,
    ):
        B, L = xt.shape
        if L != self.code_length:
            raise ValueError(f"Expected code_length={self.code_length}, got {L}")
        if x0_context.shape != xt.shape:
            raise ValueError("x0_context and xt must have the same shape")

        x_in = torch.cat([xt, x0_context], dim=1).clamp(min=0, max=self.num_classes - 1)  # [B, 2L]
        S = x_in.size(1)
        if S > self.max_positions:
            raise ValueError(f"sequence length {S} exceeds max_position_embeddings={self.max_positions}")

        emb = self.token_embed(x_in)
        pos_ids = torch.arange(S, device=xt.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        emb = emb + self.position_embed(pos_ids)

        seg_ids = torch.cat([
            torch.zeros(B, L, device=xt.device, dtype=torch.long),
            torch.ones(B, L, device=xt.device, dtype=torch.long),
        ], dim=1)
        emb = emb + self.segment_embed(seg_ids)

        if t_block.dim() == 1:
            noisy_t = t_block[:, None].expand(B, L)
            clean_t = torch.zeros(B, L, device=xt.device, dtype=torch.long)
        else:
            if block_spans is not None:
                noisy_t = expand_block_values_by_spans(t_block, block_spans=block_spans, length=L)
            else:
                noisy_t = expand_block_values(t_block, block_size=block_size, length=L)
            clean_t = torch.zeros(B, L, device=xt.device, dtype=torch.long)
        if self.use_time_embedding:
            token_t = torch.cat([noisy_t, clean_t], dim=1).long().clamp(min=0, max=self.time_step)
            emb = emb + self.token_time_embed(token_t)

        active_block_len = None
        if active_span is not None:
            active_start, active_end = map(int, active_span)
            if not (0 <= active_start < active_end <= L):
                raise ValueError(f"Invalid active_span={active_span} for length={L}")
            # For a current noisy block, HDGR permits attention only to noisy
            # tokens in that block and clean tokens in earlier blocks. Retain
            # exactly those positions and preserve their original embeddings.
            active_idx = torch.cat([
                torch.arange(active_start, active_end, device=xt.device),
                torch.arange(L, L + active_start, device=xt.device),
            ])
            emb = emb.index_select(1, active_idx)
            attention_mask = attention_mask.index_select(1, active_idx).index_select(2, active_idx)
            active_block_len = active_end - active_start

        x = self.dropout(self.input_ln(emb))
        prefix = prefix if prefix_is_normalized else self.prefix_ln(prefix)
        for block_idx, block in enumerate(self.blocks):
            block_kv = None if prefix_kv_cache is None else prefix_kv_cache[block_idx]
            x = block(x, prefix=prefix, attention_mask=attention_mask, prefix_kv=block_kv)
        x = self.final_ln(x)
        logits = self.lm_head(x) if project_logits else None
        noisy_hidden = x[:, :L, :]
        if active_span is not None:
            full_noisy_hidden = x.new_zeros((B, L, x.size(-1)))
            full_noisy_hidden[:, active_start:active_end, :] = x[:, :active_block_len, :]
            noisy_hidden = full_noisy_hidden
        return SimpleNamespace(
            loss=None,
            logits=logits[:, :L, :] if logits is not None else None,
            full_logits=logits,
            past_key_values=None,
            hidden_states=noisy_hidden,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_prefix_kv_cache(self, prefix: torch.Tensor):
        """Normalize a base query prefix and pre-project K/V for every layer."""
        normalized = self.prefix_ln(prefix)
        return normalized, [block.project_prefix_kv(normalized) for block in self.blocks]


# =========================================================
# main retriever wrapper
# =========================================================


class T5ForGenerativeRetrieval(nn.Module):
    """
    Legacy name, new implementation.

    The outer interface remains compatible with GENIUS.  Internally, however,
    ``id_generator`` is now a HDGR denoising generator instead of a T5 decoder.
    """

    def __init__(self, config=None, tokenizer=None, clip_model=None, new_tokenizer: bool = True, init_rq_codebook: bool = True):
        super().__init__()

        self.config = config
        seed = int(cfg_get(config, "seed", 2023))
        set_global_seed(seed)

        # -------- CLIP wrapper, optional in training path --------
        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()

        # -------- frozen residual quantizer --------
        self.quantizer = RQ(config=config, clip_model=clip_model)
        codebook_cfg = cfg_get(config, "codebook_config", {})
        rq_model_path = resolve_quantizer_checkpoint(
            config,
            cfg_get(codebook_cfg, "quantizer_path", ""),
        )
        rq_checkpoint = torch.load(rq_model_path, map_location=torch.device("cpu"), weights_only=False)
        if "model" not in rq_checkpoint:
            raise KeyError(
                f"Quantizer checkpoint does not contain key 'model': {rq_model_path}. "
                "Please use a checkpoint saved by src/models/residual_quantization/train.py."
            )
        missing, unexpected = self.quantizer.load_state_dict(rq_checkpoint["model"], strict=False)
        if missing:
            print(f"[HDGR generator] Warning: missing quantizer keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[HDGR generator] Warning: unexpected quantizer keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        print(f"[HDGR generator] Loaded frozen RQ quantizer from {rq_model_path}")
        self.quantizer.eval()
        for _, param in self.quantizer.named_parameters():
            param.requires_grad = False

        self.modality_index = self.quantizer.modality_index
        self.codebook_vocab = int(self.quantizer.codebook_vocab)
        self.codebook_level = int(self.quantizer.codebook_level)
        if self.quantizer.unique_code:
            self.codebook_level += 1

        # -------- tokenizer containing code tokens --------
        # This is an internal semantic-ID tokenizer, not a T5 tokenizer.  The
        # optional tokenizer argument is kept only for backward-compatible calls.
        self.tokenizer = tokenizer
        self.code_tokens: list[str] = []
        if new_tokenizer or self.tokenizer is None:
            self._initialize_tokenizer(tokenizer)
        self._initialize_codebook_tokens()

        # -------- generator hyperparameters --------
        model_cfg = cfg_get(config, "model", {})
        self.d_model = int(cfg_get(model_cfg, "d_model", 0) or 0)
        if self.d_model <= 0:
            try:
                self.d_model = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg_get(model_cfg, "t5_model_name", "google-t5/t5-small"),
                    local_files_only=bool(cfg_get(model_cfg, "local_files_only", True)),
                ).config.d_model
            except Exception:
                self.d_model = 512

        self.num_prefix = int(cfg_get(model_cfg, "num_prefix", 30))
        self.time_step = int(cfg_get(model_cfg, "time_step", self.codebook_level))
        self.block_size = int(cfg_get(model_cfg, "block_size", 4))
        self.block_size = max(1, min(self.block_size, self.codebook_level))
        self.block_spans, self.block_names = build_hierarchy_block_spans(
            codebook_level=self.codebook_level,
            model_cfg=model_cfg,
            modality_index=bool(self.modality_index),
            block_size=self.block_size,
        )
        self.block_sizes = [end - start for start, end in self.block_spans]
        self.num_blocks = len(self.block_spans)
        block_desc = ", ".join(
            f"{name}:levels {start}-{end - 1}" if end - start > 1 else f"{name}:level {start}"
            for name, (start, end) in zip(self.block_names, self.block_spans)
        )
        print(f"[HDGR generator] hierarchy-aware block partition: {self.block_sizes} ({block_desc})")

        self.guidance_scale = float(cfg_get(model_cfg, "guidance_scale", 3.0))
        self.cond_drop_prob = float(cfg_get(model_cfg, "cond_drop_prob", 0.1))
        self.top_p = float(cfg_get(model_cfg, "top_p", 1.0))
        self.temperature = float(cfg_get(model_cfg, "temperature", 1.0))
        self.visible_loss_weight = float(cfg_get(model_cfg, "visible_loss_weight", 0.0))
        self.denoising_loss_weighting = str(cfg_get(model_cfg, "denoising_loss_weighting", "masked_ce")).lower()
        self.loss_weight_eps = float(cfg_get(model_cfg, "loss_weight_eps", 1.0e-3))
        self.sampling_eps_min = float(cfg_get(model_cfg, "sampling_eps_min", 1.0e-3))
        self.sampling_eps_max = float(cfg_get(model_cfg, "sampling_eps_max", 1.0))
        # Noising mode used only during training.  The original HDGR recipe
        # samples an independent mask probability for each semantic-ID block.
        # BD3-LM/MDLM log-linear training instead samples one global p per
        # sequence, then masks every token independently with that same p.
        self.noising_mode = str(cfg_get(model_cfg, "noising_mode", "blockwise")).lower()
        self.bd3_global_noising = bool(
            cfg_get(
                model_cfg,
                "bd3_global_noising",
                self.noising_mode in {"bd3", "bd3_global", "global", "global_iid", "bd3_global_iid"},
            )
        )
        if self.bd3_global_noising:
            print(
                "[HDGR generator] BD3-LM global noising enabled: "
                "one mask probability per sample, iid token masking over the full semantic-ID sequence."
            )
        self.diffusion_steps = int(cfg_get(model_cfg, "diffusion_steps", self.time_step))
        self.decode_strategy = str(cfg_get(model_cfg, "decode_strategy", "argmax"))
        # Tree constraints are applied exactly at block boundaries by the
        # block-level Trie decoder.  Keeping token-level constraints inside the
        # temporary denoising loop is optional because it can over-constrain
        # partially masked blocks.
        self.tree_constraint_in_diffusion = bool(cfg_get(model_cfg, "tree_constraint_in_diffusion", True))
        self.tree_constraint_during_refinement = bool(cfg_get(model_cfg, "tree_constraint_during_refinement", False))
        # 0 / negative means exact scoring over every legal next-block in the Trie.
        # Set a positive value only when the candidate pool is very large and eval
        # speed is more important than exact constrained decoding.
        self.block_trie_max_candidates = int(cfg_get(model_cfg, "block_trie_max_candidates", 0))
        # Important for hierarchy-aware block decoding: after the modality block
        # there may be only one legal prefix.  Padding it to num_beams creates
        # many identical parents; if those duplicates are expanded independently,
        # the next top-k can be filled by copies of the same sequence.  Deduping
        # keeps beams as distinct semantic-ID candidates and prevents collapse.
        self.deduplicate_block_beams = bool(cfg_get(model_cfg, "deduplicate_block_beams", True))

        # -------- vocab / mask class --------
        self.vocab_size = len(self.tokenizer)      # real output vocab only
        self.num_classes = self.vocab_size + 1     # plus [MASK] class
        self.mask_token_id = self.num_classes - 1

        # -------- condition projector --------
        self.embed_projector = nn.Linear(768, self.d_model * self.num_prefix)

        self.id_generator = HDGRBlockDenoisingGenerator(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            d_model=self.d_model,
            code_length=self.codebook_level,
            num_prefix=self.num_prefix,
            time_step=self.time_step,
            num_layers=int(cfg_get(model_cfg, "hdgr_num_layers", cfg_get(model_cfg, "nar_num_layers", 6))),
            num_heads=int(cfg_get(model_cfg, "hdgr_num_heads", cfg_get(model_cfg, "nar_num_heads", 8))),
            dropout=float(cfg_get(model_cfg, "hdgr_dropout", cfg_get(model_cfg, "nar_dropout", 0.1))),
            use_time_embedding=bool(cfg_get(model_cfg, "use_time_embedding", True)),
            code_tied_output_head=bool(cfg_get(model_cfg, "hdgr_code_tied_output_head", False)),
            tied_checkpoint_merge=str(cfg_get(model_cfg, "hdgr_tied_checkpoint_merge", "average")),
        )

        self.hdgr_code_tied_output_head = bool(cfg_get(model_cfg, "hdgr_code_tied_output_head", False))
        self.hdgr_tied_checkpoint_merge = str(cfg_get(model_cfg, "hdgr_tied_checkpoint_merge", "average"))
        if self.hdgr_code_tied_output_head:
            print(
                "[Code-Tied HDGR] Enabled shared input/output code matrix; "
                f"MASK row remains input-only; legacy_merge={self.hdgr_tied_checkpoint_merge}."
            )

        self.null_condition = nn.Parameter(torch.randn(1, self.num_prefix, self.d_model) * 0.02)
        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        self.iter = 0
        self.alpha = cfg_get(cfg_get(config, "hyperparameter_config", {}), "alpha", 2)

        self.tree_index = None
        self.cand_token_ids = None
        self.cand_codes_cache = None
        self.cand_codes_signature = None
        self.block_transition_cache: dict[tuple[int, ...], torch.Tensor] = {}

        self._build_level_token_buffers()
        self.init_rq_codebook_embeddings = bool(cfg_get(model_cfg, "init_rq_codebook_embeddings", True))
        self.codebook_init_scale = float(cfg_get(model_cfg, "codebook_init_scale", 1.0))
        self.codebook_init_target = str(cfg_get(model_cfg, "codebook_init_target", "input_and_lm_head"))
        if init_rq_codebook and self.init_rq_codebook_embeddings:
            self._initialize_codebook_embeddings(scale=self.codebook_init_scale, target=self.codebook_init_target)
        else:
            print("[HDGR generator] RQ codebook embedding initialization disabled.")

    # =====================================================
    # tokenizer / codebook init
    # =====================================================

    def _initialize_tokenizer(self, tokenizer=None):
        """Build a tiny semantic-ID tokenizer used only for code tokens.

        HDGR does not tokenize natural language here; query text is already
        represented by CLIP-SF embeddings.  This tokenizer only maps strings
        such as <a0>, <b1024>, ... to token IDs for the code generator.
        """
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
        new_vocab = {}
        for tok in special_tokens.values():
            if tok not in new_vocab:
                new_vocab[tok] = len(new_vocab)

        tokenizer_model = WordLevel(vocab=new_vocab, unk_token="<unk>")
        tokenizer_fast = Tokenizer(tokenizer_model)
        tokenizer_fast.pre_tokenizer = Whitespace()
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer_fast, **special_tokens)
        self.tokenizer = tokenizer_fast

    def _initialize_codebook_tokens(self):
        self.level_indicators = list(string.ascii_lowercase[: self.codebook_level])
        for level_idx, level in enumerate(self.level_indicators):
            if self.modality_index and level_idx == 0:
                for i in range(3):
                    self.code_tokens.append(f"<{level}{i}>")
                continue
            for i in range(self.codebook_vocab):
                self.code_tokens.append(f"<{level}{i}>")
        _ = self.tokenizer.add_tokens(self.code_tokens)

    def _build_level_token_buffers(self):
        vocab_size = len(self.tokenizer)
        max_code_size = self.codebook_vocab
        level_token_ids = torch.full((self.codebook_level, max_code_size), -1, dtype=torch.long)
        level_vocab_mask = torch.zeros((self.codebook_level, vocab_size), dtype=torch.bool)
        token_id_to_code = torch.full((vocab_size,), -1, dtype=torch.long)
        level_vocab_sizes = []

        cursor = 0
        for level_idx in range(self.codebook_level):
            level_size = 3 if (self.modality_index and level_idx == 0) else self.codebook_vocab
            level_vocab_sizes.append(level_size)
            cur_tokens = self.code_tokens[cursor : cursor + level_size]
            cur_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(cur_tokens), dtype=torch.long)
            level_token_ids[level_idx, :level_size] = cur_token_ids
            level_vocab_mask[level_idx, cur_token_ids] = True
            token_id_to_code[cur_token_ids] = torch.arange(level_size, dtype=torch.long)
            cursor += level_size

        self.register_buffer("level_token_ids", level_token_ids, persistent=False)
        self.register_buffer("level_vocab_mask", level_vocab_mask, persistent=False)
        self.register_buffer("token_id_to_code", token_id_to_code, persistent=False)
        self.register_buffer("level_vocab_sizes", torch.tensor(level_vocab_sizes, dtype=torch.long), persistent=False)

    def _initialize_codebook_embeddings(self, scale: float = 1.0, target: str = "input_and_lm_head"):
        """Initialize semantic-ID token embeddings from the frozen RQ codebooks.

        Earlier HDGR patches already copied normalized RQ codebook vectors into
        the HDGR token embedding and LM head with scale=1.0.  The official
        GENIUS AR implementation uses the same random linear projection but
        multiplies mapped codebook embeddings by a much larger factor (100).
        This method makes that prior explicit and configurable so we can run a
        clean HDGR+codebook-init ablation without changing the original HDGR
        checkpoint directory.
        """
        new_token_ids = self.tokenizer.convert_tokens_to_ids(self.code_tokens)
        linear_layer = nn.Linear(768, self.d_model, bias=False)

        if self.modality_index:
            first_layer_codebook = self.quantizer.residual_rq.layers[0]._codebook.embed[0, :3, :]
            mapped_first_layer_embeddings = linear_layer(F.normalize(first_layer_codebook))
            other_layers_codebooks = [layer._codebook.embed for layer in self.quantizer.residual_rq.layers[1:]]
            other_layers_codebooks = torch.stack(other_layers_codebooks, dim=0)
            other_layers_codebooks = rearrange(other_layers_codebooks, "q 1 c d -> q c d")
            other_layers_codebook = other_layers_codebooks.reshape(-1, other_layers_codebooks.shape[-1])
            mapped_other_layers_embeddings = linear_layer(F.normalize(other_layers_codebook))
            mapped_embeddings = torch.cat([mapped_first_layer_embeddings, mapped_other_layers_embeddings], dim=0)
        else:
            codebook_vectors = self.quantizer.residual_rq.codebooks.reshape(-1, 768)
            mapped_embeddings = linear_layer(F.normalize(codebook_vectors))

        mapped_embeddings = mapped_embeddings * float(scale)
        target = (target or "input_and_lm_head").lower()
        write_input = target in {"input", "input_embed", "input_and_lm_head", "both", "all"}
        write_lm_head = target in {"lm_head", "head", "input_and_lm_head", "both", "all"}
        if not write_input and not write_lm_head:
            raise ValueError(
                f"Unsupported model.codebook_init_target={target!r}; use input, lm_head, or input_and_lm_head"
            )

        with torch.no_grad():
            for idx, token_id in enumerate(new_token_ids):
                if write_input:
                    self.id_generator.input_embed.weight[token_id].copy_(mapped_embeddings[idx])
                if write_lm_head:
                    self.id_generator.lm_head.weight[token_id].copy_(mapped_embeddings[idx])

        print(
            f"[HDGR generator] Initialized {len(new_token_ids)} semantic-ID token embeddings "
            f"from RQ codebook | scale={float(scale):g} target={target}"
        )

    # =====================================================
    # misc helpers
    # =====================================================

    def transform_row(self, row, separator=""):
        return separator.join(f"<{self.level_indicators[l]}{int(row[l])}>" for l in range(self.codebook_level))

    def detransform_row(self, row):
        splited_row = re.findall(r"<.*?>", row)
        detransformed_row = []
        for token in splited_row:
            match = re.match(r"<([a-z])(\d+)>", token)
            if match:
                _, value = match.groups()
                detransformed_row.append(int(value))
        return detransformed_row

    def _codes_to_token_ids(self, codes: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(codes)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        if codes.dim() != 2:
            raise ValueError(f"`codes` must be 1D or 2D, got shape={tuple(codes.shape)}")
        if codes.size(1) != self.codebook_level:
            raise ValueError(f"Expected codes shape [B, {self.codebook_level}], got {tuple(codes.shape)}")
        if torch.is_floating_point(codes):
            if not torch.allclose(codes, codes.round()):
                bad = torch.nonzero(codes != codes.round(), as_tuple=False)[:10]
                raise ValueError(f"Non-integer floating codes found at {bad.tolist()}")
            codes = codes.round()

        codes = codes.long()
        device = codes.device
        level_vocab_sizes = self.level_vocab_sizes.to(device).unsqueeze(0)
        valid_mask = (codes >= 0) & (codes < level_vocab_sizes)
        if not torch.all(valid_mask):
            bad = torch.nonzero(~valid_mask, as_tuple=False)[:10]
            bad_info = []
            for b, level_idx in bad.tolist():
                bad_info.append({
                    "batch": b,
                    "level": level_idx,
                    "value": int(codes[b, level_idx].item()),
                    "valid_range": [0, int(level_vocab_sizes[0, level_idx].item()) - 1],
                })
            raise ValueError(f"Invalid code index found: {bad_info}")

        level_token_ids = self.level_token_ids.to(device)
        level_idx = torch.arange(self.codebook_level, device=device).unsqueeze(0).expand(codes.size(0), -1)
        return level_token_ids[level_idx, codes]

    def _token_ids_to_codes(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_id_to_code.to(token_ids.device)[token_ids.long()]

    def _mask_invalid_logits(self, logits: torch.Tensor) -> torch.Tensor:
        valid_mask = self.level_vocab_mask.to(logits.device).unsqueeze(0)  # [1, L, V]
        return logits.masked_fill(~valid_mask, VERY_NEGATIVE)

    def _project_condition(self, emb: torch.Tensor) -> torch.Tensor:
        return self.embed_projector(F.normalize(emb)).reshape(emb.size(0), self.num_prefix, self.d_model)

    def _hdgr_attention(self, batch_size: int, device: torch.device, valid_mask: Optional[torch.Tensor] = None):
        return build_hdgr_attention_mask(
            length=self.codebook_level,
            block_size=self.block_size,
            batch_size=batch_size,
            device=device,
            valid_mask=valid_mask,
            block_spans=self.block_spans,
        )

    def _clean_context_from_prefix(self, seq: torch.Tensor, prefix_end: int) -> torch.Tensor:
        clean = torch.full_like(seq, self.mask_token_id)
        if prefix_end > 0:
            clean[:, :prefix_end] = seq[:, :prefix_end]
        return clean

    def _candidate_codes_signature(self, cand_codes):
        """Cheap fingerprint used to avoid rebuilding the same candidate Trie per batch."""
        if cand_codes is None:
            return None
        try:
            if torch.is_tensor(cand_codes):
                x = cand_codes.detach()
                shape = tuple(int(v) for v in x.shape)
                dtype = str(x.dtype)
                if x.numel() == 0:
                    return (shape, dtype, "empty")
                rows = [0, max(0, shape[0] // 2), max(0, shape[0] - 1)] if len(shape) >= 1 else [0]
                rows = sorted(set(r for r in rows if 0 <= r < shape[0]))
                sample = x[rows].contiguous().cpu().numpy() if len(shape) >= 2 else x.reshape(-1)[:64].contiguous().cpu().numpy()
            else:
                arr = np.asarray(cand_codes)
                shape = tuple(int(v) for v in arr.shape)
                dtype = str(arr.dtype)
                if arr.size == 0:
                    return (shape, dtype, "empty")
                rows = [0, max(0, shape[0] // 2), max(0, shape[0] - 1)] if len(shape) >= 1 else [0]
                rows = sorted(set(r for r in rows if 0 <= r < shape[0]))
                sample = np.ascontiguousarray(arr[rows] if len(shape) >= 2 else arr.reshape(-1)[:64])
            digest = hashlib.sha1(sample.tobytes()).hexdigest()
            return (shape, dtype, digest)
        except Exception:
            # Last-resort identity fallback: safe for avoiding rebuilds inside a
            # single eval loop, but different objects will still rebuild.
            return ("object", id(cand_codes))

    def _build_candidate_tree(self, cand_codes):
        cand_codes = torch.as_tensor(cand_codes, dtype=torch.long)
        cand_token_ids = self._codes_to_token_ids(cand_codes).cpu()
        tree = {}
        for seq in cand_token_ids.tolist():
            cur = tree
            for tok in seq:
                if tok not in cur:
                    cur[tok] = {}
                cur = cur[tok]
        return tree, cand_token_ids.cpu(), cand_codes.cpu()

    def _get_tree_node_for_prefix(self, prefix: tuple[int, ...]) -> Optional[dict]:
        """Return the Trie node reached by a committed block prefix."""
        if self.tree_index is None:
            return None
        cur = self.tree_index
        for tok in prefix:
            if tok == self.mask_token_id:
                return None
            cur = cur.get(int(tok), None)
            if cur is None:
                return None
        return cur

    def _enumerate_next_blocks_from_node(
        self,
        node: Optional[dict],
        block_len: int,
        max_candidates: int = 0,
    ) -> list[tuple[int, ...]]:
        """Enumerate legal next semantic-ID blocks under a Trie node."""
        if node is None:
            return []
        out: list[tuple[int, ...]] = []
        limit = int(max_candidates or 0)

        def dfs(cur_node: dict, depth: int, path: list[int]) -> None:
            if limit > 0 and len(out) >= limit:
                return
            if depth == int(block_len):
                out.append(tuple(path))
                return
            for tok, child in cur_node.items():
                path.append(int(tok))
                dfs(child, depth + 1, path)
                path.pop()
                if limit > 0 and len(out) >= limit:
                    return

        dfs(node, 0, [])
        return out

    def _get_next_block_candidates(self, prefix: tuple[int, ...], block_len: int, device: torch.device) -> torch.Tensor:
        """Return legal block continuations [N, block_len] for a committed prefix."""
        key = tuple(int(x) for x in prefix)
        cached = self.block_transition_cache.get(key)
        if cached is None:
            node = self._get_tree_node_for_prefix(key)
            blocks = self._enumerate_next_blocks_from_node(
                node,
                block_len=block_len,
                max_candidates=self.block_trie_max_candidates,
            )
            if len(blocks) == 0:
                cached = torch.empty((0, int(block_len)), dtype=torch.long)
            else:
                cached = torch.tensor(blocks, dtype=torch.long)
            self.block_transition_cache[key] = cached
        return cached.to(device=device, non_blocking=True)

    def _get_allowed_mask(self, cur_tokens: torch.Tensor, layer_idx: int, vocab_size: int):
        B = cur_tokens.size(0)
        device = cur_tokens.device
        allowed_mask = torch.zeros(B, vocab_size, device=device, dtype=torch.bool)

        if self.tree_index is None:
            allowed_mask[:] = self.level_vocab_mask[layer_idx].to(device)
            return allowed_mask

        for b in range(B):
            nodes = get_layer_nodes_from_tree(self.tree_index, cur_tokens[b], layer_idx, self.mask_token_id)
            if nodes is None or len(nodes) == 0:
                continue
            for tok_id in nodes:
                if 0 <= tok_id < vocab_size:
                    allowed_mask[b, tok_id] = True

        # Robust fallback: if a partial hypothesis misses the tree due to an
        # earlier diffusion guess, keep level constraints instead of producing NaNs.
        empty = allowed_mask.sum(dim=1) == 0
        if empty.any():
            allowed_mask[empty] = self.level_vocab_mask[layer_idx].to(device)
        return allowed_mask

    def _sample_from_logp(self, logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampled/argmax token ids and their log probabilities."""
        if self.decode_strategy == "sample":
            probs = torch.softmax(logp, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(1)
            score = logp.gather(1, token[:, None]).squeeze(1)
            return token, score
        score, token = torch.max(logp, dim=-1)
        return token, score

    def _apply_top_p(self, logp: torch.Tensor) -> torch.Tensor:
        if self.top_p >= 1.0:
            return logp
        sorted_logits, sorted_indices = torch.sort(logp, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        remove_mask = cum_probs > self.top_p
        remove_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, torch.finfo(logp.dtype).min)
        return torch.full_like(logp, torch.finfo(logp.dtype).min).scatter(1, sorted_indices, sorted_logits)

    def _cfg_logits(
        self,
        xt: torch.Tensor,
        clean_context: torch.Tensor,
        cond_prefix: torch.Tensor,
        uncond_prefix: torch.Tensor,
        t_block: torch.Tensor,
    ) -> torch.Tensor:
        B = xt.size(0)
        attention = self._hdgr_attention(B, xt.device)
        out_c = self.id_generator(
            x0_context=clean_context,
            xt=xt,
            prefix=cond_prefix,
            t_block=t_block,
            attention_mask=attention,
            block_size=self.block_size,
            block_spans=self.block_spans,
        )
        out_u = self.id_generator(
            x0_context=clean_context,
            xt=xt,
            prefix=uncond_prefix,
            t_block=t_block,
            attention_mask=attention,
            block_size=self.block_size,
            block_spans=self.block_spans,
        )
        logits_c = self._mask_invalid_logits(out_c.logits)
        logits_u = self._mask_invalid_logits(out_u.logits)
        if self.temperature != 1.0:
            logits_c = logits_c / self.temperature
            logits_u = logits_u / self.temperature
        return self.guidance_scale * (logits_c - logits_u) + logits_u

    def _refine_current_block(
        self,
        xt: torch.Tensor,
        logits: torch.Tensor,
        start: int,
        end: int,
        keep_fraction: float,
    ) -> torch.Tensor:
        """Commit a temporary block sample and re-mask low-confidence positions."""
        tmp = xt.clone()
        confidences = torch.empty(xt.size(0), end - start, device=xt.device, dtype=logits.dtype)
        logp = F.log_softmax(logits, dim=-1)

        for local_idx, pos in enumerate(range(start, end)):
            step_logp = logp[:, pos, :]
            if self.tree_constraint_in_diffusion and self.tree_constraint_during_refinement:
                allowed = self._get_allowed_mask(tmp, pos, self.vocab_size)
            else:
                allowed = self.level_vocab_mask[pos].to(step_logp.device).unsqueeze(0).expand_as(step_logp)
            step_logp = torch.where(allowed, step_logp, torch.full_like(step_logp, torch.finfo(step_logp.dtype).min))
            step_logp = self._apply_top_p(step_logp)
            token, score = self._sample_from_logp(step_logp)
            tmp[:, pos] = token
            confidences[:, local_idx] = score

        block_len = end - start
        keep_count = int(math.ceil(block_len * max(0.0, min(1.0, keep_fraction))))
        keep_count = max(1, min(block_len, keep_count))
        if keep_count < block_len:
            rank = torch.argsort(confidences, dim=1, descending=True)
            keep = torch.zeros_like(confidences, dtype=torch.bool)
            keep.scatter_(1, rank[:, :keep_count], True)
            for local_idx, pos in enumerate(range(start, end)):
                tmp[~keep[:, local_idx], pos] = self.mask_token_id
        return tmp

    # =====================================================
    # training
    # =====================================================

    def compute_single_batch(self, batch, gpu_id=None):
        if not isinstance(gpu_id, torch.device):
            gpu_id = torch.device(
                f"cuda:{gpu_id}" if isinstance(gpu_id, int)
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

        if len(batch) == 4:
            query, pool, instruct, h_qid = batch
            # In the HDGR generator this tensor is intentionally unused.  It is
            # accepted only for compatibility with old GENIUS dataloaders.
            if torch.is_tensor(instruct):
                _ = instruct.view(-1, instruct.size(-1)).to(gpu_id, non_blocking=True)
        elif len(batch) == 3:
            query, pool, h_qid = batch
        else:
            raise ValueError(f"Unexpected training batch format with {len(batch)} fields; expected 3 or 4.")
        h_qid = h_qid.view(-1)

        q_img_mask = query["img_mask"].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query["txt_mask"].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool["img_mask"].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool["txt_mask"].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        q_img_emb = query["img_emb"].view(-1, query["img_emb"].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query["txt_emb"].view(-1, query["txt_emb"].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool["img_emb"].view(-1, pool["img_emb"].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool["txt_emb"].view(-1, pool["txt_emb"].size(-1)).to(gpu_id, non_blocking=True)

        device = q_img_emb.device
        bs = q_img_emb.size(0)

        with torch.no_grad():
            q_output = self.quantizer.inference(q_img_emb, q_txt_emb, q_img_mask, q_txt_mask)
            p_output = self.quantizer.inference(p_img_emb, p_txt_emb, p_img_mask, p_txt_mask)

        q_emb = F.normalize(q_output["encode"])
        p_codes = p_output["code"].long().to(device)
        target = self._codes_to_token_ids(p_codes).to(device)

        tokens, valid_mask, gt = process_token(target, self.codebook_level, pad_token_id=IGNORE_INDEX)
        tokens = tokens.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        cond_prefix = self._project_condition(q_emb)
        null_prefix = self.null_condition.expand(bs, -1, -1).to(device=device, dtype=cond_prefix.dtype)
        drop_mask = torch.rand(bs, 1, 1, device=device) < self.cond_drop_prob
        prefix = torch.where(drop_mask, null_prefix, cond_prefix)

        if self.bd3_global_noising:
            # BD3-LM / MDLM log-linear noising: sample one global mask
            # probability p per sequence, then mask every semantic-ID token iid
            # with that same p.  We still expand p/t to [B, num_blocks] because
            # the HDGR attention/backbone API is block-aware.  Since every block
            # receives the same p, q_xt ultimately uses a global token-level p.
            t_global, p_global = sample_block_timesteps(
                batch_size=bs,
                num_blocks=1,
                num_timesteps=self.time_step,
                device=device,
                eps_min=self.sampling_eps_min,
                eps_max=self.sampling_eps_max,
            )
            t_block = t_global.expand(bs, self.num_blocks).contiguous()
            p_block = p_global.expand(bs, self.num_blocks).contiguous()
        else:
            t_block, p_block = sample_block_timesteps(
                batch_size=bs,
                num_blocks=self.num_blocks,
                num_timesteps=self.time_step,
                device=device,
                eps_min=self.sampling_eps_min,
                eps_max=self.sampling_eps_max,
            )
        xt, mask_pos, p_token = q_xt(
            x0=tokens,
            p_block=p_block,
            valid_mask=valid_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            block_spans=self.block_spans,
        )

        attention = self._hdgr_attention(bs, device, valid_mask=valid_mask)
        model_outputs = self.id_generator(
            x0_context=tokens,
            xt=xt,
            prefix=prefix,
            t_block=t_block,
            attention_mask=attention,
            block_size=self.block_size,
            block_spans=self.block_spans,
        )
        logits = self._mask_invalid_logits(model_outputs.logits)

        flat_loss = self.criterion(logits.reshape(-1, logits.size(-1)), gt.reshape(-1)).view(bs, self.codebook_level)

        # Training objective variants:
        #   masked_ce          : original HDGR objective, each masked code has unit weight.
        #   bd3_loglinear      : BD3/MDLM-style masked diffusion weighting for the
        #                        log-linear schedule, i.e. NLL weight proportional
        #                        to 1 / mask_probability.  This is enabled only by
        #                        the faithful comparison config and leaves HDGR's
        #                        default behavior unchanged.
        if self.denoising_loss_weighting in {"bd3", "bd3_loglinear", "loglinear"}:
            masked_weight = (1.0 / p_token.clamp_min(self.loss_weight_eps)).detach()
        else:
            masked_weight = torch.ones_like(p_token)

        loss_weight = mask_pos.float() * masked_weight
        if self.visible_loss_weight > 0:
            loss_weight = loss_weight + self.visible_loss_weight * (~mask_pos & (valid_mask > 0)).float()
        loss_weight = loss_weight * (valid_mask > 0).float()
        loss = (flat_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        pred_token_ids = logits.argmax(dim=-1)
        valid_bool = valid_mask > 0
        masked_bool = mask_pos & valid_bool

        # ``R_at_1`` in the training loop is *not* final retrieval Recall@1.
        # It is a strict full semantic-ID exact-match accuracy under randomly
        # masked diffusion training.  For an 8-level ID it can remain 0.0000 for
        # a long time even when token-level denoising is already improving, so
        # we log more diagnostic metrics below.
        code_exact_acc = ((pred_token_ids == gt) | (~valid_bool)).all(dim=1).float().mean()
        token_acc = (((pred_token_ids == gt) & valid_bool).sum().float() / valid_bool.sum().clamp_min(1).float())
        masked_token_acc = (((pred_token_ids == gt) & masked_bool).sum().float() / masked_bool.sum().clamp_min(1).float())

        masked_seq_hits = []
        for row in range(bs):
            row_mask = masked_bool[row]
            if row_mask.any():
                masked_seq_hits.append((pred_token_ids[row, row_mask] == gt[row, row_mask]).all().float())
            else:
                masked_seq_hits.append(torch.tensor(1.0, device=device))
        masked_seq_acc = torch.stack(masked_seq_hits).mean()

        # Hierarchy-aware block metrics replace the old Level1/Level12/Level123
        # prefix exact-match metrics.  Prefix metrics are misleading for HDGR
        # because generation is organized as semantic blocks such as
        # modality/coarse/middle/fine rather than left-to-right token prefixes.
        hierarchy_metrics: dict[str, torch.Tensor] = {}
        for block_idx, ((start, end), raw_name) in enumerate(zip(self.block_spans, self.block_names)):
            start, end = int(start), int(end)
            block_valid = valid_bool[:, start:end]
            block_correct = (pred_token_ids[:, start:end] == gt[:, start:end]) & block_valid
            block_token_acc = block_correct.sum().float() / block_valid.sum().clamp_min(1).float()
            block_exact_acc = (block_correct | (~block_valid)).all(dim=1).float().mean()

            metric_name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw_name).lower()).strip("_")
            metric_name = metric_name or f"block{block_idx}"
            if metric_name == "modality" and end - start == 1:
                hierarchy_metrics["modality_acc"] = block_exact_acc
            else:
                hierarchy_metrics[f"{metric_name}_token_acc"] = block_token_acc
                hierarchy_metrics[f"{metric_name}_block_acc"] = block_exact_acc

        outputs = {
            "loss": loss,
            "lm_loss": loss,
            # Backward-compatible name used by the old engine.  This is strict
            # full-code exact-match, not final retrieval Recall@1.
            "R_at_1": code_exact_acc,
            "code_exact_acc": code_exact_acc,
            "token_acc": token_acc,
            "masked_token_acc": masked_token_acc,
            "masked_seq_acc": masked_seq_acc,
            "hdgr_mask_rate": mask_pos.float().mean().detach(),
        }
        outputs.update(hierarchy_metrics)

        if self.iter % 200 == 0:
            example = self.tokenizer.decode(pred_token_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            example_labels = self.tokenizer.decode(gt[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            block_metric_str = " ".join(
                f"{name}={float(value):.4f}" for name, value in hierarchy_metrics.items()
            )
            print(
                f"HDGR Train Diagnostics | blocks={self.block_sizes} "
                f"mask_rate={float(outputs['hdgr_mask_rate']):.3f} "
                f"token_acc={float(token_acc):.4f} masked_token_acc={float(masked_token_acc):.4f} "
                f"code_exact_acc={float(code_exact_acc):.4f} {block_metric_str} "
                f"| Pred: {example} | Ans: {example_labels}"
            )

        self.iter += 1
        return outputs

    # =====================================================
    # getters
    # =====================================================

    def get_img_preprocess_fn(self):
        return self.clip_model.get_img_preprocess_fn()

    def get_clip_tokenizer(self):
        return self.clip_model.get_tokenizer()

    def get_seq2seq_tokenizer(self):
        # No T5/seq2seq tokenizer is required by HDGR inference.  Returning
        # None tells the MBEIR collator to skip text-id/instruction-id creation.
        return None

    def get_code_tokenizer(self):
        """Return the internal semantic-ID tokenizer for debugging only."""
        return self.tokenizer

    # =====================================================
    # candidate tree cache
    # =====================================================

    def generative_index(self, cand_codes):
        return None

    def distribute_trie(self, cand_codes, trie_save_path):
        """Load or build the candidate Trie once per candidate pool, then reuse it.

        The saved payload stores a cheap candidate-code fingerprint.  If the
        cached file was created by an older quantizer or a different candidate
        pool, rank 0 overwrites it before other ranks load the tree.
        """
        if cand_codes is None:
            return

        save_path = trie_save_path or os.path.join(self.config.genir_dir, "candidate_tree.pkl")
        signature = self._candidate_codes_signature(cand_codes)

        # If this process already holds the same candidate pool, do nothing.
        if self.tree_index is not None and self.cand_codes_signature == signature:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            rebuild = True
            if os.path.exists(save_path):
                try:
                    with open(save_path, "rb") as f:
                        old_payload = pickle.load(f)
                    rebuild = old_payload.get("signature") != signature
                    if rebuild:
                        print(f"Log: Existing candidate Trie at {save_path} is stale; rebuilding.")
                except Exception:
                    rebuild = True
                    print(f"Log: Existing candidate Trie at {save_path} could not be read; rebuilding.")

            if rebuild:
                tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                payload = {
                    "tree": tree,
                    "cand_token_ids": cand_token_ids,
                    "cand_codes": cand_codes_cache,
                    "signature": signature,
                }
                with open(save_path, "wb") as f:
                    pickle.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"Log: Saved candidate Trie to {save_path}.")

        if dist.is_initialized():
            dist.barrier()

        with open(save_path, "rb") as f:
            payload = pickle.load(f)

        payload_signature = payload.get("signature")
        if payload_signature is not None and payload_signature != signature:
            raise RuntimeError(
                "Loaded candidate Trie does not match the provided candidate codes. "
                f"Please remove stale file: {save_path}"
            )

        self.tree_index = payload["tree"]
        self.cand_token_ids = payload["cand_token_ids"]
        self.cand_codes_cache = payload["cand_codes"]
        self.cand_codes_signature = payload_signature or signature
        self.block_transition_cache = {}
        print(f"Log: Loaded candidate Trie from {save_path}.")

    # =====================================================
    # HDGR semi-autoregressive block sampler
    # =====================================================

    def _expand_block_token_beam_fallback(
        self,
        seqs: torch.Tensor,
        scores: torch.Tensor,
        final_logp: torch.Tensor,
        start: int,
        end: int,
        keep_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback used when no candidate Trie is available."""
        B, K0, L = seqs.shape
        V = final_logp.size(-1)
        final_logp = final_logp.view(B, K0, L, V)

        cur_seq = seqs.clone()
        cur_scores = scores.clone()
        origin = torch.arange(K0, device=seqs.device).unsqueeze(0).expand(B, -1).clone()

        for pos in range(start, end):
            cur_k = cur_seq.size(1)
            gather_idx = origin[:, :, None, None].expand(B, cur_k, 1, V)
            step_logp = final_logp[:, :, pos : pos + 1, :].gather(1, gather_idx).squeeze(2)
            allowed = self.level_vocab_mask[pos].to(seqs.device).view(1, 1, V).expand(B, cur_k, V)
            step_logp = torch.where(allowed, step_logp, torch.full_like(step_logp, torch.finfo(step_logp.dtype).min))
            step_logp = self._apply_top_p(step_logp.reshape(B * cur_k, V)).view(B, cur_k, V)

            total_scores = cur_scores.unsqueeze(-1) + step_logp
            flat_total = total_scores.reshape(B, cur_k * V)
            next_k = min(int(keep_k), flat_total.size(1))
            top_scores, top_idx = torch.topk(flat_total, k=next_k, dim=1)
            parent_idx = top_idx // V
            token_idx = top_idx % V

            gathered_seq = cur_seq.gather(1, parent_idx.unsqueeze(-1).expand(-1, -1, L)).clone()
            gathered_seq[:, :, pos] = token_idx
            gathered_origin = origin.gather(1, parent_idx)

            cur_seq = gathered_seq
            cur_scores = top_scores
            origin = gathered_origin

        return cur_seq, cur_scores

    def _deduplicate_sequence_scores(self, seqs: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep one copy of each semantic-ID sequence, using its best score.

        Beam padding and block-level Trie expansion can create many identical
        hypotheses, especially for hierarchy partitions where the first block is
        a single modality token.  Without this step, 50 beams may collapse to a
        few unique candidates even though all beams are legal.  This helper is
        deliberately CPU-keyed because K is small after pruning; it is only used
        inside per-query top-k selection.
        """
        if seqs.numel() == 0 or seqs.size(0) <= 1:
            return seqs, scores
        best = {}
        seqs_cpu = seqs.detach().cpu().tolist()
        scores_cpu = scores.detach().float().cpu().tolist()
        for i, seq in enumerate(seqs_cpu):
            key = tuple(int(x) for x in seq)
            score = float(scores_cpu[i])
            prev = best.get(key)
            if prev is None or score > prev[0]:
                best[key] = (score, i)
        if len(best) == seqs.size(0):
            return seqs, scores
        keep_indices = [idx for _, idx in best.values()]
        idx_tensor = torch.as_tensor(keep_indices, device=seqs.device, dtype=torch.long)
        return seqs.index_select(0, idx_tensor), scores.index_select(0, idx_tensor)

    def _expand_block_with_constraints(
        self,
        seqs: torch.Tensor,          # [B, K0, L]
        scores: torch.Tensor,        # [B, K0]
        final_logp: torch.Tensor,    # [B*K0, L, V]
        start: int,
        end: int,
        keep_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Block-level Trie decoding.

        For each committed prefix, enumerate legal next semantic-ID blocks from
        the candidate Trie, score each whole block by the HDGR denoising
        log-probabilities, and keep the best block continuations.  This is the
        retrieval adaptation of block-by-block denoising sampler: diffusion
        proposes token distributions inside the current block, while the Trie is
        consulted only at the block boundary to ensure every retained hypothesis
        is a legal database item prefix.
        """
        if self.tree_index is None:
            return self._expand_block_token_beam_fallback(seqs, scores, final_logp, start, end, keep_k)

        B, K0, L = seqs.shape
        V = final_logp.size(-1)
        block_len = int(end - start)
        final_logp = final_logp.view(B, K0, L, V)
        device = seqs.device
        dtype = scores.dtype
        keep_k = max(1, int(keep_k))

        out_seqs = []
        out_scores = []

        for b in range(B):
            batch_seq_chunks = []
            batch_score_chunks = []

            # Remove duplicate parent beams before expansion.  This is crucial
            # after the modality block: a single legal modality would otherwise
            # be repeated K times, and every next-block candidate would be scored
            # K times, causing top-k to be dominated by duplicates.
            parent_seqs = seqs[b]
            parent_scores = scores[b]
            parent_indices = list(range(K0))
            if self.deduplicate_block_beams and K0 > 1:
                best_parent = {}
                parent_cpu = parent_seqs[:, :start].detach().cpu().tolist() if start > 0 else [[] for _ in range(K0)]
                parent_score_cpu = parent_scores.detach().float().cpu().tolist()
                for k in range(K0):
                    key = tuple(int(x) for x in parent_cpu[k])
                    score = float(parent_score_cpu[k])
                    prev = best_parent.get(key)
                    if prev is None or score > prev[0]:
                        best_parent[key] = (score, k)
                parent_indices = [idx for _, idx in best_parent.values()]

            for k in parent_indices:
                prefix = tuple(int(x) for x in seqs[b, k, :start].detach().cpu().tolist())
                block_cands = self._get_next_block_candidates(prefix, block_len=block_len, device=device)
                if block_cands.numel() == 0:
                    continue

                # Score a complete next block as sum_j log p(x_{start+j} | prefix, query).
                block_scores = scores[b, k].expand(block_cands.size(0)).clone()
                for j, pos in enumerate(range(start, end)):
                    tok = block_cands[:, j].clamp(min=0, max=V - 1)
                    block_scores = block_scores + final_logp[b, k, pos, tok]

                parent = seqs[b, k].unsqueeze(0).expand(block_cands.size(0), -1).clone()
                parent[:, start:end] = block_cands
                batch_seq_chunks.append(parent)
                batch_score_chunks.append(block_scores)

            if len(batch_seq_chunks) == 0:
                # Defensive fallback: should only happen when the Trie was built
                # from a mismatched candidate pool.  Preserve progress with
                # level-constrained decoding instead of crashing evaluation.
                fallback_seq, fallback_score = self._expand_block_token_beam_fallback(
                    seqs[b : b + 1], scores[b : b + 1], final_logp[b : b + 1].reshape(K0, L, V), start, end, keep_k
                )
                out_seqs.append(fallback_seq.squeeze(0))
                out_scores.append(fallback_score.squeeze(0))
                continue

            all_batch_seqs = torch.cat(batch_seq_chunks, dim=0)
            all_batch_scores = torch.cat(batch_score_chunks, dim=0)
            if self.deduplicate_block_beams:
                all_batch_seqs, all_batch_scores = self._deduplicate_sequence_scores(all_batch_seqs, all_batch_scores)
            top_n = min(keep_k, all_batch_scores.numel())
            top_scores, top_idx = torch.topk(all_batch_scores, k=top_n, dim=0)
            top_seqs = all_batch_seqs.index_select(0, top_idx)

            if top_n < keep_k:
                # Pad by repeating the best hypothesis to keep tensor shapes stable.
                pad_n = keep_k - top_n
                top_seqs = torch.cat([top_seqs, top_seqs[:1].expand(pad_n, -1)], dim=0)
                top_scores = torch.cat([top_scores, top_scores[:1].expand(pad_n)], dim=0)

            out_seqs.append(top_seqs)
            out_scores.append(top_scores.to(dtype=dtype))

        return torch.stack(out_seqs, dim=0), torch.stack(out_scores, dim=0)

    @torch.no_grad()
    def constrained_beam_search(self, inputs_embeds, attention_mask=None, num_beams: int = 10, cand_codes=None):
        """
        HDGR block-wise sampler.

        Blocks are generated from left to right.  Within a block, the model runs
        masked diffusion refinement jointly, then a block-level beam expansion
        applies level masks and, when available, the candidate tree.
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        B0 = inputs_embeds.size(0)
        L = self.codebook_level
        K = max(1, int(num_beams))

        if cand_codes is not None:
            signature = self._candidate_codes_signature(cand_codes)
            if self.tree_index is None or self.cand_codes_signature != signature:
                tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
                self.tree_index = tree
                self.cand_token_ids = cand_token_ids
                self.cand_codes_cache = cand_codes_cache
                self.cand_codes_signature = signature
                self.block_transition_cache = {}
                print("Log: Built candidate Trie inside inference because no reusable Trie was loaded.")

        cur_seq = torch.full((B0, 1, L), self.mask_token_id, device=device, dtype=torch.long)
        beam_scores = torch.zeros(B0, 1, device=device, dtype=dtype)

        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(B0, -1, -1).to(device=device, dtype=dtype)

        for block_index, (start, end) in enumerate(self.block_spans):
            cur_k = cur_seq.size(1)
            flat_seq = cur_seq.reshape(B0 * cur_k, L)

            xt = torch.full_like(flat_seq, self.mask_token_id)
            if start > 0:
                xt[:, :start] = flat_seq[:, :start]
            clean_context = self._clean_context_from_prefix(flat_seq, start)

            cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(B0 * cur_k, self.num_prefix, self.d_model)
            uncond = base_uncond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(B0 * cur_k, self.num_prefix, self.d_model)

            final_logits = None
            steps = max(1, int(self.diffusion_steps))
            for step_idx in range(steps, 0, -1):
                t_val = max(1, int(math.ceil(step_idx / steps * self.time_step)))
                t_block = torch.zeros(B0 * cur_k, self.num_blocks, device=device, dtype=torch.long)
                t_block[:, block_index] = t_val
                logits = self._cfg_logits(xt, clean_context, cond, uncond, t_block)
                final_logits = logits

                if step_idx > 1:
                    keep_fraction = 1.0 - float(step_idx - 1) / float(steps)
                    xt = self._refine_current_block(xt, logits, start, end, keep_fraction=keep_fraction)
                    if start > 0:
                        xt[:, :start] = flat_seq[:, :start]
                    xt[:, end:] = self.mask_token_id

            assert final_logits is not None
            final_logp = F.log_softmax(final_logits, dim=-1)
            cur_seq, beam_scores = self._expand_block_with_constraints(
                seqs=cur_seq,
                scores=beam_scores,
                final_logp=final_logp,
                start=start,
                end=end,
                keep_k=K,
            )

        return cur_seq

    # =====================================================
    # inference
    # =====================================================

    @torch.no_grad()
    def inference(self, img_emb, txt_emb, img_mask, txt_mask, inst_ids, num_beams: int = 10, cand_codes=None):
        q_output = self.quantizer.inference(img_emb, txt_emb, img_mask, txt_mask)
        emb = F.normalize(q_output["encode"])
        cond_prefix = self._project_condition(emb)
        output_token_ids = self.constrained_beam_search(
            inputs_embeds=cond_prefix,
            num_beams=num_beams,
            cand_codes=(cand_codes if self.tree_index is None else None),
        )
        output_codes = self._token_ids_to_codes(output_token_ids)
        return output_codes.detach(), F.normalize(img_emb * img_mask + txt_emb * txt_mask)

    # =====================================================
    # encode batch
    # =====================================================

    def encode_mbeir_batch(self, batch, num_beams=10, cand_codes=None, trie_save_path=None, init_dataset=True):
        if init_dataset:
            self.distribute_trie(cand_codes, trie_save_path)

        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        img_emb, txt_emb = self.clip_model.encode_multimodal_input(batch["image_batched"], batch["txt_batched"])
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)
        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."

        output, embeddings = self.inference(
            img_emb,
            txt_emb,
            img_mask,
            txt_mask,
            batch["inst_ids"],
            num_beams=num_beams,
            cand_codes=cand_codes,
        )
        return output, embeddings, torch.LongTensor(id_list)

    # =====================================================
    # forward
    # =====================================================

    def forward(
        self,
        input=None,
        encode_mbeir_batch: bool = False,
        num_beams: int = 10,
        cand_codes=None,
        init_dataset: bool = True,
        trie_save_path=None,
        gpu_id=None,
    ):
        if isinstance(gpu_id, torch.device):
            target_device = gpu_id
        elif isinstance(gpu_id, int):
            target_device = torch.device(f"cuda:{gpu_id}")
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(target_device)
        if hasattr(self, "quantizer"):
            self.quantizer.to(target_device)
        if hasattr(self, "id_generator"):
            self.id_generator.to(target_device)
        if hasattr(self, "embed_projector"):
            self.embed_projector.to(target_device)

        if encode_mbeir_batch:
            return self.encode_mbeir_batch(
                input,
                cand_codes=cand_codes,
                num_beams=num_beams,
                init_dataset=init_dataset,
                trie_save_path=trie_save_path,
            )
        return self.compute_single_batch(input, target_device)
