"""
Generative Retriever Model Implementation
Diffusion / Non-Autoregressive / GPT-style Version
- keep original T5ForGenerativeRetrieval external interface
- training follows: process_token -> sample_time -> forward_noising -> attention mask -> denoise
- inference follows: iterative decoding + CFG + tree constraint + diverse beam
"""

# Standard library
import os
import re
import math
import random
import string
import pickle
import hashlib
from types import SimpleNamespace
from typing import Optional, Tuple, List, Dict, Any

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# Local modules
from models.uniir_clip import utils
from models.residual_quantization.residual_quantization import RQ
from models.residual_quantization.loss import ClipLoss
from models.hdgr_comparison.tf_adpt_grad import GPT2LMHeadModel
from models.hdgr_comparison.soundstorm_diffusion import SoundStormTokenDiffusion
from models.hdgr_comparison.soundstorm_full_model import ConditionalSoundStormDALLE
from models.hdgr_comparison.residual_reconstruction_gain import (
    residual_reconstruction_gain,
    selected_residual_reconstruction_gain,
)
from models.hdgr_comparison.complete_id_selection import (
    inbatch_complete_id_ranking_loss,
    sample_prefix_neighbor_codes,
    sampled_hard_negative_ranking_loss,
)
from models.hdgr_comparison.tree_risk import (
    CompactTrieAsset,
    compute_multi_positive_branch_loss,
    compute_inbatch_prefix_risk_loss,
    tree_risk_weight_scale,
)
from models.hdgr_comparison.adaptive_residual_composition import (
    AdaptiveResidualComposer,
    composition_weight_scale,
    compute_counterfactual_composition_loss,
)
from models.hdgr_comparison.gpu_flat_trie import FlatPrefixTrie

IGNORE_INDEX = -100


# =========================================================
# small helpers
# =========================================================

def cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def sample_time(batch_size: int, num_timesteps, device: torch.device):
    """
    Return:
        t: [B], values in [1, num_timesteps]
    """
    if torch.is_tensor(num_timesteps):
        num_timesteps = int(num_timesteps.item())
    else:
        num_timesteps = int(num_timesteps)

    t = torch.randint(1, num_timesteps + 1, (batch_size,), device=device, dtype=torch.long)
    return t


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

    B, L = target.shape
    if L != code_length:
        raise ValueError(f"Expected code_length={code_length}, got {L}")

    tokens = target.clone().long()
    gt = target.clone().long()
    mask = (target != pad_token_id).long()
    return tokens, mask, gt


def _maskgit_schedule_ratio_from_t(
    t: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    random_ratio: bool = True,
) -> torch.Tensor:
    """Return a MaskGIT-style ratio in [0, 1) tied to the sampled timestep.

    Official MaskGIT samples a continuous ``ratio ~ Uniform(0, 1)`` during
    training and then applies a mask schedule such as ``cos(pi / 2 * ratio)``.
    GPT-HDGR already has a discrete timestep embedding.  To keep the timestep
    meaningful while matching MaskGIT's continuous corruption distribution, we
    sample a uniform value inside the selected timestep bin:

        ratio = (t - 1 + u) / T,  u ~ Uniform(0, 1)

    Marginally this is Uniform(0, 1) when t is sampled uniformly from 1..T.
    """
    T = max(1, int(num_timesteps))
    if t.dim() > 1:
        t = t.view(t.size(0), -1)[:, 0]
    t = t.to(device=device, dtype=torch.float32).clamp(min=1.0, max=float(T))

    if random_ratio:
        ratio = (t - 1.0 + torch.rand_like(t)) / float(T)
    else:
        # Deterministic bin centers are useful for debugging/reproducibility.
        ratio = (t - 0.5) / float(T)
    return ratio.clamp(min=0.0, max=1.0 - 1.0e-6).unsqueeze(1)


def _sample_exact_mask_positions(
    valid_mask: torch.Tensor,
    mask_ratio: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample exactly round(mask_ratio * valid_length) positions per sample.

    This follows MaskGIT's corruption style more closely than iid Bernoulli
    masking: first decide how many tokens to mask, then uniformly sample that
    many token positions without replacement.
    """
    valid = valid_mask.to(device=device).bool()
    B, L = valid.shape
    if mask_ratio.dim() > 1:
        mask_ratio = mask_ratio.view(B, -1)[:, 0]
    mask_ratio = mask_ratio.to(device=device, dtype=torch.float32).clamp(min=0.0, max=1.0)

    noise_pos = torch.zeros((B, L), device=device, dtype=torch.bool)
    for b in range(B):
        valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
        n_valid = int(valid_idx.numel())
        if n_valid <= 0:
            continue
        n_mask = int(torch.round(mask_ratio[b] * n_valid).item())
        n_mask = max(1, min(n_mask, n_valid))
        perm = torch.randperm(n_valid, device=device)[:n_mask]
        noise_pos[b, valid_idx[perm]] = True
    return noise_pos


def forward_noising(tokens: torch.Tensor,
                    t: torch.Tensor,
                    mask: torch.Tensor,
                    model,
                    device: torch.device):
    """
    tokens: [B, L], clean token ids
    t:      [B, 1] or [B]
    mask:   [B, L], valid positions
    Return:
        noised_x: [B, L], masked positions replaced by model.mask_token_id

    Supported training noising schedules:
      - ``linear``: legacy GPT-HDGR iid Bernoulli mask with ratio t / T.
      - ``maskgit_cosine`` / ``maskgit`` / ``cosine``: MaskGIT-style cosine
        schedule.  By default it samples an exact number of masked tokens per
        sample using mask_ratio = cos(pi / 2 * r), where r is uniform in [0, 1)
        and tied to the discrete timestep bin.
    """
    module = model.module if hasattr(model, "module") else model
    mask_token_id = module.mask_token_id
    num_timesteps = max(1, int(module.time_step))

    B, L = tokens.shape
    if t.dim() == 1:
        t = t.unsqueeze(1)

    schedule = str(getattr(module, "training_mask_schedule", "linear")).lower()
    use_maskgit = schedule in {"maskgit", "maskgit_cosine", "cosine"}

    if use_maskgit:
        ratio = _maskgit_schedule_ratio_from_t(
            t=t,
            num_timesteps=num_timesteps,
            device=device,
            random_ratio=bool(getattr(module, "maskgit_random_ratio", True)),
        )
        mask_ratio = torch.cos(0.5 * math.pi * ratio).clamp(min=1.0e-6, max=1.0)
        if bool(getattr(module, "maskgit_exact_num_mask", True)):
            noise_pos = _sample_exact_mask_positions(mask, mask_ratio, device)
        else:
            rand = torch.rand(B, L, device=device)
            noise_pos = (rand < mask_ratio) & (mask > 0)
    else:
        ratio = t.float() / float(num_timesteps)
        ratio = ratio.clamp(min=0.0, max=1.0)
        rand = torch.rand(B, L, device=device)
        noise_pos = (rand < ratio) & (mask > 0)

    # ensure at least one valid masked position per sample
    none_noised = noise_pos.sum(dim=1) == 0
    if none_noised.any():
        rows = torch.nonzero(none_noised, as_tuple=False).squeeze(1)
        for r in rows.tolist():
            valid_idx = torch.nonzero(mask[r] > 0, as_tuple=False).squeeze(1)
            if len(valid_idx) > 0:
                pick = valid_idx[torch.randint(0, len(valid_idx), (1,), device=device)]
                noise_pos[r, pick] = True

    noised_x = tokens.clone()
    noised_x[noise_pos] = mask_token_id
    return noised_x


def forward_prefix_suffix_noising(tokens: torch.Tensor,
                                  t: torch.Tensor,
                                  mask: torch.Tensor,
                                  model,
                                  device: torch.device):
    """Mask one contiguous suffix per sample.

    The number of masked tokens follows the same ``t / T`` schedule used by
    :func:`forward_noising`.  This preserves the non-autoregressive denoising
    objective while exposing the model to the exact prefix-visible / suffix-
    masked states encountered by prefix-regeneration decoding.

    Args:
        tokens: Clean token ids with shape ``[B, L]``.
        t: Diffusion timestep with shape ``[B]`` or ``[B, 1]``.
        mask: Valid-token mask with shape ``[B, L]``.
        model: Retriever or DDP-wrapped retriever.
        device: Target device.

    Returns:
        A noised tensor with at least one valid suffix token masked in every
        non-empty sample.
    """
    module = model.module if hasattr(model, "module") else model
    mask_token_id = int(module.mask_token_id)
    num_timesteps = max(1, int(module.time_step))

    if t.dim() > 1:
        t = t.view(t.size(0), -1)[:, 0]
    t = t.to(device=device, dtype=torch.long).clamp(min=1, max=num_timesteps)

    valid = mask.to(device=device).bool()
    valid_lengths = valid.sum(dim=1)
    ratios = t.float() / float(num_timesteps)
    num_to_mask = torch.ceil(ratios * valid_lengths.float()).long()
    num_to_mask = torch.where(valid_lengths > 0, num_to_mask.clamp_min(1), torch.zeros_like(num_to_mask))
    num_to_mask = torch.minimum(num_to_mask, valid_lengths)

    # The semantic-ID sequences used here are left aligned.  Defining the
    # suffix relative to each valid length also keeps this helper safe if
    # variable-length/padded codes are introduced later.
    start = valid_lengths - num_to_mask
    positions = torch.arange(tokens.size(1), device=device).unsqueeze(0)
    suffix_mask = (positions >= start.unsqueeze(1)) & (positions < valid_lengths.unsqueeze(1)) & valid

    noised_x = tokens.clone()
    noised_x[suffix_mask] = mask_token_id
    return noised_x


def build_concentrate_attention_mask(mask_tokens: torch.Tensor,
                                     valid_mask: torch.Tensor,
                                     mask_token_id: int):
    """
    Same masking idea as the reference code.

    Input:
        mask_tokens: [B, L]
        valid_mask:  [B, L], 1 valid / 0 padding
    Return:
        all_mask: [B, L, L], additive mask in [-10000, 0]
    """
    device = mask_tokens.device
    B, L = mask_tokens.shape

    ex_mask = torch.zeros_like(mask_tokens, dtype=torch.float32) - 10000.0
    ex_nomask = torch.zeros_like(mask_tokens, dtype=torch.float32)

    all_mask = torch.where(mask_tokens == mask_token_id, ex_mask, ex_nomask)  # [B, L]
    all_mask = all_mask.unsqueeze(1).repeat_interleave(repeats=L, dim=1)      # [B, L, L]

    eye = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
    all_mask = all_mask.masked_fill(eye, 0.0)

    padding_mask = valid_mask.unsqueeze(1).float()
    padding_mask = (1.0 - padding_mask) * -10000.0

    all_mask = torch.clamp(all_mask + padding_mask, -10000.0, 0.0)
    return all_mask


def get_layer_nodes_from_tree(tree: dict, tokens_1d: torch.Tensor, layer_idx: int, mask_token_id: int):
    """
    tree: nested dict {token_id: subtree}
    tokens_1d: [L]
    layer_idx: current level index
    """
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
# loss
# =========================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.01, metric: str = 'cos', bidirection: bool = True, gather: bool = True):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.gather = gather
        self.bidirection = bidirection

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def forward(self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logit is None:
            if utils.get_world_size() > 1 and self.gather:
                x = torch.cat(utils.GatherLayer.apply(x), dim=0)
                y = torch.cat(utils.GatherLayer.apply(y), dim=0)

            assert x is not None and y is not None
            if self.metric == 'cos':
                logits_per_x = F.linear(F.normalize(x), F.normalize(y))
            elif self.metric == 'euclid':
                logits_per_x = -torch.cdist(x, y) ** 2
            else:
                raise ValueError(f'Invalid metric: {self.metric}')
            labels = self.get_ground_truth(x.device, x.shape[0])
        else:
            logits_per_x = logit
            labels = self.get_ground_truth(logit.device, logit.shape[0])

        logits_per_x = logits_per_x / self.temperature
        logits_per_y = logits_per_x.T

        if self.bidirection:
            total_loss = (F.cross_entropy(logits_per_x, labels) + F.cross_entropy(logits_per_y, labels)) / 2
        else:
            total_loss = F.cross_entropy(logits_per_x, labels)
        return total_loss


# =========================================================
# GPT-style diffusion backbone
# =========================================================

class RetrieverDiffusionGPT(nn.Module):
    """
    Continue the forward semantics of ClipCaptionModel:
      - tokens:      clean token ids
      - mask_tokens: current xt, positions == mask_token_id are treated as masked
      - prefix:      conditional prefix [B, P, D]
      - mask:        custom additive attention mask [B, L, L]
      - t:           timestep [B]
    """

    def __init__(
        self,
        vocab_size: int,        # predicted vocab size (real vocab only)
        num_classes: int,       # input classes = vocab_size + 1, last class is mask
        d_model: int,
        code_length: int,
        num_prefix: int,
        time_step: int = 8,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.mask_token_id = num_classes - 1
        self.d_model = d_model
        self.code_length = code_length
        self.num_prefix = num_prefix
        self.time_step = time_step

        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max(code_length + 8, 32),
            n_ctx=max(code_length + 8, 32),
            n_embd=d_model,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        cfg = cfg.__dict__.copy()
        cfg.update({'scale_attn_by_inverse_layer_idx': False})
        cfg.update({'reorder_and_upcast_attn': False})
        cfg.update({'use_cache': False})
        cfg = GPT2Config(**cfg)

        self.gpt = GPT2LMHeadModel(cfg)

        # input embedding allows extra mask class
        self.input_embed = nn.Embedding(num_classes, d_model)

        # output head only predicts real vocab, not the mask class
        self.gpt.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.token_ln = nn.LayerNorm(d_model)
        self.prefix_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,                 # [B, L]
        mask_tokens: torch.Tensor,            # [B, L]
        prefix: torch.Tensor,                 # [B, P, D]
        mask: Optional[torch.Tensor] = None,  # [B, L, L]
        t: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.Tensor] = None,
    ):
        B, L = tokens.shape
        if L != self.code_length:
            raise ValueError(f"Expected code_length={self.code_length}, got {L}")
        if mask_tokens.shape != tokens.shape:
            raise ValueError("mask_tokens and tokens must have the same shape")

        embedding_text = self.input_embed(tokens)  # [B, L, D]

        mask_pos = (mask_tokens == self.mask_token_id).unsqueeze(-1)
        mask_emb = self.mask_embedding.view(1, 1, -1).expand_as(embedding_text)
        embedding_text = torch.where(mask_pos, mask_emb, embedding_text)

        embedding_text = self.token_ln(embedding_text)
        prefix = self.prefix_ln(prefix)

        out = self.gpt(
            t=t,
            inputs_embeds=embedding_text,
            labels=labels,
            attention_mask=mask,
            encoder_hidden_states=prefix
        )
        return out

    def hidden_at_position(
        self,
        tokens: torch.Tensor,
        mask_tokens: torch.Tensor,
        prefix: torch.Tensor,
        position: int,
        mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states only at one semantic-ID position.

        During tree-constrained decoding we only need logits for the position
        being committed.  The original non-AR eval path materialized full
        [B*K, L, vocab] logits, which is unnecessarily expensive for a
        google-t5/t5-small tokenizer plus added semantic-ID tokens.  This method
        mirrors GENIUS/HF generate more closely by computing the Transformer
        hidden state and letting the caller project only the legal level tokens.
        """
        B, L = tokens.shape
        if L != self.code_length:
            raise ValueError(f"Expected code_length={self.code_length}, got {L}")
        if mask_tokens.shape != tokens.shape:
            raise ValueError("mask_tokens and tokens must have the same shape")
        if not 0 <= int(position) < L:
            raise ValueError(f"position must be in [0, {L}), got {position}")

        embedding_text = self.input_embed(tokens)
        mask_pos = (mask_tokens == self.mask_token_id).unsqueeze(-1)
        mask_emb = self.mask_embedding.view(1, 1, -1).expand_as(embedding_text)
        embedding_text = torch.where(mask_pos, mask_emb, embedding_text)

        embedding_text = self.token_ln(embedding_text)
        prefix = self.prefix_ln(prefix)

        transformer_outputs = self.gpt.transformer(
            t,
            input_ids=None,
            past_key_values=None,
            attention_mask=mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=embedding_text,
            encoder_hidden_states=prefix,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs[0]
        if getattr(self.gpt, "model_parallel", False):
            hidden_states = hidden_states.to(self.gpt.lm_head.weight.device)
        return hidden_states[:, int(position), :]


# =========================================================
# main model
# =========================================================

class T5ForGenerativeRetrieval(nn.Module):
    def __init__(self, config=None, tokenizer=None, clip_model=None, new_tokenizer=True, init_rq_codebook=True):
        super().__init__()

        # -------- clip --------
        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()

        self.config = config

        # -------- quantizer --------
        self.quantizer = RQ(config=config, clip_model=clip_model)
        runtime_q = cfg_get(cfg_get(config, "runtime", {}), "quantizer_path", "")
        q_path = runtime_q or cfg_get(config.codebook_config, "quantizer_path", "")
        rq_model_path = q_path if os.path.isabs(str(q_path)) else os.path.join(config.genir_dir, str(q_path))
        if not os.path.exists(rq_model_path):
            # Keep the GPT baseline convenient under the HDGR repo layout.
            candidates = []
            ckpt_root = os.path.join(config.genir_dir, "checkpoint")
            for root, _, files in os.walk(ckpt_root):
                for fn in files:
                    if fn.startswith("rq_clip") and fn.endswith(".pth"):
                        candidates.append(os.path.join(root, fn))
            if candidates:
                rq_model_path = sorted(candidates, key=os.path.getmtime)[-1]
                print(f"[GPT baseline] Auto-discovered RQ quantizer: {rq_model_path}")
        if not os.path.exists(rq_model_path):
            raise FileNotFoundError(f"Missing RQ quantizer checkpoint for GPT baseline: {rq_model_path}")
        self.quantizer.load_state_dict(
            torch.load(rq_model_path, map_location=torch.device('cpu'), weights_only=False)["model"],
            strict=False
        )
        self.quantizer.eval()
        for _, param in self.quantizer.named_parameters():
            param.requires_grad = False

        self.modality_index = self.quantizer.modality_index
        self.codebook_vocab = self.quantizer.codebook_vocab
        self.codebook_level = self.quantizer.codebook_level
        if self.quantizer.unique_code:
            self.codebook_level = self.codebook_level + 1

        # -------- tokenizer --------
        self.tokenizer = tokenizer
        self.code_tokens = []
        if new_tokenizer:
            self._initialize_tokenizer(tokenizer)
        self._initialize_codebook_tokens()

        # -------- backbone config --------
        # GPT baseline does not need a T5 backbone.  The original experimental
        # file loaded google-t5/t5-small only to read d_model=512; this breaks
        # in offline/local environments, so prefer config.model.d_model.
        model_cfg = cfg_get(config, "model", {})
        self.d_model = int(cfg_get(model_cfg, "d_model", 512))
        self.num_prefix = cfg_get(model_cfg, "num_prefix", 30)
        self.time_step = cfg_get(model_cfg, "time_step", self.codebook_level)
        self.guidance_scale = cfg_get(model_cfg, "guidance_scale", 3)
        self.cond_drop_prob = cfg_get(model_cfg, "cond_drop_prob", 0.1)
        self.top_p = cfg_get(model_cfg, "top_p", 1.0)
        self.temperature = cfg_get(model_cfg, "temperature", 2.0)
        self.diversity_strength = cfg_get(model_cfg, "diversity_strength", 0.4)
        self.delayed_diversity_start = cfg_get(model_cfg, "delayed_diversity_start", 2)
        self.decode_order = "ltr"
        # Memory-efficient tree decoding: compute logits only for the current
        # semantic-ID level / candidate-tree-allowed tokens instead of full
        # [B*K, L, vocab] logits.  This mirrors GENIUS constrained generation
        # more closely and is important for google-t5/t5-small tokenizer ablations.
        self.memory_efficient_tree_decode = bool(
            cfg_get(model_cfg, "memory_efficient_tree_decode", False)
        )
        # Experimental systems baseline: preserve the level-wise sequential
        # scoring rule while replacing per-beam Python-dict Trie traversal by
        # a level-wise CSR representation and batched tensor gathers.  It is
        # deliberately opt-in and is not enabled by paper-facing configs.
        self.gpu_flat_tree_decode = bool(
            cfg_get(model_cfg, "gpu_flat_tree_decode", False)
        )

        # -------- training/inference alignment --------
        # These options are disabled by default so old GPT_diffusion configs and
        # checkpoints keep their original behavior.  The optimized GPT-HDGR
        # config enables them explicitly.
        default_mix_alpha = cfg_get(cfg_get(config, "hyperparameter_config", {}), "alpha", 2.0)
        self.query_positive_mix_prob = float(cfg_get(model_cfg, "query_positive_mix_prob", 0.0))
        self.query_positive_mix_alpha = float(cfg_get(model_cfg, "query_positive_mix_alpha", default_mix_alpha))
        self.query_positive_mix_elementwise = bool(cfg_get(model_cfg, "query_positive_mix_elementwise", False))
        self.prefix_suffix_noising_prob = float(cfg_get(model_cfg, "prefix_suffix_noising_prob", 0.0))
        self.visible_token_loss_weight = float(cfg_get(model_cfg, "visible_token_loss_weight", 0.1))
        self.training_mask_schedule = str(cfg_get(model_cfg, "training_mask_schedule", "linear")).lower()
        maskgit_default_exact = self.training_mask_schedule in {"maskgit", "maskgit_cosine", "cosine"}
        self.maskgit_exact_num_mask = bool(cfg_get(model_cfg, "maskgit_exact_num_mask", maskgit_default_exact))
        self.maskgit_random_ratio = bool(cfg_get(model_cfg, "maskgit_random_ratio", True))
        self.training_objective = str(cfg_get(model_cfg, "training_objective", "masked_ce")).lower()
        self.soundstorm_diffusion_step = int(cfg_get(model_cfg, "soundstorm_diffusion_step", 100))
        self.soundstorm_alpha_init_type = str(cfg_get(model_cfg, "soundstorm_alpha_init_type", "alpha1"))
        self.soundstorm_auxiliary_loss_weight = float(cfg_get(model_cfg, "soundstorm_auxiliary_loss_weight", 5.0e-4))
        self.soundstorm_adaptive_auxiliary_loss = bool(cfg_get(model_cfg, "soundstorm_adaptive_auxiliary_loss", True))
        self.soundstorm_mask_weight = tuple(cfg_get(model_cfg, "soundstorm_mask_weight", [1.0, 1.0]))
        self.soundstorm_full_num_embed = int(cfg_get(model_cfg, "soundstorm_num_embed", self.codebook_vocab))
        self.soundstorm_full_n_layer = int(cfg_get(model_cfg, "soundstorm_n_layer", 16))
        self.soundstorm_full_n_embd = int(cfg_get(model_cfg, "soundstorm_n_embd", self.d_model))
        self.soundstorm_full_n_head = int(cfg_get(model_cfg, "soundstorm_n_head", 8))
        self.soundstorm_full_attn_type = str(cfg_get(model_cfg, "soundstorm_attn_type", "selfcross"))
        self.soundstorm_full_attn_pdrop = float(cfg_get(model_cfg, "soundstorm_attn_pdrop", 0.0))
        self.soundstorm_full_resid_pdrop = float(cfg_get(model_cfg, "soundstorm_resid_pdrop", 0.0))
        self.soundstorm_full_timestep_type = str(cfg_get(model_cfg, "soundstorm_timestep_type", "adalayernorm"))
        self.soundstorm_full_mlp_hidden_times = int(cfg_get(model_cfg, "soundstorm_mlp_hidden_times", 4))
        self.soundstorm_full_block_activate = str(cfg_get(model_cfg, "soundstorm_block_activate", "GELU2"))
        self.soundstorm_full_max_size = int(cfg_get(model_cfg, "soundstorm_content_max_size", max(16, self.codebook_level)))
        self.soundstorm_sample_steps = int(cfg_get(model_cfg, "soundstorm_sample_steps", self.soundstorm_diffusion_step))
        self.soundstorm_decode_mode = str(
            cfg_get(model_cfg, "soundstorm_decode_mode", "free_sample")
        ).lower()
        # Prefix-regeneration decoding is the SoundStorm analogue of the old
        # GPT-HDGR Trie-in-the-loop protocol: for semantic-ID level j, keep the
        # committed prefix, mask the suffix, rerun the denoiser, and expand only
        # legal children of the current candidate-Trie node.
        self.soundstorm_prefix_regen_chunk_size = int(
            cfg_get(model_cfg, "soundstorm_prefix_regen_chunk_size", 128)
        )
        self.soundstorm_prefix_regen_use_cfg = bool(
            cfg_get(model_cfg, "soundstorm_prefix_regen_use_cfg", True)
        )
        self.soundstorm_prefix_regen_timestep_mode = str(
            cfg_get(model_cfg, "soundstorm_prefix_regen_timestep_mode", "mask_fraction")
        ).lower()
        self.loss_normalization = str(cfg_get(model_cfg, "loss_normalization", "global")).lower()
        self.beam_score_normalization = str(cfg_get(model_cfg, "beam_score_normalization", "none")).lower()
        self.one_pass_full_mask_probability = float(
            cfg_get(model_cfg, "one_pass_full_mask_probability", 0.0)
        )
        self.one_pass_ranking_weight = float(cfg_get(model_cfg, "one_pass_ranking_weight", 0.0))
        self.one_pass_ranking_temperature = float(
            cfg_get(model_cfg, "one_pass_ranking_temperature", 1.0)
        )
        self.one_pass_candidate_codes_path = str(
            cfg_get(model_cfg, "one_pass_candidate_codes_path", "") or ""
        )
        self.one_pass_candidate_sample_size = int(
            cfg_get(model_cfg, "one_pass_candidate_sample_size", 4096)
        )
        self.one_pass_hard_negatives = int(cfg_get(model_cfg, "one_pass_hard_negatives", 32))
        self.one_pass_sorted_candidate_codes_path = str(
            cfg_get(model_cfg, "one_pass_sorted_candidate_codes_path", "") or ""
        )
        self.one_pass_prefix_depths = tuple(
            int(x) for x in cfg_get(model_cfg, "one_pass_prefix_depths", [1, 2, 3, 4, 5, 6, 7])
        )
        self.one_pass_candidates_per_depth = int(
            cfg_get(model_cfg, "one_pass_candidates_per_depth", 16)
        )
        self._one_pass_candidate_codes = None
        self._one_pass_sorted_candidate_codes = None
        if not 0.0 <= self.one_pass_full_mask_probability <= 1.0:
            raise ValueError("model.one_pass_full_mask_probability must be in [0,1]")

        # Residual-Quantization Compatibility (RQC). ``rrg_*`` remains the
        # serialized field prefix for backward compatibility.
        self.use_rrg = bool(
            cfg_get(model_cfg, "use_rrg", cfg_get(model_cfg, "use_pro_geometric_fusion", False))
        )
        self.rrg_weight = float(
            cfg_get(model_cfg, "rrg_weight", cfg_get(model_cfg, "pro_geometric_weight", 0.0))
        )
        self.rrg_normalize = bool(
            cfg_get(model_cfg, "rrg_normalize", cfg_get(model_cfg, "pro_geometric_normalize_bias", False))
        )
        self.rrg_skip_modality = bool(
            cfg_get(model_cfg, "rrg_skip_modality", cfg_get(model_cfg, "pro_geometric_skip_modality", True))
        )
        self.rrg_eps = float(
            cfg_get(model_cfg, "rrg_eps", cfg_get(model_cfg, "pro_geometric_eps", 1.0e-6))
        )
        self.rqc_score_mode = str(cfg_get(model_cfg, "rqc_score_mode", "gain")).lower()
        if self.rqc_score_mode not in {"gain", "alignment", "cosine"}:
            raise ValueError("model.rqc_score_mode must be gain, alignment, or cosine")
        self.rqc_apply_from_level = int(cfg_get(model_cfg, "rqc_apply_from_level", 1))
        self.rqc_prefix_start_level = int(cfg_get(model_cfg, "rqc_prefix_start_level", 0))
        level_weights = cfg_get(model_cfg, "rqc_level_weights", None)
        if level_weights is None:
            self.rqc_level_weights = tuple(1.0 for _ in range(self.codebook_level))
        else:
            self.rqc_level_weights = tuple(float(x) for x in level_weights)
            if len(self.rqc_level_weights) != self.codebook_level:
                raise ValueError(
                    f"model.rqc_level_weights needs {self.codebook_level} values"
                )
        self._rrg_codebook_moment_cache = {}

        # CompoGR: semantic IDs from ResidualVQ are generated as residual
        # components, not as unrelated flat categories.  A small controller
        # learns a state-dependent mixture over several residual-explanation
        # strengths. Fixed RRG remains available as an ablation.
        self.use_adaptive_residual_composition = bool(
            cfg_get(model_cfg, "use_adaptive_residual_composition", False)
        )
        self.residual_composition_experts = tuple(
            float(x) for x in cfg_get(
                model_cfg, "residual_composition_experts", [0.0, 10.0, 50.0, 100.0]
            )
        )
        self.residual_composition_hidden_dim = int(
            cfg_get(model_cfg, "residual_composition_hidden_dim", 128)
        )
        self.residual_composition_level_dim = int(
            cfg_get(model_cfg, "residual_composition_level_dim", 24)
        )
        self.residual_composition_dropout = float(
            cfg_get(model_cfg, "residual_composition_dropout", 0.1)
        )
        self.residual_composition_init_strength = float(
            cfg_get(model_cfg, "residual_composition_init_strength", 10.0)
        )
        self.residual_composition_init_probabilities = tuple(
            float(x) for x in cfg_get(
                model_cfg,
                "residual_composition_init_probabilities",
                [0.10, 0.86, 0.03, 0.01],
            )
        )
        self.residual_composition_detach_state_features = bool(
            cfg_get(model_cfg, "residual_composition_detach_state_features", True)
        )
        self.residual_composition_route_weight = float(
            cfg_get(model_cfg, "residual_composition_route_weight", 0.10)
        )
        self.residual_composition_branch_weight = float(
            cfg_get(model_cfg, "residual_composition_branch_weight", 0.05)
        )
        self.residual_composition_oracle_temperature = float(
            cfg_get(model_cfg, "residual_composition_oracle_temperature", 0.5)
        )
        self.residual_composition_normalize_oracle_utility = bool(
            cfg_get(model_cfg, "residual_composition_normalize_oracle_utility", False)
        )
        self.residual_composition_advantage_temperature = float(
            cfg_get(model_cfg, "residual_composition_advantage_temperature", 0.10)
        )
        self.residual_composition_min_utility_gain = float(
            cfg_get(model_cfg, "residual_composition_min_utility_gain", 0.02)
        )
        self.residual_composition_min_winner_gap = float(
            cfg_get(model_cfg, "residual_composition_min_winner_gap", 0.01)
        )
        self.residual_composition_no_harm_weight = float(
            cfg_get(model_cfg, "residual_composition_no_harm_weight", 2.0)
        )
        self.residual_composition_clear_ce_weight = float(
            cfg_get(model_cfg, "residual_composition_clear_ce_weight", 0.25)
        )
        self.residual_composition_max_queries = int(
            cfg_get(model_cfg, "residual_composition_max_queries", 64)
        )
        self.residual_composition_levels_per_query = int(
            cfg_get(model_cfg, "residual_composition_levels_per_query", 1)
        )
        self.residual_composition_min_level = int(
            cfg_get(model_cfg, "residual_composition_min_level", 1 if self.modality_index else 0)
        )
        self.residual_composition_max_level = int(
            cfg_get(model_cfg, "residual_composition_max_level", self.codebook_level - 1)
        )
        self.residual_composition_use_cfg = bool(
            cfg_get(model_cfg, "residual_composition_use_cfg", False)
        )
        self.residual_composition_warmup_steps = int(
            cfg_get(model_cfg, "residual_composition_warmup_steps", 500)
        )
        self.residual_composition_ramp_steps = int(
            cfg_get(model_cfg, "residual_composition_ramp_steps", 2000)
        )
        self.residual_composition_asset_path = str(
            cfg_get(model_cfg, "residual_composition_asset_path", "")
        )
        self.residual_composition_asset = None
        self.residual_composer = None

        # Tree-Risk learning.  Both terms are evaluated on exact
        # prefix-regeneration states rather than ordinary iid masks.
        self.use_tree_risk = bool(cfg_get(model_cfg, "use_tree_risk", False))
        self.tree_risk_asset_path = str(cfg_get(model_cfg, "tree_risk_asset_path", ""))
        self.tree_branch_weight = float(cfg_get(model_cfg, "tree_branch_weight", 0.2))
        self.tree_prefix_risk_weight = float(cfg_get(model_cfg, "tree_prefix_risk_weight", 0.1))
        self.tree_levels_per_sample = int(cfg_get(model_cfg, "tree_levels_per_sample", 1))
        self.tree_branch_queries_per_batch = int(
            cfg_get(model_cfg, "tree_branch_queries_per_batch", 64)
        )
        self.tree_min_level = int(cfg_get(model_cfg, "tree_min_level", 1 if self.modality_index else 0))
        self.tree_max_level = int(cfg_get(model_cfg, "tree_max_level", self.codebook_level - 1))
        self.tree_branch_temperature = float(cfg_get(model_cfg, "tree_branch_temperature", 1.0))
        self.tree_branch_require_competition = bool(
            cfg_get(model_cfg, "tree_branch_require_competition", True)
        )
        self.tree_risk_queries_per_batch = int(
            cfg_get(model_cfg, "tree_risk_queries_per_batch", 8)
        )
        self.tree_risk_hard_negatives = int(
            cfg_get(model_cfg, "tree_risk_hard_negatives", 4)
        )
        self.tree_risk_max_positive_paths = int(
            cfg_get(model_cfg, "tree_risk_max_positive_paths", 4)
        )
        self.tree_risk_margin = float(cfg_get(model_cfg, "tree_risk_margin", 0.2))
        self.tree_risk_temperature = float(cfg_get(model_cfg, "tree_risk_temperature", 1.0))
        self.tree_risk_use_cfg = bool(cfg_get(model_cfg, "tree_risk_use_cfg", True))
        self.tree_risk_warmup_steps = int(cfg_get(model_cfg, "tree_risk_warmup_steps", 0))
        self.tree_risk_ramp_steps = int(cfg_get(model_cfg, "tree_risk_ramp_steps", 1000))
        self.tree_risk_asset = None

        self.codebook_init_scale = float(cfg_get(model_cfg, "codebook_init_scale", 1.0))
        # GENIUS-style semantic-ID string tokenization ablation.
        # Default remains the faster internal direct code->token-id mapping.
        # When enabled, codes are converted to strings such as
        # "<a3> <b17> <c91>" and then passed through the code-token tokenizer,
        # matching the official GENIUS target construction style.
        self.use_genius_string_tokenizer_path = bool(
            cfg_get(model_cfg, "use_genius_string_tokenizer_path", False)
        )
        self.genius_string_tokenizer_separator = str(
            cfg_get(model_cfg, "genius_string_tokenizer_separator", " ")
        )
        self.strict_genius_tokenizer_check = bool(
            cfg_get(model_cfg, "strict_genius_tokenizer_check", True)
        )
        self.semantic_tokenizer_backend = str(
            cfg_get(model_cfg, "semantic_tokenizer_backend", "wordlevel")
        ).lower()
        self.use_google_t5_tokenizer = bool(
            cfg_get(model_cfg, "use_google_t5_tokenizer", False)
        ) or self.semantic_tokenizer_backend in {"google_t5", "google-t5", "t5", "t5-small"}
        self.google_t5_tokenizer_name = str(
            cfg_get(model_cfg, "google_t5_tokenizer_name", "google-t5/t5-small")
        )
        self.google_t5_local_files_only = bool(
            cfg_get(model_cfg, "google_t5_local_files_only", False)
        )

        if not 0.0 <= self.query_positive_mix_prob <= 1.0:
            raise ValueError("model.query_positive_mix_prob must be in [0, 1]")
        if self.query_positive_mix_alpha <= 0:
            raise ValueError("model.query_positive_mix_alpha must be positive")
        if not 0.0 <= self.prefix_suffix_noising_prob <= 1.0:
            raise ValueError("model.prefix_suffix_noising_prob must be in [0, 1]")
        if self.visible_token_loss_weight < 0:
            raise ValueError("model.visible_token_loss_weight must be non-negative")
        allowed_mask_schedules = {"linear", "maskgit", "maskgit_cosine", "cosine"}
        if self.training_mask_schedule not in allowed_mask_schedules:
            raise ValueError(
                f"model.training_mask_schedule must be one of {sorted(allowed_mask_schedules)}, "
                f"got {self.training_mask_schedule!r}"
            )
        allowed_objectives = {"masked_ce", "ce", "soundstorm", "soundstorm_vqdiffusion", "vqdiffusion", "soundstorm_full", "soundstorm_fullmodel", "soundstorm_dalle"}
        if self.training_objective not in allowed_objectives:
            raise ValueError(
                f"model.training_objective must be one of {sorted(allowed_objectives)}, "
                f"got {self.training_objective!r}"
            )
        if len(self.soundstorm_mask_weight) != 2:
            raise ValueError("model.soundstorm_mask_weight must contain exactly two weights [mask, non_mask]")
        if self.loss_normalization not in {"global", "per_sample"}:
            raise ValueError("model.loss_normalization must be 'global' or 'per_sample'")
        if self.beam_score_normalization not in {"none", "mean"}:
            raise ValueError("model.beam_score_normalization must be 'none' or 'mean'")
        if not math.isfinite(self.rrg_weight):
            raise ValueError("model.rrg_weight must be finite")
        if self.rrg_eps <= 0.0 or not math.isfinite(self.rrg_eps):
            raise ValueError("model.rrg_eps must be finite and positive")
        if len(self.residual_composition_experts) < 2:
            raise ValueError("model.residual_composition_experts needs at least two strengths")
        if any(not math.isfinite(x) for x in self.residual_composition_experts):
            raise ValueError("model.residual_composition_experts must be finite")
        if abs(float(self.residual_composition_experts[0])) > 1.0e-8:
            raise ValueError("model.residual_composition_experts must start with 0.0")
        if any(
            self.residual_composition_experts[i] > self.residual_composition_experts[i + 1]
            for i in range(len(self.residual_composition_experts) - 1)
        ):
            raise ValueError("model.residual_composition_experts must be sorted")
        if self.residual_composition_hidden_dim <= 0 or self.residual_composition_level_dim <= 0:
            raise ValueError("CompoGR hidden and level dimensions must be positive")
        if not 0.0 <= self.residual_composition_dropout < 1.0:
            raise ValueError("model.residual_composition_dropout must be in [0,1)")
        if self.residual_composition_route_weight < 0.0 or self.residual_composition_branch_weight < 0.0:
            raise ValueError("CompoGR loss weights must be non-negative")
        if self.residual_composition_oracle_temperature <= 0.0:
            raise ValueError("model.residual_composition_oracle_temperature must be positive")
        if len(self.residual_composition_init_probabilities) != len(self.residual_composition_experts):
            raise ValueError("residual_composition_init_probabilities must match experts")
        if any((not math.isfinite(x)) or x < 0.0 for x in self.residual_composition_init_probabilities):
            raise ValueError("residual_composition_init_probabilities must be finite and non-negative")
        if sum(self.residual_composition_init_probabilities) <= 0.0:
            raise ValueError("residual_composition_init_probabilities must have positive mass")
        if self.residual_composition_advantage_temperature <= 0.0:
            raise ValueError("residual_composition_advantage_temperature must be positive")
        if self.residual_composition_min_utility_gain < 0.0 or self.residual_composition_min_winner_gap < 0.0:
            raise ValueError("CompoGR V2 utility thresholds must be non-negative")
        if self.residual_composition_no_harm_weight < 0.0 or self.residual_composition_clear_ce_weight < 0.0:
            raise ValueError("CompoGR V2 objective weights must be non-negative")
        if self.residual_composition_max_queries < 0 or self.residual_composition_levels_per_query <= 0:
            raise ValueError("CompoGR query count must be non-negative and levels_per_query positive")
        if self.use_adaptive_residual_composition and self.use_rrg:
            raise ValueError(
                "Use either adaptive residual composition or fixed RRG, not both. "
                "Set model.use_rrg=false for CompoGR."
            )
        if self.use_adaptive_residual_composition and self.training_objective in {
            "soundstorm_full", "soundstorm_fullmodel", "soundstorm_dalle"
        }:
            raise ValueError("CompoGR V1 supports native GPT-HDGR/HDGR, not SoundStorm full model")
        if self.tree_branch_weight < 0.0 or self.tree_prefix_risk_weight < 0.0:
            raise ValueError("Tree-Risk loss weights must be non-negative")
        if self.tree_levels_per_sample <= 0:
            raise ValueError("model.tree_levels_per_sample must be positive")
        if self.tree_branch_temperature <= 0.0 or self.tree_risk_temperature <= 0.0:
            raise ValueError("Tree-Risk temperatures must be positive")
        if self.tree_branch_queries_per_batch < 0:
            raise ValueError("model.tree_branch_queries_per_batch must be non-negative")
        if self.tree_risk_queries_per_batch < 0 or self.tree_risk_hard_negatives < 0:
            raise ValueError("Tree-Risk query/negative counts must be non-negative")
        if self.soundstorm_prefix_regen_chunk_size <= 0:
            raise ValueError("model.soundstorm_prefix_regen_chunk_size must be positive")
        if self.soundstorm_prefix_regen_timestep_mode not in {"mask_fraction", "linear"}:
            raise ValueError(
                "model.soundstorm_prefix_regen_timestep_mode must be "
                "'mask_fraction' or 'linear'"
            )

        # -------- vocab / mask class --------
        self.vocab_size = len(self.tokenizer)      # tokenizer vocab for legacy GPT-HDGR path
        self.num_classes = self.vocab_size + 1     # plus one extra mask class for legacy path
        self.mask_token_id = self.num_classes - 1
        self.use_soundstorm_full_model = self.training_objective in {
            "soundstorm_full", "soundstorm_fullmodel", "soundstorm_dalle"
        }

        # -------- condition projector --------
        self.embed_projector = nn.Linear(768, self.d_model * self.num_prefix)

        # Level buffers are required both by legacy tokenizer IDs and by the
        # compact SoundStorm full model for level-specific output masking.
        self._build_level_token_buffers()

        if self.use_soundstorm_full_model:
            # Full SoundStorm model path: compact code classes [0, codebook_vocab)
            # rather than global tokenizer IDs.  This mirrors SoundStorm's per-quantizer codebook heads and avoids a huge tokenizer-class diffusion.
            level_sizes = [int(x) for x in self.level_vocab_sizes.detach().cpu().tolist()]
            self.soundstorm_full_model = ConditionalSoundStormDALLE(
                num_embed=self.soundstorm_full_num_embed,
                n_q=self.codebook_level,
                condition_dim=self.d_model,
                diffusion_step=self.soundstorm_diffusion_step,
                alpha_init_type=self.soundstorm_alpha_init_type,
                auxiliary_loss_weight=self.soundstorm_auxiliary_loss_weight,
                adaptive_auxiliary_loss=self.soundstorm_adaptive_auxiliary_loss,
                mask_weight=self.soundstorm_mask_weight,
                n_layer=self.soundstorm_full_n_layer,
                n_embd=self.soundstorm_full_n_embd,
                n_head=self.soundstorm_full_n_head,
                attn_pdrop=self.soundstorm_full_attn_pdrop,
                resid_pdrop=self.soundstorm_full_resid_pdrop,
                mlp_hidden_times=self.soundstorm_full_mlp_hidden_times,
                block_activate=self.soundstorm_full_block_activate,
                attn_type=self.soundstorm_full_attn_type,
                timestep_type=self.soundstorm_full_timestep_type,
                max_size=self.soundstorm_full_max_size,
                level_vocab_sizes=level_sizes,
            )
            self.soundstorm_diffusion = self.soundstorm_full_model.diffusion
            self.id_generator = None
            # For generic logging/noising helpers; full model has its own mask id.
            self.mask_token_id = self.soundstorm_full_model.mask_token_id
        else:
            # Optional SoundStorm / VQ-Diffusion objective on top of the legacy
            # GPT-HDGR denoiser.
            self.soundstorm_diffusion = SoundStormTokenDiffusion(
                num_classes=self.num_classes,
                num_timesteps=self.soundstorm_diffusion_step,
                alpha_init_type=self.soundstorm_alpha_init_type,
                auxiliary_loss_weight=self.soundstorm_auxiliary_loss_weight,
                adaptive_auxiliary_loss=self.soundstorm_adaptive_auxiliary_loss,
                mask_weight=self.soundstorm_mask_weight,
            )
            self.id_generator = RetrieverDiffusionGPT(
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
                d_model=self.d_model,
                code_length=self.codebook_level,
                num_prefix=self.num_prefix,
                time_step=self.time_step,
                num_layers=cfg_get(model_cfg, "nar_num_layers", 6),
                num_heads=cfg_get(model_cfg, "nar_num_heads", 8),
                dropout=cfg_get(model_cfg, "nar_dropout", 0.1),
            )

        # unconditional prefix
        self.null_condition = nn.Parameter(
            torch.randn(1, self.num_prefix, self.d_model) * 0.02
        )

        if self.use_adaptive_residual_composition:
            self.residual_composer = AdaptiveResidualComposer(
                embedding_dim=768,
                num_levels=self.codebook_level,
                expert_strengths=self.residual_composition_experts,
                hidden_dim=self.residual_composition_hidden_dim,
                level_dim=self.residual_composition_level_dim,
                dropout=self.residual_composition_dropout,
                init_strength=self.residual_composition_init_strength,
                init_probabilities=self.residual_composition_init_probabilities,
                detach_state_features=self.residual_composition_detach_state_features,
            )

        self.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
        self.iter = 0
        self.alpha = cfg_get(cfg_get(config, 'hyperparameter_config', {}), 'alpha', 2)

        retrieval_cfg = cfg_get(config, "retrieval_config", {})
        self.save_beam_scores = bool(
            cfg_get(
                retrieval_cfg,
                "save_beam_scores",
                cfg_get(retrieval_cfg, "use_hybrid_score", False),
            )
        )

        # candidate-tree cache
        self.tree_index = None
        self.cand_token_ids = None
        self.cand_codes_cache = None
        self.flat_tree_index = None
        self._flat_tree_device = None

        if self.use_tree_risk:
            if not self.tree_risk_asset_path:
                raise ValueError(
                    "model.use_tree_risk=true requires model.tree_risk_asset_path"
                )
            asset_path = self.tree_risk_asset_path
            if not os.path.isabs(asset_path):
                asset_path = os.path.join(config.genir_dir, asset_path)
            if not os.path.exists(asset_path):
                raise FileNotFoundError(
                    f"Tree-Risk asset not found: {asset_path}. "
                    "Tree-Risk is not included in this cleaned release."
                )
            self.tree_risk_asset = CompactTrieAsset.load(asset_path)
            if int(self.tree_risk_asset.code_length) != int(self.codebook_level):
                raise ValueError(
                    "Tree-Risk asset code length mismatch: "
                    f"asset={self.tree_risk_asset.code_length}, model={self.codebook_level}"
                )
            print(
                "[Tree-Risk] Loaded training Trie asset: "
                f"path={asset_path} nodes={self.tree_risk_asset.num_nodes} "
                f"qids={len(self.tree_risk_asset.qid_positive_paths)} "
                f"branch_w={self.tree_branch_weight:.3f} "
                f"risk_w={self.tree_prefix_risk_weight:.3f} "
                f"use_cfg={self.tree_risk_use_cfg}"
            )

        if self.use_adaptive_residual_composition and self.residual_composition_asset_path:
            comp_asset_path = self.residual_composition_asset_path
            if not os.path.isabs(comp_asset_path):
                comp_asset_path = os.path.join(config.genir_dir, comp_asset_path)
            if not os.path.exists(comp_asset_path):
                raise FileNotFoundError(
                    f"CompoGR supervision asset not found: {comp_asset_path}. "
                    "The existing Tree-Risk asset may be reused."
                )
            if (
                self.tree_risk_asset is not None
                and os.path.abspath(comp_asset_path)
                == os.path.abspath(self.tree_risk_asset_path if os.path.isabs(self.tree_risk_asset_path) else os.path.join(config.genir_dir, self.tree_risk_asset_path))
            ):
                self.residual_composition_asset = self.tree_risk_asset
            else:
                self.residual_composition_asset = CompactTrieAsset.load(comp_asset_path)
            if int(self.residual_composition_asset.code_length) != int(self.codebook_level):
                raise ValueError(
                    "CompoGR asset code length mismatch: "
                    f"asset={self.residual_composition_asset.code_length}, model={self.codebook_level}"
                )
            print(
                "[CompoGR] Loaded multi-positive composition asset: "
                f"path={comp_asset_path} nodes={self.residual_composition_asset.num_nodes} "
                f"qids={len(self.residual_composition_asset.qid_positive_paths)}"
            )

        # level buffers were built before model construction.
        if self.use_genius_string_tokenizer_path:
            self._validate_genius_string_tokenizer_path()

        if init_rq_codebook and not self.use_soundstorm_full_model:
            self._initialize_codebook_embeddings()

        print(
            f"[GPT baseline] Initialized GPT-style diffusion generator | "
            f"code_length={self.codebook_level} vocab={self.vocab_size} "
            f"layers={cfg_get(model_cfg, 'nar_num_layers', 6)} heads={cfg_get(model_cfg, 'nar_num_heads', 8)} "
            f"prefix={self.num_prefix} query_mix={self.query_positive_mix_prob:.2f} "
            f"suffix_noising={self.prefix_suffix_noising_prob:.2f} "
            f"visible_loss={self.visible_token_loss_weight:.3f} "
            f"mask_schedule={self.training_mask_schedule} "
            f"maskgit_exact={self.maskgit_exact_num_mask} "
            f"objective={self.training_objective} "
            f"soundstorm_full={self.use_soundstorm_full_model} "
            f"soundstorm_layers={getattr(self, 'soundstorm_full_n_layer', 0)} "
            f"soundstorm_steps={self.soundstorm_diffusion_step} "
            f"soundstorm_decode={self.soundstorm_decode_mode} "
            f"ss_prefix_chunk={self.soundstorm_prefix_regen_chunk_size} "
            f"ss_prefix_cfg={self.soundstorm_prefix_regen_use_cfg} "
            f"ss_prefix_t={self.soundstorm_prefix_regen_timestep_mode} "
            f"soundstorm_aux={self.soundstorm_auxiliary_loss_weight:.1e} "
            f"codebook_scale={self.codebook_init_scale:.3f} "
            f"genius_string_tokenizer={self.use_genius_string_tokenizer_path} "
            f"tokenizer_backend={getattr(self, 'semantic_tokenizer_backend', 'unknown')} "
            f"lowmem_tree_decode={getattr(self, 'memory_efficient_tree_decode', False)} "
            f"gpu_flat_tree_decode={getattr(self, 'gpu_flat_tree_decode', False)} "
            f"rrg={self.use_rrg} "
            f"rrg_weight={self.rrg_weight:.4f} "
            f"rrg_normalize={self.rrg_normalize} "
            f"rrg_skip_modality={self.rrg_skip_modality} "
            f"adaptive_residual={self.use_adaptive_residual_composition} "
            f"residual_experts={list(self.residual_composition_experts)} "
            f"residual_init={self.residual_composition_init_strength:.2f} "
            f"tree_risk={self.use_tree_risk} "
            f"tree_branch_w={self.tree_branch_weight:.3f} "
            f"tree_prefix_w={self.tree_prefix_risk_weight:.3f}"
        )

    # =====================================================
    # tokenizer / codebook init
    # =====================================================

    def _initialize_tokenizer(self, tokenizer):
        model_cfg = cfg_get(self.config, "model", {})
        semantic_backend = str(cfg_get(model_cfg, "semantic_tokenizer_backend", "wordlevel")).lower()
        use_google_t5 = bool(cfg_get(model_cfg, "use_google_t5_tokenizer", False)) or semantic_backend in {
            "google_t5",
            "google-t5",
            "t5",
            "t5-small",
        }

        if use_google_t5:
            name = str(cfg_get(model_cfg, "google_t5_tokenizer_name", "google-t5/t5-small"))
            local_only = bool(cfg_get(model_cfg, "google_t5_local_files_only", False))
            print(
                f"[Tokenizer] Loading google T5 tokenizer for semantic IDs: {name} "
                f"(local_files_only={local_only})"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                model_max_length=42,
                local_files_only=local_only,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.semantic_tokenizer_backend = "google_t5"
            return

        special_tokens = {
            'pad_token': '<pad>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
        }

        # create a fresh minimal tokenizer
        new_vocab = {}
        for tok in special_tokens.values():
            if tok not in new_vocab:
                new_vocab[tok] = len(new_vocab)

        tokenizer_model = WordLevel(vocab=new_vocab, unk_token='<unk>')
        tokenizer_fast = Tokenizer(tokenizer_model)
        tokenizer_fast.pre_tokenizer = Whitespace()
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer_fast, **special_tokens)
        self.tokenizer = tokenizer_fast
        self.semantic_tokenizer_backend = "wordlevel"

    def _initialize_codebook_tokens(self):
        self.level_indicators = list(string.ascii_lowercase[:self.codebook_level])

        for l, level in enumerate(self.level_indicators):
            if self.modality_index and l == 0:
                for i in range(3):
                    self.code_tokens.append(f'<{level}{i}>')
                continue
            for i in range(self.codebook_vocab):
                self.code_tokens.append(f'<{level}{i}>')

        # For the google-t5/t5-small tokenizer path, add semantic-ID tokens
        # as special added tokens so strings like "<a123>" stay atomic.
        if getattr(self, "semantic_tokenizer_backend", "wordlevel") == "google_t5":
            _ = self.tokenizer.add_tokens(self.code_tokens, special_tokens=True)
        else:
            _ = self.tokenizer.add_tokens(self.code_tokens)

    def _build_level_token_buffers(self):
        vocab_size = len(self.tokenizer)
        max_code_size = self.codebook_vocab

        level_token_ids = torch.full((self.codebook_level, max_code_size), -1, dtype=torch.long)
        level_vocab_mask = torch.zeros((self.codebook_level, vocab_size), dtype=torch.bool)
        token_id_to_code = torch.full((vocab_size,), -1, dtype=torch.long)
        level_vocab_sizes = []

        cursor = 0
        for l in range(self.codebook_level):
            level_size = 3 if (self.modality_index and l == 0) else self.codebook_vocab
            level_vocab_sizes.append(level_size)

            cur_tokens = self.code_tokens[cursor: cursor + level_size]
            cur_token_ids = self.tokenizer.convert_tokens_to_ids(cur_tokens)
            cur_token_ids = torch.tensor(cur_token_ids, dtype=torch.long)

            level_token_ids[l, :level_size] = cur_token_ids
            level_vocab_mask[l, cur_token_ids] = True
            token_id_to_code[cur_token_ids] = torch.arange(level_size, dtype=torch.long)

            cursor += level_size

        self.register_buffer("level_token_ids", level_token_ids, persistent=False)
        self.register_buffer("level_vocab_mask", level_vocab_mask, persistent=False)
        self.register_buffer("token_id_to_code", token_id_to_code, persistent=False)
        self.register_buffer("level_vocab_sizes", torch.tensor(level_vocab_sizes, dtype=torch.long), persistent=False)

    def _initialize_codebook_embeddings(self):
        if getattr(self, "id_generator", None) is None:
            return
        new_token_ids = self.tokenizer.convert_tokens_to_ids(self.code_tokens)
        embedding_dim = self.d_model
        linear_layer = nn.Linear(768, embedding_dim, bias=False)

        if self.modality_index:
            first_layer_codebook = self.quantizer.residual_rq.layers[0]._codebook.embed[0, :3, :]
            mapped_first_layer_embeddings = linear_layer(F.normalize(first_layer_codebook))

            other_layers_codebooks = [layer._codebook.embed for layer in self.quantizer.residual_rq.layers[1:]]
            other_layers_codebooks = torch.stack(other_layers_codebooks, dim=0)
            other_layers_codebooks = rearrange(other_layers_codebooks, 'q 1 c d -> q c d')
            other_layers_codebook = other_layers_codebooks.reshape(-1, other_layers_codebooks.shape[-1])
            mapped_other_layers_embeddings = linear_layer(F.normalize(other_layers_codebook))

            mapped_embeddings = torch.cat([mapped_first_layer_embeddings, mapped_other_layers_embeddings], dim=0)
        else:
            codebook_vectors = self.quantizer.residual_rq.codebooks.reshape(-1, 768)
            mapped_embeddings = linear_layer(F.normalize(codebook_vectors))

        with torch.no_grad():
            scale = float(getattr(self, "codebook_init_scale", 1.0))
            for idx, token_id in enumerate(new_token_ids):
                self.id_generator.input_embed.weight[token_id].copy_(mapped_embeddings[idx] * scale)
                self.id_generator.gpt.lm_head.weight[token_id].copy_(mapped_embeddings[idx] * scale)

    # =====================================================
    # misc helpers
    # =====================================================

    def transform_row(self, row, separator=''):
        transformed_row = []
        for l, level_indicator in enumerate(self.level_indicators):
            new_value = row[l]
            transformed_row.append(f"<{level_indicator}{new_value}>")
        return separator.join(transformed_row)

    def detransform_row(self, row):
        splited_row = re.findall(r'<.*?>', row)
        detransformed_row = []
        for token in splited_row:
            match = re.match(r'<([a-z])(\d+)>', token)
            if match:
                _, value = match.groups()
                detransformed_row.append(int(value))
        return detransformed_row


    def _encode_code_strings_with_tokenizer(self, codes: torch.Tensor) -> torch.Tensor:
        """GENIUS-style code -> string -> tokenizer id path.

        This is an ablation path.  It intentionally routes semantic IDs through
        code strings such as "<a3> <b17> <c91>" and the code-token tokenizer,
        instead of directly indexing `level_token_ids`.

        When `semantic_tokenizer_backend=google_t5`, the tokenizer is
        `google-t5/t5-small` with semantic-ID tokens added as special tokens.
        This mirrors GENIUS target construction using a real T5 tokenizer.
        """
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        device = codes.device
        rows = codes.detach().cpu().long().tolist()
        code_texts = [
            self.transform_row(row, separator=self.genius_string_tokenizer_separator)
            for row in rows
        ]
        encoded = self.tokenizer(
            code_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]

        bad = []
        for i, ids in enumerate(input_ids):
            if len(ids) != self.codebook_level:
                bad.append((i, code_texts[i], ids))
            elif any((tok is None or tok < 0) for tok in ids):
                bad.append((i, code_texts[i], ids))
        if bad:
            preview = bad[:5]
            raise ValueError(
                "GENIUS-style tokenizer path produced invalid code-token IDs. "
                f"Expected exactly {self.codebook_level} atomic tokens per row. "
                f"Examples: {preview}"
            )
        return torch.tensor(input_ids, dtype=torch.long, device=device)

    def _validate_genius_string_tokenizer_path(self):
        """Fail early if code special tokens are not atomic for string tokenization."""
        if not self.strict_genius_tokenizer_check:
            return

        # Check the first few tokens from each level and the last token of each
        # level.  This catches tokenizer splitting / unk routing immediately.
        samples = []
        cursor = 0
        for level, size in enumerate(self.level_vocab_sizes.tolist()):
            size = int(size)
            if size <= 0:
                continue
            idxs = sorted(set([0, min(1, size - 1), min(2, size - 1), size - 1]))
            for idx in idxs:
                samples.append(self.code_tokens[cursor + idx])
            cursor += size

        for tok in samples:
            ids = self.tokenizer(tok, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(
                    f"Semantic token {tok!r} is not atomic under the tokenizer: {ids}"
                )
            back = self.tokenizer.convert_ids_to_tokens(ids[0])
            if back != tok:
                raise ValueError(
                    f"Semantic token {tok!r} maps to {ids[0]} -> {back!r}; expected identity."
                )

        # End-to-end row check with a valid zero code.
        zero_code = torch.zeros((1, self.codebook_level), dtype=torch.long)
        ids = self._encode_code_strings_with_tokenizer(zero_code)
        if tuple(ids.shape) != (1, self.codebook_level):
            raise ValueError(
                f"Tokenizer row sanity check failed: got shape {tuple(ids.shape)}"
            )
        print(
            "[GENIUS-tokenizer ablation] Enabled string semantic-ID path: "
            "code -> '<a0> <b0> ...' -> tokenizer ids"
        )


    def _codes_to_token_ids(self, codes: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(codes)

        if codes.dim() == 1:
            codes = codes.unsqueeze(0)

        if codes.dim() != 2:
            raise ValueError(f"`codes` must be 1D or 2D, got shape={tuple(codes.shape)}")

        if codes.size(1) != self.codebook_level:
            raise ValueError(
                f"Expected codes shape [B, {self.codebook_level}], got {tuple(codes.shape)}"
            )

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
            for b, l in bad.tolist():
                bad_info.append({
                    "batch": b,
                    "level": l,
                    "value": int(codes[b, l].item()),
                    "valid_range": [0, int(level_vocab_sizes[0, l].item()) - 1],
                })
            raise ValueError(f"Invalid code index found: {bad_info}")

        if getattr(self, "use_genius_string_tokenizer_path", False):
            return self._encode_code_strings_with_tokenizer(codes)

        level_token_ids = self.level_token_ids.to(device)
        level_idx = torch.arange(self.codebook_level, device=device).unsqueeze(0).expand(codes.size(0), -1)
        token_ids = level_token_ids[level_idx, codes]
        return token_ids

    def _token_ids_to_codes(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_id_to_code.to(token_ids.device)[token_ids.long()]

    # =====================================================
    # Residual Reconstruction Gain helpers
    # =====================================================

    def _rrg_enabled_for_level(self, level: int) -> bool:
        # The residual-explanation score is shared by the fixed-weight ablation
        # and CompoGR's learned state-dependent composition.
        if not (self.use_rrg or self.use_adaptive_residual_composition):
            return False
        if (
            self.use_rrg
            and not self.use_adaptive_residual_composition
            and abs(float(self.rrg_weight)) <= 0.0
        ):
            return False
        level = int(level)
        if level < 0 or level >= int(self.codebook_level):
            return False
        if level < self.rqc_apply_from_level:
            return False
        if self.modality_index and self.rrg_skip_modality and level == 0:
            return False
        return True

    def _rqc_level_weight(self, level: int) -> float:
        return float(self.rqc_level_weights[int(level)])

    def _rrg_level_codebook(
        self,
        level: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return the exact RQ codebook used at one semantic-ID level."""
        level = int(level)
        layer = self.quantizer.residual_rq.layers[level]
        embed = layer._codebook.embed
        while embed.dim() > 2:
            embed = embed[0]
        level_size = int(self.level_vocab_sizes[level].item())
        if embed.dim() != 2 or embed.size(0) < level_size:
            raise RuntimeError(
                f"Invalid RQ codebook at level {level}: shape={tuple(embed.shape)}, "
                f"required_size={level_size}"
            )
        return embed[:level_size].detach().to(device=device, dtype=dtype)

    def _rrg_prefix_reconstruction_from_token_ids(
        self,
        token_rows: torch.Tensor,
        level: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sum committed codewords before ``level`` for every beam row.

        Candidate Tries store tokenizer IDs, while the RQ codebooks are indexed
        by compact per-level codes.  This helper performs the mapping explicitly
        and ignores local suffix sentinels/mask IDs.  The modality
        routing token is skipped as a *fusion target* by default, but its
        codeword is still included in later prefix reconstructions because the
        current GENIUS RQ implements modality routing as quantizer layer 0.
        """
        if token_rows.dim() != 2 or token_rows.size(1) != self.codebook_level:
            raise ValueError(
                f"token_rows must be [N,{self.codebook_level}], got {tuple(token_rows.shape)}"
            )

        level = int(level)
        current_codebook = self._rrg_level_codebook(
            level, device=token_rows.device, dtype=dtype
        )
        reconstruction = torch.zeros(
            token_rows.size(0),
            current_codebook.size(1),
            device=token_rows.device,
            dtype=dtype,
        )

        # The modality token is the first actual ResidualVQ codebook, so it
        # contributes to the
        # residual seen by semantic level 1 and must be included in the prefix
        # reconstruction even when fusion itself is disabled at level 0.
        first_level = max(0, int(self.rqc_prefix_start_level))
        token_to_code = self.token_id_to_code.to(token_rows.device)
        token_vocab = int(token_to_code.numel())

        for prefix_level in range(first_level, level):
            token_ids = token_rows[:, prefix_level].long()
            valid_token = (token_ids >= 0) & (token_ids < token_vocab)
            if not bool(valid_token.any().item()):
                continue

            safe_tokens = token_ids.clamp(min=0, max=max(token_vocab - 1, 0))
            code_ids = token_to_code.index_select(0, safe_tokens)
            codebook = self._rrg_level_codebook(
                prefix_level, device=token_rows.device, dtype=dtype
            )
            valid_code = valid_token & (code_ids >= 0) & (code_ids < codebook.size(0))
            if bool(valid_code.any().item()):
                reconstruction[valid_code] += codebook.index_select(
                    0, code_ids[valid_code]
                )

        return reconstruction

    def _rrg_selected_scores_for_rows(
        self,
        query_rows: torch.Tensor,
        token_rows: torch.Tensor,
        level: int,
    ) -> torch.Tensor | None:
        """Return exact RRG scores only for tokens selected in each row."""
        if not self._rrg_enabled_for_level(level):
            return None
        query_rows = F.normalize(query_rows.float(), dim=-1)
        codebook = self._rrg_level_codebook(level, device=token_rows.device)
        token_ids = token_rows[:, level].long()
        code_ids = self.token_id_to_code.to(token_rows.device).index_select(0, token_ids)
        code_ids = code_ids.clamp(min=0, max=codebook.size(0) - 1)
        prefix = self._rrg_prefix_reconstruction_from_token_ids(token_rows, level)
        moments = None
        if self.rrg_normalize:
            key = (int(level), str(token_rows.device), int(codebook.size(0)), int(codebook.size(1)))
            moments = self._rrg_codebook_moment_cache.get(key)
            if moments is None:
                norms = codebook.square().sum(-1)
                moments = (
                    codebook.mean(0),
                    norms.mean(),
                    codebook.t().matmul(codebook) / float(codebook.size(0)),
                    (codebook * norms[:, None]).mean(0),
                    norms.square().mean(),
                )
                self._rrg_codebook_moment_cache[key] = moments
        return selected_residual_reconstruction_gain(
            query_rows,
            prefix,
            codebook,
            code_ids,
            normalize=self.rrg_normalize,
            eps=self.rrg_eps,
            moments=moments,
        )

    def _rrg_scores_for_rows(
        self,
        query_rows: Optional[torch.Tensor],
        token_rows: torch.Tensor,
        level: int,
    ) -> Optional[torch.Tensor]:
        """Return ``[N,K_level]`` RRG scores, or ``None`` when disabled."""
        if query_rows is None or not self._rrg_enabled_for_level(level):
            return None
        if query_rows.dim() != 2 or query_rows.size(0) != token_rows.size(0):
            raise ValueError(
                "RRG query rows must be [N,D] and align with beam rows: "
                f"query={tuple(query_rows.shape)}, tokens={tuple(token_rows.shape)}"
            )

        prefix = self._rrg_prefix_reconstruction_from_token_ids(
            token_rows, level, dtype=torch.float32
        )
        return self._rrg_scores_from_prefix_reconstruction(query_rows, prefix, level)

    def _rrg_scores_from_prefix_reconstruction(
        self,
        query_rows: Optional[torch.Tensor],
        prefix_reconstruction: torch.Tensor,
        level: int,
    ) -> Optional[torch.Tensor]:
        """Return RRG scores using a caller-maintained prefix reconstruction."""
        if query_rows is None or not self._rrg_enabled_for_level(level):
            return None
        codebook = self._rrg_level_codebook(
            level, device=prefix_reconstruction.device, dtype=torch.float32
        )
        query_rows = F.normalize(query_rows.float(), dim=-1)
        if query_rows.size(1) != codebook.size(1):
            raise ValueError(
                f"RRG query/codebook dimension mismatch: {query_rows.size(1)} vs {codebook.size(1)}"
            )
        return residual_reconstruction_gain(
            query_rows,
            prefix_reconstruction,
            codebook,
            normalize=self.rrg_normalize,
            mode=self.rqc_score_mode,
            eps=self.rrg_eps,
        )

    def _residual_composition_strength_for_rows(
        self,
        query_rows: Optional[torch.Tensor],
        token_rows: torch.Tensor,
        level: int,
        generator_level_scores: torch.Tensor,
        residual_scores: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return one residual-composition strength per beam row."""
        if residual_scores is None or query_rows is None:
            return None
        if self.use_adaptive_residual_composition:
            prefix = self._rrg_prefix_reconstruction_from_token_ids(
                token_rows, int(level), dtype=torch.float32
            )
            composition = self.residual_composer(
                query_embedding=query_rows,
                prefix_reconstruction=prefix,
                generator_scores=generator_level_scores,
                residual_scores=residual_scores,
                level=int(level),
                valid_mask=valid_mask,
            )
            return composition.strength.to(
                device=generator_level_scores.device,
                dtype=generator_level_scores.dtype,
            )
        if self.use_rrg:
            return torch.full(
                (generator_level_scores.size(0),),
                float(self.rrg_weight) * self._rqc_level_weight(level),
                device=generator_level_scores.device,
                dtype=generator_level_scores.dtype,
            )
        return None

    def _allowed_token_lists_to_level_mask(
        self,
        allowed_lists: List[List[int]],
        level: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert Trie-legal tokenizer IDs to a compact per-level mask."""
        level_size = int(self.level_vocab_sizes[int(level)].item())
        mask = torch.zeros(
            (len(allowed_lists), level_size), device=device, dtype=torch.bool
        )
        token_to_code = self.token_id_to_code.to(device)
        token_vocab = int(token_to_code.numel())
        for row, allowed in enumerate(allowed_lists):
            if not allowed:
                continue
            token_ids = torch.as_tensor(allowed, device=device, dtype=torch.long)
            valid_token = (token_ids >= 0) & (token_ids < token_vocab)
            if not bool(valid_token.any().item()):
                continue
            token_ids = token_ids[valid_token]
            code_ids = token_to_code.index_select(0, token_ids)
            valid_code = (code_ids >= 0) & (code_ids < level_size)
            if bool(valid_code.any().item()):
                mask[row, code_ids[valid_code]] = True
        return mask

    def _mask_invalid_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, L, vocab_size]
        """
        valid_mask = self.level_vocab_mask.to(logits.device).unsqueeze(0)  # [1, L, vocab_size]
        return logits.masked_fill(~valid_mask, -1e4)

    def _get_decode_order(self, L: int):
        if self.decode_order == "ltr":
            return list(range(L))
        # reference-style fallback
        return [L - 1] + list(range(L - 1))

    def _project_condition(self, emb: torch.Tensor) -> torch.Tensor:
        return self.embed_projector(F.normalize(emb)).reshape(emb.size(0), self.num_prefix, self.d_model)

    def _build_training_condition(self, q_emb: torch.Tensor, p_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a query condition and the realized augmentation rate.

        Query-positive mixing is an embedding-space augmentation only; it does
        not change the masked-diffusion target or introduce an autoregressive
        objective.  A scalar Beta coefficient per sample is the default because
        it preserves a coherent direction in the shared embedding space.  The
        legacy element-wise variant can be enabled from config for ablations.
        """
        bs, dim = q_emb.shape
        if self.query_positive_mix_prob <= 0.0:
            return q_emb, q_emb.new_zeros(())

        concentration = torch.tensor(
            self.query_positive_mix_alpha,
            device=q_emb.device,
            dtype=torch.float32,
        )
        beta = torch.distributions.Beta(concentration, concentration)
        sample_shape = (bs, dim) if self.query_positive_mix_elementwise else (bs, 1)
        s = beta.sample(sample_shape).to(dtype=q_emb.dtype)
        mixed = torch.sqrt(s) * q_emb + torch.sqrt(1.0 - s) * p_emb
        mixed = F.normalize(mixed, dim=-1)

        use_mix = torch.rand(bs, 1, device=q_emb.device) < self.query_positive_mix_prob
        condition = torch.where(use_mix, mixed, q_emb)
        return condition, use_mix.float().mean()

    def _compute_soundstorm_full_objective(
        self,
        codes: torch.Tensor,
        valid_mask: torch.Tensor,
        prefix: torch.Tensor,
        query_mix_rate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train the copied SoundStorm DALLE/VQ-Diffusion model on compact RQ codes."""
        ss = self.soundstorm_full_model(
            target_codes=codes,
            cond_emb=prefix,
            valid_mask=valid_mask,
            is_train=self.training,
        )
        pred_codes = ss["pred_token_ids"].long()
        gt = codes.long()
        valid = valid_mask.bool()
        token_correct = pred_codes == gt
        seq_acc = (token_correct | ~valid).all(dim=1).float().mean()
        outputs = {
            "loss": ss["loss"],
            "lm_loss": ss["loss"].detach(),
            "denoise_loss": ss["loss"].detach(),
            "R_at_1": seq_acc,
            "token_acc": ss["token_acc"],
            "masked_token_acc": ss["masked_token_acc"],
            "masked_seq_acc": ss["masked_seq_acc"],
            "code_exact_acc": seq_acc,
            "hdgr_mask_rate": ss["soundstorm_mask_rate"],
            "prefix_suffix_rate": codes.new_zeros((), dtype=torch.float32),
            "query_mix_rate": query_mix_rate,
            "Level1_acc": ss["level1_acc"],
            "Level12_acc": ss["level12_acc"],
            "Level123_acc": ss["level123_acc"],
            "soundstorm_t_mean": ss["t_mean"],
            "soundstorm_changed_rate": ss["soundstorm_changed_rate"],
            "soundstorm_random_rate": ss["soundstorm_random_rate"],
        }
        if self.iter % 200 == 0:
            print("SoundStorm FullModel Retrieval Task: Pred codes:", pred_codes[0].detach().cpu().tolist(), "Ans:", gt[0].detach().cpu().tolist())
        return outputs

    def _sample_training_noising(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix iid diffusion masks with inference-aligned suffix masks."""
        if self.prefix_suffix_noising_prob <= 0.0:
            iid_noised = forward_noising(tokens, t.unsqueeze(1), valid_mask, self, device)
            return iid_noised, tokens.new_zeros((), dtype=torch.float32)
        if self.prefix_suffix_noising_prob >= 1.0:
            suffix_noised = forward_prefix_suffix_noising(tokens, t, valid_mask, self, device)
            return suffix_noised, tokens.new_ones((), dtype=torch.float32)

        iid_noised = forward_noising(tokens, t.unsqueeze(1), valid_mask, self, device)
        suffix_noised = forward_prefix_suffix_noising(tokens, t, valid_mask, self, device)
        use_suffix = torch.rand(tokens.size(0), 1, device=device) < self.prefix_suffix_noising_prob
        noised = torch.where(use_suffix, suffix_noised, iid_noised)
        return noised, use_suffix.float().mean()

    def _select_diverse_expansions(
        self,
        total_scores: torch.Tensor,
        keep_k: int,
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select beam expansions, optionally penalizing repeated next tokens.

        The returned scores are the *unpenalized* cumulative log-probabilities,
        so saved beam scores remain meaningful for downstream calibration.  The
        diversity penalty only affects which hypotheses survive.
        """
        B, cur_k, vocab_size = total_scores.shape
        flat_total = total_scores.view(B, cur_k * vocab_size)
        if (
            self.diversity_strength <= 0.0
            or keep_k <= 1
            or step_idx < int(self.delayed_diversity_start)
        ):
            return torch.topk(flat_total, k=keep_k, dim=1)

        selected_scores = []
        selected_indices = []
        neg_inf = torch.finfo(flat_total.dtype).min
        finite_floor = neg_inf / 2
        for b in range(B):
            work = flat_total[b].clone()
            chosen_scores = []
            chosen_indices = []
            for _ in range(keep_k):
                flat_idx = int(torch.argmax(work).item())
                raw_score = flat_total[b, flat_idx]
                if (not torch.isfinite(raw_score)) or raw_score <= finite_floor:
                    break
                chosen_indices.append(flat_idx)
                chosen_scores.append(raw_score)

                token_idx = flat_idx % vocab_size
                work[flat_idx] = neg_inf
                sibling_positions = torch.arange(token_idx, cur_k * vocab_size, vocab_size, device=work.device)
                work[sibling_positions] -= float(self.diversity_strength)

            # A legal Trie should always provide at least keep_k expansions.  If
            # the optional diversity loop stopped early, backfill from raw top-k.
            if len(chosen_indices) < keep_k:
                raw_order = torch.argsort(flat_total[b], descending=True)
                chosen_set = set(chosen_indices)
                for idx_tensor in raw_order:
                    idx = int(idx_tensor.item())
                    if idx in chosen_set:
                        continue
                    chosen_indices.append(idx)
                    chosen_scores.append(flat_total[b, idx])
                    chosen_set.add(idx)
                    if len(chosen_indices) == keep_k:
                        break

            selected_indices.append(torch.tensor(chosen_indices, device=total_scores.device, dtype=torch.long))
            selected_scores.append(torch.stack(chosen_scores))

        return torch.stack(selected_scores, dim=0), torch.stack(selected_indices, dim=0)

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

    def _get_allowed_mask(self, cur_tokens: torch.Tensor, layer_idx: int, vocab_size: int):
        """
        cur_tokens: [B, L], prefix positions contain chosen token ids,
                    suffix positions are usually self.mask_token_id
        return: [B, vocab_size] bool

        Strict candidate-tree constraint:
        - layer 0: only root nodes are allowed
        - invalid prefix: no token allowed
        """
        B = cur_tokens.size(0)
        device = cur_tokens.device
        allowed_mask = torch.zeros(B, vocab_size, device=device, dtype=torch.bool)

        # no candidate tree -> no extra constraint
        if self.tree_index is None:
            allowed_mask[:] = self.level_vocab_mask[layer_idx].to(device)
            return allowed_mask

        for b in range(B):
            nodes = get_layer_nodes_from_tree(
                self.tree_index,
                cur_tokens[b],
                layer_idx,
                self.mask_token_id
            )

            # strict mode: invalid prefix => keep all False
            if nodes is None or len(nodes) == 0:
                continue

            for tok_id in nodes:
                if 0 <= tok_id < vocab_size:
                    allowed_mask[b, tok_id] = True

        return allowed_mask

    def _build_prefix_regen_xt(self, seqs: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        seqs: [B, K, L] or [B, L]
              only positions < step_idx are considered fixed / valid
        return:
            xt with positions [step_idx:] all masked
        """
        xt = seqs.clone()
        xt[..., step_idx:] = self.mask_token_id
        return xt
    def _prepare_infer_tokens_from_xt(self, xt: torch.Tensor) -> torch.Tensor:
        """
        xt contains mask class at masked positions.
        tokens fed into embedding path cannot use undefined semantic there,
        so replace mask positions temporarily with 0. Actual mask semantics are
        controlled by mask_tokens in id_generator.forward().
        """
        tokens = xt.clone()
        tokens[tokens == self.mask_token_id] = 0
        return tokens

    # =====================================================
    # training
    # =====================================================

    def _compute_soundstorm_objective(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        prefix: torch.Tensor,
        query_mix_rate: torch.Tensor,
    ):
        """SoundStorm/VQ-Diffusion copied objective on retrieval semantic-ID tokens.

        Interface adaptation only:
          - SoundStorm target_acoustics -> tokens [B, L]
          - SoundStorm condition dict -> query-conditioned prefix embeddings
          - SoundStorm transformer(x_t, condition, t) -> self.id_generator(...)

        The denoiser sees corrupted x_t, not clean x0.  Mask tokens are carried
        through ``mask_tokens``; for GPT embeddings, positions equal to the mask
        class are replaced by a safe placeholder in the ``tokens`` argument and
        then swapped to the learned mask embedding inside id_generator.
        """

        def model_forward_fn(xt_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # The real-token embedding table cannot index the extra mask class in
            # plain GPT.  The mask class is still preserved in mask_tokens so the
            # generator substitutes its learned mask embedding at those positions.
            xt_for_embedding = xt_ids.clone()
            xt_for_embedding[xt_for_embedding == self.mask_token_id] = 0

            model_outputs = self.id_generator(
                tokens=xt_for_embedding,
                mask_tokens=xt_ids,
                prefix=prefix,
                mask=None,  # SoundStorm uses full self-attention; no mask-token attention suppression.
                t=t,
                labels=None,
            )
            return self._mask_invalid_logits(model_outputs.logits)

        ss = self.soundstorm_diffusion.train_loss(
            x_start=tokens,
            valid_mask=valid_mask,
            model_forward_fn=model_forward_fn,
            is_train=self.training,
        )

        pred_token_ids = ss["pred_token_ids"]
        gt = tokens
        seq_acc = ss["seq_acc"]

        outputs = {
            "loss": ss["loss"],
            "lm_loss": ss["loss"].detach(),
            "denoise_loss": ss["loss"].detach(),
            "R_at_1": seq_acc,
            "token_acc": ss["token_acc"],
            "masked_token_acc": ss["masked_token_acc"],
            "masked_seq_acc": ss["masked_seq_acc"],
            "code_exact_acc": seq_acc,
            # Keep the existing engine/log key.  Here it is SoundStorm's mask-token rate in x_t.
            "hdgr_mask_rate": ss["soundstorm_mask_rate"],
            "prefix_suffix_rate": tokens.new_zeros((), dtype=torch.float32),
            "query_mix_rate": query_mix_rate,
            "Level1_acc": ss["level1_acc"],
            "Level12_acc": ss["level12_acc"],
            "Level123_acc": ss["level123_acc"],
            "soundstorm_t_mean": ss["t_mean"],
            "soundstorm_changed_rate": ss["soundstorm_changed_rate"],
            "soundstorm_random_rate": ss["soundstorm_random_rate"],
        }

        if self.iter % 200 == 0:
            example = self.tokenizer.decode(pred_token_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            example_labels = self.tokenizer.decode(gt[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("SoundStorm Retrieval Task: " + "Pred: " + str(example) + " Ans: " + str(example_labels))

        return outputs

    def compute_single_batch(self, batch, gpu_id=None):
        if not isinstance(gpu_id, torch.device):
            gpu_id = torch.device(
                f'cuda:{gpu_id}' if isinstance(gpu_id, int)
                else 'cuda' if torch.cuda.is_available() else 'cpu'
            )

        # Current HDGR/MBEIR dataloader returns 3 fields:
        #   (query, positive_pool, h_qid)
        # Older GENIUS-style instruction dataloaders may return 4 fields:
        #   (query, positive_pool, instruct, h_qid)
        # The GPT baseline does not consume `instruct` directly because the
        # query embedding has already been instruction-conditioned upstream.
        if len(batch) == 4:
            query, pool, instruct, h_qid = batch
            if torch.is_tensor(instruct):
                _ = instruct.view(-1, instruct.size(-1)).to(gpu_id, non_blocking=True)
        elif len(batch) == 3:
            query, pool, h_qid = batch
        else:
            raise ValueError(
                f"Unexpected training batch format with {len(batch)} fields; "
                "expected 3 (query, pool, h_qid) or 4 (query, pool, instruct, h_qid)."
            )

        h_qid = h_qid.view(-1)

        q_img_mask = query['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        q_img_emb = query['img_emb'].view(-1, query['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query['txt_emb'].view(-1, query['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        device = q_img_emb.device
        bs = len(q_img_emb)

        with torch.no_grad():
            q_output = self.quantizer.inference(q_img_emb, q_txt_emb, q_img_mask, q_txt_mask)
            p_output = self.quantizer.inference(p_img_emb, p_txt_emb, p_img_mask, p_txt_mask)

        q_emb = F.normalize(q_output['encode'])
        p_emb = F.normalize(p_output['encode'])

        # target compact RQ codes.  Legacy GPT-HDGR converts these to tokenizer IDs;
        # the full SoundStorm model keeps compact per-level code indices, matching
        # SoundStorm's per-quantizer acoustic-code heads.
        p_codes = p_output['code'].long().to(device)           # [B, L]

        if self.use_soundstorm_full_model:
            tokens = p_codes.clamp(min=0, max=self.soundstorm_full_model.num_embed - 1)
            valid_mask = torch.ones_like(tokens, device=device, dtype=torch.long)
            gt = tokens
        else:
            target = self._codes_to_token_ids(p_codes).to(device)  # [B, L], in [0, vocab_size-1]
            tokens, valid_mask, gt = process_token(
                target,
                self.codebook_level,
                pad_token_id=IGNORE_INDEX
            )
            tokens = tokens.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

        # Condition augmentation: mix query and positive embeddings for only a
        # configurable subset of samples.  This keeps pure-query conditions in
        # every batch and reduces train/inference distribution shift relative to
        # always-on positive mixing.
        condition_emb, query_mix_rate = self._build_training_condition(q_emb, p_emb)
        cond_prefix = self._project_condition(condition_emb)

        # CFG training: randomly drop condition
        null_prefix = self.null_condition.expand(bs, -1, -1).to(device=device, dtype=cond_prefix.dtype)
        drop_mask = (torch.rand(bs, 1, 1, device=device) < self.cond_drop_prob)
        prefix = torch.where(drop_mask, null_prefix, cond_prefix)

        # Full copied SoundStorm DALLE/VQ-Diffusion model.
        if self.use_soundstorm_full_model:
            outputs = self._compute_soundstorm_full_objective(
                codes=tokens,
                valid_mask=valid_mask,
                prefix=prefix,
                query_mix_rate=query_mix_rate,
            )
            self.iter += 1
            return outputs

        # Optional copied SoundStorm / VQ-Diffusion objective using legacy GPT-HDGR denoiser.
        if self.training_objective in {"soundstorm", "soundstorm_vqdiffusion", "vqdiffusion"}:
            outputs = self._compute_soundstorm_objective(
                tokens=tokens,
                valid_mask=valid_mask,
                prefix=prefix,
                query_mix_rate=query_mix_rate,
            )
            self.iter += 1
            return outputs

        # timestep
        t = sample_time(bs, self.time_step, device)
        if isinstance(t, tuple):
            t = t[0]

        # Forward noising: combine ordinary iid masked diffusion with structured
        # suffix masks that match prefix-regeneration inference.
        mask_tokens, prefix_suffix_rate = self._sample_training_noising(
            tokens=tokens,
            t=t,
            valid_mask=valid_mask,
            device=device,
        )
        if self.training and self.one_pass_full_mask_probability > 0.0:
            force_full_mask = (
                torch.rand(bs, 1, device=device) < self.one_pass_full_mask_probability
            )
            mask_tokens = torch.where(
                force_full_mask,
                torch.full_like(mask_tokens, self.mask_token_id),
                mask_tokens,
            )
        else:
            force_full_mask = torch.zeros((bs, 1), device=device, dtype=torch.bool)

        # custom attention mask
        all_mask = build_concentrate_attention_mask(
            mask_tokens=mask_tokens,
            valid_mask=valid_mask,
            mask_token_id=self.mask_token_id
        )

        # forward
        model_outputs = self.id_generator(
            tokens=tokens,
            mask_tokens=mask_tokens,
            prefix=prefix,
            mask=all_mask,
            t=t,
            labels=None
        )

        logits = self._mask_invalid_logits(model_outputs.logits)  # [B, L, vocab_size]

        flat_loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            gt.reshape(-1)
        ).view(bs, self.codebook_level)

        # Masked positions carry the denoising objective.  A small configurable
        # visible-token term preserves context reconstruction without allowing
        # easy visible positions to dominate the loss.
        masked_pos = (mask_tokens == self.mask_token_id) & (valid_mask > 0)
        visible_pos = (~masked_pos) & (valid_mask > 0)
        base_weight = masked_pos.float() + visible_pos.float() * self.visible_token_loss_weight
        loss_weight = base_weight * (valid_mask > 0).float()

        if self.loss_normalization == "per_sample":
            per_sample_loss = (flat_loss * loss_weight).sum(dim=1) / loss_weight.sum(dim=1).clamp_min(1.0)
            denoise_loss = per_sample_loss.mean()
        else:
            denoise_loss = (flat_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        loss = denoise_loss
        one_pass_rank_loss = loss.new_zeros(())
        one_pass_rank_acc = loss.new_zeros(())
        one_pass_hard_margin = loss.new_zeros(())
        full_mask_rows = force_full_mask.squeeze(1)
        if (
            self.training
            and self.one_pass_ranking_weight > 0.0
            and int(full_mask_rows.sum().item()) >= 2
        ):
            if self.one_pass_candidate_codes_path:
                if self._one_pass_candidate_codes is None:
                    path = self.one_pass_candidate_codes_path
                    if not os.path.isabs(path):
                        path = os.path.join(str(cfg_get(self.config, "genir_dir", ".")), path)
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"One-Pass candidate codes not found: {path}")
                    self._one_pass_candidate_codes = np.load(path, mmap_mode="r")
                    if self._one_pass_candidate_codes.ndim != 2 or self._one_pass_candidate_codes.shape[1] != self.codebook_level:
                        raise ValueError(
                            f"One-Pass candidate codes must be [N,{self.codebook_level}], "
                            f"got {self._one_pass_candidate_codes.shape}"
                        )
                    print(f"[One-Pass training] Loaded legal candidate codes: {path} shape={self._one_pass_candidate_codes.shape}")
                if self.one_pass_sorted_candidate_codes_path:
                    if self._one_pass_sorted_candidate_codes is None:
                        sorted_path = self.one_pass_sorted_candidate_codes_path
                        if not os.path.isabs(sorted_path):
                            sorted_path = os.path.join(str(cfg_get(self.config, "genir_dir", ".")), sorted_path)
                        self._one_pass_sorted_candidate_codes = np.load(sorted_path, mmap_mode="r")
                        print(
                            "[One-Pass training] Loaded prefix candidate index: "
                            f"{sorted_path} shape={self._one_pass_sorted_candidate_codes.shape}"
                        )
                    structured = sample_prefix_neighbor_codes(
                        self._one_pass_sorted_candidate_codes,
                        p_codes[full_mask_rows].detach().cpu().numpy(),
                        prefix_depths=self.one_pass_prefix_depths,
                        candidates_per_depth=self.one_pass_candidates_per_depth,
                    )
                else:
                    structured = np.empty((0, self.codebook_level), dtype=np.int64)
                pool_size = int(self._one_pass_candidate_codes.shape[0])
                sample_size = min(max(0, self.one_pass_candidate_sample_size), pool_size)
                if sample_size > 0:
                    sampled_rows = np.random.randint(0, pool_size, size=sample_size)
                    random_codes = np.asarray(self._one_pass_candidate_codes[sampled_rows])
                    sampled_array = np.unique(
                        np.concatenate((structured, random_codes), axis=0), axis=0
                    )
                else:
                    sampled_array = structured
                if sampled_array.shape[0] == 0:
                    raise RuntimeError("One-Pass hard-negative mining produced no legal candidates")
                sampled_codes = torch.from_numpy(np.asarray(sampled_array).copy()).long().to(device)
                sampled_tokens = self._codes_to_token_ids(sampled_codes)
                one_pass_rank_loss, one_pass_rank_acc, one_pass_hard_margin = (
                    sampled_hard_negative_ranking_loss(
                        logits[full_mask_rows],
                        gt[full_mask_rows],
                        sampled_tokens,
                        num_hard_negatives=self.one_pass_hard_negatives,
                        temperature=self.one_pass_ranking_temperature,
                    )
                )
            else:
                one_pass_rank_loss, one_pass_rank_acc = inbatch_complete_id_ranking_loss(
                    logits[full_mask_rows],
                    gt[full_mask_rows],
                    temperature=self.one_pass_ranking_temperature,
                )
            loss = loss + self.one_pass_ranking_weight * one_pass_rank_loss
        residual_composition = {
            "route_loss": loss.new_zeros(()),
            "branch_loss": loss.new_zeros(()),
            "regret_loss": loss.new_zeros(()),
            "no_harm_loss": loss.new_zeros(()),
            "clear_ce_loss": loss.new_zeros(()),
            "route_acc": loss.new_zeros(()),
            "clear_fraction": loss.new_zeros(()),
            "positive_gain_fraction": loss.new_zeros(()),
            "branch_acc": loss.new_zeros(()),
            "baseline_branch_acc": loss.new_zeros(()),
            "strength_mean": loss.new_zeros(()),
            "strength_std": loss.new_zeros(()),
            "oracle_strength_mean": loss.new_zeros(()),
            "expert0_probability": loss.new_zeros(()),
            "policy_entropy": loss.new_zeros(()),
            "utility_gain": loss.new_zeros(()),
            "best_utility_gain": loss.new_zeros(()),
            "utility_gap": loss.new_zeros(()),
            "expert0_win_rate": loss.new_zeros(()),
            "expert10_win_rate": loss.new_zeros(()),
            "expert50_win_rate": loss.new_zeros(()),
            "expert100_win_rate": loss.new_zeros(()),
            "count": loss.new_zeros(()),
            "clear_count": loss.new_zeros(()),
        }
        residual_composition_scale = 0.0
        if self.use_adaptive_residual_composition and self.training:
            residual_composition_scale = composition_weight_scale(
                self.iter,
                warmup_steps=self.residual_composition_warmup_steps,
                ramp_steps=self.residual_composition_ramp_steps,
            )
            if residual_composition_scale > 0.0 and self.residual_composition_max_queries > 0:
                residual_composition = compute_counterfactual_composition_loss(
                    self,
                    target_tokens=tokens,
                    h_qid=h_qid,
                    cond_prefix=cond_prefix,
                    query_embedding=q_emb,
                    asset=self.residual_composition_asset,
                    max_queries=self.residual_composition_max_queries,
                    levels_per_query=self.residual_composition_levels_per_query,
                    min_level=self.residual_composition_min_level,
                    max_level=self.residual_composition_max_level,
                    oracle_temperature=self.residual_composition_oracle_temperature,
                    use_cfg=self.residual_composition_use_cfg,
                    normalize_oracle_utility=self.residual_composition_normalize_oracle_utility,
                    advantage_temperature=self.residual_composition_advantage_temperature,
                    min_utility_gain=self.residual_composition_min_utility_gain,
                    min_winner_gap=self.residual_composition_min_winner_gap,
                    no_harm_weight=self.residual_composition_no_harm_weight,
                    clear_ce_weight=self.residual_composition_clear_ce_weight,
                )
                loss = loss + float(residual_composition_scale) * (
                    self.residual_composition_route_weight * residual_composition["route_loss"]
                    + self.residual_composition_branch_weight * residual_composition["branch_loss"]
                )

        tree_branch = {
            "loss": loss.new_zeros(()),
            "accuracy": loss.new_zeros(()),
            "positive_mass": loss.new_zeros(()),
            "count": loss.new_zeros(()),
        }
        tree_prefix_risk = {
            "loss": loss.new_zeros(()),
            "accuracy": loss.new_zeros(()),
            "margin": loss.new_zeros(()),
            "count": loss.new_zeros(()),
        }
        tree_scale = 0.0
        if self.use_tree_risk and self.training:
            tree_scale = tree_risk_weight_scale(
                self.iter,
                warmup_steps=self.tree_risk_warmup_steps,
                ramp_steps=self.tree_risk_ramp_steps,
            )
            if tree_scale > 0.0 and self.tree_branch_weight > 0.0:
                tree_branch = compute_multi_positive_branch_loss(
                    self,
                    self.tree_risk_asset,
                    target_tokens=tokens,
                    h_qid=h_qid,
                    cond_prefix=cond_prefix,
                    levels_per_sample=self.tree_levels_per_sample,
                    min_level=self.tree_min_level,
                    max_level=self.tree_max_level,
                    temperature=self.tree_branch_temperature,
                    use_cfg=self.tree_risk_use_cfg,
                    require_competition=self.tree_branch_require_competition,
                    max_samples=self.tree_branch_queries_per_batch,
                )
                loss = loss + float(tree_scale) * self.tree_branch_weight * tree_branch["loss"]
            if (
                tree_scale > 0.0
                and self.tree_prefix_risk_weight > 0.0
                and self.tree_risk_queries_per_batch > 0
                and self.tree_risk_hard_negatives > 0
            ):
                tree_prefix_risk = compute_inbatch_prefix_risk_loss(
                    self,
                    self.tree_risk_asset,
                    target_tokens=tokens,
                    h_qid=h_qid,
                    cond_prefix=cond_prefix,
                    q_emb=q_emb,
                    p_emb=p_emb,
                    max_queries=self.tree_risk_queries_per_batch,
                    hard_negatives=self.tree_risk_hard_negatives,
                    max_positive_paths=self.tree_risk_max_positive_paths,
                    min_level=self.tree_min_level,
                    max_level=self.tree_max_level,
                    margin=self.tree_risk_margin,
                    temperature=self.tree_risk_temperature,
                    use_cfg=self.tree_risk_use_cfg,
                )
                loss = loss + float(tree_scale) * self.tree_prefix_risk_weight * tree_prefix_risk["loss"]

        pred_token_ids = logits.argmax(dim=-1)

        token_correct = pred_token_ids == gt
        seq_acc = (token_correct | (valid_mask <= 0)).all(dim=1).float().mean()
        level1_acc = (pred_token_ids[:, :1] == gt[:, :1]).all(dim=1).float().mean()
        level12_acc = (pred_token_ids[:, :min(2, self.codebook_level)] == gt[:, :min(2, self.codebook_level)]).all(dim=1).float().mean()
        level123_acc = (pred_token_ids[:, :min(3, self.codebook_level)] == gt[:, :min(3, self.codebook_level)]).all(dim=1).float().mean()

        valid_count = (valid_mask > 0).sum().clamp_min(1)
        masked_count = masked_pos.sum().clamp_min(1)
        token_acc = (token_correct & (valid_mask > 0)).sum().float() / valid_count.float()
        masked_token_acc = (token_correct & masked_pos).sum().float() / masked_count.float()
        masked_seq_acc = ((token_correct | ~masked_pos).all(dim=1)).float().mean()
        mask_rate = masked_pos.sum().float() / valid_count.float()

        outputs = {
            'loss': loss,
            'lm_loss': denoise_loss.detach(),
            'denoise_loss': denoise_loss.detach(),
            'one_pass_rank_loss': one_pass_rank_loss.detach(),
            'one_pass_rank_acc': one_pass_rank_acc.detach(),
            'one_pass_hard_margin': one_pass_hard_margin.detach(),
            'one_pass_full_mask_rate': full_mask_rows.float().mean().detach(),
            'residual_composition_scale': loss.new_tensor(float(residual_composition_scale)),
            'residual_route_loss': residual_composition["route_loss"].detach(),
            'residual_branch_loss': residual_composition["branch_loss"].detach(),
            'residual_regret_loss': residual_composition["regret_loss"].detach(),
            'residual_no_harm_loss': residual_composition["no_harm_loss"].detach(),
            'residual_clear_ce_loss': residual_composition["clear_ce_loss"].detach(),
            'residual_route_acc': residual_composition["route_acc"].detach(),
            'residual_clear_fraction': residual_composition["clear_fraction"].detach(),
            'residual_positive_gain_fraction': residual_composition["positive_gain_fraction"].detach(),
            'residual_branch_acc': residual_composition["branch_acc"].detach(),
            'residual_baseline_branch_acc': residual_composition["baseline_branch_acc"].detach(),
            'residual_strength_mean': residual_composition["strength_mean"].detach(),
            'residual_strength_std': residual_composition["strength_std"].detach(),
            'residual_oracle_strength_mean': residual_composition["oracle_strength_mean"].detach(),
            'residual_expert0_probability': residual_composition["expert0_probability"].detach(),
            'residual_policy_entropy': residual_composition["policy_entropy"].detach(),
            'residual_utility_gain': residual_composition["utility_gain"].detach(),
            'residual_best_utility_gain': residual_composition["best_utility_gain"].detach(),
            'residual_utility_gap': residual_composition["utility_gap"].detach(),
            'residual_expert0_win_rate': residual_composition["expert0_win_rate"].detach(),
            'residual_expert10_win_rate': residual_composition["expert10_win_rate"].detach(),
            'residual_expert50_win_rate': residual_composition["expert50_win_rate"].detach(),
            'residual_expert100_win_rate': residual_composition["expert100_win_rate"].detach(),
            'residual_composition_count': residual_composition["count"].detach(),
            'residual_clear_count': residual_composition["clear_count"].detach(),
            'tree_risk_scale': loss.new_tensor(float(tree_scale)),
            'tree_branch_loss': tree_branch["loss"].detach(),
            'tree_branch_acc': tree_branch["accuracy"].detach(),
            'tree_branch_positive_mass': tree_branch["positive_mass"].detach(),
            'tree_branch_count': tree_branch["count"].detach(),
            'tree_prefix_risk_loss': tree_prefix_risk["loss"].detach(),
            'tree_prefix_risk_acc': tree_prefix_risk["accuracy"].detach(),
            'tree_prefix_risk_margin': tree_prefix_risk["margin"].detach(),
            'tree_prefix_risk_count': tree_prefix_risk["count"].detach(),
            'R_at_1': seq_acc,
            'token_acc': token_acc,
            'masked_token_acc': masked_token_acc,
            'masked_seq_acc': masked_seq_acc,
            'code_exact_acc': seq_acc,
            'hdgr_mask_rate': mask_rate,
            'prefix_suffix_rate': prefix_suffix_rate,
            'query_mix_rate': query_mix_rate,
            'Level1_acc': level1_acc,
            'Level12_acc': level12_acc,
            'Level123_acc': level123_acc
        }

        if self.iter % 200 == 0:
            example = self.tokenizer.decode(pred_token_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            example_labels = self.tokenizer.decode(gt[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print('Diffusion Retrieval Task: ' + 'Pred: ' + str(example) + ' Ans: ' + str(example_labels))

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
        return self.tokenizer

    # =====================================================
    # candidate tree cache (keep legacy interface name)
    # =====================================================

    def generative_index(self, cand_codes):
        return None

    @staticmethod
    def _candidate_codes_signature(cand_codes):
        """Return a cheap, stable fingerprint for a candidate-code array.

        Candidate trees are persisted across evaluation runs.  A path alone is
        not a sufficient cache key because older experiments may have written a
        tree for a different quantizer or candidate-code payload to that path.
        """
        if cand_codes is None:
            return None
        arr = np.asarray(cand_codes)
        shape = tuple(int(v) for v in arr.shape)
        dtype = str(arr.dtype)
        if arr.size == 0:
            return (shape, dtype, "empty")
        if arr.ndim >= 2:
            # Evenly sample enough rows to make accidental reuse of a stale
            # tree vanishingly unlikely without hashing multi-GB arrays.
            rows = np.linspace(0, shape[0] - 1, num=min(64, shape[0]), dtype=np.int64)
            rows = np.unique(rows)
            sample = np.ascontiguousarray(arr[rows])
        else:
            sample = np.ascontiguousarray(arr.reshape(-1)[:64])
        return (shape, dtype, hashlib.sha1(sample.tobytes()).hexdigest())

    def distribute_trie(self, cand_codes, trie_save_path):
        """
        Keep old interface name for compatibility.
        Internally build/load nested candidate tree instead of Trie object.
        """
        if cand_codes is None:
            return

        signature = self._candidate_codes_signature(cand_codes)

        save_path = trie_save_path
        if save_path is None:
            save_path = os.path.join(self.config.genir_dir, "candidate_tree.pkl")

        # The same candidate pool is reused for every query dataset in an
        # all-M-BEIR phase.  Do not reload a multi-GB tree that is already live.
        normalized_save_path = os.path.abspath(save_path)
        if (
            getattr(self, "_loaded_candidate_tree_path", None) == normalized_save_path
            and getattr(self, "cand_codes_signature", None) == signature
            and self.tree_index is not None
            and (
                not getattr(self, "gpu_flat_tree_decode", False)
                or self.flat_tree_index is not None
            )
        ):
            print(f"Log: Reusing candidate tree already loaded from {save_path}.")
            return

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            rebuild = True
            if os.path.exists(save_path):
                try:
                    with open(save_path, "rb") as f:
                        old_payload = pickle.load(f)
                    rebuild = old_payload.get("signature") != signature
                    if rebuild:
                        print(f"Log: Existing candidate tree at {save_path} is stale; rebuilding.")
                except Exception as exc:
                    print(f"Log: Candidate tree cache at {save_path} is unreadable ({exc}); rebuilding.")
            if rebuild:
                tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
                payload = {
                    "tree": tree,
                    "cand_token_ids": cand_token_ids,
                    "cand_codes": cand_codes_cache,
                    "signature": signature,
                }
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                tmp_path = save_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, save_path)
                print(f"Log: Saved candidate tree from {save_path}.")

        if dist.is_initialized():
            dist.barrier()

        with open(save_path, "rb") as f:
            payload = pickle.load(f)

        if payload.get("signature") != signature:
            raise RuntimeError(
                "Loaded candidate tree does not match the provided candidate codes: "
                f"{save_path}"
            )

        self.tree_index = payload["tree"]
        self.cand_token_ids = payload["cand_token_ids"]
        self.cand_codes_cache = payload["cand_codes"]
        self.cand_codes_signature = signature
        self._loaded_candidate_tree_path = normalized_save_path

        if getattr(self, "gpu_flat_tree_decode", False):
            flat_save_path = save_path + ".flatgpu.pt"
            if rank == 0:
                rebuild_flat = True
                if os.path.exists(flat_save_path):
                    try:
                        FlatPrefixTrie.load(
                            flat_save_path, expected_signature=signature
                        )
                        rebuild_flat = False
                    except Exception as exc:
                        print(
                            "Log: GPU-flat candidate tree cache at "
                            f"{flat_save_path} is stale or unreadable ({exc}); rebuilding."
                        )
                if rebuild_flat:
                    flat_tree = FlatPrefixTrie.from_token_ids(self.cand_token_ids)
                    flat_tree.save(flat_save_path, signature=signature)
                    print(
                        "Log: Saved GPU-flat candidate tree to "
                        f"{flat_save_path} ({flat_tree.storage_bytes() / (1024 ** 2):.2f} MiB)."
                    )
            if dist.is_initialized():
                dist.barrier()
            self.flat_tree_index = FlatPrefixTrie.load(
                flat_save_path, expected_signature=signature
            )
            # Candidate-index setup is called before the online generation
            # timer. Upload the compact Trie here so the first query does not
            # absorb an index-transfer cost that other cached indexes exclude.
            model_device = next(self.parameters()).device
            self.flat_tree_index = self.flat_tree_index.to(model_device)
            self._flat_tree_device = model_device
        # Continuations and leaf counts are properties of the active Trie.
        # Keeping these memoized values while moving from one LOCAL candidate
        # pool to another can return a legal continuation from the previous
        # dataset, which then becomes an empty continuation at the next block.
        self.block_transition_cache = {}
        self.trie_leaf_count_cache = {}

        print(f"Log: Loaded candidate tree from {save_path}.")

    # =====================================================
    # iterative decoding (keep legacy interface name)
    # =====================================================


    def _get_allowed_token_lists(self, cur_tokens: torch.Tensor, layer_idx: int):
        """Return legal next-token IDs per row under the candidate tree.

        This list form avoids allocating [B*K, vocab] boolean masks.
        """
        B = cur_tokens.size(0)

        if self.tree_index is None:
            level_size = int(self.level_vocab_sizes[layer_idx].item())
            level_ids = self.level_token_ids[layer_idx, :level_size].detach().cpu().tolist()
            return [level_ids for _ in range(B)]

        out = []
        for b in range(B):
            nodes = get_layer_nodes_from_tree(
                self.tree_index,
                cur_tokens[b],
                layer_idx,
                self.mask_token_id,
            )
            if nodes is None:
                out.append([])
            else:
                out.append([int(x) for x in nodes])
        return out

    def _select_lowmem_expansions_for_batch(
        self,
        raw_scores: torch.Tensor,
        parent_ids: torch.Tensor,
        token_ids: torch.Tensor,
        keep_k: int,
        step_idx: int,
    ):
        """Top-k with the same optional sibling-token diversity idea as the full-V path."""
        if raw_scores.numel() == 0:
            raise RuntimeError("No legal low-memory beam expansion candidates.")
        keep_k = min(int(keep_k), int(raw_scores.numel()))
        if (
            self.diversity_strength <= 0.0
            or keep_k <= 1
            or step_idx < int(self.delayed_diversity_start)
        ):
            top_scores, top_pos = torch.topk(raw_scores, k=keep_k, dim=0)
            return top_scores, parent_ids[top_pos], token_ids[top_pos]

        work = raw_scores.clone()
        neg_inf = torch.finfo(work.dtype).min
        chosen = []
        for _ in range(keep_k):
            pos = int(torch.argmax(work).item())
            if work[pos] <= neg_inf / 2:
                break
            chosen.append(pos)
            tok = token_ids[pos]
            work[pos] = neg_inf
            work[token_ids == tok] -= float(self.diversity_strength)

        if len(chosen) < keep_k:
            raw_order = torch.argsort(raw_scores, descending=True)
            chosen_set = set(chosen)
            for pos_t in raw_order:
                pos = int(pos_t.item())
                if pos in chosen_set:
                    continue
                chosen.append(pos)
                chosen_set.add(pos)
                if len(chosen) == keep_k:
                    break

        pos = torch.tensor(chosen, device=raw_scores.device, dtype=torch.long)
        return raw_scores[pos], parent_ids[pos], token_ids[pos]

    def _flat_tree_on_device(self, device: torch.device) -> FlatPrefixTrie:
        """Return the active flat Trie on ``device`` without timing its upload."""
        if self.flat_tree_index is None:
            if self.cand_token_ids is None:
                raise RuntimeError(
                    "GPU-flat decoding requires a loaded candidate index. "
                    "Call distribute_trie before inference."
                )
            self.flat_tree_index = FlatPrefixTrie.from_token_ids(self.cand_token_ids)
            self._flat_tree_device = None
        if self._flat_tree_device != device:
            self.flat_tree_index = self.flat_tree_index.to(device)
            self._flat_tree_device = device
        return self.flat_tree_index

    @staticmethod
    def _segmented_flat_topk(
        scores: torch.Tensor,
        owners: torch.Tensor,
        token_ids: torch.Tensor,
        child_nodes: torch.Tensor,
        batch_size: int,
        parents_per_query: int,
        keep_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a common top-k from variable-length expansion segments.

        ``owners`` indexes a flattened ``[batch, parent]`` array.  Flat-Trie
        edges are emitted in parent-major/token-major order, so compacting the
        legal candidates preserves the ordering used by dense beam search.
        """
        query_ids = torch.div(owners, int(parents_per_query), rounding_mode="floor")
        counts = torch.bincount(query_ids, minlength=int(batch_size))
        # Fetch the two shape scalars with one device synchronization.  They
        # are required to size the dense segmented-top-k workspace, but do
        # not otherwise participate in scoring.
        min_count_t, max_count_t = torch.aminmax(counts)
        min_count, max_count = (
            int(value) for value in torch.stack((min_count_t, max_count_t)).tolist()
        )
        if min_count <= 0:
            raise RuntimeError("GPU-flat Trie expansion produced an empty query segment.")
        keep_k = min(int(keep_limit), min_count)

        starts = torch.cumsum(counts, dim=0) - counts
        local_pos = torch.arange(scores.numel(), device=scores.device) - torch.repeat_interleave(
            starts, counts
        )
        neg_inf = torch.finfo(scores.dtype).min
        dense_scores = torch.full(
            (int(batch_size), max_count), neg_inf, device=scores.device, dtype=scores.dtype
        )
        dense_scores[query_ids, local_pos] = scores

        top_scores, top_local = torch.topk(dense_scores, k=keep_k, dim=1)
        dense_parent = torch.zeros(
            (int(batch_size), max_count), device=owners.device, dtype=torch.long
        )
        dense_token = torch.zeros_like(dense_parent)
        dense_child = torch.zeros_like(dense_parent)
        dense_parent[query_ids, local_pos] = owners.remainder(int(parents_per_query))
        dense_token[query_ids, local_pos] = token_ids.long()
        dense_child[query_ids, local_pos] = child_nodes.long()
        return (
            top_scores,
            dense_parent.gather(1, top_local),
            dense_token.gather(1, top_local),
            dense_child.gather(1, top_local),
        )

    @torch.no_grad()
    def _constrained_beam_search_gpu_flat(
        self,
        inputs_embeds,
        attention_mask=None,
        num_beams=10,
        cand_codes=None,
        return_scores: bool = False,
        rrg_query_embedding: Optional[torch.Tensor] = None,
    ):
        """Sequential semantic-ID search with a tensorized GPU Trie.

        This experimental baseline preserves the score, level-wise
        normalization, PCAA contribution, beam width, and pruning schedule of
        the canonical sequential decoder.  Only the physical representation
        and expansion of the valid-child relation are changed.
        """
        if self.top_p < 1.0:
            raise NotImplementedError("GPU-flat decoding currently requires top_p=1.")
        if self.diversity_strength > 0.0:
            raise NotImplementedError(
                "GPU-flat decoding currently requires diversity_strength=0."
            )
        if getattr(self, "_semantic_prefix_expansion_hook", None) is not None:
            raise NotImplementedError(
                "GPU-flat decoding does not support semantic-prefix expansion hooks."
            )

        device = inputs_embeds.device
        model_dtype = inputs_embeds.dtype
        score_dtype = torch.float32
        batch_size = int(inputs_embeds.size(0))
        code_length = int(self.codebook_level)
        beam_limit = max(1, int(num_beams))

        if cand_codes is not None:
            tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
            self.tree_index = tree
            self.cand_token_ids = cand_token_ids
            self.cand_codes_cache = cand_codes_cache
            self.flat_tree_index = FlatPrefixTrie.from_token_ids(cand_token_ids)
            self._flat_tree_device = None
        flat_tree = self._flat_tree_on_device(device)
        if flat_tree.code_length != code_length:
            raise RuntimeError(
                f"Flat Trie depth {flat_tree.code_length} does not match model length {code_length}."
            )

        cur_seq = torch.full(
            (batch_size, 1, code_length),
            self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        trie_nodes = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        beam_scores = torch.zeros(batch_size, 1, device=device, dtype=score_dtype)
        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(batch_size, -1, -1).to(
            device=device, dtype=model_dtype
        )
        use_cfg = abs(float(self.guidance_scale) - 1.0) > 1e-8
        token_id_to_code = self.token_id_to_code.to(device)

        for level in range(code_length):
            cur_k = int(cur_seq.size(1))
            xt = self._build_prefix_regen_xt(cur_seq, level)
            flat_xt = xt.reshape(batch_size * cur_k, code_length)
            flat_tokens_for_embed = self._prepare_infer_tokens_from_xt(flat_xt)
            valid_mask = torch.ones_like(flat_xt, device=device, dtype=torch.long)
            all_mask = build_concentrate_attention_mask(
                mask_tokens=flat_xt,
                valid_mask=valid_mask,
                mask_token_id=self.mask_token_id,
            )
            t_val = max(
                (code_length - level) * self.time_step // max(code_length, 1), 1
            )
            timestep = torch.full(
                (batch_size * cur_k,), t_val, device=device, dtype=torch.long
            )
            flat_cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                batch_size * cur_k, self.num_prefix, self.d_model
            )
            hidden_c = self.id_generator.hidden_at_position(
                tokens=flat_tokens_for_embed,
                mask_tokens=flat_xt,
                prefix=flat_cond,
                mask=all_mask,
                t=timestep,
                position=level,
            ).float()
            if use_cfg:
                flat_uncond = base_uncond.unsqueeze(1).expand(
                    -1, cur_k, -1, -1
                ).reshape(batch_size * cur_k, self.num_prefix, self.d_model)
                hidden_u = self.id_generator.hidden_at_position(
                    tokens=flat_tokens_for_embed,
                    mask_tokens=flat_xt,
                    prefix=flat_uncond,
                    mask=all_mask,
                    t=timestep,
                    position=level,
                ).float()
            else:
                hidden_u = None

            level_size = (
                3
                if self.modality_index and level == 0
                else int(self.codebook_vocab)
            )
            level_token_ids = self.level_token_ids[level, :level_size].to(
                device=device, dtype=torch.long
            )
            level_weight = self.id_generator.gpt.lm_head.weight.index_select(
                0, level_token_ids
            ).float()
            logits_c = F.linear(hidden_c, level_weight)
            if use_cfg:
                logits_u = F.linear(hidden_u, level_weight)
                level_logits = logits_u + float(self.guidance_scale) * (
                    logits_c - logits_u
                )
            else:
                level_logits = logits_c
            if self.temperature != 1.0:
                level_logits = level_logits / max(float(self.temperature), 1e-6)
            level_log_denom = torch.logsumexp(level_logits, dim=-1)

            owner, edge_tokens, child_nodes, _ = flat_tree.expand(
                trie_nodes.reshape(-1), level
            )
            if owner.numel() == 0:
                raise RuntimeError(
                    f"GPU-flat Trie decoding reached a dead end at position {level}."
                )
            compact_ids = token_id_to_code.index_select(0, edge_tokens)
            # The Flat Trie is constructed from `_codes_to_token_ids`, whose
            # level-specific mapping is validated before serialization.  A
            # device-side validity reduction here would therefore only repeat
            # that offline invariant and force one host synchronization per
            # sequential level.

            edge_scores = level_logits[owner, compact_ids] - level_log_denom[owner]
            if rrg_query_embedding is not None:
                rrg_query_rows = rrg_query_embedding.unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(batch_size * cur_k, -1)
            else:
                rrg_query_rows = None
            rrg_score_rows = self._rrg_scores_for_rows(
                rrg_query_rows, flat_xt, level
            )
            if rrg_score_rows is not None:
                # The composition gate is defined over the full level and the
                # legal-child mask.  Constructing this modest [rows, level]
                # mask preserves the canonical PCAA scoring semantics without
                # materializing full-tokenizer logits.
                legal_level_mask = torch.zeros(
                    (batch_size * cur_k, level_size),
                    device=device,
                    dtype=torch.bool,
                )
                legal_level_mask[owner, compact_ids] = True
                composition_strength = self._residual_composition_strength_for_rows(
                    rrg_query_rows,
                    flat_xt,
                    level,
                    level_logits,
                    rrg_score_rows[:, :level_size],
                    valid_mask=legal_level_mask,
                )
                if composition_strength is not None:
                    edge_scores = edge_scores + composition_strength[owner] * rrg_score_rows[
                        owner, compact_ids
                    ]

            total_scores = beam_scores.reshape(-1)[owner] + edge_scores.to(score_dtype)
            top_scores, parent_idx, token_idx, selected_nodes = self._segmented_flat_topk(
                scores=total_scores,
                owners=owner,
                token_ids=edge_tokens,
                child_nodes=child_nodes,
                batch_size=batch_size,
                parents_per_query=cur_k,
                keep_limit=beam_limit,
            )
            gathered = cur_seq.gather(
                1, parent_idx.unsqueeze(-1).expand(-1, -1, code_length)
            ).clone()
            gathered[:, :, level] = token_idx
            cur_seq = gathered
            trie_nodes = selected_nodes
            beam_scores = top_scores

        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(code_length, 1))
        if return_scores:
            return cur_seq, beam_scores
        return cur_seq

    @torch.no_grad()
    def _constrained_beam_search_lowmem(
        self,
        inputs_embeds,
        attention_mask=None,
        num_beams=10,
        cand_codes=None,
        return_scores: bool = False,
        rrg_query_embedding: Optional[torch.Tensor] = None,
    ):
        """GENIUS-style low-memory constrained search for GPT-HDGR.

        GENIUS uses HF `generate` with `prefix_allowed_tokens_fn`, so it does
        not materialize full [B*K, L, vocab] logits.  For the non-AR GPT-HDGR
        decoder, we mimic that memory behavior by:
          1. computing hidden state only at the current semantic position j;
          2. projecting only the current level's semantic tokens;
          3. selecting only candidate-tree-allowed expansions.

        Scores are still normalized over the full valid level vocabulary, not
        only the Trie-allowed subset, matching the old full-logit path:
            masked level logits -> log_softmax -> Trie mask -> top-k.
        """
        device = inputs_embeds.device
        model_dtype = inputs_embeds.dtype
        score_dtype = torch.float32

        B0 = inputs_embeds.size(0)
        L = self.codebook_level
        K = max(1, int(num_beams))

        if cand_codes is not None:
            tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
            self.tree_index = tree
            self.cand_token_ids = cand_token_ids
            self.cand_codes_cache = cand_codes_cache

        cur_seq = torch.full((B0, 1, L), self.mask_token_id, device=device, dtype=torch.long)
        beam_scores = torch.zeros(B0, 1, device=device, dtype=score_dtype)

        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(B0, -1, -1).to(device=device, dtype=model_dtype)
        use_cfg = abs(float(self.guidance_scale) - 1.0) > 1e-8

        for j in range(L):
            cur_k = cur_seq.size(1)
            xt = self._build_prefix_regen_xt(cur_seq, j)
            flat_xt = xt.reshape(B0 * cur_k, L)
            flat_tokens_for_embed = self._prepare_infer_tokens_from_xt(flat_xt)

            valid_mask = torch.ones_like(flat_xt, device=device, dtype=torch.long)
            all_mask = build_concentrate_attention_mask(
                mask_tokens=flat_xt,
                valid_mask=valid_mask,
                mask_token_id=self.mask_token_id,
            )

            t_val = max((L - j) * self.time_step // max(L, 1), 1)
            t = torch.full((B0 * cur_k,), t_val, device=device, dtype=torch.long)

            flat_cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                B0 * cur_k, self.num_prefix, self.d_model
            )

            hidden_c = self.id_generator.hidden_at_position(
                tokens=flat_tokens_for_embed,
                mask_tokens=flat_xt,
                prefix=flat_cond,
                mask=all_mask,
                t=t,
                position=j,
            ).float()

            if use_cfg:
                flat_uncond = base_uncond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                    B0 * cur_k, self.num_prefix, self.d_model
                )
                hidden_u = self.id_generator.hidden_at_position(
                    tokens=flat_tokens_for_embed,
                    mask_tokens=flat_xt,
                    prefix=flat_uncond,
                    mask=all_mask,
                    t=t,
                    position=j,
                ).float()
            else:
                hidden_u = None

            # Compute logits only over the valid semantic tokens for this level.
            level_size = int(self.level_vocab_sizes[j].item())
            level_token_ids = self.level_token_ids[j, :level_size].to(device=device, dtype=torch.long)
            level_weight = self.id_generator.gpt.lm_head.weight.index_select(0, level_token_ids).float()

            logits_c_level = F.linear(hidden_c, level_weight)
            if use_cfg:
                logits_u_level = F.linear(hidden_u, level_weight)
                level_logits = logits_u_level + float(self.guidance_scale) * (logits_c_level - logits_u_level)
                del logits_u_level, hidden_u
            else:
                level_logits = logits_c_level

            if self.temperature != 1.0:
                level_logits = level_logits / max(float(self.temperature), 1e-6)

            # Normalization denominator over all level-valid tokens, like the
            # old _mask_invalid_logits(...)->log_softmax path.
            level_log_denom = torch.logsumexp(level_logits, dim=-1)

            allowed_lists = self._get_allowed_token_lists(flat_xt, j)
            if rrg_query_embedding is not None:
                rrg_query_rows = rrg_query_embedding.unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(B0 * cur_k, -1)
            else:
                rrg_query_rows = None
            rrg_score_rows = self._rrg_scores_for_rows(
                rrg_query_rows, flat_xt, j
            )
            composition_valid_mask = self._allowed_token_lists_to_level_mask(
                allowed_lists, j, device=device
            ) if rrg_score_rows is not None else None
            composition_strength_rows = self._residual_composition_strength_for_rows(
                rrg_query_rows,
                flat_xt,
                j,
                level_logits,
                rrg_score_rows,
                valid_mask=composition_valid_mask,
            )

            per_b_num_candidates = []
            batch_parent_chunks = []
            batch_token_chunks = []
            batch_score_chunks = []

            for b in range(B0):
                parent_chunks = []
                token_chunks = []
                score_chunks = []
                for kk in range(cur_k):
                    row = b * cur_k + kk
                    allowed = allowed_lists[row]
                    if not allowed:
                        continue
                    allowed_toks = torch.tensor(allowed, device=device, dtype=torch.long)
                    allowed_level_idx = self.token_id_to_code.to(device).index_select(0, allowed_toks)
                    valid = (allowed_level_idx >= 0) & (allowed_level_idx < level_size)
                    if not torch.all(valid):
                        allowed_toks = allowed_toks[valid]
                        allowed_level_idx = allowed_level_idx[valid]
                    if allowed_toks.numel() == 0:
                        continue

                    logp = level_logits[row].index_select(0, allowed_level_idx) - level_log_denom[row]
                    if rrg_score_rows is not None and composition_strength_rows is not None:
                        logp = logp + composition_strength_rows[row] * rrg_score_rows[
                            row
                        ].index_select(0, allowed_level_idx)
                    total = beam_scores[b, kk] + logp.to(score_dtype)

                    parent_chunks.append(torch.full_like(allowed_toks, kk, dtype=torch.long))
                    token_chunks.append(allowed_toks)
                    score_chunks.append(total)

                if not score_chunks:
                    raise RuntimeError(
                        f"Trie-constrained low-memory decoding reached a dead end at position {j}; "
                        "check candidate codes, tokenizer ids, and stale Trie caches."
                    )

                parent_cat = torch.cat(parent_chunks, dim=0)
                token_cat = torch.cat(token_chunks, dim=0)
                score_cat = torch.cat(score_chunks, dim=0)
                # Optional research hook for decision-aligned semantic-prefix
                # planning. It is absent in the canonical model, so the frozen
                # decoding path and scores remain bitwise unchanged by default.
                expansion_hook = getattr(
                    self, "_semantic_prefix_expansion_hook", None
                )
                if expansion_hook is not None:
                    score_cat = expansion_hook(
                        batch_index=b,
                        step_index=j,
                        parent_sequences=cur_seq[b],
                        parent_ids=parent_cat,
                        token_ids=token_cat,
                        raw_scores=score_cat,
                    )
                    if score_cat.shape != parent_cat.shape:
                        raise ValueError(
                            "Semantic-prefix expansion hook changed score shape: "
                            f"{tuple(score_cat.shape)} vs {tuple(parent_cat.shape)}"
                        )

                batch_parent_chunks.append(parent_cat)
                batch_token_chunks.append(token_cat)
                batch_score_chunks.append(score_cat)
                per_b_num_candidates.append(int(score_cat.numel()))

            keep_k = min(K, min(per_b_num_candidates))
            if keep_k <= 0:
                raise RuntimeError(f"No legal beam expansions at position {j}.")

            new_seq = []
            new_scores = []
            for b in range(B0):
                top_scores, parent_idx, token_idx = self._select_lowmem_expansions_for_batch(
                    raw_scores=batch_score_chunks[b],
                    parent_ids=batch_parent_chunks[b],
                    token_ids=batch_token_chunks[b],
                    keep_k=keep_k,
                    step_idx=j,
                )
                gathered = cur_seq[b].index_select(0, parent_idx).clone()
                gathered[:, j] = token_idx
                new_seq.append(gathered)
                new_scores.append(top_scores)

            cur_seq = torch.stack(new_seq, dim=0)
            beam_scores = torch.stack(new_scores, dim=0)

            del level_logits, logits_c_level, hidden_c, level_weight

        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(L, 1))
        if return_scores:
            return cur_seq, beam_scores
        return cur_seq


    @torch.no_grad()
    def constrained_beam_search(
        self,
        inputs_embeds,
        attention_mask=None,
        num_beams=10,
        cand_codes=None,
        return_scores: bool = False,
        rrg_query_embedding: Optional[torch.Tensor] = None,
    ):
        """
        Prefix-preserving suffix-regeneration decoding

        For step j:
          1) keep prefix [0, ..., j-1]
          2) mask positions [j, ..., L-1]
          3) run the model once to predict the whole sequence
          4) only commit token at position j
          5) repeat

        The training objective remains non-autoregressive masked denoising;
        only the constrained search commits one semantic-ID position per round.
        """
        if getattr(self, "gpu_flat_tree_decode", False):
            return self._constrained_beam_search_gpu_flat(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=num_beams,
                cand_codes=cand_codes,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )
        if getattr(self, "memory_efficient_tree_decode", False):
            return self._constrained_beam_search_lowmem(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=num_beams,
                cand_codes=cand_codes,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )

        device = inputs_embeds.device
        model_dtype = inputs_embeds.dtype
        score_dtype = torch.float32

        B0 = inputs_embeds.size(0)
        L = self.codebook_level
        V = self.vocab_size
        K = max(1, int(num_beams))

        # if cand_codes explicitly provided, rebuild tree every time to avoid stale cache
        if cand_codes is not None:
            tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
            self.tree_index = tree
            self.cand_token_ids = cand_token_ids
            self.cand_codes_cache = cand_codes_cache

        # beam state:
        # cur_seq stores the committed prefix tokens; uncommitted suffix stays as mask_token_id
        cur_seq = torch.full((B0, 1, L), self.mask_token_id, device=device, dtype=torch.long)
        # Keep cumulative search scores in fp32 even under autocast.  Nine
        # semantic-ID levels are short, but fp16 still loses useful differences
        # between close beams and can underflow after strict Trie masking.
        beam_scores = torch.zeros(B0, 1, device=device, dtype=score_dtype)

        base_cond = inputs_embeds  # [B0, P, D]
        base_uncond = self.null_condition.expand(B0, -1, -1).to(device=device, dtype=model_dtype)

        for j in range(L):
            cur_k = cur_seq.size(1)  # current beam width

            # --------------------------------------------------
            # Build xt = [fixed prefix | masked suffix]
            # --------------------------------------------------
            xt = self._build_prefix_regen_xt(cur_seq, j)  # [B0, K, L]
            flat_xt = xt.reshape(B0 * cur_k, L)  # [B0*K, L]

            # tokens for embedding path: masked positions temporarily replaced with 0
            flat_tokens_for_embed = self._prepare_infer_tokens_from_xt(flat_xt)

            # valid mask: all sequence positions are structurally valid during inference
            valid_mask = torch.ones_like(flat_xt, device=device, dtype=torch.long)
            all_mask = build_concentrate_attention_mask(
                mask_tokens=flat_xt,
                valid_mask=valid_mask,
                mask_token_id=self.mask_token_id
            )  # [B0*K, L, L]

            # timestep schedule: larger t for earlier refinement, smaller t for later refinement
            t_val = max((L - j) * self.time_step // max(L, 1), 1)
            t = torch.full((B0 * cur_k,), t_val, device=device, dtype=torch.long)

            flat_cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(B0 * cur_k, self.num_prefix,
                                                                                 self.d_model)
            flat_uncond = base_uncond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(B0 * cur_k, self.num_prefix,
                                                                                     self.d_model)

            # --------------------------------------------------
            # One full-sequence regeneration pass
            # --------------------------------------------------
            out_c = self.id_generator(
                tokens=flat_tokens_for_embed,
                mask_tokens=flat_xt,
                prefix=flat_cond,
                mask=all_mask,
                t=t,
                labels=None
            )
            out_u = self.id_generator(
                tokens=flat_tokens_for_embed,
                mask_tokens=flat_xt,
                prefix=flat_uncond,
                mask=all_mask,
                t=t,
                labels=None
            )

            logits_c = self._mask_invalid_logits(out_c.logits.float())  # [B0*K, L, V]
            logits_u = self._mask_invalid_logits(out_u.logits.float())

            # Classifier-free guidance is applied in logit space and normalized
            # once afterwards.  Combining separately normalized log-probabilities
            # adds parent-dependent constants that can distort beam comparisons.
            guided_logits = logits_u + float(self.guidance_scale) * (logits_c - logits_u)
            if self.temperature != 1.0:
                guided_logits = guided_logits / max(float(self.temperature), 1e-6)

            # We only commit position j in this round.
            step_logp = F.log_softmax(guided_logits[:, j, :], dim=-1)  # [B0*K, V]

            # Strict Trie legality is also part of the composition state.
            allowed_mask = self._get_allowed_mask(flat_xt, j, V)  # [B0*K, V]

            if rrg_query_embedding is not None:
                rrg_query_rows = rrg_query_embedding.unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(B0 * cur_k, -1)
            else:
                rrg_query_rows = None
            rrg_score_rows = self._rrg_scores_for_rows(
                rrg_query_rows, flat_xt, j
            )
            if rrg_score_rows is not None:
                level_size = int(self.level_vocab_sizes[j].item())
                level_token_ids = self.level_token_ids[j, :level_size].to(
                    device=device, dtype=torch.long
                )
                generator_level_scores = guided_logits[:, j, :].index_select(
                    1, level_token_ids
                )
                composition_strength_rows = self._residual_composition_strength_for_rows(
                    rrg_query_rows,
                    flat_xt,
                    j,
                    generator_level_scores,
                    rrg_score_rows[:, :level_size],
                    valid_mask=allowed_mask.index_select(1, level_token_ids),
                )
                if composition_strength_rows is not None:
                    step_logp[:, level_token_ids] = (
                        step_logp[:, level_token_ids]
                        + composition_strength_rows.unsqueeze(1) * rrg_score_rows[:, :level_size]
                    )

            # --------------------------------------------------
            # Strict tree constraint at the current prefix
            # --------------------------------------------------
            very_small = torch.finfo(step_logp.dtype).min
            step_logp = torch.where(allowed_mask, step_logp, torch.full_like(step_logp, very_small))

            # optional top-p
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logp, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = probs.cumsum(dim=-1)

                remove_mask = cum_probs > self.top_p
                remove_mask[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(remove_mask, very_small)
                step_logp = torch.full_like(step_logp, very_small).scatter(1, sorted_indices, sorted_logits)

            # reshape back to [B0, cur_k, V]
            step_logp = step_logp.view(B0, cur_k, V)

            # --------------------------------------------------
            # Standard beam expansion:
            #   total score = previous beam score + current position logp
            # --------------------------------------------------
            total_scores = beam_scores.unsqueeze(-1) + step_logp  # [B0, cur_k, V]

            # Do not pad a narrow Trie level with impossible hypotheses.  This
            # matters at the modality/root level, where only a handful of legal
            # tokens may exist even when num_beams is 50 or 100.  Beam width can
            # grow again at the next level as the number of legal prefixes grows.
            finite_floor = torch.finfo(total_scores.dtype).min / 2
            legal_counts = (total_scores > finite_floor).sum(dim=(1, 2))
            min_legal = int(legal_counts.min().item())
            if min_legal <= 0:
                raise RuntimeError(
                    f"Trie-constrained decoding reached a dead end at position {j}; "
                    "check candidate codes, tokenizer offsets, and stale Trie caches."
                )
            keep_k = min(K, cur_k * V, min_legal)
            top_scores, top_idx = self._select_diverse_expansions(
                total_scores=total_scores,
                keep_k=keep_k,
                step_idx=j,
            )

            parent_idx = top_idx // V
            token_idx = top_idx % V

            # gather parent beams
            gathered_seq = cur_seq.gather(
                1,
                parent_idx.unsqueeze(-1).expand(-1, -1, L)
            ).clone()  # [B0, keep_k, L]

            # commit current position j
            gathered_seq[:, :, j] = token_idx

            cur_seq = gathered_seq
            beam_scores = top_scores

        # cur_seq is already sorted by cumulative beam score when diversity is
        # disabled (the default).  With diversity enabled, preserve the selected
        # order because it reflects the configured coverage trade-off.
        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(L, 1))
        if return_scores:
            return cur_seq, beam_scores
        return cur_seq

    def _soundstorm_prefix_regen_timestep(self, level: int) -> int:
        """Choose the SoundStorm timestep that matches the masked-suffix ratio.

        The prefix-regeneration state at level ``j`` has exactly ``L-j`` masked
        positions.  ``mask_fraction`` selects the diffusion timestep whose
        cumulative mask probability is closest to that deterministic fraction.
        ``linear`` is a simple remaining-level schedule kept for ablation.
        """
        L = max(int(self.codebook_level), 1)
        remaining_fraction = float(L - int(level)) / float(L)
        diffusion = self.soundstorm_full_model.diffusion
        T = int(diffusion.num_timesteps)

        if self.soundstorm_prefix_regen_timestep_mode == "linear":
            return max(0, min(T - 1, int(round(remaining_fraction * (T - 1)))))

        # log_cumprod_ct contains the cumulative mask probabilities.  The last
        # sentinel entry is not a valid training timestep, so restrict to [:T].
        mask_probs = diffusion.log_cumprod_ct[:T].detach().float().exp()
        target = mask_probs.new_tensor(remaining_fraction)
        return int(torch.argmin((mask_probs - target).abs()).item())

    @torch.no_grad()
    def _soundstorm_prefix_regen_trie(
        self,
        cond_prefix: torch.Tensor,
        num_beams: int,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trie-in-the-loop prefix regeneration for the SoundStorm denoiser.

        For semantic-ID level ``j`` this decoder:
          1. keeps the already committed compact RQ prefix;
          2. sets positions ``j:`` to SoundStorm's MASK class;
          3. reruns the x0 denoiser (conditional + optional CFG unconditional);
          4. scores only level ``j``;
          5. expands only legal children of the current candidate-Trie node.

        Unlike final-logits projection, later token distributions are therefore
        recomputed for every surviving legal prefix.  This mirrors the original
        GPT-HDGR constrained suffix-regeneration test protocol while retaining
        the trained SoundStorm x0 parameterization.
        """
        if self.tree_index is None:
            raise RuntimeError(
                "SoundStorm prefix regeneration requires a loaded candidate tree. "
                "Call distribute_trie(cand_codes, trie_save_path) before inference."
            )
        if not self.use_soundstorm_full_model:
            raise RuntimeError("SoundStorm prefix regeneration requires soundstorm_full.")

        device = cond_prefix.device
        model_dtype = cond_prefix.dtype
        score_dtype = torch.float32
        B = int(cond_prefix.size(0))
        L = int(self.codebook_level)
        V = int(self.soundstorm_full_model.num_embed)
        K = max(1, int(num_beams))
        chunk_size = max(1, int(self.soundstorm_prefix_regen_chunk_size))
        compact_mask_id = int(self.soundstorm_full_model.mask_token_id)
        token_id_to_code = self.token_id_to_code.to(device)

        # Store tokenizer IDs because the candidate Trie is built over the
        # level-specific tokenizer tokens.  Uncommitted positions use -1 only
        # as a local sentinel and are never passed to the SoundStorm model.
        cur_seq = torch.full((B, 1, L), -1, device=device, dtype=torch.long)
        beam_scores = torch.zeros(B, 1, device=device, dtype=score_dtype)
        base_uncond = self.null_condition.expand(B, -1, -1).to(
            device=device, dtype=model_dtype
        )

        for level in range(L):
            cur_k = int(cur_seq.size(1))
            flat_token_seq = cur_seq.reshape(B * cur_k, L)

            # Convert the committed tokenizer prefix to compact per-level RQ
            # indices and mask the whole uncommitted suffix.
            flat_xt = torch.full(
                (B * cur_k, L),
                compact_mask_id,
                device=device,
                dtype=torch.long,
            )
            if level > 0:
                committed_token_ids = flat_token_seq[:, :level]
                committed_codes = token_id_to_code.index_select(
                    0, committed_token_ids.reshape(-1)
                ).reshape(B * cur_k, level)
                if bool((committed_codes < 0).any().item()):
                    raise RuntimeError(
                        "SoundStorm prefix regeneration found an invalid "
                        "tokenizer-to-code mapping in a committed Trie prefix."
                    )
                flat_xt[:, :level] = committed_codes

            # Traverse the Python Trie from a single CPU copy rather than doing
            # one GPU .item() synchronization per prefix token.
            flat_token_rows = flat_token_seq.detach().cpu().tolist()
            allowed_lists = []
            for row_tokens in flat_token_rows:
                node = self.tree_index
                for prefix_level in range(level):
                    token_id = int(row_tokens[prefix_level])
                    node = node.get(token_id) if node is not None else None
                    if node is None:
                        break
                allowed_lists.append([] if node is None else [int(x) for x in node.keys()])

            t_val = self._soundstorm_prefix_regen_timestep(level)
            query_index = torch.arange(B, device=device).unsqueeze(1).expand(
                B, cur_k
            ).reshape(-1)

            # Only retain the current-level logits.  Chunking avoids materializing
            # [B*K, L, V] activations for all 50 beams at once.
            level_logits_chunks = []
            for start in range(0, B * cur_k, chunk_size):
                end = min(start + chunk_size, B * cur_k)
                xt_chunk = flat_xt[start:end]
                qidx = query_index[start:end]
                t = torch.full(
                    (end - start,),
                    t_val,
                    device=device,
                    dtype=torch.long,
                )
                cond_chunk = cond_prefix.index_select(0, qidx)
                logits_c = self.soundstorm_full_model.denoise_logits(
                    xt_chunk, cond_chunk, t
                )[:, level, :].float()

                if self.soundstorm_prefix_regen_use_cfg:
                    uncond_chunk = base_uncond.index_select(0, qidx)
                    logits_u = self.soundstorm_full_model.denoise_logits(
                        xt_chunk, uncond_chunk, t
                    )[:, level, :].float()
                    guided = logits_u + float(self.guidance_scale) * (
                        logits_c - logits_u
                    )
                else:
                    guided = logits_c

                guided = torch.nan_to_num(
                    guided,
                    nan=0.0,
                    posinf=30.0,
                    neginf=-30.0,
                ).clamp(-30.0, 30.0)
                if self.temperature != 1.0:
                    guided = guided / max(float(self.temperature), 1.0e-6)
                level_logits_chunks.append(F.log_softmax(guided, dim=-1))

            level_logp = torch.cat(level_logits_chunks, dim=0)  # [B*cur_k,V]

            if query_embedding is not None:
                rrg_query_rows = query_embedding.unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(B * cur_k, -1)
            else:
                rrg_query_rows = None
            rrg_score_rows = self._rrg_scores_for_rows(
                rrg_query_rows, flat_token_seq, level
            )
            if rrg_score_rows is not None:
                usable = min(level_logp.size(-1), rrg_score_rows.size(-1))
                composition_valid_mask = self._allowed_token_lists_to_level_mask(
                    allowed_lists, level, device=device
                )[:, :usable]
                composition_strength_rows = self._residual_composition_strength_for_rows(
                    rrg_query_rows,
                    flat_token_seq,
                    level,
                    level_logp[:, :usable],
                    rrg_score_rows[:, :usable],
                    valid_mask=composition_valid_mask,
                )
                if composition_strength_rows is not None:
                    level_logp[:, :usable] = (
                        level_logp[:, :usable]
                        + composition_strength_rows.unsqueeze(1) * rrg_score_rows[:, :usable]
                    )

            batch_parent = []
            batch_token = []
            batch_score = []
            candidate_counts = []

            for b in range(B):
                parent_chunks = []
                token_chunks = []
                score_chunks = []
                for k in range(cur_k):
                    row = b * cur_k + k
                    allowed = allowed_lists[row]
                    if not allowed:
                        continue

                    allowed_token_ids = torch.as_tensor(
                        allowed, device=device, dtype=torch.long
                    )
                    compact_ids = token_id_to_code.index_select(
                        0, allowed_token_ids
                    )
                    valid = (compact_ids >= 0) & (compact_ids < V)
                    if not bool(valid.all().item()):
                        allowed_token_ids = allowed_token_ids[valid]
                        compact_ids = compact_ids[valid]
                    if allowed_token_ids.numel() == 0:
                        continue

                    child_logp = level_logp[row].index_select(0, compact_ids)
                    if self.top_p < 1.0 and child_logp.numel() > 1:
                        order = torch.argsort(child_logp, descending=True)
                        ordered_prob = torch.softmax(child_logp.index_select(0, order), dim=0)
                        remove = ordered_prob.cumsum(dim=0) > float(self.top_p)
                        remove[0] = False
                        keep = torch.ones_like(remove, dtype=torch.bool)
                        keep[order] = ~remove
                        allowed_token_ids = allowed_token_ids[keep]
                        child_logp = child_logp[keep]
                    if allowed_token_ids.numel() == 0:
                        continue

                    scores = beam_scores[b, k] + child_logp.to(score_dtype)
                    parent_chunks.append(
                        torch.full(
                            (allowed_token_ids.numel(),),
                            k,
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    token_chunks.append(allowed_token_ids)
                    score_chunks.append(scores)

                if not score_chunks:
                    raise RuntimeError(
                        f"SoundStorm prefix regeneration reached a dead end at "
                        f"semantic-ID level {level}. Delete stale candidate Trie "
                        "files and regenerate the candidate-code cache."
                    )

                parents = torch.cat(parent_chunks, dim=0)
                tokens = torch.cat(token_chunks, dim=0)
                scores = torch.cat(score_chunks, dim=0)
                batch_parent.append(parents)
                batch_token.append(tokens)
                batch_score.append(scores)
                candidate_counts.append(int(scores.numel()))

            # Keep a common dense beam width.  It may be narrow at the modality
            # root and grow again as legal candidate prefixes branch.
            keep_k = min(K, min(candidate_counts))
            if keep_k <= 0:
                raise RuntimeError(
                    f"No legal SoundStorm prefix expansions at level {level}."
                )

            new_seq = []
            new_scores = []
            for b in range(B):
                top_scores, parent_idx, token_idx = (
                    self._select_lowmem_expansions_for_batch(
                        raw_scores=batch_score[b],
                        parent_ids=batch_parent[b],
                        token_ids=batch_token[b],
                        keep_k=keep_k,
                        step_idx=level,
                    )
                )
                seq = cur_seq[b].index_select(0, parent_idx).clone()
                seq[:, level] = token_idx
                new_seq.append(seq)
                new_scores.append(top_scores)

            cur_seq = torch.stack(new_seq, dim=0)
            beam_scores = torch.stack(new_scores, dim=0)

        output_codes = self._token_ids_to_codes(cur_seq)
        if bool((output_codes < 0).any().item()):
            raise RuntimeError(
                "SoundStorm prefix regeneration produced an invalid final code."
            )
        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(L, 1))
        return output_codes.long(), beam_scores.float()

    @torch.no_grad()
    def _soundstorm_trie_project_logits(
        self,
        logits_blv: torch.Tensor,
        num_beams: int,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project SoundStorm x0 logits onto legal candidate semantic IDs.

        SoundStorm predicts all RQ levels non-autoregressively.  Independent
        per-level sampling almost never lands exactly on a semantic ID that
        exists in a large retrieval pool.  This method keeps the SoundStorm
        denoiser, but performs a low-memory beam search over the candidate
        trie using the final predicted clean-token log probabilities.

        Args:
            logits_blv: final SoundStorm x0 logits, shape [B, L, V], where V
                is the compact RQ code vocabulary (not tokenizer vocabulary).
            num_beams: number of legal candidate codes to return per query.

        Returns:
            codes: [B, K, L] compact RQ codes, every row guaranteed to exist
                in the currently loaded candidate pool.
            scores: [B, K] cumulative log-probability scores, sorted high to
                low.
        """
        if self.tree_index is None:
            raise RuntimeError(
                "SoundStorm trie projection requires a loaded candidate tree. "
                "Call distribute_trie(cand_codes, trie_save_path) before inference."
            )

        if logits_blv.dim() != 3:
            raise ValueError(
                f"Expected SoundStorm logits [B,L,V], got {tuple(logits_blv.shape)}"
            )

        B, L, V = logits_blv.shape
        if L != self.codebook_level:
            raise ValueError(
                f"Expected code length {self.codebook_level}, got {L}"
            )

        # Use the exact SoundStorm x0 normalization used by the diffusion
        # objective.  Drop the extra MASK class; candidate codes only contain
        # real RQ tokens.
        log_x0 = self.soundstorm_full_model.diffusion.predict_start_from_logits(
            logits_blv
        )
        level_logp = log_x0[:, :-1, :].transpose(1, 2).contiguous()  # [B,L,V]

        device = logits_blv.device
        score_dtype = torch.float32
        K = max(1, int(num_beams))

        # Store tokenizer IDs because the existing candidate trie is built over
        # tokenizer IDs.  Convert back to compact RQ indices at the end.
        # -1 is only a local, uncommitted suffix sentinel.  Do not reuse the
        # SoundStorm MASK class ID here: the candidate trie is built from the
        # much larger tokenizer vocabulary, where integer 4096 can itself be a
        # valid semantic token ID.
        cur_seq = torch.full(
            (B, 1, L),
            -1,
            device=device,
            dtype=torch.long,
        )
        beam_scores = torch.zeros(B, 1, device=device, dtype=score_dtype)
        token_id_to_code = self.token_id_to_code.to(device)

        for level in range(L):
            cur_k = int(cur_seq.size(1))
            flat_seq = cur_seq.reshape(B * cur_k, L)

            # Traverse only the committed prefix.  This deliberately avoids
            # get_layer_nodes_from_tree(..., mask_token_id), because the
            # SoundStorm compact MASK class can numerically collide with a
            # valid tokenizer token ID used by the candidate trie.
            allowed_lists = []
            for row in range(flat_seq.size(0)):
                node = self.tree_index
                for prefix_level in range(level):
                    token_id = int(flat_seq[row, prefix_level].item())
                    node = node.get(token_id) if node is not None else None
                    if node is None:
                        break
                allowed_lists.append([] if node is None else list(node.keys()))

            if query_embedding is not None:
                rrg_query_rows = query_embedding.unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(B * cur_k, -1)
            else:
                rrg_query_rows = None
            rrg_score_rows = self._rrg_scores_for_rows(
                rrg_query_rows, flat_seq, level
            )
            if rrg_score_rows is not None:
                generator_level_rows = level_logp[:, level, :].unsqueeze(1).expand(
                    -1, cur_k, -1
                ).reshape(B * cur_k, -1)
                composition_valid_mask = self._allowed_token_lists_to_level_mask(
                    allowed_lists, level, device=device
                )[:, : generator_level_rows.size(-1)]
                composition_strength_rows = self._residual_composition_strength_for_rows(
                    rrg_query_rows,
                    flat_seq,
                    level,
                    generator_level_rows,
                    rrg_score_rows[:, : generator_level_rows.size(-1)],
                    valid_mask=composition_valid_mask,
                )
            else:
                composition_strength_rows = None

            batch_parent = []
            batch_token = []
            batch_score = []
            candidate_counts = []

            for b in range(B):
                parent_chunks = []
                token_chunks = []
                score_chunks = []

                for k in range(cur_k):
                    allowed = allowed_lists[b * cur_k + k]
                    if not allowed:
                        continue

                    allowed_token_ids = torch.as_tensor(
                        allowed, device=device, dtype=torch.long
                    )
                    compact_ids = token_id_to_code.index_select(
                        0, allowed_token_ids
                    )
                    valid = (compact_ids >= 0) & (compact_ids < V)
                    if not bool(valid.all().item()):
                        allowed_token_ids = allowed_token_ids[valid]
                        compact_ids = compact_ids[valid]
                    if allowed_token_ids.numel() == 0:
                        continue

                    child_score = level_logp[b, level].index_select(
                        0, compact_ids
                    ).to(score_dtype)
                    if rrg_score_rows is not None and composition_strength_rows is not None:
                        row = b * cur_k + k
                        child_score = child_score + composition_strength_rows[row].to(score_dtype) * (
                            rrg_score_rows[row].index_select(0, compact_ids).to(score_dtype)
                        )
                    scores = beam_scores[b, k] + child_score
                    parent_chunks.append(
                        torch.full(
                            (allowed_token_ids.numel(),),
                            k,
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    token_chunks.append(allowed_token_ids)
                    score_chunks.append(scores)

                if not score_chunks:
                    raise RuntimeError(
                        f"SoundStorm trie projection reached a dead end at level {level}. "
                        "Delete stale candidate trie files and regenerate candidate codes."
                    )

                parents = torch.cat(parent_chunks, dim=0)
                tokens = torch.cat(token_chunks, dim=0)
                scores = torch.cat(score_chunks, dim=0)
                batch_parent.append(parents)
                batch_token.append(tokens)
                batch_score.append(scores)
                candidate_counts.append(int(scores.numel()))

            # Keep a common beam width so outputs can stay dense [B,K,L].
            keep_k = min(K, min(candidate_counts))
            if keep_k <= 0:
                raise RuntimeError(
                    f"No legal SoundStorm candidate expansions at level {level}."
                )

            new_seq = []
            new_scores = []
            for b in range(B):
                top_scores, top_pos = torch.topk(
                    batch_score[b], k=keep_k, largest=True, sorted=True
                )
                parent_idx = batch_parent[b].index_select(0, top_pos)
                token_idx = batch_token[b].index_select(0, top_pos)
                seq = cur_seq[b].index_select(0, parent_idx).clone()
                seq[:, level] = token_idx
                new_seq.append(seq)
                new_scores.append(top_scores)

            cur_seq = torch.stack(new_seq, dim=0)
            beam_scores = torch.stack(new_scores, dim=0)

        output_codes = self._token_ids_to_codes(cur_seq)
        if (output_codes < 0).any():
            raise RuntimeError(
                "SoundStorm trie projection produced an invalid tokenizer-to-code mapping."
            )

        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(L, 1))
        return output_codes.long(), beam_scores.float()

    # =====================================================
    # inference
    # =====================================================

    @torch.no_grad()
    def inference(
        self,
        img_emb,
        txt_emb,
        img_mask,
        txt_mask,
        inst_ids,
        num_beams=10,
        cand_codes=None,
        return_beam_scores: bool = False,
    ):
        device = img_emb.device
        bs = len(img_emb)

        q_output = self.quantizer.inference(img_emb, txt_emb, img_mask, txt_mask)
        emb = F.normalize(q_output['encode'])

        cond_prefix = self._project_condition(emb)

        if self.use_soundstorm_full_model:
            K = max(1, int(num_beams))

            use_prefix_regen = (
                self.tree_index is not None
                and self.soundstorm_decode_mode in {
                    "trie_prefix_regen",
                    "prefix_regen",
                    "prefix_regeneration",
                    "trie_in_loop",
                }
            )
            use_trie_projection = (
                self.tree_index is not None
                and self.soundstorm_decode_mode in {
                    "trie_project",
                    "trie_projection",
                    "candidate_trie",
                    "candidate_constrained",
                }
            )

            if use_prefix_regen:
                if not getattr(self, "_soundstorm_prefix_regen_logged", False):
                    schedule = [
                        self._soundstorm_prefix_regen_timestep(j)
                        for j in range(self.codebook_level)
                    ]
                    print(
                        "[SoundStorm eval] Enabled Trie-in-the-loop prefix regeneration: "
                        f"legal_beams={K} cfg={self.soundstorm_prefix_regen_use_cfg} "
                        f"guidance={float(self.guidance_scale):.3f} "
                        f"chunk={self.soundstorm_prefix_regen_chunk_size} "
                        f"timesteps={schedule}. Every returned semantic ID is "
                        "constrained to the current candidate pool."
                    )
                    self._soundstorm_prefix_regen_logged = True
                sampled_codes, beam_scores = self._soundstorm_prefix_regen_trie(
                    cond_prefix,
                    num_beams=K,
                    query_embedding=emb,
                )
            elif use_trie_projection:
                if not getattr(self, "_soundstorm_trie_projection_logged", False):
                    print(
                        "[SoundStorm eval] Enabled candidate-trie projection: "
                        f"reverse_steps={self.soundstorm_sample_steps} legal_beams={K}. "
                        "Every returned semantic ID is constrained to the current candidate pool."
                    )
                    self._soundstorm_trie_projection_logged = True

                # Run one SoundStorm reverse chain per query, then project the
                # final clean-token distribution onto the candidate trie.  K is
                # the number of legal candidate beams, not K independent free
                # samples.  This guarantees that every returned semantic ID is
                # present in the current candidate pool.
                _, final_logits = self.soundstorm_full_model.sample(
                    cond_prefix,
                    num_iter=self.soundstorm_sample_steps,
                    return_logits=True,
                )
                sampled_codes, beam_scores = self._soundstorm_trie_project_logits(
                    final_logits,
                    num_beams=K,
                    query_embedding=emb,
                )
            else:
                # Fallback for standalone/interactive inference where no
                # candidate pool is available.
                cond_rep = cond_prefix.repeat_interleave(K, dim=0)
                sampled_codes = self.soundstorm_full_model.sample(
                    cond_rep,
                    num_iter=self.soundstorm_sample_steps,
                    return_logits=False,
                ).reshape(bs, K, self.codebook_level)
                beam_scores = torch.zeros(
                    bs, K, device=device, dtype=torch.float32
                )

            query_embedding = F.normalize(img_emb * img_mask + txt_emb * txt_mask)
            if return_beam_scores:
                return (
                    sampled_codes.detach(),
                    query_embedding,
                    beam_scores.detach(),
                )
            return sampled_codes.detach(), query_embedding

        search_output = self.constrained_beam_search(
            inputs_embeds=cond_prefix,
            num_beams=num_beams,
            cand_codes=cand_codes,
            return_scores=return_beam_scores,
            rrg_query_embedding=emb,
        )

        if return_beam_scores:
            output_token_ids, beam_scores = search_output
        else:
            output_token_ids = search_output

        output_codes = self._token_ids_to_codes(output_token_ids)
        query_embedding = F.normalize(img_emb * img_mask + txt_emb * txt_mask)
        if return_beam_scores:
            return output_codes.detach(), query_embedding, beam_scores.detach()
        return output_codes.detach(), query_embedding
    # =====================================================
    # encode batch
    # =====================================================

    def encode_mbeir_batch(self, batch, num_beams=10, cand_codes=None, trie_save_path=None, init_dataset=True):
        if init_dataset:
            self.distribute_trie(cand_codes, trie_save_path)

        # distribute_trie() owns candidate-index initialization.  Passing the
        # full candidate array into inference made the decoder rebuild the
        # entire Python tree once per query batch (catastrophic for UNION).
        decode_cand_codes = None if self.tree_index is not None else cand_codes

        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        img_emb, txt_emb = self.clip_model.encode_multimodal_input(
            batch["image_batched"],
            batch["txt_batched"],
        )
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)
        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."

        if self.save_beam_scores:
            output, embeddings, beam_scores = self.inference(
                img_emb,
                txt_emb,
                img_mask,
                txt_mask,
                batch["inst_ids"],
                num_beams=num_beams,
                cand_codes=decode_cand_codes,
                return_beam_scores=True,
            )
            return output, embeddings, torch.LongTensor(id_list), beam_scores

        output, embeddings = self.inference(
            img_emb,
            txt_emb,
            img_mask,
            txt_mask,
            batch["inst_ids"],
            num_beams=num_beams,
            cand_codes=decode_cand_codes,
            return_beam_scores=False,
        )
        return output, embeddings, torch.LongTensor(id_list)

    # =====================================================
    # forward
    # =====================================================

    def forward(
        self,
        input=None,
        encode_mbeir_batch=False,
        num_beams=10,
        cand_codes=None,
        init_dataset=True,
        trie_save_path=None,
        gpu_id=None
    ):
        if isinstance(gpu_id, torch.device):
            target_device = gpu_id
        elif isinstance(gpu_id, int):
            target_device = torch.device(f'cuda:{gpu_id}')
        else:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(target_device)

        if hasattr(self, 'quantizer'):
            self.quantizer.to(target_device)
        if hasattr(self, 'id_generator'):
            self.id_generator.to(target_device)
        if hasattr(self, 'embed_projector'):
            self.embed_projector.to(target_device)

        if encode_mbeir_batch:
            return self.encode_mbeir_batch(
                input,
                cand_codes=cand_codes,
                num_beams=num_beams,
                init_dataset=init_dataset,
                trie_save_path=trie_save_path
            )
        else:
            return self.compute_single_batch(input, target_device)
