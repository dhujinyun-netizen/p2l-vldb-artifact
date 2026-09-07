"""Conditional SoundStorm full-model adapter for GPT-HDGR / M-BEIR.

This file copies the SoundStorm S2 / DALLE-VQ-Diffusion model family at the
model-structure level, with only interface-level changes:

  SoundStorm original:
      condition = prompt_semantics + target_semantics + prompt_acoustics
      target    = target_acoustics  [B, n_q, T]

  Retrieval adapter:
      condition = query-conditioned prefix embeddings [B, P, D]
      target    = RQ semantic-ID code tokens [B, n_q]  (T=1)

The diffusion math is delegated to ``SoundStormTokenDiffusion``, which is a
separate copy/adaptation of SoundStorm's ``DiffusionTransformer`` objective.
The denoiser below is the SoundStorm Text2ImageTransformer/DalleMaskEmbedding
architecture adapted to compact semantic-code classes rather than tokenizer IDs.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.hdgr_comparison.soundstorm_diffusion import (
    SoundStormTokenDiffusion,
    index_to_log_onehot,
    log_onehot_to_index,
)


# -----------------------------------------------------------------------------
# Copied/adapted from SoundStorm: mask_embedding.py
# -----------------------------------------------------------------------------

class DalleMaskImageEmbedding(nn.Module):
    """SoundStorm DalleMaskImageEmbedding adapted for compact code classes.

    ``num_embed`` is the number of real code classes.  The module internally adds
    one extra class at ``num_embed`` for the mask token, exactly like SoundStorm.
    """

    def __init__(
        self,
        num_embed: int = 1024,
        max_size: int = 1024,
        embed_dim: int = 512,
        n_q: int = 4,
        trainable: bool = True,
        pos_emb_type: str = "embedding",
    ):
        super().__init__()
        self.max_size = int(max_size)
        self.num_embed = int(num_embed) + 1
        self.embed_dim = int(embed_dim)
        self.trainable = bool(trainable)
        self.n_q = int(n_q)
        self.pos_emb_type = str(pos_emb_type)
        assert self.pos_emb_type in ["embedding", "parameter"]

        self.embs = nn.ModuleDict({
            str(i): nn.Embedding(self.num_embed, self.embed_dim)
            for i in range(self.n_q)
        })
        self.register_buffer("position_ids", torch.arange(self.max_size), persistent=False)
        if self.pos_emb_type == "embedding":
            self.pos_emb = nn.Embedding(self.max_size, self.embed_dim)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_size, self.embed_dim))

        if not self.trainable:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, index: torch.Tensor, **kwargs):
        # index: [B, n_q * T]
        assert index.dim() == 2, f"expected [B, L], got {tuple(index.shape)}"
        B, L = index.shape
        if L % self.n_q != 0:
            raise ValueError(f"sequence length {L} must be divisible by n_q={self.n_q}")
        T = L // self.n_q
        index = index.reshape(B, self.n_q, T).long().clamp(min=0, max=self.num_embed - 1)

        emb_per_q = []
        for q in range(self.n_q):
            emb_per_q.append(self.embs[str(q)](index[:, q, :]).unsqueeze(1))  # [B,1,T,D]
        target_emb = torch.cat(emb_per_q, dim=1)  # [B,n_q,T,D]

        if self.pos_emb_type == "embedding":
            position_ids = self.position_ids[:T]
            pos = self.pos_emb(position_ids)[None, :, :]  # [1,T,D]
        else:
            pos = self.pos_emb[:, :T, :]
        return target_emb, pos


# -----------------------------------------------------------------------------
# Copied/adapted from SoundStorm: transformer_utils.py
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=(1, 2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor=(1, 2), bilinear: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode="nearest")
        # The copied SoundStorm UNet assumes exact shape recovery.  With T=1 and
        # odd n_q, MaxPool2d/nearest can differ by one cell depending on PyTorch;
        # crop/pad to preserve the original skip shape before concatenation.
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="nearest")
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FullAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.1, resid_pdrop: float = 0.1, causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = int(n_head)
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            slf_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        else:
            slf_mask = None
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if slf_mask is not None:
            att = att.masked_fill(slf_mask, -torch.finfo(att.dtype).max)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att_mean = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att_mean


class CrossAttention(nn.Module):
    def __init__(self, n_embd: int, condition_embd: int, n_head: int, attn_pdrop: float = 0.1, resid_pdrop: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = int(n_head)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            slf_mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        else:
            slf_mask = None
        B, T, C = x.size()
        B2, T_E, _ = encoder_output.size()
        assert B == B2
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if slf_mask is not None:
            att = att.masked_fill(slf_mask, -torch.finfo(att.dtype).max)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att_mean = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att_mean


class GELU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = int(dim)
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd: int, diffusion_step: int, emb_type: str = "adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type or emb_type == "adalayernorm":
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class Block(nn.Module):
    def __init__(
        self,
        class_type: str = "adalayernorm",
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        activate: str = "GELU2",
        attn_type: str = "selfcross",
        condition_dim: int = 1024,
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm",
        mlp_type: str = "fc",
    ):
        super().__init__()
        self.attn_type = attn_type
        if attn_type in ["selfcross", "selfcondition", "self"]:
            self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if attn_type in ["self", "selfcondition"]:
            self.attn = FullAttention(n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        elif attn_type == "selfcross":
            self.attn1 = FullAttention(n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
            self.attn2 = CrossAttention(n_embd=n_embd, condition_embd=condition_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
            self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
        else:
            raise ValueError(f"Unsupported attn_type={attn_type}")

        if isinstance(activate, str):
            act = nn.GELU() if activate.upper() == "GELU" else GELU2()
        else:
            act = GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, x_mask: Optional[torch.Tensor], cond_emb_mask: Optional[torch.Tensor], timestep: torch.Tensor):
        max_len = x.shape[1]
        slf_x_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1) if x_mask is not None else None

        if self.attn_type == "selfcross":
            a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
            a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=None)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a
            x = x + self.mlp(x + encoder_output)
            return x, att
        else:
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=slf_x_attn_mask)
            if x_mask is not None:
                a = a.masked_fill(x_mask.unsqueeze(-1), 0)
            x = x + a

        x = x + self.mlp(self.ln2(x))
        if x_mask is not None:
            x = x.masked_fill(x_mask.unsqueeze(-1), 0)
        return x, att


class SoundStormText2ImageTransformer(nn.Module):
    """SoundStorm Text2ImageTransformer with condition already embedded.

    Original SoundStorm builds condition embeddings from audio semantic/acoustic
    token dictionaries inside ``forward``.  For retrieval, this is the only
    intentional interface change: ``condition`` is directly a query-prefix tensor
    [B, P, D].  The content embedding, U-Net-like q-axis conv, selfcross blocks,
    AdaLayerNorm timestep conditioning, and per-codebook output heads are kept.
    """

    def __init__(
        self,
        n_layer: int = 16,
        n_q: int = 9,
        n_embd: int = 512,
        n_head: int = 8,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU2",
        attn_type: str = "selfcross",
        condition_dim: int = 512,
        diffusion_step: int = 100,
        timestep_type: str = "adalayernorm",
        mlp_type: str = "fc",
        content_num_embed: int = 4096,
        content_max_size: int = 16,
        content_pos_emb_type: str = "embedding",
    ):
        super().__init__()
        self.n_q = int(n_q)
        self.n_embd = int(n_embd)
        self.condition_dim = int(condition_dim)
        self.content_emb = DalleMaskImageEmbedding(
            num_embed=content_num_embed,
            max_size=content_max_size,
            embed_dim=n_embd,
            n_q=n_q,
            trainable=True,
            pos_emb_type=content_pos_emb_type,
        )

        self.inc = DoubleConv(condition_dim, condition_dim)
        self.down1 = Down(condition_dim, condition_dim, kernel_size=(self.n_q, 1))
        self.up1 = Up(condition_dim * 2, condition_dim, scale_factor=(self.n_q, 1))

        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=attn_type,
                condition_dim=condition_dim,
                diffusion_step=diffusion_step,
                timestep_type=timestep_type,
                mlp_type=mlp_type,
            )
            for _ in range(n_layer)
        ])

        self.out_cls = self.content_emb.num_embed - 1
        self.register_buffer("cls_ids", torch.arange(self.out_cls), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input: torch.Tensor, condition: torch.Tensor, x_mask: Optional[torch.Tensor], cond_emb_mask: Optional[torch.Tensor], t: torch.Tensor):
        # input: [B, n_q * T]
        cont_emb, pos_emb = self.content_emb(input)  # [B,n_q,T,D], [1,T,D]
        cont_emb = cont_emb.permute(0, 3, 1, 2)      # [B,D,n_q,T]
        x1 = self.inc(cont_emb)
        x2 = self.down1(x1)                          # [B,D,1,T]

        emb = x2.transpose(1, 3).reshape(x2.shape[0], -1, x2.shape[1])  # [B,T,D]
        emb = emb + pos_emb[:, :emb.shape[1], :].to(dtype=emb.dtype, device=emb.device)
        cond_emb = condition

        if x_mask is None:
            x_mask = torch.zeros(emb.shape[0], emb.shape[1], device=emb.device, dtype=torch.bool)
        for block in self.blocks:
            emb, _ = block(emb, cond_emb, x_mask, cond_emb_mask, t.to(emb.device))

        x3 = emb.unsqueeze(1).permute(0, 3, 1, 2)     # [B,D,1,T]
        x = self.up1(x3, x1).permute(0, 2, 3, 1)      # [B,n_q,T,D]

        logits = []
        # Memory-safe classifier head.  The first full-model patch followed the
        # SoundStorm-style per-level embedding lookup too literally and expanded
        # the codebook embedding to [B, V, D].  With B=512, V=4096, D=512 this
        # alone allocates about 4 GiB and OOMs on a 24GB GPU.  The result is
        # mathematically the same if we use the shared embedding table directly
        # as a classifier matrix [V, D] and multiply [B, T, D] @ [D, V].
        for q in range(self.n_q):
            weight = self.content_emb.embs[str(q)].weight[: self.out_cls]  # [V,D]
            tmp_logit = torch.matmul(x[:, q, :, :], weight.transpose(0, 1))  # [B,T,V]
            logits.append(tmp_logit)
        logits = torch.cat(logits, dim=1)             # [B,n_q*T,V]
        out = rearrange(logits, "b l c -> b c l")    # [B,V,L]
        return out


# -----------------------------------------------------------------------------
# DALLE wrapper + diffusion interface
# -----------------------------------------------------------------------------

class ConditionalSoundStormDALLE(nn.Module):
    """SoundStorm DALLE wrapper for query-conditioned RQ semantic IDs."""

    def __init__(
        self,
        num_embed: int,
        n_q: int,
        condition_dim: int = 512,
        diffusion_step: int = 100,
        alpha_init_type: str = "alpha1",
        auxiliary_loss_weight: float = 5.0e-4,
        adaptive_auxiliary_loss: bool = True,
        mask_weight: Sequence[float] = (1.0, 1.0),
        n_layer: int = 16,
        n_embd: int = 512,
        n_head: int = 8,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU2",
        attn_type: str = "selfcross",
        timestep_type: str = "adalayernorm",
        mlp_type: str = "fc",
        max_size: int = 16,
        level_vocab_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.n_q = int(n_q)
        self.num_embed = int(num_embed)
        self.num_classes = self.num_embed + 1
        self.mask_token_id = self.num_embed

        self.transformer = SoundStormText2ImageTransformer(
            n_layer=n_layer,
            n_q=n_q,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            block_activate=block_activate,
            attn_type=attn_type,
            condition_dim=condition_dim,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
            mlp_type=mlp_type,
            content_num_embed=num_embed,
            content_max_size=max_size,
            content_pos_emb_type="embedding",
        )
        self.diffusion = SoundStormTokenDiffusion(
            num_classes=self.num_classes,
            num_timesteps=diffusion_step,
            alpha_init_type=alpha_init_type,
            auxiliary_loss_weight=auxiliary_loss_weight,
            adaptive_auxiliary_loss=adaptive_auxiliary_loss,
            mask_weight=mask_weight,
        )
        if level_vocab_sizes is None:
            level_vocab_sizes = [self.num_embed] * self.n_q
        sizes = torch.tensor(list(level_vocab_sizes), dtype=torch.long)
        if sizes.numel() != self.n_q:
            raise ValueError(f"level_vocab_sizes length must equal n_q={self.n_q}")
        self.register_buffer("level_vocab_sizes", sizes.clamp(min=1, max=self.num_embed), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def _mask_invalid_level_logits(self, logits_bvl: torch.Tensor) -> torch.Tensor:
        # logits_bvl: [B, V, L]
        B, V, L = logits_bvl.shape
        if L != self.n_q:
            # In the original audio use case L=n_q*T.  For retrieval T=1.
            return logits_bvl
        out = logits_bvl.clone()
        for level in range(L):
            valid = int(self.level_vocab_sizes[level].item())
            if valid < V:
                out[:, valid:, level] = -1.0e4
        return out

    def denoise_logits(self, xt_ids: torch.Tensor, cond_emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # xt_ids: [B,L], cond_emb: [B,P,D], returns [B,L,V]
        B, L = xt_ids.shape
        target_frames = max(1, L // self.n_q)
        x_mask = torch.zeros(B, target_frames, device=xt_ids.device, dtype=torch.bool)
        logits_bvl = self.transformer(xt_ids, cond_emb, x_mask, None, t)
        logits_bvl = self._mask_invalid_level_logits(logits_bvl)
        return logits_bvl.transpose(1, 2).contiguous()

    def forward(self, target_codes: torch.Tensor, cond_emb: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, is_train: bool = True):
        # target_codes: [B,n_q] compact RQ code indices, not tokenizer IDs.
        target_codes = target_codes.long().clamp(min=0, max=self.num_embed - 1)
        if valid_mask is None:
            valid_mask = torch.ones_like(target_codes, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        def model_forward_fn(xt_ids: torch.Tensor, t: torch.Tensor):
            return self.denoise_logits(xt_ids, cond_emb, t)

        out = self.diffusion.train_loss(
            x_start=target_codes,
            valid_mask=valid_mask,
            model_forward_fn=model_forward_fn,
            is_train=is_train,
        )
        return out

    @torch.no_grad()
    def sample(self, cond_emb: torch.Tensor, num_iter: Optional[int] = None, return_logits: bool = False):
        """SoundStorm all-mask discrete diffusion sampling adapted for retrieval."""
        self.eval()
        B = cond_emb.size(0)
        L = self.n_q
        device = cond_emb.device
        num_iter = int(num_iter or self.diffusion.num_timesteps)
        num_iter = max(1, min(num_iter, self.diffusion.num_timesteps))

        zero_logits = torch.zeros((B, self.num_classes - 1, L), device=device)
        one_logits = torch.ones((B, 1, L), device=device)
        log_z = torch.log(torch.cat((zero_logits, one_logits), dim=1).clamp_min(1.0e-30))

        # Follow SoundStorm's reverse loop.  If fewer iterations are requested,
        # step through the same diffusion range with a fixed stride.
        if num_iter == self.diffusion.num_timesteps:
            diffusion_list = list(range(self.diffusion.num_timesteps - 1, -1, -1))
        else:
            diffusion_list = torch.linspace(self.diffusion.num_timesteps - 1, 0, steps=num_iter).long().unique_consecutive().tolist()
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)

        last_logits = None
        for diffusion_index in diffusion_list:
            t = torch.full((B,), int(diffusion_index), device=device, dtype=torch.long)
            xt = log_onehot_to_index(log_z)
            logits_blv = self.denoise_logits(xt, cond_emb, t)
            last_logits = logits_blv
            log_x0 = self.diffusion.predict_start_from_logits(logits_blv)
            model_log_prob = self.diffusion.q_posterior(log_x_start=log_x0, log_x_t=log_z, t=t)
            log_z = self.diffusion.log_sample_categorical(model_log_prob)

        codes = log_onehot_to_index(log_z).clamp(min=0, max=self.num_embed - 1)
        # Enforce level-specific validity after stochastic sampling.
        for level in range(min(L, self.level_vocab_sizes.numel())):
            valid = int(self.level_vocab_sizes[level].item())
            if valid < self.num_embed:
                codes[:, level].clamp_(max=valid - 1)
        if return_logits:
            return codes, last_logits
        return codes
