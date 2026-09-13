"""
GPT-style diffusion baseline with an HDGR Transformer backbone.

This is an ablation of ``retriever_gpt.py``:
  - keep GPT_diffusion data flow, noising, loss, CFG, prefix-regeneration
    decoding, and Trie-constrained beam search;
  - replace the GPT2LMHeadModel denoising network with the explicit HDGR
    ``[x_t | x_0]`` Transformer backbone.

The goal is to isolate the effect of the decoder/backbone architecture while
keeping the GPT_diffusion objective/eval protocol as unchanged as possible.
"""
from __future__ import annotations

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.hdgr_comparison.retriever_gpt import (
    T5ForGenerativeRetrieval as GPTDiffusionRetriever,
    cfg_get,
)
from models.hdgr_comparison.retriever_hdgr import (
    HDGRBlockDenoisingGenerator,
    build_hdgr_attention_mask,
    build_hierarchy_block_spans,
)
from models.hdgr_comparison.shared_code_head import migrate_legacy_untied_state_dict
from models.hdgr_comparison.complete_id_selection import topk_complete_id_scores


class RetrieverDiffusionHDGRBackbone(nn.Module):
    """Adapter exposing the GPT_diffusion generator forward signature.

    ``retriever_gpt.T5ForGenerativeRetrieval`` calls its generator as

        id_generator(tokens=..., mask_tokens=..., prefix=..., mask=..., t=...)

    The HDGR backbone instead consumes ``x0_context`` and ``xt`` and builds
    logits from the noisy half of ``[xt | x0_context]``.  This adapter maps the
    GPT-style arguments into the HDGR-style backbone.

    During GPT prefix-regeneration decoding, uncommitted suffix tokens in
    ``tokens`` are placeholders.  We therefore default to token-level HDGR blocks
    (block_size=1), so noisy position j can attend only to its own noisy token
    and previously committed clean positions, matching the AR-prefix spirit of
    the GPT baseline.
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
        block_size: int = 1,
        use_time_embedding: bool = True,
        block_spans: list[tuple[int, int]] | None = None,
        block_names: list[str] | None = None,
        code_tied_output_head: bool = False,
        tied_checkpoint_merge: str = "average",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.mask_token_id = int(num_classes) - 1
        self.d_model = int(d_model)
        self.code_length = int(code_length)
        self.num_prefix = int(num_prefix)
        self.time_step = int(time_step)
        self.block_size = max(1, int(block_size))
        self.block_spans = block_spans
        self.block_names = block_names
        self.code_tied_output_head = bool(code_tied_output_head)
        self.tied_checkpoint_merge = str(tied_checkpoint_merge).lower().strip()

        self.hdgr = HDGRBlockDenoisingGenerator(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            d_model=self.d_model,
            code_length=self.code_length,
            num_prefix=self.num_prefix,
            time_step=self.time_step,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            use_time_embedding=bool(use_time_embedding),
            code_tied_output_head=self.code_tied_output_head,
            tied_checkpoint_merge=self.tied_checkpoint_merge,
        )

        # Legacy aliases used by initialization and diagnostics.
        self.input_embed = self.hdgr.input_embed
        self.lm_head = self.hdgr.lm_head

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
            # Old adapters registered both hdgr.* and top-level alias paths.
            head_candidates = ("hdgr.lm_head.weight", "lm_head.weight")
            token_candidates = (
                "hdgr.token_embed.weight",
                "hdgr.input_embed.weight",
                "input_embed.weight",
            )
            migrated = migrate_legacy_untied_state_dict(
                state_dict,
                prefix=prefix,
                vocab_size=self.vocab_size,
                strategy=self.tied_checkpoint_merge,
                token_key_suffixes=token_candidates,
                head_key_suffixes=head_candidates,
            )
            if migrated:
                print(
                    "[Code-Tied HDGR] Migrated GPT-HDGR adapter checkpoint at "
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
        tokens: torch.Tensor,
        mask_tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        project_logits: bool = True,
        prefix_kv_cache=None,
        prefix_is_normalized: bool = False,
        active_span=None,
        block_spans_override: list[tuple[int, int]] | None = None,
    ):
        del mask, labels  # HDGR uses its own explicit [xt|x0] mask below.
        B, L = tokens.shape
        if L != self.code_length:
            raise ValueError(f"Expected code_length={self.code_length}, got {L}")
        if mask_tokens.shape != tokens.shape:
            raise ValueError("mask_tokens and tokens must have the same shape")

        # In training, tokens are clean x0.  In GPT prefix-regeneration inference,
        # uncommitted suffix positions are placeholders; token-level HDGR masking
        # prevents those future placeholders from leaking into current predictions.
        x0_context = tokens.clamp(min=0, max=self.num_classes - 1)
        xt = mask_tokens.clamp(min=0, max=self.num_classes - 1)

        valid_mask = torch.ones_like(tokens, dtype=torch.long, device=tokens.device)
        effective_spans = (
            block_spans_override
            if block_spans_override is not None
            else self.block_spans
        )
        attn = build_hdgr_attention_mask(
            length=L,
            block_size=self.block_size,
            batch_size=B,
            device=tokens.device,
            valid_mask=valid_mask,
            clean_visible_mask=x0_context.ne(self.mask_token_id),
            block_spans=effective_spans,
        )

        if t is None:
            t_block = torch.ones(B, device=tokens.device, dtype=torch.long)
        else:
            t_block = t.long().view(B)

        return self.hdgr(
            x0_context=x0_context,
            xt=xt,
            prefix=prefix,
            t_block=t_block,
            attention_mask=attn,
            block_size=self.block_size,
            block_spans=effective_spans,
            project_logits=project_logits,
            prefix_kv_cache=prefix_kv_cache,
            prefix_is_normalized=prefix_is_normalized,
            active_span=active_span,
        )

    def prepare_prefix_kv_cache(self, prefix: torch.Tensor):
        return self.hdgr.prepare_prefix_kv_cache(prefix)


class T5ForGenerativeRetrieval(GPTDiffusionRetriever):
    """GPT_diffusion wrapper whose generator backbone is HDGR."""

    def __init__(self, config=None, tokenizer=None, clip_model=None, new_tokenizer=True, init_rq_codebook=True):
        # Build all GPT_diffusion wrapper components/tokenizer/quantizer, but do
        # not initialize codebook embeddings on the temporary GPT generator.
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            clip_model=clip_model,
            new_tokenizer=new_tokenizer,
            init_rq_codebook=False,
        )

        model_cfg = cfg_get(config, "model", {})
        hierarchy_on = bool(cfg_get(model_cfg, "gpt_hdgr_hierarchy_aware_blocks", False))
        block_spans = None
        block_names = None
        if hierarchy_on:
            block_spans, block_names = build_hierarchy_block_spans(
                codebook_level=self.codebook_level,
                model_cfg=model_cfg,
                modality_index=self.modality_index,
                block_size=cfg_get(model_cfg, "gpt_hdgr_block_size", 1),
            )
            print(f"[GPT-HDGR] Hierarchy block spans enabled: {block_spans} names={block_names}")

        self.id_generator = RetrieverDiffusionHDGRBackbone(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            d_model=self.d_model,
            code_length=self.codebook_level,
            num_prefix=self.num_prefix,
            time_step=self.time_step,
            num_layers=cfg_get(model_cfg, "hdgr_num_layers", cfg_get(model_cfg, "nar_num_layers", 6)),
            num_heads=cfg_get(model_cfg, "hdgr_num_heads", cfg_get(model_cfg, "nar_num_heads", 8)),
            dropout=cfg_get(model_cfg, "hdgr_dropout", cfg_get(model_cfg, "nar_dropout", 0.1)),
            block_size=cfg_get(model_cfg, "gpt_hdgr_block_size", 1),
            use_time_embedding=cfg_get(model_cfg, "use_time_embedding", True),
            block_spans=block_spans,
            block_names=block_names,
            code_tied_output_head=bool(cfg_get(model_cfg, "hdgr_code_tied_output_head", False)),
            tied_checkpoint_merge=str(cfg_get(model_cfg, "hdgr_tied_checkpoint_merge", "average")),
        )

        self.gpt_hdgr_hierarchy_aware_blocks = bool(cfg_get(model_cfg, "gpt_hdgr_hierarchy_aware_blocks", False))
        self.use_tcis = bool(
            cfg_get(
                model_cfg,
                "use_tcis",
                cfg_get(model_cfg, "use_block_decoding", cfg_get(model_cfg, "gpt_hdgr_block_decoding", False)),
            )
        )
        self.use_one_pass_tcis = bool(cfg_get(model_cfg, "use_one_pass_tcis", False))
        self.one_pass_tcis_chunk_size = int(cfg_get(model_cfg, "one_pass_tcis_chunk_size", 8192))
        self.block_spans = block_spans or [(i, i + 1) for i in range(self.codebook_level)]
        self.block_names = block_names or [f"level{i}" for i in range(self.codebook_level)]
        self.block_transition_cache = {}
        self.trie_leaf_count_cache = {}
        self.block_trie_max_candidates = int(cfg_get(model_cfg, "block_trie_max_candidates", 0))
        self.tcis_max_leaves = int(
            cfg_get(model_cfg, "tcis_max_leaves", cfg_get(model_cfg, "adaptive_suffix_max_leaves", 0))
        )
        # Query-level, label-free switch planner.  This is intentionally
        # separate from the legacy ``tcis_max_leaves`` path so existing fixed
        # P2L configurations and historical measurements remain unchanged.
        self.tcis_frontier_leaf_budget = int(
            cfg_get(model_cfg, "tcis_frontier_leaf_budget", 0)
        )
        self.tcis_frontier_min_prefix = max(
            1, int(cfg_get(model_cfg, "tcis_frontier_min_prefix", 2))
        )
        self.tcis_frontier_max_prefix = min(
            self.codebook_level - 1,
            max(
                self.tcis_frontier_min_prefix,
                int(cfg_get(model_cfg, "tcis_frontier_max_prefix", 5)),
            ),
        )
        self.deduplicate_block_beams = bool(cfg_get(model_cfg, "deduplicate_block_beams", True))
        # Analytically score only the selected codeword while retaining the
        # exact full-codebook mean/variance used by normalized RRG. This avoids
        # materializing [num_legal_ids, 4096] tensors at every suffix level.
        self.tcis_selected_rrg = bool(cfg_get(model_cfg, "tcis_selected_rrg", False))
        self.tcis_batched_cfg = bool(cfg_get(model_cfg, "tcis_batched_cfg", False))
        self.tcis_compact_head = bool(cfg_get(model_cfg, "tcis_compact_head", False))
        self.tcis_dynamic_intermediate_beams = bool(
            cfg_get(model_cfg, "tcis_dynamic_intermediate_beams", False)
        )
        self.tcis_intermediate_beam_multiplier = max(
            1, int(cfg_get(model_cfg, "tcis_intermediate_beam_multiplier", 1))
        )
        # Optional independent width for the sequential prefix frontier.
        # ``num_beams`` remains the number of complete IDs returned by the
        # final block.  A value <=0 preserves the historical shared-width rule.
        self.tcis_prefix_beams = int(cfg_get(model_cfg, "tcis_prefix_beams", 0))
        self.tcis_prefix_kv_cache = bool(cfg_get(model_cfg, "tcis_prefix_kv_cache", False))
        self.tcis_active_token_pruning = bool(cfg_get(model_cfg, "tcis_active_token_pruning", False))
        # Research control for separating P2L's delayed complete-leaf ranking
        # from latent interaction inside its multi-position suffix block.  The
        # default preserves the canonical P2L execution.  When enabled, the
        # suffix logits are still produced in one forward and scored over the
        # same complete Trie leaves, but every identifier position uses a
        # singleton attention block.
        self.tcis_isolate_suffix_states = bool(
            cfg_get(model_cfg, "tcis_isolate_suffix_states", False)
        )
        self.tcis_force_legal_token_pruning = bool(
            cfg_get(model_cfg, "tcis_force_legal_token_pruning", False)
        )
        self.gpt_hdgr_block_diffusion_steps = int(
            cfg_get(model_cfg, "gpt_hdgr_block_diffusion_steps", self.time_step)
        )
        self.gpt_hdgr_block_score_normalization = str(
            cfg_get(model_cfg, "gpt_hdgr_block_score_normalization", "mean")
        ).lower()
        if self.gpt_hdgr_block_score_normalization not in {"sum", "mean", "last", "prefix"}:
            raise ValueError(
                "model.gpt_hdgr_block_score_normalization must be "
                "'sum', 'mean', 'last', or 'prefix'."
            )

        self.hdgr_code_tied_output_head = bool(cfg_get(model_cfg, "hdgr_code_tied_output_head", False))
        self.hdgr_tied_checkpoint_merge = str(cfg_get(model_cfg, "hdgr_tied_checkpoint_merge", "average"))
        if self.hdgr_code_tied_output_head:
            print(
                "[Code-Tied HDGR] Enabled shared input/output code matrix; "
                f"MASK row remains input-only; legacy_merge={self.hdgr_tied_checkpoint_merge}."
            )

        self.init_rq_codebook_embeddings = bool(cfg_get(model_cfg, "init_rq_codebook_embeddings", True))
        self.codebook_init_scale = float(cfg_get(model_cfg, "codebook_init_scale", 1.0))
        if init_rq_codebook and self.init_rq_codebook_embeddings:
            self._initialize_hdgr_codebook_embeddings(scale=self.codebook_init_scale)
        else:
            print("[GPT-HDGR] RQ codebook embedding initialization disabled.")

        print(
            f"[GPT-HDGR] Initialized GPT_diffusion objective with HDGR backbone | "
            f"code_length={self.codebook_level} vocab={self.vocab_size} "
            f"layers={cfg_get(model_cfg, 'hdgr_num_layers', cfg_get(model_cfg, 'nar_num_layers', 6))} "
            f"heads={cfg_get(model_cfg, 'hdgr_num_heads', cfg_get(model_cfg, 'nar_num_heads', 8))} "
            f"prefix={self.num_prefix} block_size={cfg_get(model_cfg, 'gpt_hdgr_block_size', 1)} "
            f"hierarchy_on={bool(cfg_get(model_cfg, 'gpt_hdgr_hierarchy_aware_blocks', False))} "
            f"tcis={self.use_tcis} "
            f"one_pass_tcis={self.use_one_pass_tcis} "
            f"code_tied={self.hdgr_code_tied_output_head}"
        )

    @staticmethod
    def _expand_prefix_kv_cache(prefix_kv_cache, batch_size: int, beam_size: int):
        if prefix_kv_cache is None or beam_size == 1:
            return prefix_kv_cache
        expanded = []
        for key, value in prefix_kv_cache:
            _, heads, prefix_len, head_dim = key.shape
            key = key.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(
                batch_size * beam_size, heads, prefix_len, head_dim
            )
            value = value.unsqueeze(1).expand(-1, beam_size, -1, -1, -1).reshape(
                batch_size * beam_size, heads, prefix_len, head_dim
            )
            expanded.append((key, value))
        return expanded

    def _clean_context_from_prefix(self, seq: torch.Tensor, prefix_end: int) -> torch.Tensor:
        """Return x0 context with only committed previous blocks visible."""
        clean = torch.full_like(seq, self.mask_token_id)
        if prefix_end > 0:
            clean[:, :prefix_end] = seq[:, :prefix_end]
        return clean

    # -----------------------------
    # Trie-Constrained Complete-ID Selection (TCIS)
    # -----------------------------

    def _get_tree_node_for_prefix(self, prefix: tuple[int, ...]):
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

    def _enumerate_next_blocks_from_node(self, node, block_len: int, max_candidates: int = 0):
        if node is None:
            return []
        out = []
        limit = int(max_candidates or 0)

        def dfs(cur_node, depth, path):
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
        prefix_key = tuple(int(x) for x in prefix)
        # Adaptive decoding may request different continuation lengths for the
        # same prefix across queries, so block length is part of the cache key.
        cache_key = (prefix_key, int(block_len))
        cached = self.block_transition_cache.get(cache_key)
        if cached is None:
            node = self._get_tree_node_for_prefix(prefix_key)
            blocks = self._enumerate_next_blocks_from_node(
                node,
                block_len=block_len,
                max_candidates=self.block_trie_max_candidates,
            )
            if len(blocks) == 0:
                cached = torch.empty((0, int(block_len)), dtype=torch.long)
            else:
                cached = torch.tensor(blocks, dtype=torch.long)
            self.block_transition_cache[cache_key] = cached
        return cached.to(device=device, non_blocking=True)

    def _count_descendant_leaves(self, prefix: tuple[int, ...], stop_after: int = 0) -> int:
        """Count real candidate leaves below a prefix, optionally with early exit."""
        key = tuple(int(x) for x in prefix)
        cached = self.trie_leaf_count_cache.get(key)
        if cached is not None:
            return int(cached)
        node = self._get_tree_node_for_prefix(key)
        if node is None:
            return 0
        limit = max(0, int(stop_after))
        count = 0
        stack = [node]
        while stack:
            cur = stack.pop()
            if not cur:
                count += 1
                if limit and count >= limit:
                    return count
            else:
                stack.extend(cur.values())
        self.trie_leaf_count_cache[key] = count
        return count

    def _count_exposed_leaves(
        self, prefixes: set[tuple[int, ...]], stop_after: int = 0
    ) -> int:
        """Count unique complete leaves exposed by a prefix frontier.

        Prefixes at the same depth correspond to disjoint Trie subtrees.  The
        optional limit permits a cheap budget test without materializing the
        complete descendants of an over-budget frontier.
        """
        limit = max(0, int(stop_after))
        total = 0
        for prefix in prefixes:
            remaining = 0 if limit == 0 else max(1, limit - total)
            total += self._count_descendant_leaves(prefix, stop_after=remaining)
            if limit and total >= limit:
                return total
        return total

    @staticmethod
    def _planner_block_spans(length: int, start: int, finish: bool):
        """Return attention blocks for one planner decision.

        Committed positions remain singleton blocks.  A continuation step also
        keeps all future positions isolated; a finish step groups the complete
        unresolved suffix into one mutually isolated prediction block.
        """
        length, start = int(length), int(start)
        if not (0 <= start < length):
            raise ValueError(f"Invalid planner start={start} for length={length}")
        spans = [(i, i + 1) for i in range(start)]
        if finish:
            spans.append((start, length))
        else:
            spans.extend((i, i + 1) for i in range(start, length))
        return spans

    def _apply_top_p_block(self, logp: torch.Tensor) -> torch.Tensor:
        if self.top_p >= 1.0:
            return logp
        sorted_logits, sorted_indices = torch.sort(logp, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        remove_mask = cum_probs > self.top_p
        remove_mask[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, torch.finfo(logp.dtype).min)
        return torch.full_like(logp, torch.finfo(logp.dtype).min).scatter(1, sorted_indices, sorted_logits)

    def _project_compact_level_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project every position only onto its valid RQ-level vocabulary."""
        max_size = int(self.level_vocab_sizes.max().item())
        rows = []
        weight = self.id_generator.lm_head.weight
        for pos in range(self.codebook_level):
            level_size = int(self.level_vocab_sizes[pos].item())
            token_ids = self.level_token_ids[pos, :level_size].to(hidden.device)
            level_weight = weight.index_select(0, token_ids)
            logits = F.linear(hidden[:, pos, :], level_weight).float()
            if level_size < max_size:
                logits = F.pad(logits, (0, max_size - level_size), value=torch.finfo(logits.dtype).min)
            rows.append(logits)
        return torch.stack(rows, dim=1)

    def _block_cfg_logits_compact(
        self,
        xt: torch.Tensor,
        clean_context: torch.Tensor,
        cond_prefix: torch.Tensor,
        uncond_prefix: torch.Tensor,
        t: torch.Tensor,
        cond_prefix_kv=None,
        uncond_prefix_kv=None,
        prefixes_normalized: bool = False,
        active_span=None,
        block_spans_override=None,
    ) -> torch.Tensor:
        common = dict(tokens=clean_context, mask_tokens=xt, mask=None, t=t, labels=None, project_logits=False)
        if self.tcis_batched_cfg:
            batch = clean_context.size(0)
            out = self.id_generator(
                tokens=torch.cat((clean_context, clean_context), dim=0),
                mask_tokens=torch.cat((xt, xt), dim=0),
                prefix=torch.cat((cond_prefix, uncond_prefix), dim=0),
                mask=None,
                t=torch.cat((t, t), dim=0),
                labels=None,
                project_logits=False,
                active_span=active_span,
                block_spans_override=block_spans_override,
            )
            hidden_c, hidden_u = out.hidden_states.split(batch, dim=0)
        else:
            hidden_c = self.id_generator(
                prefix=cond_prefix, prefix_kv_cache=cond_prefix_kv,
                prefix_is_normalized=prefixes_normalized, **common
                , active_span=active_span,
                block_spans_override=block_spans_override
            ).hidden_states
            hidden_u = self.id_generator(
                prefix=uncond_prefix, prefix_kv_cache=uncond_prefix_kv,
                prefix_is_normalized=prefixes_normalized, **common
                , active_span=active_span,
                block_spans_override=block_spans_override
            ).hidden_states
        logits_c = self._project_compact_level_logits(hidden_c)
        logits_u = self._project_compact_level_logits(hidden_u)
        # Optional read-only research capture of the *actual* conditional
        # generator state.  This is default-off and does not participate in
        # canonical scoring.
        if getattr(self, "_capture_semantic_hidden_states", False):
            self._latest_semantic_cond_hidden = hidden_c.detach()
        guided = logits_u + float(self.guidance_scale) * (logits_c - logits_u)
        if self.temperature != 1.0:
            guided = guided / max(float(self.temperature), 1e-6)
        return guided

    def _block_cfg_logits(
        self,
        xt: torch.Tensor,
        clean_context: torch.Tensor,
        cond_prefix: torch.Tensor,
        uncond_prefix: torch.Tensor,
        t: torch.Tensor,
        cond_prefix_kv=None,
        uncond_prefix_kv=None,
        prefixes_normalized: bool = False,
        active_span=None,
        block_spans_override=None,
    ) -> torch.Tensor:
        if self.tcis_compact_head:
            return self._block_cfg_logits_compact(
                xt, clean_context, cond_prefix, uncond_prefix, t,
                cond_prefix_kv, uncond_prefix_kv, prefixes_normalized,
                active_span, block_spans_override,
            )
        if self.tcis_batched_cfg:
            # Both CFG branches share token/time inputs. Evaluating them as one
            # larger batch preserves the formula while removing one dispatch.
            batch = clean_context.size(0)
            out = self.id_generator(
                tokens=torch.cat((clean_context, clean_context), dim=0),
                mask_tokens=torch.cat((xt, xt), dim=0),
                prefix=torch.cat((cond_prefix, uncond_prefix), dim=0),
                mask=None,
                t=torch.cat((t, t), dim=0),
                labels=None,
                active_span=active_span,
                block_spans_override=block_spans_override,
            )
            logits_c, logits_u = out.logits.float().split(batch, dim=0)
        else:
            out_c = self.id_generator(
                tokens=clean_context,
                mask_tokens=xt,
                prefix=cond_prefix,
                mask=None,
                t=t,
                labels=None,
                prefix_kv_cache=cond_prefix_kv,
                prefix_is_normalized=prefixes_normalized,
                active_span=active_span,
                block_spans_override=block_spans_override,
            )
            out_u = self.id_generator(
                tokens=clean_context,
                mask_tokens=xt,
                prefix=uncond_prefix,
                mask=None,
                t=t,
                labels=None,
                prefix_kv_cache=uncond_prefix_kv,
                prefix_is_normalized=prefixes_normalized,
                active_span=active_span,
                block_spans_override=block_spans_override,
            )
            logits_c = out_c.logits.float()
            logits_u = out_u.logits.float()
        logits_c = self._mask_invalid_logits(logits_c)
        logits_u = self._mask_invalid_logits(logits_u)
        guided = logits_u + float(self.guidance_scale) * (logits_c - logits_u)
        if self.temperature != 1.0:
            guided = guided / max(float(self.temperature), 1e-6)
        return guided

    def _refine_current_block(self, xt: torch.Tensor, logits: torch.Tensor, start: int, end: int, keep_fraction: float):
        tmp = xt.clone()
        logp = F.log_softmax(logits, dim=-1)
        confidences = torch.empty(xt.size(0), end - start, device=xt.device, dtype=logits.dtype)
        for local_idx, pos in enumerate(range(start, end)):
            step_logp = logp[:, pos, :]
            compact_logits = (
                self.tcis_compact_head
                and step_logp.size(1) != self.level_vocab_mask.size(1)
            )
            if compact_logits:
                level_size = int(self.level_vocab_sizes[pos].item())
                allowed = (
                    torch.arange(step_logp.size(1), device=step_logp.device)
                    < level_size
                ).unsqueeze(0).expand_as(step_logp)
            else:
                allowed = self.level_vocab_mask[pos].to(step_logp.device).unsqueeze(0).expand_as(step_logp)
            step_logp = torch.where(allowed, step_logp, torch.full_like(step_logp, torch.finfo(step_logp.dtype).min))
            step_logp = self._apply_top_p_block(step_logp)
            score, token = torch.max(step_logp, dim=-1)
            if compact_logits:
                token = self.level_token_ids[pos].to(step_logp.device).index_select(0, token.long())
            tmp[:, pos] = token
            confidences[:, local_idx] = score

        block_len = end - start
        keep_count = int(math.ceil(block_len * max(0.0, min(1.0, float(keep_fraction)))))
        keep_count = max(1, min(block_len, keep_count))
        if keep_count < block_len:
            rank = torch.argsort(confidences, dim=1, descending=True)
            keep = torch.zeros_like(confidences, dtype=torch.bool)
            keep.scatter_(1, rank[:, :keep_count], True)
            for local_idx, pos in enumerate(range(start, end)):
                tmp[~keep[:, local_idx], pos] = self.mask_token_id
        return tmp

    def _expand_block_token_beam_fallback(self, seqs, scores, final_logp, start: int, end: int, keep_k: int):
        B, K0, L = seqs.shape
        V = final_logp.size(-1)
        final_logp = final_logp.view(B, K0, L, V)
        cur_seq = seqs.clone()
        cur_scores = scores.clone()
        origin = torch.arange(K0, device=seqs.device).unsqueeze(0).expand(B, -1).clone()

        for pos in range(start, end):
            cur_k = cur_seq.size(1)
            gather_idx = origin[:, :, None, None].expand(B, cur_k, 1, V)
            step_logp = final_logp[:, :, pos:pos + 1, :].gather(1, gather_idx).squeeze(2)
            allowed = self.level_vocab_mask[pos].to(seqs.device).view(1, 1, V).expand(B, cur_k, V)
            step_logp = torch.where(allowed, step_logp, torch.full_like(step_logp, torch.finfo(step_logp.dtype).min))
            step_logp = self._apply_top_p_block(step_logp.reshape(B * cur_k, V)).view(B, cur_k, V)
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

    def _expand_block_legal_token_beam(
        self, seqs, scores, final_logp, start: int, end: int, keep_k: int
    ):
        """Prune after every suffix position using one shared masked forward.

        This control keeps the neural evidence identical to P2L while changing
        only the pruning schedule. Every expansion follows an actual Trie edge.
        """
        B, K0, L = seqs.shape
        V = final_logp.size(-1)
        final_logp = final_logp.view(B, K0, L, V)
        code_map = self.token_id_to_code.to(seqs.device) if self.tcis_compact_head else None
        out_sequences, out_scores = [], []
        for b in range(B):
            cur_seq = seqs[b].clone()
            cur_scores = scores[b].clone()
            origin = torch.arange(K0, device=seqs.device)
            for pos in range(start, end):
                parents, tokens, token_scores = [], [], []
                for parent in range(cur_seq.size(0)):
                    prefix = tuple(int(x) for x in cur_seq[parent, :pos].detach().cpu().tolist())
                    node = self._get_tree_node_for_prefix(prefix)
                    if not node:
                        continue
                    child_tokens = torch.tensor(
                        list(node.keys()), dtype=torch.long, device=seqs.device
                    )
                    score_tokens = child_tokens
                    if code_map is not None:
                        score_tokens = code_map.index_select(0, child_tokens)
                    score_tokens = score_tokens.clamp(min=0, max=V - 1)
                    local = final_logp[b, origin[parent], pos].index_select(0, score_tokens)
                    parents.append(
                        torch.full_like(child_tokens, parent, dtype=torch.long)
                    )
                    tokens.append(child_tokens)
                    token_scores.append(cur_scores[parent] + local)
                if not parents:
                    raise RuntimeError(
                        f"legal token-pruning control reached an empty Trie frontier at position {pos}"
                    )
                parent_idx = torch.cat(parents)
                token_idx = torch.cat(tokens)
                candidate_scores = torch.cat(token_scores)
                next_k = min(max(1, int(keep_k)), candidate_scores.numel())
                top_scores, top_idx = torch.topk(candidate_scores, k=next_k)
                selected_parent = parent_idx.index_select(0, top_idx)
                next_seq = cur_seq.index_select(0, selected_parent).clone()
                next_seq[:, pos] = token_idx.index_select(0, top_idx)
                origin = origin.index_select(0, selected_parent)
                cur_seq, cur_scores = next_seq, top_scores
            if cur_seq.size(0) < keep_k:
                pad = keep_k - cur_seq.size(0)
                cur_seq = torch.cat((cur_seq, cur_seq[-1:].expand(pad, -1)), dim=0)
                cur_scores = torch.cat((cur_scores, cur_scores[-1:].expand(pad)), dim=0)
            out_sequences.append(cur_seq[:keep_k])
            out_scores.append(cur_scores[:keep_k])
        return torch.stack(out_sequences), torch.stack(out_scores)

    def _expand_block_with_constraints(
        self, seqs, scores, final_logp, start: int, end: int, keep_k: int,
        rrg_query_embedding: torch.Tensor | None = None,
        pad_to_keep_k: bool = True,
    ):
        if self.tree_index is None:
            return self._expand_block_token_beam_fallback(seqs, scores, final_logp, start, end, keep_k)
        if self.tcis_force_legal_token_pruning and end - start > 1:
            return self._expand_block_legal_token_beam(
                seqs, scores, final_logp, start, end, keep_k
            )

        B, K0, L = seqs.shape
        V = final_logp.size(-1)
        block_len = int(end - start)
        final_logp = final_logp.view(B, K0, L, V)
        device = seqs.device
        dtype = scores.dtype
        keep_k = max(1, int(keep_k))

        # Transfer the tiny prefix/score metadata once per block.  The former
        # implementation called ``.cpu().tolist()`` inside the query/beam
        # loops, forcing hundreds of CUDA synchronizations per batch.
        if start > 0:
            prefix_rows = seqs[:, :, :start].detach().cpu().tolist()
        else:
            prefix_rows = [[[] for _ in range(K0)] for _ in range(B)]
        score_rows = scores.detach().float().cpu().tolist()
        out_seqs = []
        out_scores = []
        profile_candidates = bool(int(os.environ.get("STRUCTNAR_PROFILE_TCIS", "0"))) and not getattr(
            self, "_tcis_profile_emitted", False
        )
        profile_candidate_counts = []
        profile_parent_counts = []
        profile_query_budget = bool(int(os.environ.get("STRUCTNAR_PROFILE_QUERY_BUDGET", "0")))
        profile_components = bool(int(os.environ.get("STRUCTNAR_PROFILE_COMPONENTS", "0")))
        for b in range(B):
            parent_indices = list(range(K0))
            if self.deduplicate_block_beams and K0 > 1:
                best_parent = {}
                parent_prefix = prefix_rows[b]
                parent_scores = score_rows[b]
                for k in range(K0):
                    key = tuple(int(x) for x in parent_prefix[k])
                    sc = float(parent_scores[k])
                    old = best_parent.get(key)
                    if old is None or sc > old[0]:
                        best_parent[key] = (sc, k)
                parent_indices = [idx for _, idx in best_parent.values()]
            if profile_candidates:
                profile_parent_counts.append(len(parent_indices))

            # Build the ragged legal-continuation table on CPU, then transfer
            # and score it once. This avoids one GPU copy and several indexing
            # kernels for every parent beam.
            cand_chunks_cpu = []
            cand_parent_cpu = []
            enumeration_started_at = time.perf_counter()
            for k in parent_indices:
                prefix = tuple(int(x) for x in prefix_rows[b][k])
                block_cands = self._get_next_block_candidates(
                    prefix, block_len=block_len, device=torch.device("cpu")
                )
                if block_cands.numel() == 0:
                    continue
                cand_chunks_cpu.append(block_cands)
                cand_parent_cpu.append(torch.full((block_cands.size(0),), k, dtype=torch.long))
            enumeration_seconds = time.perf_counter() - enumeration_started_at

            if len(cand_chunks_cpu) == 0:
                if self.tcis_compact_head:
                    raise RuntimeError("Compact TCIS head reached an empty Trie continuation")
                fallback_seq, fallback_score = self._expand_block_token_beam_fallback(
                    seqs[b:b + 1],
                    scores[b:b + 1],
                    final_logp[b:b + 1].reshape(K0, L, V),
                    start,
                    end,
                    keep_k,
                )
                out_seqs.append(fallback_seq.squeeze(0))
                out_scores.append(fallback_score.squeeze(0))
                continue

            block_cands = torch.cat(cand_chunks_cpu, dim=0).to(device=device, non_blocking=True)
            cand_parent = torch.cat(cand_parent_cpu, dim=0).to(device=device, non_blocking=True)
            if profile_query_budget and block_len > 1:
                if not hasattr(self, "_tcis_query_budget_rows"):
                    self._tcis_query_budget_rows = []
                self._tcis_query_budget_rows.append(
                    (len(parent_indices), int(block_cands.size(0)), enumeration_seconds)
                )
            if profile_candidates:
                profile_candidate_counts.append(int(block_cands.size(0)))
            component_events = None
            if profile_components and block_len > 1:
                gather_start = torch.cuda.Event(enable_timing=True)
                gather_end = torch.cuda.Event(enable_timing=True)
                rqc_end = torch.cuda.Event(enable_timing=True)
                topk_end = torch.cuda.Event(enable_timing=True)
                gather_start.record()
            token_scores = []
            code_map = self.token_id_to_code.to(device) if self.tcis_compact_head else None
            for local_idx, pos in enumerate(range(start, end)):
                tok = block_cands[:, local_idx]
                if code_map is not None:
                    tok = code_map.index_select(0, tok.long())
                tok = tok.clamp(min=0, max=V - 1)
                token_scores.append(final_logp[b, cand_parent, pos, tok])
            block_score = torch.stack(token_scores, dim=0)
            if self.gpt_hdgr_block_score_normalization == "prefix":
                block_score = torch.zeros_like(block_score[0])
            elif self.gpt_hdgr_block_score_normalization == "mean":
                block_score = block_score.mean(dim=0)
            elif self.gpt_hdgr_block_score_normalization == "last":
                block_score = block_score[-1]
            else:
                block_score = block_score.sum(dim=0)

            all_seqs = seqs[b].index_select(0, cand_parent).clone()
            all_seqs[:, start:end] = block_cands
            all_scores = scores[b].index_select(0, cand_parent) + block_score
            if profile_components and block_len > 1:
                gather_end.record()
            # Batch RRG over every legal continuation of this query. Doing
            # this inside the parent loop repeats a 4096-way codebook score for
            # every beam and is prohibitively slow.
            if rrg_query_embedding is not None and self.use_rrg:
                query_rows = rrg_query_embedding[b:b + 1].expand(all_seqs.size(0), -1)
                running_rrg_prefix = None
                if not self.tcis_selected_rrg:
                    running_rrg_prefix = self._rrg_prefix_reconstruction_from_token_ids(
                        all_seqs, start, dtype=torch.float32
                    )
                for pos in range(start, end):
                    if self.tcis_selected_rrg:
                        selected_rrg = self._rrg_selected_scores_for_rows(
                            query_rows, all_seqs, pos
                        )
                    else:
                        # Reference implementation retained for equivalence
                        # tests and reproduction of earlier saved beams.
                        rrg_scores = self._rrg_scores_from_prefix_reconstruction(
                            query_rows, running_rrg_prefix, pos
                        )
                        if rrg_scores is None:
                            continue
                        token_ids = all_seqs[:, pos].long()
                        code_ids = self.token_id_to_code.to(device).index_select(0, token_ids)
                        code_ids = code_ids.clamp(min=0, max=rrg_scores.size(1) - 1)
                        selected_rrg = rrg_scores.gather(1, code_ids[:, None]).squeeze(1)
                    if selected_rrg is not None:
                        all_scores = all_scores + (
                            float(self.rrg_weight)
                            * self._rqc_level_weight(pos)
                            * selected_rrg
                        )
                    if running_rrg_prefix is not None and pos + 1 < end:
                        token_ids = all_seqs[:, pos].long()
                        code_ids = self.token_id_to_code.to(device).index_select(0, token_ids)
                        codebook = self._rrg_level_codebook(pos, device=device, dtype=torch.float32)
                        valid = (code_ids >= 0) & (code_ids < codebook.size(0))
                        if bool(valid.any().item()):
                            running_rrg_prefix[valid] += codebook.index_select(0, code_ids[valid])
            if profile_components and block_len > 1:
                rqc_end.record()
            # Optional read-only research trace. It is absent in the frozen
            # canonical path and is called before any world-model score hook.
            # The callback must not return or modify scores.
            trace_hook = getattr(self, "_semantic_block_trace_hook", None)
            if trace_hook is not None:
                trace_hook(
                    batch_index=b,
                    start_index=start,
                    end_index=end,
                    candidate_sequences=all_seqs.detach(),
                    canonical_scores=all_scores.detach(),
                    keep_k=keep_k,
                )
            # Optional DAD-WM hook. The canonical path has no such attribute,
            # and therefore remains bitwise unchanged. The hook scores only
            # already legal Trie continuations before the existing top-k.
            expansion_hook = getattr(
                self, "_semantic_block_expansion_hook", None
            )
            if expansion_hook is not None:
                all_scores = expansion_hook(
                    batch_index=b,
                    start_index=start,
                    end_index=end,
                    candidate_sequences=all_seqs,
                    raw_scores=all_scores,
                )
                if all_scores.ndim != 1 or all_scores.size(0) != all_seqs.size(0):
                    raise ValueError(
                        "Semantic-block expansion hook changed score shape: "
                        f"{tuple(all_scores.shape)} vs {(all_seqs.size(0),)}"
                    )
            internal_q_hook = getattr(self, "_semantic_internal_q_hook", None)
            if internal_q_hook is not None:
                latest_hidden = getattr(self, "_latest_semantic_cond_hidden", None)
                if latest_hidden is None:
                    raise RuntimeError(
                        "Internal Q requested without captured HDGR states"
                    )
                parent_hidden = latest_hidden.view(B, K0, L, -1)[
                    b
                ].index_select(0, cand_parent)
                all_scores = internal_q_hook(
                    batch_index=b,
                    start_index=start,
                    end_index=end,
                    candidate_sequences=all_seqs,
                    raw_scores=all_scores,
                    parent_hidden_states=parent_hidden[:, start:end].mean(dim=1),
                )
                if all_scores.ndim != 1 or all_scores.size(0) != all_seqs.size(0):
                    raise ValueError("Internal Q hook changed score shape")
            # Optional generator-internal decision trace. Unlike the compact
            # score trace above, this exposes the HDGR parent state actually
            # used to produce each legal continuation.
            internal_hook = getattr(
                self, "_semantic_internal_decision_trace_hook", None
            )
            if internal_hook is not None:
                latest_hidden = getattr(self, "_latest_semantic_cond_hidden", None)
                if latest_hidden is None:
                    raise RuntimeError(
                        "Internal decision trace requested without captured HDGR states"
                    )
                parent_hidden = latest_hidden.view(B, K0, L, -1)[
                    b
                ].index_select(0, cand_parent)
                internal_hook(
                    batch_index=b,
                    start_index=start,
                    end_index=end,
                    candidate_sequences=all_seqs.detach(),
                    canonical_scores=all_scores.detach(),
                    parent_hidden_states=parent_hidden[:, start:end].mean(dim=1),
                )
            # Parent beams were deduplicated by their committed prefix above,
            # and a Trie contains each child path once.  Descendants of two
            # distinct prefixes are disjoint, so ``all_seqs`` is already
            # unique here.  Re-running sequence deduplication forced every
            # candidate and score through GPU->CPU conversion once per query.
            top_n = min(keep_k, all_scores.numel())
            top_scores, top_idx = torch.topk(all_scores, k=top_n, dim=0)
            top_seqs = all_seqs.index_select(0, top_idx)
            # Optional, read-only trace for joint-suffix research.  It runs
            # after the canonical top-k decision and cannot alter scores or
            # selected identifiers.  Unlike the older decision hook, this
            # preserves the per-position states of a multi-token block.
            joint_suffix_hook = getattr(
                self, "_semantic_joint_suffix_selected_trace_hook", None
            )
            if joint_suffix_hook is not None:
                latest_hidden = getattr(self, "_latest_semantic_cond_hidden", None)
                if latest_hidden is None:
                    raise RuntimeError(
                        "Joint-suffix trace requested without captured HDGR states"
                    )
                parent_hidden = latest_hidden.view(B, K0, L, -1)[
                    b
                ].index_select(0, cand_parent)
                joint_suffix_hook(
                    batch_index=b,
                    start_index=start,
                    end_index=end,
                    candidate_sequences=top_seqs.detach(),
                    canonical_scores=top_scores.detach(),
                    hidden_states_by_position=parent_hidden.index_select(
                        0, top_idx
                    )[:, start:end].detach(),
                )
            if profile_components and block_len > 1:
                topk_end.record()
                if not hasattr(self, "_tcis_component_event_rows"):
                    self._tcis_component_event_rows = []
                self._tcis_component_event_rows.append(
                    (gather_start, gather_end, rqc_end, topk_end)
                )

            if pad_to_keep_k and top_n < keep_k:
                pad_n = keep_k - top_n
                top_seqs = torch.cat([top_seqs, top_seqs[:1].expand(pad_n, -1)], dim=0)
                top_scores = torch.cat([top_scores, top_scores[:1].expand(pad_n)], dim=0)

            out_seqs.append(top_seqs)
            out_scores.append(top_scores.to(dtype=dtype))
        if profile_candidates and profile_candidate_counts:
            print(
                f"[TCIS candidates] span={start}:{end} queries={B} "
                f"parents_total={sum(profile_parent_counts)} "
                f"candidates_total={sum(profile_candidate_counts)} "
                f"candidates_mean={sum(profile_candidate_counts)/len(profile_candidate_counts):.2f} "
                f"candidates_max={max(profile_candidate_counts)}"
            )
        # A batched tensor still needs a common beam dimension. Dynamic mode
        # pads only to the largest number of live beams in this batch, instead
        # of eagerly expanding every query to the final K after a narrow root.
        target_k = keep_k if pad_to_keep_k else max(row.size(0) for row in out_seqs)
        for b in range(len(out_seqs)):
            if out_seqs[b].size(0) < target_k:
                pad_n = target_k - out_seqs[b].size(0)
                out_seqs[b] = torch.cat(
                    [out_seqs[b], out_seqs[b][:1].expand(pad_n, -1)], dim=0
                )
                out_scores[b] = torch.cat(
                    [out_scores[b], out_scores[b][:1].expand(pad_n)], dim=0
                )
        return torch.stack(out_seqs, dim=0), torch.stack(out_scores, dim=0)

    @torch.no_grad()
    def one_pass_tcis_search(
        self,
        inputs_embeds,
        num_beams=10,
        cand_codes=None,
        return_scores: bool = False,
        rrg_query_embedding: torch.Tensor | None = None,
    ):
        """Select legal complete IDs from one full-mask generator prediction."""
        if rrg_query_embedding is not None and self.use_rrg:
            raise NotImplementedError(
                "One-Pass TCIS currently implements the generator-only baseline; "
                "normalized RRG fusion will be added after this path is validated."
            )
        if cand_codes is not None:
            tree, cand_token_ids, cand_codes_cache = self._build_candidate_tree(cand_codes)
            self.tree_index = tree
            self.cand_token_ids = cand_token_ids
            self.cand_codes_cache = cand_codes_cache
        if self.cand_token_ids is None:
            raise RuntimeError("One-Pass TCIS requires candidate semantic IDs.")

        device = inputs_embeds.device
        batch = inputs_embeds.size(0)
        length = self.codebook_level
        masked = torch.full(
            (batch, length), self.mask_token_id, device=device, dtype=torch.long
        )
        uncond = self.null_condition.expand(batch, -1, -1).to(
            device=device, dtype=inputs_embeds.dtype
        )
        timestep = torch.full((batch,), self.time_step, device=device, dtype=torch.long)
        logits = self._block_cfg_logits(masked, masked, inputs_embeds, uncond, timestep)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # A Trie represents unique complete IDs; match that behavior when raw
        # candidate arrays contain duplicate semantic codes.
        unique_candidates = torch.unique(self.cand_token_ids, dim=0)
        top_indices, top_scores = topk_complete_id_scores(
            log_probs,
            unique_candidates,
            num_beams,
            chunk_size=self.one_pass_tcis_chunk_size,
        )
        selected = unique_candidates.to(device=device).unsqueeze(0).expand(batch, -1, -1)
        selected = selected.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, length)
        )
        if self.beam_score_normalization == "mean":
            top_scores = top_scores / float(max(length, 1))
        if return_scores:
            return selected, top_scores
        return selected

    @torch.no_grad()
    def tcis_search(
        self,
        inputs_embeds,
        attention_mask=None,
        num_beams=10,
        cand_codes=None,
        return_scores: bool = False,
        rrg_query_embedding: torch.Tensor | None = None,
    ):
        """Score legal Trie continuations after each committed prefix.

        This is an inference-only candidate-scoring routine, not blockwise
        training or free non-autoregressive generation.  Multi-token
        continuations are enumerated from the candidate Trie and scored from
        one model pass before beam selection.
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
            self.block_transition_cache = {}
            self.trie_leaf_count_cache = {}

        if self.tcis_frontier_leaf_budget > 0:
            return self._tcis_frontier_planned_search(
                inputs_embeds=inputs_embeds,
                num_beams=num_beams,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )

        if self.tcis_max_leaves > 0:
            return self._tcis_adaptive_search(
                inputs_embeds=inputs_embeds,
                num_beams=num_beams,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )

        cur_seq = torch.full((B0, 1, L), self.mask_token_id, device=device, dtype=torch.long)
        beam_scores = torch.zeros(B0, 1, device=device, dtype=score_dtype)
        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(B0, -1, -1).to(device=device, dtype=model_dtype)
        base_cond_kv = base_uncond_kv = None
        prefixes_normalized = False
        if self.tcis_prefix_kv_cache:
            if self.tcis_batched_cfg:
                raise ValueError("tcis_prefix_kv_cache and tcis_batched_cfg are mutually exclusive")
            base_cond, base_cond_kv = self.id_generator.prepare_prefix_kv_cache(base_cond)
            base_uncond, base_uncond_kv = self.id_generator.prepare_prefix_kv_cache(base_uncond)
            prefixes_normalized = True
        spans = self.block_spans or [(i, i + 1) for i in range(L)]
        singleton_spans = [(i, i + 1) for i in range(L)]
        profile_all_calls = bool(int(os.environ.get("STRUCTNAR_PROFILE_ALL_TCIS", "0")))
        profile_this_call = profile_all_calls or (
            bool(int(os.environ.get("STRUCTNAR_PROFILE_TCIS", "0")))
            and not getattr(self, "_tcis_profile_emitted", False)
        )
        profile_rows = []

        for block_index, (start, end) in enumerate(spans):
            start, end = int(start), int(end)
            cur_k = cur_seq.size(1)
            flat_seq = cur_seq.reshape(B0 * cur_k, L)

            xt = torch.full_like(flat_seq, self.mask_token_id)
            if start > 0:
                xt[:, :start] = flat_seq[:, :start]
            clean_context = self._clean_context_from_prefix(flat_seq, start)

            cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                B0 * cur_k, self.num_prefix, self.d_model
            )
            # Optional world-state conditioner. Unlike a score hook, this
            # changes the neural condition used to predict the next legal
            # block after committed RQ decisions update the latent state.
            # The attribute is absent in the canonical model, preserving the
            # frozen path exactly.
            dynamic_condition_hook = getattr(
                self, "_semantic_dynamic_condition_hook", None
            )
            if dynamic_condition_hook is not None:
                cond = dynamic_condition_hook(
                    block_index=block_index,
                    start_index=start,
                    end_index=end,
                    parent_sequences=flat_seq,
                    canonical_condition=cond,
                    batch_size=B0,
                    beam_size=cur_k,
                )
                if cond.shape != (
                    B0 * cur_k, self.num_prefix, self.d_model
                ):
                    raise ValueError(
                        "Dynamic world conditioner changed prefix shape: "
                        f"{tuple(cond.shape)}"
                    )
            uncond = base_uncond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                B0 * cur_k, self.num_prefix, self.d_model
            )
            cond_kv = self._expand_prefix_kv_cache(base_cond_kv, B0, cur_k)
            uncond_kv = self._expand_prefix_kv_cache(base_uncond_kv, B0, cur_k)

            final_logits = None
            if profile_this_call:
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                expand_end = torch.cuda.Event(enable_timing=True)
                model_start.record()
            steps = max(1, int(self.gpt_hdgr_block_diffusion_steps))
            for step_idx in range(steps, 0, -1):
                t_val = max(1, int(math.ceil(step_idx / steps * self.time_step)))
                t = torch.full((B0 * cur_k,), t_val, device=device, dtype=torch.long)
                logits = self._block_cfg_logits(
                    xt=xt,
                    clean_context=clean_context,
                    cond_prefix=cond,
                    uncond_prefix=uncond,
                    t=t,
                    cond_prefix_kv=cond_kv,
                    uncond_prefix_kv=uncond_kv,
                    prefixes_normalized=prefixes_normalized,
                    active_span=((start, end) if self.tcis_active_token_pruning else None),
                    block_spans_override=(
                        singleton_spans
                        if self.tcis_isolate_suffix_states and end - start > 1
                        else None
                    ),
                )
                final_logits = logits
                if step_idx > 1:
                    keep_fraction = 1.0 - float(step_idx - 1) / float(steps)
                    xt = self._refine_current_block(xt, logits, start, end, keep_fraction=keep_fraction)
                    if start > 0:
                        xt[:, :start] = flat_seq[:, :start]
                    xt[:, end:] = self.mask_token_id

            if profile_this_call:
                model_end.record()

            final_logp = F.log_softmax(final_logits.float(), dim=-1)
            is_final_block = block_index == len(spans) - 1
            if (
                is_final_block
                and bool(int(os.environ.get("STRUCTNAR_PROFILE_QUERY_BUDGET", "0")))
            ):
                if not hasattr(self, "_tcis_frontier_prefix_batches"):
                    self._tcis_frontier_prefix_batches = []
                self._tcis_frontier_prefix_batches.append(
                    cur_seq[:, :, :start].detach().cpu().numpy()
                )
            if is_final_block:
                block_keep_k = K
            elif self.tcis_prefix_beams > 0:
                block_keep_k = self.tcis_prefix_beams
            else:
                block_keep_k = K * self.tcis_intermediate_beam_multiplier
            cur_seq, beam_scores = self._expand_block_with_constraints(
                seqs=cur_seq,
                scores=beam_scores,
                final_logp=final_logp,
                start=start,
                end=end,
                keep_k=block_keep_k,
                rrg_query_embedding=rrg_query_embedding,
                pad_to_keep_k=(
                    not self.tcis_dynamic_intermediate_beams
                    or is_final_block
                ),
            )
            if profile_this_call:
                expand_end.record()
                profile_rows.append((block_index, start, end, cur_k, model_start, model_end, expand_end))

        if profile_this_call and profile_all_calls:
            if not hasattr(self, "_tcis_profile_event_rows"):
                self._tcis_profile_event_rows = []
            self._tcis_profile_event_rows.extend(profile_rows)
        elif profile_this_call:
            torch.cuda.synchronize(device)
            for block_index, start, end, cur_k, model_start, model_end, expand_end in profile_rows:
                print(
                    f"[TCIS profile] block={block_index} span={start}:{end} input_beams={cur_k} "
                    f"model_ms={model_start.elapsed_time(model_end):.3f} "
                    f"expand_ms={model_end.elapsed_time(expand_end):.3f}"
                )
            self._tcis_profile_emitted = True

        if self.beam_score_normalization == "mean":
            beam_scores = beam_scores / float(max(L, 1))
        if return_scores:
            return cur_seq, beam_scores
        return cur_seq

    @torch.no_grad()
    def _tcis_frontier_planned_search(
        self,
        inputs_embeds,
        num_beams=10,
        return_scores: bool = False,
        rrg_query_embedding: torch.Tensor | None = None,
    ):
        """Choose a query-level switch using only the exposed Trie frontier.

        At each prefix depth, queries whose unique live-prefix descendants fit
        the configured leaf budget finish in one suffix round.  Remaining
        queries take one legal prefix step and are compacted into the next
        neural batch.  The decision uses no relevance label or model score
        threshold.  Dynamic attention blocks ensure that a continuation step
        and a suffix-completion step retain their respective fixed-P2L
        semantics even when different queries stop at different depths.
        """
        if int(self.gpt_hdgr_block_diffusion_steps) != 1:
            raise ValueError(
                "Frontier-planned P2L currently supports exactly one suffix "
                "prediction step; set gpt_hdgr_block_diffusion_steps=1."
            )
        device = inputs_embeds.device
        model_dtype = inputs_embeds.dtype
        batch_size = int(inputs_embeds.size(0))
        length = int(self.codebook_level)
        output_width = max(1, int(num_beams))
        budget = int(self.tcis_frontier_leaf_budget)
        min_prefix = int(self.tcis_frontier_min_prefix)
        max_prefix = int(self.tcis_frontier_max_prefix)

        active_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        active_seq = torch.full(
            (batch_size, 1, length),
            self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        active_scores = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        finished_seq = [None] * batch_size
        finished_scores = [None] * batch_size
        stop_depths = torch.full((batch_size,), -1, dtype=torch.long)

        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(batch_size, -1, -1).to(
            device=device, dtype=model_dtype
        )

        for start in range(length):
            if active_ids.numel() == 0:
                break
            current_batch, current_width = active_seq.shape[:2]
            finish_flags = []
            exposed_counts = []
            prefix_rows = active_seq[:, :, :start].detach().cpu().tolist()
            for rows in prefix_rows:
                prefixes = {tuple(int(token) for token in row) for row in rows}
                if start < min_prefix:
                    exposed = budget + 1
                    finish = False
                else:
                    exposed = self._count_exposed_leaves(
                        prefixes, stop_after=budget + 1
                    )
                    finish = exposed <= budget
                if start >= max_prefix or start == length - 1:
                    finish = True
                exposed_counts.append(int(exposed))
                finish_flags.append(bool(finish))

            finish_mask = torch.tensor(finish_flags, device=device, dtype=torch.bool)
            next_active_seq = None
            next_active_scores = None
            next_active_ids = None

            for finish in (False, True):
                group_mask = finish_mask if finish else ~finish_mask
                group_indices = torch.nonzero(group_mask, as_tuple=False).flatten()
                if group_indices.numel() == 0:
                    continue
                seq_group = active_seq.index_select(0, group_indices)
                score_group = active_scores.index_select(0, group_indices)
                ids_group = active_ids.index_select(0, group_indices)
                group_size = int(group_indices.numel())
                flat_seq = seq_group.reshape(group_size * current_width, length)
                end = length if finish else start + 1

                xt = torch.full_like(flat_seq, self.mask_token_id)
                if start > 0:
                    xt[:, :start] = flat_seq[:, :start]
                clean_context = self._clean_context_from_prefix(flat_seq, start)
                cond_base = base_cond.index_select(0, ids_group)
                uncond_base = base_uncond.index_select(0, ids_group)
                cond = cond_base.unsqueeze(1).expand(
                    -1, current_width, -1, -1
                ).reshape(group_size * current_width, self.num_prefix, self.d_model)
                uncond = uncond_base.unsqueeze(1).expand(
                    -1, current_width, -1, -1
                ).reshape(group_size * current_width, self.num_prefix, self.d_model)
                timestep = torch.full(
                    (group_size * current_width,),
                    self.time_step,
                    device=device,
                    dtype=torch.long,
                )
                dynamic_spans = self._planner_block_spans(length, start, finish)
                logits = self._block_cfg_logits(
                    xt=xt,
                    clean_context=clean_context,
                    cond_prefix=cond,
                    uncond_prefix=uncond,
                    t=timestep,
                    block_spans_override=dynamic_spans,
                )
                final_logp = F.log_softmax(logits.float(), dim=-1)
                if finish:
                    keep_width = output_width
                elif self.tcis_prefix_beams > 0:
                    keep_width = self.tcis_prefix_beams
                else:
                    keep_width = output_width * self.tcis_intermediate_beam_multiplier
                rrg_group = (
                    None
                    if rrg_query_embedding is None
                    else rrg_query_embedding.index_select(0, ids_group)
                )
                expanded_seq, expanded_scores = self._expand_block_with_constraints(
                    seqs=seq_group,
                    scores=score_group,
                    final_logp=final_logp,
                    start=start,
                    end=end,
                    keep_k=keep_width,
                    rrg_query_embedding=rrg_group,
                    pad_to_keep_k=True,
                )

                if finish:
                    for local_index, original_id in enumerate(ids_group.detach().cpu().tolist()):
                        finished_seq[original_id] = expanded_seq[local_index]
                        finished_scores[original_id] = expanded_scores[local_index]
                        stop_depths[original_id] = start
                else:
                    next_active_seq = expanded_seq
                    next_active_scores = expanded_scores
                    next_active_ids = ids_group

            if next_active_ids is None:
                active_ids = active_ids[:0]
            else:
                active_ids = next_active_ids
                active_seq = next_active_seq
                active_scores = next_active_scores

        if any(row is None for row in finished_seq):
            raise RuntimeError("Frontier planner terminated with unfinished queries")
        result_seq = torch.stack(finished_seq, dim=0)
        result_scores = torch.stack(finished_scores, dim=0)
        if not hasattr(self, "_tcis_frontier_stop_depths"):
            self._tcis_frontier_stop_depths = []
        self._tcis_frontier_stop_depths.extend(stop_depths.tolist())
        call_index = len(self._tcis_frontier_stop_depths) // max(batch_size, 1)
        if call_index == 1 or call_index % 50 == 0:
            observed = torch.tensor(self._tcis_frontier_stop_depths, dtype=torch.long)
            hist = torch.bincount(observed, minlength=length + 1)
            hist_text = ", ".join(
                f"s{depth}={int(count)}"
                for depth, count in enumerate(hist)
                if count
            )
            print(
                f"[Frontier planner] budget={budget} min_s={min_prefix} "
                f"max_s={max_prefix} queries={observed.numel()} {hist_text}"
            )

        if self.beam_score_normalization == "mean":
            result_scores = result_scores / float(max(length, 1))
        if return_scores:
            return result_seq, result_scores
        return result_seq

    @torch.no_grad()
    def _tcis_adaptive_search(
        self, inputs_embeds, num_beams=10, return_scores: bool = False,
        rrg_query_embedding: torch.Tensor | None = None,
    ):
        """Run query-level adaptive TCIS.

        A query switches from one-token Trie expansion to full-suffix expansion
        only when every live beam prefix has at most K descendant leaves.  All
        beams of that query therefore have scores covering the same number of
        positions, avoiding comparisons between partial and completed paths.
        """
        device = inputs_embeds.device
        model_dtype = inputs_embeds.dtype
        B0 = inputs_embeds.size(0)
        L = self.codebook_level
        K = max(1, int(num_beams))
        threshold = int(self.tcis_max_leaves)
        cur_seq = torch.full((B0, 1, L), self.mask_token_id, device=device, dtype=torch.long)
        beam_scores = torch.zeros(B0, 1, device=device, dtype=torch.float32)
        done = torch.zeros(B0, device=device, dtype=torch.bool)
        stop_depths = torch.full((B0,), L, device=device, dtype=torch.long)
        base_cond = inputs_embeds
        base_uncond = self.null_condition.expand(B0, -1, -1).to(device=device, dtype=model_dtype)

        for start in range(L):
            if bool(done.all()):
                break
            cur_k = cur_seq.size(1)
            flat_seq = cur_seq.reshape(B0 * cur_k, L)
            xt = torch.full_like(flat_seq, self.mask_token_id)
            if start > 0:
                xt[:, :start] = flat_seq[:, :start]
            clean_context = self._clean_context_from_prefix(flat_seq, start)
            cond = base_cond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                B0 * cur_k, self.num_prefix, self.d_model
            )
            uncond = base_uncond.unsqueeze(1).expand(-1, cur_k, -1, -1).reshape(
                B0 * cur_k, self.num_prefix, self.d_model
            )
            t = torch.full((B0 * cur_k,), self.time_step, device=device, dtype=torch.long)
            logits = self._block_cfg_logits(xt, clean_context, cond, uncond, t)
            final_logp = F.log_softmax(logits.float(), dim=-1).view(B0, cur_k, L, -1)

            next_seqs, next_scores = [], []
            for b in range(B0):
                if bool(done[b]):
                    next_seqs.append(cur_seq[b])
                    next_scores.append(beam_scores[b])
                    continue
                prefixes = {
                    tuple(int(x) for x in row[:start])
                    for row in cur_seq[b].detach().cpu().tolist()
                }
                can_finish = start > 0 and all(
                    0 < self._count_descendant_leaves(p, stop_after=threshold + 1) <= threshold
                    for p in prefixes
                )
                end = L if can_finish else start + 1
                seq_b, score_b = self._expand_block_with_constraints(
                    seqs=cur_seq[b:b + 1],
                    scores=beam_scores[b:b + 1],
                    final_logp=final_logp[b:b + 1].reshape(cur_k, L, -1),
                    start=start,
                    end=end,
                    keep_k=K,
                    rrg_query_embedding=(rrg_query_embedding[b:b + 1] if rrg_query_embedding is not None else None),
                )
                next_seqs.append(seq_b[0])
                next_scores.append(score_b[0])
                if can_finish or end == L:
                    done[b] = True
                    stop_depths[b] = start
            cur_seq = torch.stack(next_seqs, dim=0)
            beam_scores = torch.stack(next_scores, dim=0)

        hist = torch.bincount(stop_depths.detach().cpu(), minlength=L + 1)
        print(
            f"[Adaptive suffix] K={threshold} queries={B0} "
            f"mean_stop_depth={stop_depths.float().mean().item():.3f} "
            f"hist={{{', '.join(f'{i}:{int(n)}' for i, n in enumerate(hist) if n)}}}"
        )
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
        rrg_query_embedding: torch.Tensor | None = None,
    ):
        if getattr(self, "use_one_pass_tcis", False):
            return self.one_pass_tcis_search(
                inputs_embeds=inputs_embeds,
                num_beams=num_beams,
                cand_codes=cand_codes,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )
        if getattr(self, "use_tcis", False):
            if self.use_adaptive_residual_composition:
                raise ValueError(
                    "Residual-structured composition currently supports token-level Trie "
                    "decoding, SoundStorm prefix regeneration, and SoundStorm Trie "
                    "projection. Set model.use_tcis=false for this experiment."
                )
            return self.tcis_search(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=num_beams,
                cand_codes=cand_codes,
                return_scores=return_scores,
                rrg_query_embedding=rrg_query_embedding,
            )
        return super().constrained_beam_search(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            cand_codes=cand_codes,
            return_scores=return_scores,
            rrg_query_embedding=rrg_query_embedding,
        )


    def _initialize_hdgr_codebook_embeddings(self, scale: float = 1.0) -> None:
        """GENIUS-style RQ codebook initialization for the HDGR adapter."""
        new_token_ids = self.tokenizer.convert_tokens_to_ids(self.code_tokens)
        linear_layer = nn.Linear(768, self.d_model, bias=False)

        if self.modality_index:
            first_layer_codebook = self.quantizer.residual_rq.layers[0]._codebook.embed[0, :3, :]
            mapped_first = linear_layer(F.normalize(first_layer_codebook))

            other_layers_codebooks = [layer._codebook.embed for layer in self.quantizer.residual_rq.layers[1:]]
            other_layers_codebooks = torch.stack(other_layers_codebooks, dim=0)
            other_layers_codebooks = rearrange(other_layers_codebooks, "q 1 c d -> q c d")
            other_codebook = other_layers_codebooks.reshape(-1, other_layers_codebooks.shape[-1])
            mapped_other = linear_layer(F.normalize(other_codebook))
            mapped_embeddings = torch.cat([mapped_first, mapped_other], dim=0)
        else:
            codebook_vectors = self.quantizer.residual_rq.codebooks.reshape(-1, 768)
            mapped_embeddings = linear_layer(F.normalize(codebook_vectors))

        with torch.no_grad():
            for idx, token_id in enumerate(new_token_ids):
                vec = mapped_embeddings[idx] * float(scale)
                self.id_generator.input_embed.weight[token_id].copy_(vec)
                self.id_generator.lm_head.weight[token_id].copy_(vec)
