"""Shared semantic-code output head utilities for StructNAR.

The HDGR input embedding contains ``vocab_size`` real output-token rows plus
one input-only MASK row.  A standard ``nn.Linear`` output head duplicates the
real-token rows and lets the two spaces drift during training.  This module
implements a parameter-free output head whose classifier matrix is the first
``vocab_size`` rows of the input embedding.
"""

from __future__ import annotations

from typing import MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_LEGACY_MERGE_STRATEGIES = {"average", "input", "output"}


class CodeTiedOutputHead(nn.Module):
    """Parameter-free classifier tied to the real-token rows of an embedding.

    ``embedding`` is intentionally stored without registering it as a child
    module.  The owning HDGR module already registers the embedding, and
    registering it again here would duplicate state-dict paths.
    """

    def __init__(self, embedding: nn.Embedding, vocab_size: int):
        super().__init__()
        if not isinstance(embedding, nn.Embedding):
            raise TypeError(f"embedding must be nn.Embedding, got {type(embedding)!r}")
        vocab_size = int(vocab_size)
        if vocab_size <= 0 or vocab_size > embedding.num_embeddings:
            raise ValueError(
                f"vocab_size must be in [1, {embedding.num_embeddings}], got {vocab_size}"
            )
        object.__setattr__(self, "_embedding", embedding)
        self.vocab_size = vocab_size

    @property
    def weight(self) -> torch.Tensor:
        """The shared output matrix; excludes the input-only MASK row."""
        return self._embedding.weight[: self.vocab_size]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, tied_to=input_embedding"


def merge_legacy_untied_weights(
    token_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    vocab_size: int,
    strategy: str = "average",
) -> torch.Tensor:
    """Merge an old untied HDGR checkpoint into one shared embedding matrix.

    The returned tensor keeps any input-only rows (notably MASK) from
    ``token_weight`` and replaces only the first ``vocab_size`` rows.
    """

    strategy = str(strategy).lower().strip()
    if strategy not in VALID_LEGACY_MERGE_STRATEGIES:
        raise ValueError(
            f"Unsupported legacy merge strategy {strategy!r}; "
            f"choose one of {sorted(VALID_LEGACY_MERGE_STRATEGIES)}"
        )

    vocab_size = int(vocab_size)
    if token_weight.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("token_weight and lm_head_weight must both be rank-2 tensors")
    if token_weight.size(0) < vocab_size:
        raise ValueError(
            f"token_weight has {token_weight.size(0)} rows, fewer than vocab_size={vocab_size}"
        )
    if lm_head_weight.shape != (vocab_size, token_weight.size(1)):
        raise ValueError(
            "lm_head_weight shape mismatch: expected "
            f"({vocab_size}, {token_weight.size(1)}), got {tuple(lm_head_weight.shape)}"
        )

    merged = token_weight.clone()
    input_rows = token_weight[:vocab_size]
    output_rows = lm_head_weight.to(device=token_weight.device, dtype=token_weight.dtype)

    if strategy == "average":
        shared_rows = 0.5 * (input_rows + output_rows)
    elif strategy == "input":
        shared_rows = input_rows
    else:  # output
        shared_rows = output_rows

    merged[:vocab_size] = shared_rows
    return merged


def migrate_legacy_untied_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    prefix: str,
    vocab_size: int,
    strategy: str,
    token_key_suffixes: tuple[str, ...] = ("token_embed.weight", "input_embed.weight"),
    head_key_suffixes: tuple[str, ...] = ("lm_head.weight",),
) -> bool:
    """In-place migration helper used during ``load_state_dict``.

    Returns ``True`` when an old independent LM-head tensor was found and
    merged.  All alias token-embedding keys are updated consistently so an
    alias cannot overwrite the merged value later in recursive loading.
    """

    head_keys = [prefix + suffix for suffix in head_key_suffixes]
    token_keys = [prefix + suffix for suffix in token_key_suffixes]

    head_weight = next((state_dict[k] for k in head_keys if k in state_dict), None)
    token_weight = next((state_dict[k] for k in token_keys if k in state_dict), None)
    if head_weight is None or token_weight is None:
        return False

    merged = merge_legacy_untied_weights(
        token_weight,
        head_weight,
        vocab_size=vocab_size,
        strategy=strategy,
    )
    for key in token_keys:
        if key in state_dict:
            state_dict[key] = merged.clone()
    for key in head_keys:
        state_dict.pop(key, None)
    return True
