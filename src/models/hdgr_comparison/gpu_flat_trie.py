"""Level-wise flat Trie tensors for GPU-constrained semantic-ID search.

The representation is deliberately small and execution-oriented.  At depth
``j``, each node owns a contiguous range of outgoing edges.  An edge stores
only its token ID; its position in the edge array is also the node ID at depth
``j + 1``.  This lets constrained beam search expand a batch of Trie nodes
with tensor gathers instead of walking Python dictionaries beam by beam.

Building and serializing the structure are offline operations.  They must not
be included in online decoder latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FlatTrieLevel:
    """CSR-like outgoing edges for one identifier depth."""

    offsets: torch.Tensor
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        if self.offsets.ndim != 1 or self.tokens.ndim != 1:
            raise ValueError("Flat Trie offsets and tokens must be one-dimensional")
        if self.offsets.numel() < 2:
            raise ValueError("A Flat Trie level must contain at least one parent node")
        if int(self.offsets[0].item()) != 0:
            raise ValueError("Flat Trie offsets must start at zero")
        if int(self.offsets[-1].item()) != int(self.tokens.numel()):
            raise ValueError("Final Flat Trie offset must equal the number of edges")

    @property
    def num_nodes(self) -> int:
        return int(self.offsets.numel() - 1)

    @property
    def num_edges(self) -> int:
        return int(self.tokens.numel())


class FlatPrefixTrie:
    """A fixed-depth Trie stored as level-wise CSR tensors."""

    FORMAT_VERSION = 1

    def __init__(self, levels: Sequence[FlatTrieLevel], code_length: int) -> None:
        self.levels = tuple(levels)
        self.code_length = int(code_length)
        if self.code_length <= 0 or len(self.levels) != self.code_length:
            raise ValueError("Flat Trie must contain one level per identifier position")
        for depth in range(1, self.code_length):
            expected = self.levels[depth - 1].num_edges
            actual = self.levels[depth].num_nodes
            if expected != actual:
                raise ValueError(
                    f"Flat Trie node mismatch at depth {depth}: "
                    f"expected {expected}, found {actual}"
                )

    @classmethod
    def from_token_ids(cls, token_ids: torch.Tensor | np.ndarray) -> "FlatPrefixTrie":
        """Build a deterministic flat Trie from fixed-length token IDs.

        Duplicate complete identifiers are removed because Trie leaves index
        unique semantic keys; item collisions remain represented by the
        separate exact-ID buckets.
        """

        if torch.is_tensor(token_ids):
            values = token_ids.detach().cpu().numpy()
        else:
            values = np.asarray(token_ids)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("token_ids must be a non-empty [N,L] matrix")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("token_ids must contain integers")

        values = np.asarray(values, dtype=np.int64)
        code_length = int(values.shape[1])
        # np.lexsort uses the last key as primary, hence the reversed column
        # order below gives standard left-to-right identifier ordering.
        order = np.lexsort(tuple(values[:, j] for j in reversed(range(code_length))))
        values = values[order]
        unique = np.ones(values.shape[0], dtype=np.bool_)
        if values.shape[0] > 1:
            unique[1:] = np.any(values[1:] != values[:-1], axis=1)
        values = values[unique]

        levels: list[FlatTrieLevel] = []
        parent_starts = np.asarray([0], dtype=np.int64)
        prefix_change = np.zeros(max(values.shape[0] - 1, 0), dtype=np.bool_)

        for depth in range(code_length):
            if values.shape[0] > 1:
                prefix_change |= values[1:, depth] != values[:-1, depth]
            child_starts = np.flatnonzero(
                np.concatenate((np.asarray([True]), prefix_change))
            ).astype(np.int64, copy=False)
            parent_ids = np.searchsorted(
                parent_starts, child_starts, side="right"
            ) - 1
            counts = np.bincount(parent_ids, minlength=parent_starts.size)
            offsets = np.empty(parent_starts.size + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            tokens = values[child_starts, depth]

            if tokens.size and (
                int(tokens.min()) < np.iinfo(np.int32).min
                or int(tokens.max()) > np.iinfo(np.int32).max
            ):
                raise OverflowError("Token IDs do not fit in int32")
            if int(offsets[-1]) > np.iinfo(np.int32).max:
                raise OverflowError("Flat Trie edge count exceeds int32 capacity")

            levels.append(
                FlatTrieLevel(
                    offsets=torch.from_numpy(offsets.astype(np.int32, copy=False)),
                    tokens=torch.from_numpy(tokens.astype(np.int32, copy=False)),
                )
            )
            parent_starts = child_starts

        return cls(levels=levels, code_length=code_length)

    def to(self, device: torch.device | str) -> "FlatPrefixTrie":
        return FlatPrefixTrie(
            levels=[
                FlatTrieLevel(
                    offsets=level.offsets.to(device=device, non_blocking=True),
                    tokens=level.tokens.to(device=device, non_blocking=True),
                )
                for level in self.levels
            ],
            code_length=self.code_length,
        )

    def expand(
        self, parent_nodes: torch.Tensor, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand flattened parent nodes at one depth.

        Returns ``(owner, token, child_node, counts)``.  ``owner`` indexes the
        flattened input parent array.  Child node IDs are the selected edge
        positions and are valid parent IDs at the next level.
        """

        depth = int(depth)
        if not 0 <= depth < self.code_length:
            raise ValueError(f"depth must be in [0,{self.code_length}), got {depth}")
        flat_nodes = parent_nodes.reshape(-1).long()
        level = self.levels[depth]
        if flat_nodes.numel() == 0:
            raise ValueError("parent_nodes must be non-empty")
        # CPU callers (including construction tests) retain explicit bounds
        # checking.  During GPU decoding, node IDs are produced only by the
        # preceding Flat Trie level, so synchronizing the device here solely
        # to validate that invariant would add one host round trip per depth.
        if flat_nodes.device.type == "cpu" and bool(
            ((flat_nodes < 0) | (flat_nodes >= level.num_nodes)).any().item()
        ):
            raise IndexError(f"Invalid parent node at depth {depth}")

        starts = level.offsets.index_select(0, flat_nodes).long()
        ends = level.offsets.index_select(0, flat_nodes + 1).long()
        counts = ends - starts
        owner = torch.repeat_interleave(
            torch.arange(flat_nodes.numel(), device=flat_nodes.device), counts
        )
        total = owner.numel()
        if total == 0:
            empty = torch.empty(0, device=flat_nodes.device, dtype=torch.long)
            return empty, empty, empty, counts

        segment_starts = torch.cumsum(counts, dim=0) - counts
        local = torch.arange(total, device=flat_nodes.device) - torch.repeat_interleave(
            segment_starts, counts
        )
        edge_ids = starts.index_select(0, owner) + local
        tokens = level.tokens.index_select(0, edge_ids).long()
        return owner, tokens, edge_ids, counts

    def state_dict(self, signature: str | None = None) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "signature": signature,
            "code_length": self.code_length,
            "offsets": [level.offsets.cpu() for level in self.levels],
            "tokens": [level.tokens.cpu() for level in self.levels],
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "FlatPrefixTrie":
        if int(payload.get("format_version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("Unsupported Flat Trie format version")
        offsets: Iterable[torch.Tensor] = payload["offsets"]
        tokens: Iterable[torch.Tensor] = payload["tokens"]
        levels = [
            FlatTrieLevel(offsets=o.int().cpu(), tokens=t.int().cpu())
            for o, t in zip(offsets, tokens)
        ]
        return cls(levels=levels, code_length=int(payload["code_length"]))

    def save(self, path: str | Path, signature: str | None = None) -> None:
        torch.save(self.state_dict(signature=signature), str(path))

    @classmethod
    def load(
        cls, path: str | Path, expected_signature: str | None = None
    ) -> "FlatPrefixTrie":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if expected_signature is not None and payload.get("signature") != expected_signature:
            raise ValueError("Flat Trie signature does not match the candidate codes")
        return cls.from_state_dict(payload)

    def storage_bytes(self) -> int:
        return sum(
            level.offsets.numel() * level.offsets.element_size()
            + level.tokens.numel() * level.tokens.element_size()
            for level in self.levels
        )
