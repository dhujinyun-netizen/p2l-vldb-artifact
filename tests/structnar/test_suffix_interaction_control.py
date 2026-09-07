from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.hdgr_comparison.retriever_hdgr import build_hdgr_attention_mask


def test_singleton_control_blocks_cross_suffix_noisy_attention():
    length = 9
    suffix_start = 3
    grouped = [(0, 1), (1, 2), (2, 3), (suffix_start, length)]
    isolated = [(i, i + 1) for i in range(length)]

    grouped_mask = build_hdgr_attention_mask(
        length=length,
        block_size=1,
        batch_size=1,
        device=torch.device("cpu"),
        block_spans=grouped,
    )[0]
    isolated_mask = build_hdgr_attention_mask(
        length=length,
        block_size=1,
        batch_size=1,
        device=torch.device("cpu"),
        block_spans=isolated,
    )[0]

    # Canonical P2L permits noisy suffix positions to exchange latent state.
    assert grouped_mask[suffix_start, suffix_start + 1].item() == 0.0
    # The control preserves self-attention but blocks other noisy suffix states.
    assert isolated_mask[suffix_start, suffix_start].item() == 0.0
    assert isolated_mask[suffix_start, suffix_start + 1].item() < -1_000.0
    # Both policies retain visibility of the committed prefix through x0.
    assert grouped_mask[suffix_start, length + suffix_start - 1].item() == 0.0
    assert isolated_mask[suffix_start, length + suffix_start - 1].item() == 0.0
