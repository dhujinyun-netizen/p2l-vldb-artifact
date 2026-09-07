#!/usr/bin/env python3
"""CPU unit tests for StructNAR inference components."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.hdgr_comparison.shared_code_head import (  # noqa: E402
    CodeTiedOutputHead,
    merge_legacy_untied_weights,
    migrate_legacy_untied_state_dict,
)
from models.hdgr_comparison.residual_reconstruction_gain import (  # noqa: E402
    residual_reconstruction_gain,
    residual_quantization_compatibility,
    selected_residual_reconstruction_gain,
    selected_residual_quantization_compatibility,
)
from models.hdgr_comparison.complete_id_selection import (  # noqa: E402
    inbatch_complete_id_ranking_loss,
    sample_prefix_neighbor_codes,
    sampled_hard_negative_ranking_loss,
)


def test_forward_and_gradient_tying() -> None:
    torch.manual_seed(7)
    vocab_size, d_model = 5, 4
    embedding = nn.Embedding(vocab_size + 1, d_model)
    head = CodeTiedOutputHead(embedding, vocab_size)

    hidden = torch.randn(3, 2, d_model, requires_grad=True)
    expected = F.linear(hidden, embedding.weight[:vocab_size])
    actual = head(hidden)
    torch.testing.assert_close(actual, expected)

    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
    before = embedding.weight.detach().clone()
    loss = actual.square().mean()
    loss.backward()
    optimizer.step()

    assert not torch.equal(before[:vocab_size], embedding.weight[:vocab_size])
    # The input-only MASK row is not used by the output classifier and receives no gradient.
    torch.testing.assert_close(before[vocab_size], embedding.weight[vocab_size])
    assert head.weight.data_ptr() == embedding.weight.data_ptr()
    assert list(head.parameters()) == []


def test_legacy_merge() -> None:
    token = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]],
        dtype=torch.float32,
    )
    output = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    average = merge_legacy_untied_weights(token, output, vocab_size=2, strategy="average")
    torch.testing.assert_close(average[:2], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    torch.testing.assert_close(average[2], token[2])

    input_only = merge_legacy_untied_weights(token, output, vocab_size=2, strategy="input")
    torch.testing.assert_close(input_only, token)

    output_only = merge_legacy_untied_weights(token, output, vocab_size=2, strategy="output")
    torch.testing.assert_close(output_only[:2], output)
    torch.testing.assert_close(output_only[2], token[2])


def test_state_dict_alias_migration() -> None:
    token = torch.randn(6, 3)
    output = torch.randn(5, 3)
    state = {
        "module.hdgr.token_embed.weight": token.clone(),
        "module.hdgr.input_embed.weight": token.clone(),
        "module.input_embed.weight": token.clone(),
        "module.hdgr.lm_head.weight": output.clone(),
        "module.lm_head.weight": output.clone(),
    }
    original = copy.deepcopy(state)
    migrated = migrate_legacy_untied_state_dict(
        state,
        prefix="module.",
        vocab_size=5,
        strategy="average",
        token_key_suffixes=(
            "hdgr.token_embed.weight",
            "hdgr.input_embed.weight",
            "input_embed.weight",
        ),
        head_key_suffixes=("hdgr.lm_head.weight", "lm_head.weight"),
    )
    assert migrated
    assert "module.hdgr.lm_head.weight" not in state
    assert "module.lm_head.weight" not in state
    expected = 0.5 * (
        original["module.hdgr.token_embed.weight"][:5]
        + original["module.hdgr.lm_head.weight"]
    )
    for key in (
        "module.hdgr.token_embed.weight",
        "module.hdgr.input_embed.weight",
        "module.input_embed.weight",
    ):
        torch.testing.assert_close(state[key][:5], expected)
        torch.testing.assert_close(state[key][5], token[5])


def test_residual_reconstruction_gain() -> None:
    query = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    prefix = torch.tensor([[0.5, 0.5], [-1.0, 1.0]])
    codebook = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    actual = residual_reconstruction_gain(query, prefix, codebook)
    torch.testing.assert_close(
        residual_quantization_compatibility(query, prefix, codebook), actual
    )
    residual = query[:, None, :] - prefix[:, None, :]
    expected = residual.square().sum(-1) - (residual - codebook[None]).square().sum(-1)
    torch.testing.assert_close(actual, expected)
    alignment = residual_quantization_compatibility(
        query, prefix, codebook, mode="alignment"
    )
    torch.testing.assert_close(alignment, (query - prefix) @ codebook.t())
    cosine = residual_quantization_compatibility(
        query, prefix, codebook, mode="cosine"
    )
    expected_cosine = F.normalize(query - prefix, dim=-1) @ F.normalize(
        codebook, dim=-1
    ).t()
    torch.testing.assert_close(cosine, expected_cosine)
    normalized = residual_reconstruction_gain(query, prefix, codebook, normalize=True)
    torch.testing.assert_close(normalized.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(normalized.std(-1, unbiased=False), torch.ones(2), atol=1e-6, rtol=0)
    selected_ids = torch.tensor([1, 2])
    selected = selected_residual_reconstruction_gain(
        query, prefix, codebook, selected_ids, normalize=True
    )
    torch.testing.assert_close(
        selected_residual_quantization_compatibility(
            query.expand(2, -1),
            prefix.expand(2, -1),
            codebook,
            selected_ids,
            normalize=True,
        ),
        selected,
    )
    torch.testing.assert_close(
        selected, normalized.gather(1, selected_ids[:, None]).squeeze(1),
        atol=2e-5, rtol=2e-5,
    )


def test_raw_rq_path_gain_telescopes() -> None:
    """The traditional raw-gain control equals full-path error reduction."""
    query = torch.tensor([0.7, -0.4, 1.2])
    prefix = torch.tensor([0.1, 0.2, 0.3])
    suffix_codes = (
        torch.tensor([0.2, -0.1, 0.4]),
        torch.tensor([-0.3, 0.5, 0.1]),
        torch.tensor([0.1, 0.1, -0.2]),
    )
    running = prefix.clone()
    path_gain = query.new_zeros(())
    for code in suffix_codes:
        path_gain = path_gain + residual_reconstruction_gain(
            query, running, code.unsqueeze(0), normalize=False, mode="gain"
        ).squeeze(0)
        running = running + code
    direct_gain = (query - prefix).square().sum() - (query - running).square().sum()
    torch.testing.assert_close(path_gain, direct_gain, atol=1e-6, rtol=1e-6)


def test_inbatch_complete_id_ranking_loss() -> None:
    targets = torch.tensor([[0, 1], [2, 3], [0, 1]])
    logits = torch.full((3, 2, 5), -4.0)
    for row, target_row in enumerate(targets):
        logits[row, 0, target_row[0]] = 4.0
        logits[row, 1, target_row[1]] = 4.0
    loss, accuracy = inbatch_complete_id_ranking_loss(logits, targets)
    assert torch.isfinite(loss)
    assert float(accuracy) == 1.0
    assert float(loss) < 1.0e-3


def test_sampled_hard_negative_ranking_loss() -> None:
    targets = torch.tensor([[0, 1], [2, 3]])
    candidates = torch.tensor([[0, 1], [2, 3], [4, 4], [1, 2]])
    logits = torch.full((2, 2, 5), -3.0)
    for row, target in enumerate(targets):
        logits[row, 0, target[0]] = 3.0
        logits[row, 1, target[1]] = 3.0
    loss, accuracy, margin = sampled_hard_negative_ranking_loss(
        logits, targets, candidates, num_hard_negatives=2
    )
    assert torch.isfinite(loss) and torch.isfinite(margin)
    assert float(accuracy) == 1.0
    assert float(margin) > 0.0


def test_prefix_neighbor_sampling() -> None:
    import numpy as np
    legal = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
    target = np.array([[0, 0, 0]], dtype=np.int32)
    sampled = sample_prefix_neighbor_codes(
        legal, target, prefix_depths=(1, 2), candidates_per_depth=8,
        rng=np.random.default_rng(3),
    )
    assert sampled.shape[1] == 3
    assert not np.any(np.all(sampled == target[0], axis=1))
    assert any(np.array_equal(row, [0, 0, 1]) for row in sampled)


def main() -> None:
    test_forward_and_gradient_tying()
    test_legacy_merge()
    test_state_dict_alias_migration()
    test_residual_reconstruction_gain()
    test_raw_rq_path_gain_telescopes()
    test_inbatch_complete_id_ranking_loss()
    test_sampled_hard_negative_ranking_loss()
    test_prefix_neighbor_sampling()
    print("StructNAR unit tests: PASS")


if __name__ == "__main__":
    main()
