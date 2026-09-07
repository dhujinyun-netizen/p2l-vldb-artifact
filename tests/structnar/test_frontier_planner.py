import unittest

import torch

from models.hdgr_comparison.retriever_gpt_hdgr import (
    RetrieverDiffusionHDGRBackbone,
    T5ForGenerativeRetrieval,
)


class _TrieCounter:
    _get_tree_node_for_prefix = T5ForGenerativeRetrieval._get_tree_node_for_prefix
    _count_descendant_leaves = T5ForGenerativeRetrieval._count_descendant_leaves
    _count_exposed_leaves = T5ForGenerativeRetrieval._count_exposed_leaves

    def __init__(self):
        # Three leaves: (1, 3), (1, 4), and (2, 5).
        self.tree_index = {1: {3: {}, 4: {}}, 2: {5: {}}}
        self.mask_token_id = 99
        self.trie_leaf_count_cache = {}


class FrontierPlannerTest(unittest.TestCase):
    def test_exposed_leaf_count_deduplicates_prefixes_and_stops_early(self):
        counter = _TrieCounter()
        self.assertEqual(counter._count_exposed_leaves({(1,), (2,)}), 3)
        self.assertEqual(counter._count_exposed_leaves({(1,), (1,)}), 2)
        self.assertGreaterEqual(
            counter._count_exposed_leaves({(1,), (2,)}, stop_after=2), 2
        )

    def test_dynamic_block_spans_cover_each_position_once(self):
        continue_spans = T5ForGenerativeRetrieval._planner_block_spans(9, 3, False)
        finish_spans = T5ForGenerativeRetrieval._planner_block_spans(9, 3, True)
        self.assertEqual(continue_spans, [(i, i + 1) for i in range(9)])
        self.assertEqual(finish_spans, [(0, 1), (1, 2), (2, 3), (3, 9)])

    def test_default_adapter_path_matches_explicit_static_spans(self):
        torch.manual_seed(7)
        spans = [(0, 1), (1, 2), (2, 4)]
        adapter = RetrieverDiffusionHDGRBackbone(
            vocab_size=13,
            num_classes=14,
            d_model=8,
            code_length=4,
            num_prefix=2,
            time_step=2,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            block_spans=spans,
        ).eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        masked = torch.tensor([[1, 2, 13, 13]])
        prefix = torch.randn(1, 2, 8)
        timestep = torch.tensor([2])
        implicit = adapter(tokens, masked, prefix, t=timestep).logits
        explicit = adapter(
            tokens,
            masked,
            prefix,
            t=timestep,
            block_spans_override=spans,
        ).logits
        torch.testing.assert_close(implicit, explicit, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
