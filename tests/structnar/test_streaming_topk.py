import unittest

import torch

from models.hdgr_comparison.complete_id_selection import (
    deterministic_topk_by_index,
    topk_complete_id_scores,
)


class StreamingTopKTest(unittest.TestCase):
    def test_exact_score_ties_use_candidate_index(self):
        scores = torch.tensor([1.0, 2.0, 2.0, 2.0, 0.0])
        indices = torch.tensor([4, 3, 1, 2, 0])
        top_scores, top_indices = deterministic_topk_by_index(scores, indices, 3)
        torch.testing.assert_close(top_scores, torch.tensor([2.0, 2.0, 2.0]))
        self.assertEqual(top_indices.tolist(), [1, 2, 3])

    def test_streaming_is_chunk_invariant_at_tie_boundary(self):
        # Four legal complete IDs receive the same maximal complete-key score.
        # The deterministic secondary key must therefore return rows 0..3 for
        # every chunking, including the one-candidate-at-a-time extreme.
        log_probs = torch.tensor(
            [[
                [0.0, 0.0, -10.0],
                [0.0, 0.0, -10.0],
            ]],
            dtype=torch.float32,
        )
        candidates = torch.tensor(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
            ],
            dtype=torch.long,
        )
        expected = [0, 1, 2, 3]
        expected_scores = None
        for chunk_size in (1, 2, 3, 5, 99):
            top_idx, top_scores = topk_complete_id_scores(
                log_probs, candidates, 4, chunk_size=chunk_size
            )
            self.assertEqual(top_idx[0].tolist(), expected)
            if expected_scores is None:
                expected_scores = top_scores
            else:
                torch.testing.assert_close(top_scores, expected_scores, rtol=0.0, atol=0.0)

    def test_streaming_matches_full_deterministic_selection(self):
        torch.manual_seed(17)
        batch, length, vocab, num_candidates = 3, 4, 11, 37
        log_probs = torch.randn(batch, length, vocab)
        # Force many exact ties by using a small code alphabet and then
        # duplicating position-wise score rows.
        log_probs[:, :, 1] = log_probs[:, :, 0]
        candidates = torch.randint(0, 8, (num_candidates, length))

        full_scores = torch.zeros(batch, num_candidates)
        for level in range(length):
            full_scores += log_probs[:, level, :].index_select(1, candidates[:, level])
        full_indices = torch.arange(num_candidates).unsqueeze(0).expand(batch, -1)
        ref_scores, ref_indices = deterministic_topk_by_index(full_scores, full_indices, 9)

        for chunk_size in (1, 4, 7, 16, 64):
            got_indices, got_scores = topk_complete_id_scores(
                log_probs, candidates, 9, chunk_size=chunk_size
            )
            torch.testing.assert_close(got_scores, ref_scores, rtol=0.0, atol=0.0)
            self.assertTrue(torch.equal(got_indices, ref_indices))


if __name__ == "__main__":
    unittest.main()
