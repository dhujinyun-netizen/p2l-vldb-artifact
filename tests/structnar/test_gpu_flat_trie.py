import itertools

import torch

from src.models.hdgr_comparison.gpu_flat_trie import FlatPrefixTrie


def test_segmented_flat_topk_matches_per_query_topk():
    # Import through the runtime package layout used by the evaluation entry
    # point; this also catches accidental import-time regressions.
    from src.models.hdgr_comparison.retriever_gpt import T5ForGenerativeRetrieval

    # Two queries, two parent beams each. Owners are parent-major and the
    # number of legal children varies by query and parent.
    owners = torch.tensor([0, 0, 1, 2, 3, 3, 3])
    tokens = torch.tensor([10, 11, 20, 30, 40, 41, 42])
    children = torch.arange(7)
    scores = torch.tensor([0.1, 0.7, 0.5, 0.3, 0.9, 0.2, 0.8])

    top_scores, parents, top_tokens, top_children = (
        T5ForGenerativeRetrieval._segmented_flat_topk(
            scores=scores,
            owners=owners,
            token_ids=tokens,
            child_nodes=children,
            batch_size=2,
            parents_per_query=2,
            keep_limit=3,
        )
    )
    assert torch.allclose(top_scores, torch.tensor([[0.7, 0.5, 0.1], [0.9, 0.8, 0.3]]))
    assert torch.equal(top_tokens, torch.tensor([[11, 20, 10], [40, 42, 30]]))
    assert torch.equal(parents, torch.tensor([[0, 1, 0], [1, 1, 0]]))
    assert torch.equal(top_children, torch.tensor([[1, 2, 0], [4, 6, 3]]))


def _enumerate_flat_paths(trie: FlatPrefixTrie):
    paths = [(0, ())]
    for depth in range(trie.code_length):
        nodes = torch.tensor([node for node, _ in paths], dtype=torch.long)
        owner, tokens, children, counts = trie.expand(nodes, depth)
        assert int(counts.sum().item()) == int(tokens.numel())
        paths = [
            (int(children[i]), paths[int(owner[i])][1] + (int(tokens[i]),))
            for i in range(tokens.numel())
        ]
    return [path for _, path in paths]


def test_flat_trie_matches_unique_identifiers():
    rows = torch.tensor(
        [
            [9, 2, 8, 4],
            [3, 7, 1, 6],
            [9, 2, 8, 4],
            [3, 7, 5, 0],
            [3, 1, 2, 9],
            [9, 4, 0, 1],
        ],
        dtype=torch.long,
    )
    trie = FlatPrefixTrie.from_token_ids(rows)
    expected = sorted(set(map(tuple, rows.tolist())))
    assert sorted(_enumerate_flat_paths(trie)) == expected


def test_expansion_owner_and_child_nodes_are_stable():
    rows = torch.tensor(list(itertools.product([0, 1], [10, 11], [20, 21])))
    trie = FlatPrefixTrie.from_token_ids(rows)
    owner0, token0, node1, count0 = trie.expand(torch.tensor([0]), 0)
    assert owner0.tolist() == [0, 0]
    assert token0.tolist() == [0, 1]
    assert node1.tolist() == [0, 1]
    assert count0.tolist() == [2]

    owner1, token1, node2, count1 = trie.expand(node1, 1)
    assert owner1.tolist() == [0, 0, 1, 1]
    assert token1.tolist() == [10, 11, 10, 11]
    assert node2.tolist() == [0, 1, 2, 3]
    assert count1.tolist() == [2, 2]


def test_round_trip_serialization(tmp_path):
    rows = torch.tensor([[0, 2, 4], [0, 3, 5], [1, 2, 6]])
    trie = FlatPrefixTrie.from_token_ids(rows)
    path = tmp_path / "flat_trie.pt"
    trie.save(path, signature="abc")
    loaded = FlatPrefixTrie.load(path, expected_signature="abc")
    assert _enumerate_flat_paths(loaded) == _enumerate_flat_paths(trie)
    assert loaded.storage_bytes() == trie.storage_bytes()
