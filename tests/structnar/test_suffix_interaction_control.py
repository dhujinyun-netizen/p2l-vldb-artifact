from pathlib import Path
import sys
from types import MethodType

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.hdgr_comparison.retriever_hdgr import build_hdgr_attention_mask
from models.hdgr_comparison.retriever_gpt_hdgr import (
    RetrieverDiffusionHDGRBackbone,
    T5ForGenerativeRetrieval,
)


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


def test_isolated_suffix_hides_uncommitted_clean_mask_placeholders():
    length = 9
    suffix_start = 3
    isolated = [(i, i + 1) for i in range(length)]
    visible = torch.zeros((1, length), dtype=torch.bool)
    visible[:, :suffix_start] = True
    attention = build_hdgr_attention_mask(
        length=length,
        block_size=1,
        batch_size=1,
        device=torch.device("cpu"),
        block_spans=isolated,
        clean_visible_mask=visible,
    )[0]

    # Every unresolved noisy state keeps its own noisy MASK state.
    assert attention[suffix_start + 2, suffix_start + 2].item() == 0.0
    # It sees committed clean prefix states.
    assert attention[suffix_start + 2, length + suffix_start - 1].item() == 0.0
    # It cannot see earlier or later uncommitted clean MASK placeholders.
    assert attention[suffix_start + 2, length + suffix_start].item() < -1_000.0
    assert attention[suffix_start + 2, length + suffix_start + 1].item() < -1_000.0


def test_clean_visibility_shape_is_checked():
    try:
        build_hdgr_attention_mask(
            length=9,
            block_size=1,
            batch_size=2,
            device=torch.device("cpu"),
            clean_visible_mask=torch.ones((1, 9), dtype=torch.bool),
        )
    except ValueError as error:
        assert "clean_visible_mask" in str(error)
    else:
        raise AssertionError("invalid clean_visible_mask shape was accepted")


def test_isolated_suffix_logits_do_not_depend_on_other_unresolved_states():
    torch.manual_seed(7)
    length = 5
    suffix_start = 2
    mask_token = 16
    singleton_spans = [(i, i + 1) for i in range(length)]
    model = RetrieverDiffusionHDGRBackbone(
        vocab_size=mask_token,
        num_classes=mask_token + 1,
        d_model=16,
        code_length=length,
        num_prefix=3,
        time_step=4,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        block_size=1,
        block_spans=[(0, 1), (1, 2), (2, 5)],
    ).eval()
    clean = torch.tensor([[1, 2, mask_token, mask_token, mask_token]])
    noisy_a = clean.clone()
    noisy_b = clean.clone()
    # Change other unresolved noisy states while leaving the probed position
    # untouched.  With singleton spans they must not affect its logits.
    noisy_b[:, suffix_start + 1] = 5
    noisy_b[:, suffix_start + 2] = 9
    prefix = torch.randn(1, 3, 16)
    timestep = torch.tensor([4])
    with torch.inference_mode():
        logits_a = model(
            tokens=clean,
            mask_tokens=noisy_a,
            prefix=prefix,
            t=timestep,
            block_spans_override=singleton_spans,
        ).logits
        logits_b = model(
            tokens=clean,
            mask_tokens=noisy_b,
            prefix=prefix,
            t=timestep,
            block_spans_override=singleton_spans,
        ).logits
    torch.testing.assert_close(
        logits_a[:, suffix_start], logits_b[:, suffix_start], rtol=0.0, atol=1e-6
    )


def test_one_batched_isolated_suffix_matches_repeated_singleton_forwards():
    torch.manual_seed(11)
    length = 5
    suffix_start = 2
    mask_token = 16
    singleton_spans = [(i, i + 1) for i in range(length)]
    model = RetrieverDiffusionHDGRBackbone(
        vocab_size=mask_token,
        num_classes=mask_token + 1,
        d_model=16,
        code_length=length,
        num_prefix=3,
        time_step=4,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        block_size=1,
        block_spans=[(0, 1), (1, 2), (2, 5)],
    ).eval()
    clean = torch.tensor([[1, 2, mask_token, mask_token, mask_token]])
    noisy = clean.clone()
    prefix = torch.randn(1, 3, 16)
    timestep = torch.tensor([4])
    common = dict(
        tokens=clean,
        mask_tokens=noisy,
        prefix=prefix,
        t=timestep,
        project_logits=False,
        block_spans_override=singleton_spans,
    )
    with torch.inference_mode():
        batched_hidden = model(**common).hidden_states
        for position in range(suffix_start, length):
            singleton_hidden = model(
                **common, active_span=(position, position + 1)
            ).hidden_states
            batched_logits = model.hdgr.lm_head(
                batched_hidden[:, position : position + 1]
            )
            singleton_logits = model.hdgr.lm_head(
                singleton_hidden[:, position : position + 1]
            )
            torch.testing.assert_close(
                batched_logits, singleton_logits, rtol=0.0, atol=1e-6
            )


def _tiny_tcis_retriever(isolate_suffix_states):
    """Construct only the production TCIS/HDGR pieces needed by this test."""
    torch.manual_seed(19)
    length = 4
    vocab_size = 8
    mask_token = vocab_size
    d_model = 12
    num_prefix = 2

    retriever = T5ForGenerativeRetrieval.__new__(T5ForGenerativeRetrieval)
    torch.nn.Module.__init__(retriever)
    retriever.id_generator = RetrieverDiffusionHDGRBackbone(
        vocab_size=vocab_size,
        num_classes=vocab_size + 1,
        d_model=d_model,
        code_length=length,
        num_prefix=num_prefix,
        time_step=4,
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        block_size=1,
        block_spans=[(0, 1), (1, 2), (2, 4)],
    ).eval()
    retriever.eval()

    # These are the production P2L controls, reduced only in tensor size/beam
    # width.  Compact projection maps each legal token ID to the same local
    # code index at every level.
    retriever.codebook_level = length
    retriever.vocab_size = vocab_size
    retriever.num_classes = vocab_size + 1
    retriever.mask_token_id = mask_token
    retriever.d_model = d_model
    retriever.num_prefix = num_prefix
    retriever.time_step = 4
    retriever.guidance_scale = 3.0
    retriever.temperature = 1.0
    retriever.top_p = 1.0
    retriever.null_condition = torch.zeros(1, num_prefix, d_model)
    retriever.block_spans = [(0, 1), (1, 2), (2, 4)]
    retriever.block_transition_cache = {}
    retriever.trie_leaf_count_cache = {}
    retriever.block_trie_max_candidates = 0
    retriever.tcis_frontier_leaf_budget = 0
    retriever.tcis_max_leaves = 0
    retriever.deduplicate_block_beams = True
    retriever.tcis_selected_rrg = False
    retriever.tcis_batched_cfg = False
    retriever.tcis_compact_head = True
    retriever.tcis_dynamic_intermediate_beams = True
    retriever.tcis_intermediate_beam_multiplier = 1
    retriever.tcis_prefix_beams = 2
    retriever.tcis_prefix_kv_cache = False
    retriever.tcis_active_token_pruning = False
    retriever.tcis_isolate_suffix_states = bool(isolate_suffix_states)
    retriever.tcis_force_legal_token_pruning = False
    retriever.gpt_hdgr_block_diffusion_steps = 1
    retriever.gpt_hdgr_block_score_normalization = "sum"
    retriever.beam_score_normalization = "none"
    retriever.use_rrg = False
    retriever.level_vocab_sizes = torch.full((length,), vocab_size, dtype=torch.long)
    retriever.level_token_ids = torch.arange(vocab_size).repeat(length, 1)
    retriever.level_vocab_mask = torch.ones(length, vocab_size, dtype=torch.bool)
    retriever.token_id_to_code = torch.arange(vocab_size)

    leaves = (
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 2, 3),
        (1, 0, 3, 2),
        (1, 1, 1, 1),
        (1, 1, 2, 0),
    )
    tree = {}
    for leaf in leaves:
        node = tree
        for token in leaf:
            node = node.setdefault(token, {})
    retriever.tree_index = tree
    return retriever, leaves


def _run_tcis_with_trace(retriever, condition):
    cfg_calls = []
    expansions = []
    cfg_impl = T5ForGenerativeRetrieval._block_cfg_logits.__get__(
        retriever, T5ForGenerativeRetrieval
    )
    expand_impl = T5ForGenerativeRetrieval._expand_block_with_constraints.__get__(
        retriever, T5ForGenerativeRetrieval
    )

    def traced_cfg(self, **kwargs):
        logits = cfg_impl(**kwargs)
        cfg_calls.append(
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
            | {"logits": logits.detach().clone()}
        )
        return logits

    def traced_expand(self, **kwargs):
        result = expand_impl(**kwargs)
        expansions.append(
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
            | {
                "output_sequences": result[0].detach().clone(),
                "output_scores": result[1].detach().clone(),
            }
        )
        return result

    retriever._block_cfg_logits = MethodType(traced_cfg, retriever)
    retriever._expand_block_with_constraints = MethodType(traced_expand, retriever)
    output = retriever.tcis_search(
        inputs_embeds=condition,
        num_beams=2,
        return_scores=True,
    )
    return output, cfg_calls, expansions, cfg_impl


def test_tcis_search_isolated_suffix_matches_singleton_evidence_and_legal_topk(
    monkeypatch,
):
    for name in (
        "STRUCTNAR_PROFILE_ALL_TCIS",
        "STRUCTNAR_PROFILE_TCIS",
        "STRUCTNAR_PROFILE_QUERY_BUDGET",
        "STRUCTNAR_PROFILE_COMPONENTS",
    ):
        monkeypatch.setenv(name, "0")

    retriever, leaves = _tiny_tcis_retriever(isolate_suffix_states=False)
    torch.manual_seed(23)
    condition = torch.randn(1, retriever.num_prefix, retriever.d_model)

    _, grouped_calls, grouped_expansions, _ = _run_tcis_with_trace(
        retriever, condition
    )
    retriever.tcis_isolate_suffix_states = True
    (actual_sequences, actual_scores), isolated_calls, isolated_expansions, cfg_impl = (
        _run_tcis_with_trace(retriever, condition)
    )

    # The three configured P2L blocks cause two unchanged prefix rounds and
    # exactly one suffix round.  The singleton override is confined to that
    # one multi-token suffix; it never reaches either prefix call.
    assert [(row["start"], row["end"]) for row in isolated_expansions] == [
        (0, 1),
        (1, 2),
        (2, 4),
    ]
    assert len(isolated_calls) == 3
    singleton_spans = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert [row["block_spans_override"] for row in isolated_calls] == [
        None,
        None,
        singleton_spans,
    ]
    assert [row["block_spans_override"] for row in grouped_calls] == [None, None, None]
    assert all(row["active_span"] is None for row in isolated_calls)
    for round_index in (0, 1):
        for key in ("xt", "clean_context", "cond_prefix", "uncond_prefix", "t", "logits"):
            torch.testing.assert_close(
                isolated_calls[round_index][key],
                grouped_calls[round_index][key],
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            isolated_expansions[round_index]["output_sequences"],
            grouped_expansions[round_index]["output_sequences"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            isolated_expansions[round_index]["output_scores"],
            grouped_expansions[round_index]["output_scores"],
            rtol=0.0,
            atol=0.0,
        )

    # Re-evaluate each unresolved position as an active singleton through the
    # same production CFG function.  A single isolated suffix pass must carry
    # exactly the same guided evidence at each position.
    suffix_call = isolated_calls[-1]
    repeated_logits = suffix_call["logits"].clone()
    cfg_kwargs = {
        key: suffix_call[key]
        for key in (
            "xt",
            "clean_context",
            "cond_prefix",
            "uncond_prefix",
            "t",
            "cond_prefix_kv",
            "uncond_prefix_kv",
            "prefixes_normalized",
        )
    }
    for position in (2, 3):
        singleton_logits = cfg_impl(
            **cfg_kwargs,
            active_span=(position, position + 1),
            block_spans_override=singleton_spans,
        )
        torch.testing.assert_close(
            suffix_call["logits"][:, position],
            singleton_logits[:, position],
            rtol=0.0,
            atol=1e-6,
        )
        repeated_logits[:, position] = singleton_logits[:, position]

    # Brute-force every complete Trie descendant of the live prefix frontier
    # using the repeated-singleton evidence, then compare the production top-k.
    suffix_expansion = isolated_expansions[-1]
    parent_sequences = suffix_expansion["seqs"]
    parent_scores = suffix_expansion["scores"]
    assert parent_sequences.size(0) == 1
    assert len({tuple(row[:2].tolist()) for row in parent_sequences[0]}) == parent_sequences.size(1)
    repeated_logp = torch.log_softmax(repeated_logits.float(), dim=-1)
    candidate_sequences = []
    candidate_scores = []
    for parent_index, parent in enumerate(parent_sequences[0]):
        prefix = tuple(parent[:2].tolist())
        expected_suffixes = {leaf[2:] for leaf in leaves if leaf[:2] == prefix}
        enumerated_suffixes = {
            tuple(row)
            for row in retriever._get_next_block_candidates(
                prefix, block_len=2, device=torch.device("cpu")
            ).tolist()
        }
        assert enumerated_suffixes == expected_suffixes
        for suffix in sorted(expected_suffixes):
            leaf = prefix + suffix
            candidate_sequences.append(torch.tensor(leaf, dtype=torch.long))
            candidate_scores.append(
                parent_scores[0, parent_index]
                + repeated_logp[parent_index, 2, suffix[0]]
                + repeated_logp[parent_index, 3, suffix[1]]
            )
    candidate_sequences = torch.stack(candidate_sequences)
    candidate_scores = torch.stack(candidate_scores)
    expected_scores, expected_indices = torch.topk(candidate_scores, k=2)
    expected_sequences = candidate_sequences.index_select(0, expected_indices)

    torch.testing.assert_close(actual_sequences[0], expected_sequences, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_scores[0], expected_scores, rtol=0.0, atol=1e-6)


def test_fixed_score_levelwise_changes_only_suffix_selection(monkeypatch):
    """The E1 branches must enter suffix selection with identical evidence."""
    for name in (
        "STRUCTNAR_PROFILE_ALL_TCIS",
        "STRUCTNAR_PROFILE_TCIS",
        "STRUCTNAR_PROFILE_QUERY_BUDGET",
        "STRUCTNAR_PROFILE_COMPONENTS",
    ):
        monkeypatch.setenv(name, "0")

    retriever, leaves = _tiny_tcis_retriever(isolate_suffix_states=True)
    torch.manual_seed(29)
    condition = torch.randn(1, retriever.num_prefix, retriever.d_model)

    retriever.tcis_force_legal_token_pruning = False
    _, p2l_calls, p2l_expansions, _ = _run_tcis_with_trace(retriever, condition)
    retriever.tcis_force_legal_token_pruning = True
    (levelwise_sequences, levelwise_scores), levelwise_calls, levelwise_expansions, _ = (
        _run_tcis_with_trace(retriever, condition)
    )

    # The policy flag is consulted only by constrained expansion. The query,
    # retained prefix frontier, CFG inputs, and position-wise logits are exact.
    assert len(p2l_calls) == len(levelwise_calls) == 3
    for p2l_call, levelwise_call in zip(p2l_calls, levelwise_calls):
        assert p2l_call.keys() == levelwise_call.keys()
        for key in p2l_call:
            left, right = p2l_call[key], levelwise_call[key]
            if torch.is_tensor(left):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            else:
                assert left == right
    for round_index in (0, 1):
        torch.testing.assert_close(
            p2l_expansions[round_index]["output_sequences"],
            levelwise_expansions[round_index]["output_sequences"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            p2l_expansions[round_index]["output_scores"],
            levelwise_expansions[round_index]["output_scores"],
            rtol=0.0,
            atol=0.0,
        )
    for key in ("seqs", "scores", "final_logp", "start", "end", "keep_k"):
        left, right = p2l_expansions[-1][key], levelwise_expansions[-1][key]
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        else:
            assert left == right

    # Independently reproduce legal per-position top-B pruning from the frozen
    # suffix evidence; this checks that E1's control is not unconstrained token
    # beam search and never creates an ID absent from the Trie.
    suffix = levelwise_expansions[-1]
    parent_sequences = suffix["seqs"][0]
    parent_scores = suffix["scores"][0]
    logp = suffix["final_logp"].view(
        1, parent_sequences.size(0), retriever.codebook_level, -1
    )[0]
    origin = torch.arange(parent_sequences.size(0))
    current_sequences = parent_sequences.clone()
    current_scores = parent_scores.clone()
    for position in range(int(suffix["start"]), int(suffix["end"])):
        rows = []
        scores = []
        origins = []
        for parent_index, row in enumerate(current_sequences):
            prefix = tuple(int(value) for value in row[:position].tolist())
            legal = sorted({leaf[position] for leaf in leaves if leaf[:position] == prefix})
            for token in legal:
                candidate = row.clone()
                candidate[position] = token
                rows.append(candidate)
                scores.append(current_scores[parent_index] + logp[origin[parent_index], position, token])
                origins.append(origin[parent_index])
        candidate_scores = torch.stack(scores)
        selected_scores, selected = torch.topk(
            candidate_scores, k=min(int(suffix["keep_k"]), len(rows))
        )
        current_sequences = torch.stack(rows).index_select(0, selected)
        current_scores = selected_scores
        origin = torch.stack(origins).index_select(0, selected)

    torch.testing.assert_close(levelwise_sequences[0], current_sequences, rtol=0.0, atol=0.0)
    torch.testing.assert_close(levelwise_scores[0], current_scores, rtol=0.0, atol=1e-6)
    assert all(tuple(row.tolist()) in leaves for row in levelwise_sequences[0])
