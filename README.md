# Prefix-to-Leaf (P2L)

**Deferred Selection and Shared Suffix Scoring for Semantic-ID Retrieval**

[Project homepage](https://dhujinyun-netizen.github.io/p2l-vldb-artifact/) ·
[Current code](configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml) ·
[Archived v1.0.0](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/tag/v1.0.0) ·
[中文说明](README_CN.md)

## Current manuscript-facing artifact snapshot

This repository tracks the September 2026 P2L manuscript revision:
Independent-Suffix P2L with `s=3`, `B=K=50`, and `lambda=0`.
The matched 16-task macro R@10 is **39.36% -> 43.64% (+4.28 points)**.
The primary Evaluation-only-12 comparison is **38.81% -> 42.10% (+3.29 points)**.

A separate selection-only control now covers all twelve evaluation-only
configurations. Frozen-Levelwise and Frozen-P2L use the same position-wise
Independent-Suffix score tables and complete-key scoring rule; only the
intermediate legal top-B selection schedule changes. Its macro R@10 is
**38.46% -> 42.10% (+3.64 points; 95% CI [3.42, 3.86])**, with all 12/12
point estimates positive. See [the 14 September evidence note](docs/evidence-20260914.md).

The final tie-fixed Streaming audit also covers all twelve evaluation-only
tasks at chunk sizes 4,096, 16,384, and 65,536. All 36 Streaming cells match
their Full reference exactly in ordered top-K keys, score arrays, and R@10,
while the temporary candidate buffer never exceeds the configured chunk size.
The current Python Streaming path is about 11% slower in aggregate, so this is
a bounded-memory correctness result rather than a speedup claim. See
[the 15 September streaming audit](docs/streaming-audit-20260915.md).

The repository contains source code, configurations, lightweight result files,
plot/audit utilities, and tests. It is a source-and-lightweight-evidence
research artifact; manuscript/supplement LaTeX and compiled paper PDFs remain
outside the public repository before formal submission.

## What is included

- `src/`: model and retrieval implementation;
- `scripts/`: submission-facing training, evaluation, profiling, plotting, and audit scripts;
- `configs/`: configurations used by the reported experiments;
- `docs/results/`: lightweight CSV/TSV/JSON evidence for headline results and execution diagnostics;
- `tests/`: lightweight package tests;
- `genius_env.yml` and `LICENSE`.

The archive intentionally excludes model checkpoints, M-BEIR data, extracted
features, candidate embeddings, semantic-ID caches, and large vector indexes.
These assets are subject to their original licenses and/or are too large for a
source artifact.

## Reproducibility levels

### 1. Evidence and exactness audits

The existing release audit checks the current matched-policy and validation
records:

```bash
python scripts/structnar/audit_current_release.py
```

The Frozen-12 selection-control evidence can be checked without a GPU,
dataset, or checkpoint:

```bash
python scripts/structnar/audit_fixed_score_eval12.py
```

This audit recomputes the task macro and verifies the query count, positive-task
count, task confidence-interval count, APF, and key timing/frontier summary
values from the bundled CSV/JSON files.

Given a tie-fixed Streaming campaign JSON, exact Full-versus-Streaming identity
is checked with:

```bash
python scripts/structnar/audit_streaming_exact_eval12.py <campaign.json>
```

The checker fails on any R@10 mismatch, selected-set mismatch, ordered-key or
score-array difference, cross-chunk inconsistency, or chunk-buffer violation.
The lightweight aggregate summary is stored in
`docs/results/streaming_tie_fixed_summary_20260915.csv`.

### 2. Full experiment reproduction

Full inference requires the official M-BEIR release, frozen CLIP/RQ50 assets,
the evaluated P2L checkpoint, and the realized candidate semantic-ID/Trie
artifacts. Set `MBEIR_DATA_DIR`, `CKPT_DIR`, and experiment-specific paths
before running the scripts under `scripts/structnar/`.

The reference configuration is:

`configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml`

After downloading the official M-BEIR release, link it into the checkout with:

```bash
bash scripts/shared/setup_official_mbeir_symlink.sh /path/to/M-BEIR
```

Historical release boundaries remain explicit: `v1.0.0` is the older PCAA
configuration (`lambda=20`). Historical tags must not be relabeled as
reproducing later evidence; the paper should cite an immutable commit permalink
for the submission-facing snapshot.

## Environment

The recorded environment is described by `genius_env.yml` (Python 3.10,
PyTorch, FAISS, Transformers, CLIP, and related packages). Exact latencies
depend on GPU, driver, CUDA, and storage state. Runtime comparisons must use
the same checkpoint, candidate index, batch size, precision mode, and
synchronization protocol.

## Scope

The public repository intentionally excludes the manuscript, supplementary
material, compiled manuscript PDFs, review material, private filesystem paths,
credentials, model weights, copyrighted dataset files, and large indexes.

## License

The released code is distributed under the MIT License in `LICENSE`. External
datasets, pretrained models, and third-party packages remain under their own
licenses.
