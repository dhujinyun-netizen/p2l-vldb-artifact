# Prefix-to-Leaf (P2L)

**Deferred Selection and Shared Suffix Scoring for Semantic-ID Retrieval**

[Project homepage](https://dhujinyun-netizen.github.io/p2l-vldb-artifact/) ·
[Current code](configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml) ·
[Archived v1.0.0](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/tag/v1.0.0) ·
[中文说明](README_CN.md)

## Current manuscript and archived artifact

The homepage now follows the manuscript revision of **13 September 2026**:
Independent-Suffix P2L, `s=3`, `B=K=50`, `lambda=0`; matched 16-task macro
R@10 is **39.36% → 43.64% (+4.28 points)**. See the
[current evidence/version note](docs/evidence-20260913.md) for evaluation scope.

The current branch now contains the Independent-Suffix implementation,
canonical `lambda=0` configuration, lightweight matched-policy evidence, and
its no-data audit. The immutable **v1.0.0** tag remains an archived PCAA
configuration and must not be presented as reproducing the current setup.

This repository contains the source code, configuration files, lightweight
evidence, and tests for **Prefix-to-Leaf (P2L)**. It is a code-only research
artifact; the manuscript and supplementary paper files are intentionally not
included before formal submission.

## What is included

- `src/`: model and retrieval implementation;
- `scripts/`: submission-facing training, evaluation, profiling, plotting, and
  audit scripts;
- `configs/`: the configurations used by the reported experiments;
- `docs/results/`: the small CSV/TSV/JSON evidence files used to verify the
  headline numbers;
- `tests/`: lightweight package tests;
- `genius_env.yml` and `LICENSE`.

The archive intentionally excludes model checkpoints, M-BEIR data, extracted
features, candidate embeddings, semantic-ID caches, and large vector indexes.
These assets are either subject to their original dataset/model licenses or
are too large for a source artifact. The README documents the expected paths
and command-line overrides so that an evaluator can provide them separately.

## Reproducibility levels

The package supports two levels of verification.

1. **No-data evidence audit.** This runs without a GPU, dataset, or checkpoint
   and recomputes the headline values from the bundled lightweight evidence:

   ```bash
   python scripts/structnar/audit_current_release.py
   ```

2. **Full experiment reproduction.** This requires the official M-BEIR
   release, the frozen CLIP/RQ50 assets, the P2L checkpoint, and the candidate
   semantic-ID/Trie artifacts. Set `MBEIR_DATA_DIR`, `CKPT_DIR`, and any
   experiment-specific paths before running the scripts under
   `scripts/structnar/`.

   The current reference configuration is
   `configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml`. After
   downloading the official M-BEIR release, link it into the artifact
   checkout with:

   ```bash
   bash scripts/shared/setup_official_mbeir_symlink.sh /path/to/M-BEIR
   ```

   The historical v1.0.0 PCAA configuration is
   `configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml`, using `s=3`,
   `B=K=50`, standardized PCAA with `lambda=20`, and batch size 8.
   It is not the current manuscript's default Independent-Suffix configuration.

## Environment

The recorded environment is described by `genius_env.yml` (Python 3.10,
PyTorch, FAISS, Transformers, CLIP, and the other required packages). Exact
latencies depend on GPU, driver, CUDA, and storage state. Runtime comparisons
must use the same checkpoint, candidate index, batch size, precision mode,
and synchronization protocol.

## Scope

This public repository intentionally excludes the manuscript, supplementary
material, compiled manuscript PDFs, and submission-specific LaTeX sources until
the work has been formally submitted. The project homepage includes a framework
preview. The source artifact contains no credentials, private
filesystem paths, model weights, or copyrighted dataset files.

## License

The released code is distributed under the MIT License in `LICENSE`. External
datasets, pretrained models, and third-party packages remain under their own
licenses.
