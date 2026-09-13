# P2L current public artifact state

## Current method

- Independent-Suffix P2L with prefix depth `s=3`.
- Prefix/final beam `B=K=50`; PCAA disabled (`lambda=0`).
- Canonical public configuration:
  `configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml`.
- The checkpoint, quantizer, query inputs, semantic-ID index, and reranker are
  fixed within matched policy comparisons but are external licensed/large
  assets and are not bundled.

## Lightweight verification

Run `python scripts/structnar/audit_current_release.py`. It checks the current
16-task policy table and the four-task validation-depth table without reading
the unpublished manuscript or external assets. Run
`python scripts/structnar/audit_public_release.py .` before publishing to reject
private paths, symlinks, and unexpectedly large files.

The current evidence reports macro R@10 39.36 for Sequential, 42.83 for
Interacting-Suffix P2L, and 43.64 for Independent-Suffix P2L. The latter has a
positive point estimate on all 16 matched tasks. The depth sweep is a
validation-set sensitivity study, not a claim of prospective hyperparameter
selection.

## Release boundary

The `v1.0.0` tag is immutable historical code using PCAA with `lambda=20`.
The current branch must not overwrite or relabel that tag. No manuscript,
supplement, LaTeX source, review material, model checkpoint, dataset, extracted
feature, or large index is part of this public artifact.
