# P2L current public artifact state

## Current method

- Independent-Suffix P2L with prefix depth `s=3`.
- Prefix/final beam `B=K=50`; PCAA disabled (`lambda=0`).
- Canonical configuration: `configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml`.
- Checkpoint, quantizer, query inputs, semantic-ID index, and reranker are fixed within matched policy comparisons but remain external licensed/large assets.

## Current lightweight evidence

The matched all-16 policy table reports macro R@10 39.36 for Sequential and
43.64 for Independent-Suffix P2L (+4.28 points). The primary
Evaluation-only-12 group reports 38.81 -> 42.10 (+3.29 points, paired-query
95% CI [3.07, 3.52]).

The 14 September 2026 Frozen selection-control extension keeps the
Independent-Suffix score tables and complete-key scoring rule fixed while
changing only intermediate legal top-B selection. Across all twelve
evaluation-only configurations it reports:

- macro R@10 38.46 -> 42.10 (+3.64 points; 95% CI [3.42, 3.86]);
- macro IDR@50 42.67 -> 47.12 (+4.46 points);
- 12/12 positive R@10 point estimates and 11/12 task intervals excluding zero;
- task-macro APF 7.63% (diagnostic, not an additive decomposition of R@10).

Run `python scripts/structnar/audit_fixed_score_eval12.py` for a no-data audit
of the bundled CSV/JSON evidence.

## Release boundary

The `v1.0.0` tag remains immutable historical code using PCAA with
`lambda=20`. The submission-facing paper should cite an immutable commit
snapshot containing the current source and lightweight evidence. No manuscript,
supplement, LaTeX source, review material, model checkpoint, dataset, extracted
feature, or large index is part of the public artifact.
