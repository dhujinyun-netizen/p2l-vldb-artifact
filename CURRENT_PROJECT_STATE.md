# P2L current project state

## Canonical method

- Method: P2L with prefix length `s=3` + Prefix-Conditioned Angular Affinity
  (PCAA).
- PCAA weight: `20`.
- Prefix length: `s=3`, including the modality-routing code and two
  residual-code positions.
- Prefix/final beam: `50/50`.
- Inference batch size: `8`.
- Final checkpoint: `checkpoint/code_tied/gpt_hdgr_latest.pth`.
- Final config: `configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml`.

The internal names `tcis`, `rrg`, and `rqc` remain in selected source,
configuration, and script interfaces for backward compatibility. They do not
denote additional paper methods and must not be used as manuscript terminology.

## Canonical paper

The active manuscript is the PVLDB 2027 long-paper rewrite:

```text
VLDB/vldb2027/structnar_vldb.tex
VLDB/vldb2027/build_current/structnar_vldb.pdf
VLDB/vldb2027/supplementary/structnar_vldb_supplement.tex
VLDB/vldb2027/supplementary/build_current/structnar_vldb_supplement.pdf
```

The active title is:

> P2L: Prefix-to-Leaf Search over Semantic-Identifier Indexes for Universal
> Multimodal Retrieval

The earlier TIP source is preserved as a historical venue version and is not
the active submission manuscript.

## Evidence

`docs/results/` contains the evidence retained for the current paper:

- final P2L-R and reproduced GENIUS-R 16-task LOCAL/UNION results;
- matched all-16 LOCAL decoder controls at `lambda=0`;
- five-scale CIRR-7 LOCAL-to-UNION measurements and exact GPU FlatIP;
- prefix-length, prefix-width, suffix-evidence, raw-cosine, RQ-distance, and
  PCAA controls;
- paired bootstrap, batch scaling, latency tails, component profiling,
  semantic-index resources, and position-wise quantizer diagnostics.

Run:

```bash
python scripts/structnar/audit_vldb_evidence.py
```

to verify the locked configuration, result matrices, provenance, and generated
paper artifacts. The current main-paper build uses 12 content pages plus one
reference-only continuation page, within the PVLDB regular-paper limit.

The final systems control is complete. On the full 4,170-query CIRR-7 LOCAL
manifest, a level-wise CSR/GPU-gather implementation of Sequential+PCAA is
exactly equivalent to the canonical decoder for the saved top-50 identifiers,
scores, query embeddings, and query IDs. It changes generation latency only
from 59.90 to 59.33 ms/query, while P2L+PCAA runs at 27.65 ms/query and remains
53.4% faster. The raw summary is
`docs/results/p2l_gpu_flat_trie_baseline.json`; the admission rule is recorded
in `VLDB/vldb2027/EXPERIMENT_INCLUSION_GATE.md`.

The deterministic evidence audit must report `PASS`, and the current
`PAPER_CLAIM_AUDIT.{md,json}` records the independent source-to-evidence review
for the exact manuscript snapshot included in a release.

## Recoverable archives

- `tip/packages/StructNAR_Historical_Experiments_20260727.zip` contains the
  superseded configurations, scripts, logs, results, and an earlier paper
  snapshot.
- `tip/packages/StructNAR_Legacy_Papers_Reviews_20260728.zip` contains the old
  AAAI/Journal manuscripts, duplicate figures, and review documents.

## Removed obsolete assets

- 181 generated one-off YAML configurations;
- six abandoned one-pass/smoke checkpoint families;
- AAAI and pre-canonical Journal manuscript trees;
- old root-level paper packages and duplicate figures;
- unused manuscript figures and build caches;
- superseded lambda-10, RRG/TCIS, router, stability, and failed-experiment
  records from the active results directory.

The removed smoke checkpoints were not referenced by the final configuration
or evidence audit. The canonical P2L, GENIUS, quantizer, CLIP, and
reranker checkpoints remain intact.
