# Tie-fixed exact streaming audit — 15 September 2026

This note records the final Eval-12 audit of the bounded-memory streaming realization used by the P2L manuscript.

## Contract

Full materialization and Streaming use the same frozen checkpoint, semantic-ID index, queries, scoring configuration, and runtime environment. Both paths use the same deterministic lexicographic tie rule:

1. descending complete-key score;
2. ascending semantic-ID / leaf order for exact score ties.

Streaming scores descendants in chunks and maintains a running top-K rather than materializing the whole candidate-score array at once.

## Campaign

- 12 Full-reference task cells;
- 36 Streaming cells: 12 tasks × chunk sizes 4,096, 16,384, and 65,536;
- all 48 cells completed successfully.

## Exactness

All 36 Streaming cells match their Full reference exactly:

- ordered top-K key identity: 36/36;
- score-array identity: 36/36;
- cross-chunk ordered identity: 36/36;
- cross-chunk score equality: 36/36;
- set-mismatch queries: 0;
- tie-permutation queries: 0;
- R@10 mismatches: 0.

The tie-fixed Full rerun has macro R@10 = 42.1017%, which rounds to the 42.10% manuscript value.

## Buffer and runtime

| Path | Chunk | Aggregate generation (s) | Overhead vs Full | Peak candidate buffer |
|---|---:|---:|---:|---:|
| Full | — | 4,704.99 | — | full frontier |
| Streaming | 4,096 | 5,247.18 | +11.52% | ≤4,096 |
| Streaming | 16,384 | 5,215.72 | +10.86% | ≤16,384 |
| Streaming | 65,536 | 5,224.71 | +11.05% | ≤65,536 |

The largest exposed frontier in the campaign is 78,585 entries (OVEN-6), so the chunk bound is active rather than vacuous. The present Python Streaming implementation is about 11% slower in aggregate; the supported claim is deterministic equivalence plus bounded temporary buffering, not a speedup over Full.

## Audit command

The strict checker is:

```bash
python scripts/structnar/audit_streaming_exact_eval12.py <campaign.json>
```

It fails if any task/chunk changes R@10, selected-set membership, ordered key identity, score arrays, or exceeds the configured chunk buffer.

This final tie-fixed result supersedes the earlier pre-fix campaign in which tied keys could appear in a different order despite identical scores and R@10.
