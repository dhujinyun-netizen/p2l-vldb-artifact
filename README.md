# Prefix-to-Leaf (P2L)

**Deferred Complete-Key Selection for Semantic-ID Retrieval**

[Project homepage](https://dhujinyun-netizen.github.io/p2l-vldb-artifact/) ·
[Current code](configs/structnar/p2l_independent_s3_b50_k50_lambda0.yaml) ·
[Artifact snapshot](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/tree/089d1a572aa2adfd1f34ded080c48840c62f7ff9) ·
[Archived v1.0.0](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/tag/v1.0.0)

## Current manuscript and evidence

The current private manuscript package is dated 16 September 2026. The evaluated
policy is Independent-Suffix P2L with `s=3`, `B=K=50`, `lambda=0`, and the
frozen epoch-99 checkpoint. On the twelve configurations excluded from policy
selection, macro R@10 increases from **38.81% to 42.10% (+3.29 points)**.
The all-16 value, **39.36% to 43.64%**, is descriptive rather than the primary
generalization estimate. See the [current evidence note](docs/evidence-20260916.md).

P2L retains a short Trie prefix, enumerates its stored complete descendants,
shares suffix score computation, and performs global complete-key selection.
WIDE addresses a related early-commit risk with uncertainty-triggered wildcard
expansion; P2L instead fixes a localization boundary and changes when partial-key
selection terminates. The homepage records this distinction without claiming a
matched WIDE reimplementation.

This public repository is code-only apart from the supplementary document and
the evaluated P2L weights release. The manuscript PDF/source, M-BEIR data,
extracted features, candidate embeddings, semantic-ID caches, and large indexes
are intentionally not included.

## Supplementary Material

The supplementary document referenced in the paper is available here:

[P2L_Supplementary_Material.pdf](./P2L_Supplementary_Material.pdf)

It contains additional model and training details, complete task-level
results and controls, statistical procedures, and detailed timing and
index-resource measurements.

## Public weights and asset reconstruction

The evaluated epoch-99 P2L checkpoint and its matching trained RQ quantizer are
available from the [P2L weights release](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/tag/p2l-vldb-weights-v1.0.0).
Their SHA-256 hashes and exact download commands are documented in
[`docs/P2L_PUBLIC_ASSETS.md`](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/blob/c64dc53d10026714910f36109cbd05edcc410bf6/docs/P2L_PUBLIC_ASSETS.md). M-BEIR, OpenAI CLIP,
UniIR CLIP-SF, and GENIUS third-party files are not redistributed; the same
document links to their official sources and gives the commands for rebuilding
semantic-ID and Trie assets from the public M-BEIR layout.

## Included

* `src/`: model and retrieval implementation;
* `scripts/`: training, evaluation, profiling, plotting, and audit scripts;
* `configs/`: current and historical experiment configurations;
* `docs/results/`: lightweight evidence tables and audit inputs;
* `tests/`, `genius_env.yml`, and `LICENSE`.

The immutable snapshot linked above contains the released code and supplementary
PDF. The manuscript source, datasets, and large derived assets remain separately
supplied under their respective licenses; the evaluated weights are distributed
through the release above.

## Reproducibility

No-data evidence checks can be run with the lightweight files under
`docs/results/`. Full inference requires the official M-BEIR release, the frozen
checkpoint, and the candidate semantic-ID/Trie assets. Runtime comparisons must
use the same checkpoint, index, beam/output widths, precision, and synchronization
protocol.

## License

The released code is distributed under the MIT License. External datasets,
pretrained models, and third-party packages remain under their own licenses.
