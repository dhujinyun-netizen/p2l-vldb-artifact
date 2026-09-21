# P2L public weights and asset reconstruction

The evaluated P2L weights are released as inference-only files. They contain
no optimizer, scheduler, or scaler state.

## P2L weights

Release: <https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/tag/p2l-vldb-weights-v1.0.0>

| Asset | Download | Integrity |
|---|---|---|
| Evaluated epoch-99 GPT-HDGR/P2L checkpoint | Seven release shards plus [manifest](https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/download/p2l-vldb-weights-v1.0.0/p2l_evaluated_epoch99_shards.manifest.json) | Manifest contains SHA-256 for every shard; assembled tensor state matches the released inference checkpoint |
| Matching trained RQ quantizer | <https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/download/p2l-vldb-weights-v1.0.0/p2l_rq_quantizer_epoch99_inference.pth> | `b662556d050c4fb7557feaf85e4dabcb1e56d1b9cd83e30eb92a2d920d12b5b8` |

The checkpoint corresponds to the reported epoch-99 source checkpoint
(`2035a059...944fb`) and retains the complete trained generator state. The
second file is the exact RQ state expected by the current evaluation config;
the runtime initializes the quantizer before loading the generator checkpoint.
The unsharded inference export used to create the release has SHA-256
`2fcf63e7eb211b849f1b582e3e86ca5e6be1090eff7912e5f317aeeb6eaaaf09`; the
manifest is authoritative for the uploaded shard files, and the assembler
checks tensor-key coverage and exact tensor equality after reconstruction.

```bash
mkdir -p checkpoint/code_tied \
  checkpoint/rq_clip_large/Large/Instruct/InBatch
curl -L 'https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/download/p2l-vldb-weights-v1.0.0/p2l_rq_quantizer_epoch99_inference.pth' \
  -o checkpoint/rq_clip_large/Large/Instruct/InBatch/rq_clip_large_epoch_50.pth
mkdir -p /tmp/p2l_shards
curl -L 'https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/download/p2l-vldb-weights-v1.0.0/p2l_evaluated_epoch99_shards.manifest.json' \
  -o /tmp/p2l_shards/p2l_evaluated_epoch99_shards.manifest.json
for i in 000 001 002 003 004 005 006; do
  curl -L "https://github.com/dhujinyun-netizen/p2l-vldb-artifact/releases/download/p2l-vldb-weights-v1.0.0/p2l_evaluated_epoch99_shard-${i}-of-007.pth" \
    -o "/tmp/p2l_shards/p2l_evaluated_epoch99_shard-${i}-of-007.pth"
done
python scripts/structnar/assemble_public_checkpoint.py \
  --manifest /tmp/p2l_shards/p2l_evaluated_epoch99_shards.manifest.json \
  --shard-dir /tmp/p2l_shards \
  --output checkpoint/code_tied/gpt_hdgr_latest.pth
sha256sum checkpoint/code_tied/gpt_hdgr_latest.pth \
  checkpoint/rq_clip_large/Large/Instruct/InBatch/rq_clip_large_epoch_50.pth
```

## M-BEIR (not redistributed)

Download the official dataset from <https://huggingface.co/datasets/TIGER-Lab/M-BEIR>.
The official UniIR/GENIUS instructions are mirrored at
<https://github.com/TIGER-AI-Lab/UniIR> and
<https://github.com/sung-yeon-kim/GENIUS-CVPR25>.

```bash
git lfs install
git clone https://huggingface.co/datasets/TIGER-Lab/M-BEIR /absolute/path/M-BEIR
cd /absolute/path/M-BEIR
cat mbeir_images.tar.gz.part-* > mbeir_images.tar.gz
tar -xzf mbeir_images.tar.gz
cd /path/to/GENIUS-CVPR25-main
bash scripts/shared/setup_official_mbeir_symlink.sh /absolute/path/M-BEIR
```

The expected root contains `query/`, `cand_pool/`, `qrels/`, `instructions/`,
and `mbeir_images/`. M-BEIR is not mirrored in this repository.

## Third-party models (not redistributed)

* OpenAI CLIP: <https://github.com/openai/CLIP>. The required ViT-L/14 file is
  downloaded by `clip.load("ViT-L/14", download_root="checkpoint/CLIP")`.
* UniIR CLIP-SF checkpoint used by the feature pipeline: download the official
  file from
  <https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth>
  to `checkpoint/CLIP_SF/clip_sf_large.pth`.
* GENIUS baseline code and checkpoints: <https://github.com/sung-yeon-kim/GENIUS-CVPR25>
  and <https://huggingface.co/Sungyeon/GENIUS>. These third-party files are not
  uploaded to the P2L artifact.

```bash
pip install git+https://github.com/openai/CLIP.git
python - <<'PY'
import clip
clip.load("ViT-L/14", device="cpu", download_root="checkpoint/CLIP")
PY
mkdir -p checkpoint/CLIP_SF
curl -L 'https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth' \
  -o checkpoint/CLIP_SF/clip_sf_large.pth
```

## Rebuilding semantic-ID and Trie assets

Candidate embeddings, semantic-ID arrays, candidate IDs, and serialized Tries
are derived assets and are not redistributed because they encode the M-BEIR
candidate corpus. After downloading the official data, public weights, and
third-party feature models, the complete all-32 regeneration/evaluation path is:

```bash
export MBEIR_DATA_DIR=/absolute/path/M-BEIR
CUDA_VISIBLE_DEVICES=0 \
  MBEIR_DATA_DIR="$MBEIR_DATA_DIR" \
  bash scripts/shared/extract_official_mbeir_cand_features.sh

# Generates missing local semantic-ID caches and then the UNION cache/Trie.
CUDA_VISIBLE_DEVICES=0 \
  MBEIR_DATA_DIR="$MBEIR_DATA_DIR" \
  bash scripts/structnar/eval_rrg_w20_all32.sh
```

The generated files are written below
`gen_code/STRUCTNAR/Large/Instruct/structnar_p2l_d3_rqc_cosine_w20_all32/`.
The runner uses the released checkpoint, `s=3`, `B=K=50`, cosine suffix scoring,
and the same deterministic 16-LOCAL/16-UNION protocol as the reported results.
For a single task, use `scripts/structnar/eval_tcis.sh` after the corresponding
candidate cache exists. The scripts record the generated code, ID, embedding,
manifest, and Trie paths in their run logs.
