#!/usr/bin/env bash
set -euo pipefail

# Single/multi-GPU friendly candidate feature extraction script for GENIUS.
# Override defaults with e.g.:
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash src/feature_extraction/run_feature_extraction_cand.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
genir_dir="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$genir_dir/src"
MODEL_DIR="$SRC/feature_extraction"
CONFIG_DIR="$MODEL_DIR"
CONFIG_PATH="$CONFIG_DIR/config_cand.yaml"
SCRIPT_NAME="clip_feature_extraction_cand.py"

MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$genir_dir/mbeir_data}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NPROC="${NPROC:-1}"
export PYTHONPATH="$SRC:${PYTHONPATH:-}"

# Auto-convert local Flickr30K CSV/annotations into minimal MBEIR JSONL files if needed.
if [ ! -f "$MBEIR_DATA_DIR/cand_pool/local/mbeir_flickr30k_task0_cand_pool.jsonl" ] && [ -d "$MBEIR_DATA_DIR/flickr30k" ]; then
  echo "[GENIUS patch] Flickr30K candidate jsonl not found. Preparing local Flickr30K MBEIR files..."
  python "$SRC/data/prepare_flickr30k_mbeir_local.py" \
    --flickr-root "$MBEIR_DATA_DIR/flickr30k" \
    --mbeir-data-dir "$MBEIR_DATA_DIR"
fi

echo "GENIUS_ROOT: $genir_dir"
echo "MBEIR_DATA_DIR: $MBEIR_DATA_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC: $NPROC"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

cd "$MODEL_DIR"
python3 -m torch.distributed.run --nproc_per_node="$NPROC" "$MODEL_DIR/$SCRIPT_NAME" \
  --config_path "$CONFIG_PATH" \
  --genir_dir "$genir_dir" \
  --mbeir_data_dir "$MBEIR_DATA_DIR"
