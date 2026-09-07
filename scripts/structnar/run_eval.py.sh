#!/usr/bin/env bash
set -euo pipefail

# Official M-BEIR evaluation wrapper for GPT-HDGR.
# No Flickr/local adapter is run here. It uses official query/cand_pool/qrels files directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-1337}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must be set to an official M-BEIR HDGR config}"
MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$REPO_ROOT/mbeir_data}"
CHECK_EXTRACTED="${CHECK_EXTRACTED:-1}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"

echo "[official M-BEIR HDGR eval] REPO_ROOT=$REPO_ROOT"
echo "[official M-BEIR HDGR eval] MBEIR_DATA_DIR=$MBEIR_DATA_DIR"
echo "[official M-BEIR HDGR eval] CONFIG_PATH=$CONFIG_PATH"
echo "[official M-BEIR HDGR eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC=$NPROC"
grep -nE "exp_name:|ckpt_dir:|ckpt_name:|num_beams:|rerank:|use_hybrid_score:|save_beam_scores:|cand_pool_dir_name:|test_dir_name:|union_cand_pool_dir_name:" "$CONFIG_PATH" || true

if [ "$CHECK_EXTRACTED" = "1" ]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/shared/verify_official_mbeir_layout.py"     --config-path "$CONFIG_PATH"     --mbeir-data-dir "$MBEIR_DATA_DIR"     --repo-root "$REPO_ROOT"     --check-extracted
else
  "$PYTHON_BIN" "$REPO_ROOT/scripts/shared/verify_official_mbeir_layout.py"     --config-path "$CONFIG_PATH"     --mbeir-data-dir "$MBEIR_DATA_DIR"     --repo-root "$REPO_ROOT"
fi

cd "$REPO_ROOT/src/common"
"$PYTHON_BIN" -m torch.distributed.run   --master_port "$MASTER_PORT"   --nproc_per_node="$NPROC"   "$REPO_ROOT/src/common/mbeir_generative_retriever_hdgr.py"   --config_path "$CONFIG_PATH"   --genir_dir "$REPO_ROOT"   --mbeir_data_dir "$MBEIR_DATA_DIR"   "$@"
