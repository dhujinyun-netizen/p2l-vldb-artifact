#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/structnar/train.yaml}"
MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-${REPO_ROOT}/mbeir_data}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-3141}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${REPO_ROOT}"
mkdir -p logs
LOG="${LOG:-logs/train_structnar_official_mbeir_$(date +%Y%m%d_%H%M%S).log}"

echo "[StructNAR] repo=${REPO_ROOT}"
echo "[StructNAR] data=${MBEIR_DATA_DIR}"
echo "[StructNAR] config=${CONFIG_PATH}"
echo "[StructNAR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC}"
echo "[StructNAR] log=${LOG}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/structnar/test_structnar.py"

"${PYTHON_BIN}" -m torch.distributed.run \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NPROC}" \
  "${REPO_ROOT}/src/models/hdgr_comparison/train.py" \
  --config_path "${CONFIG_PATH}" \
  --genir_dir "${REPO_ROOT}" \
  --mbeir_data_dir "${MBEIR_DATA_DIR}" \
  "$@" 2>&1 | tee "${LOG}"
