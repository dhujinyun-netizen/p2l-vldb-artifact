#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN="$(printenv PYTHON_BIN 2>/dev/null || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  export PYTHON_BIN="python"
fi

bash scripts/structnar/run_cirr_scale_curve.sh
"$PYTHON_BIN" scripts/structnar/collect_cirr_scale_results.py
"$PYTHON_BIN" scripts/structnar/profile_flatip_gpu_scaling.py
"$PYTHON_BIN" scripts/structnar/plot_cirr_scale_results.py

echo "RESULT=PASS_CIRR_SCALE_MATRIX_W20"
