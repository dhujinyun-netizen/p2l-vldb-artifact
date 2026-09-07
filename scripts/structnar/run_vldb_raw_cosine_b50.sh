#!/usr/bin/env bash
set -euo pipefail

# Re-run the raw-cosine score control with the final effective prefix width
# B=50.  An older exploratory sweep used B=100 and is not eligible for the
# matched score-design comparison.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 NPROC=1
export PYTHON_BIN="${PYTHON_BIN:-python}"
export MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$(readlink -f "$ROOT/mbeir_data")}"

RUN_ID="${RAW_COSINE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATE="$ROOT/docs/results/vldb_raw_cosine_b50/$RUN_ID"
LOG_DIR="$ROOT/logs/vldb_raw_cosine_b50/$RUN_ID"
CONFIG_DIR="$ROOT/configs/structnar/generated/vldb_raw_cosine_b50_$RUN_ID"
mkdir -p "$STATE" "$LOG_DIR" "$CONFIG_DIR"
printf '%s\n' "$RUN_ID" > "$ROOT/docs/results/structnar_vldb_raw_cosine_active_run_id.txt"

cell_valid() {
  local marker="$1" log expected
  [[ -s "$marker" ]] || return 1
  log="$(sed -n 's/^log=//p' "$marker" | head -n1)"
  expected="$(sed -n 's/^sha256=//p' "$marker" | head -n1)"
  [[ -s "$log" && "$(sha256sum "$log" | awk '{print $1}')" == "$expected" ]] || return 1
  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Generation Metrics: seconds=' "$log"
}

run_cell() {
  local split="$1" task="$2" weight="$3" index="$4"
  local exp="structnar_vldb_rawcos_b50_${split}_${task}_w${weight}"
  local config="$CONFIG_DIR/${exp}.yaml"
  local log="$LOG_DIR/${split}_${task}_w${weight}.log"
  local marker="$STATE/${split}_${task}_w${weight}.ok"
  if cell_valid "$marker"; then
    echo "[raw cosine B50] skip $split $task weight=$weight"
    return 0
  fi

  "$PYTHON_BIN" scripts/structnar/make_tcis_config.py \
    --output "$config" --switch-depth 3 --batch-size 8 --num-beams 50 \
    --prefix-beams 50 --dataset "$task" --split "$split" \
    --rrg-weight "$weight" --rrg-normalization none --rqc-score-mode cosine \
    --rqc-apply-from-level 3 --rqc-prefix-start-level 0 \
    --tcis-compact-head --tcis-dynamic-intermediate-beams \
    --deterministic-eval-sampling --exp-name "$exp"
  "$PYTHON_BIN" - "$config" <<'PY'
from pathlib import Path
import sys, yaml
p = Path(sys.argv[1])
c = yaml.safe_load(p.read_text())
c["retrieval_config"]["load_saved_beam_scores"] = False
c["retrieval_config"]["save_beam_scores"] = True
p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
PY
  "$PYTHON_BIN" scripts/structnar/prepare_candidate_cache.py \
    --repo-root "$ROOT" --exp-name "$exp" --model-short-name STRUCTNAR --link-existing
  CONFIG_PATH="$config" CHECK_EXTRACTED=0 MASTER_PORT="$((5900 + index))" \
    bash scripts/structnar/run_eval.py.sh 2>&1 | tee "$log"
  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Generation Metrics: seconds=' "$log"
  if [[ "$split" == "val" ]]; then
    rg -q '/qrels/val/' "$log"
  else
    rg -q '/qrels/test/' "$log"
  fi
  {
    printf 'split=%s\n' "$split"
    printf 'task=%s\n' "$task"
    printf 'weight=%s\n' "$weight"
    printf 'log=%s\n' "$log"
    printf 'sha256=%s\n' "$(sha256sum "$log" | awk '{print $1}')"
    printf 'config=%s\n' "$config"
    printf 'config_sha256=%s\n' "$(sha256sum "$config" | awk '{print $1}')"
  } > "$marker"
}

weights=(0 5 10 15 20 30 40)
val_tasks=(cirr_task7 nights_task4)
test_tasks=(cirr_task7 nights_task4 edis_task2 webqa_task1)
index=0
for weight in "${weights[@]}"; do
  for task in "${val_tasks[@]}"; do
    index=$((index + 1))
    run_cell val "$task" "$weight" "$index"
  done
done

selected="$("$PYTHON_BIN" scripts/structnar/collect_vldb_raw_cosine_b50.py \
  --state "$STATE" --select-only)"
echo "[raw cosine B50] selected validation weight=$selected"
for task in "${test_tasks[@]}"; do
  index=$((index + 1))
  run_cell test "$task" "$selected" "$index"
done

"$PYTHON_BIN" scripts/structnar/collect_vldb_raw_cosine_b50.py --state "$STATE"
echo "RESULT=PASS_VLDB_RAW_COSINE_B50"
echo "RUN_ID=$RUN_ID"
