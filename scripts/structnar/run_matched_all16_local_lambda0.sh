#!/usr/bin/env bash
set -euo pipefail

# Matched all-task decoder intervention for the VLDB evaluation. Every cell
# uses the same frozen StructNAR checkpoint, deterministic query instances,
# RQ50 candidate caches, batch size, beam width, and reranker. PCAA is
# disabled. The only changed variable is the inference policy: P2L retains a
# short sequential prefix and scores every unresolved position before the next
# pruning decision, whereas Sequential repeats masked next-position scoring
# through all nine identifier positions.
#
# The 32 cells are intentionally run one task at a time. Each completed cell
# receives a checksum-bearing marker, so an external GPU interruption can
# invalidate and retry only the active cell instead of discarding many hours of
# clean results.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export NPROC=1
export PYTHON_BIN="${PYTHON_BIN:-python}"
export MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$(readlink -f "$ROOT/mbeir_data")}"

BASE="$ROOT/configs/structnar/eval.yaml"
CKPT_DIR="checkpoint/code_tied"
CKPT_NAME="gpt_hdgr_latest.pth"
RUN_ID="${VLDB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT/logs/vldb_matched/$RUN_ID"
STATE_DIR="$ROOT/docs/results/vldb_matched/$RUN_ID"
CONFIG_DIR="$ROOT/configs/structnar/generated/vldb_matched_$RUN_ID"
mkdir -p "$RUN_DIR" "$STATE_DIR" "$CONFIG_DIR"

TASKS=(
  visualnews_task0 mscoco_task0 fashion200k_task0 webqa_task1
  edis_task2 webqa_task2 visualnews_task3 mscoco_task3
  fashion200k_task3 nights_task4 oven_task6 infoseek_task6
  fashioniq_task7 cirr_task7 oven_task8 infoseek_task8
)

cell_is_complete() {
  local marker="$1"
  [[ -s "$marker" ]] || return 1
  local log expected actual
  log="$(sed -n 's/^log=//p' "$marker" | head -n1)"
  expected="$(sed -n 's/^sha256=//p' "$marker" | head -n1)"
  [[ -s "$log" && -n "$expected" ]] || return 1
  actual="$(sha256sum "$log" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || return 1
  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Retrieval Metrics: seconds=' "$log"
}

run_cell() {
  local decoder="$1"
  local depth="$2"
  local task="$3"
  local index="$4"
  local exp="structnar_matched_lambda0_${decoder}_${task}"
  local cfg="$CONFIG_DIR/${exp}.yaml"
  local dummy_union="$CONFIG_DIR/${exp}_union.yaml"
  local log="$RUN_DIR/${decoder}_${task}.log"
  local marker="$STATE_DIR/${decoder}_${task}.ok"

  if cell_is_complete "$marker"; then
    printf '[matched all16] skip verified cell %02d/32 decoder=%s task=%s\n' \
      "$index" "$decoder" "$task"
    return 0
  fi

  if [[ -e "$log" ]]; then
    mkdir -p "$RUN_DIR/incomplete_attempts"
    mv "$log" "$RUN_DIR/incomplete_attempts/${decoder}_${task}_$(date +%Y%m%d_%H%M%S).log"
  fi

  "$PYTHON_BIN" scripts/structnar/make_all32_configs.py \
    --base "$BASE" \
    --local-out "$cfg" \
    --union-out "$dummy_union" \
    --ckpt-dir "$CKPT_DIR" \
    --ckpt-name "$CKPT_NAME" \
    --exp-name "$exp" \
    --batch-size 8 \
    --num-beams 50 \
    --rrg-weight 0 \
    --rrg-normalize \
    --rqc-score-mode cosine \
    --rqc-apply-from-level 3 \
    --rqc-prefix-start-level 0 \
    --switch-depth "$depth" \
    --tcis-compact-head \
    --tcis-dynamic-intermediate-beams \
    --deterministic-eval-sampling \
    --results-root "retrieval_results/vldb_matched_lambda0" \
    --datasets "$task"

  "$PYTHON_BIN" - "$cfg" <<'PY'
from pathlib import Path
import sys, yaml
p = Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text())
cfg["retrieval_config"]["load_saved_beam_scores"] = False
cfg["retrieval_config"]["save_beam_scores"] = True
cfg["model"]["rrg_weight"] = 0.0
cfg["model"]["use_rrg"] = False
cfg["data_config"]["deterministic_eval_sampling"] = True
p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY

  "$PYTHON_BIN" scripts/structnar/prepare_candidate_cache.py \
    --repo-root "$ROOT" \
    --exp-name "$exp" \
    --model-short-name STRUCTNAR \
    --link-existing

  printf '[matched all16] start %02d/32 decoder=%s task=%s time=%s\n' \
    "$index" "$decoder" "$task" "$(date --iso-8601=seconds)"
  CONFIG_PATH="$cfg" CHECK_EXTRACTED=0 MASTER_PORT="$((4100 + index))" \
    bash scripts/structnar/run_eval.py.sh 2>&1 | tee "$log"

  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Retrieval Metrics: seconds=' "$log"
  {
    printf 'decoder=%s\n' "$decoder"
    printf 'task=%s\n' "$task"
    printf 'log=%s\n' "$log"
    printf 'sha256=%s\n' "$(sha256sum "$log" | awk '{print $1}')"
    printf 'config=%s\n' "$cfg"
    printf 'config_sha256=%s\n' "$(sha256sum "$cfg" | awk '{print $1}')"
    printf 'completed=%s\n' "$(date --iso-8601=seconds)"
  } > "$marker"
  printf '[matched all16] complete %02d/32 decoder=%s task=%s\n' \
    "$index" "$decoder" "$task"
}

index=0
for decoder in p2l sequential; do
  if [[ "$decoder" == "p2l" ]]; then
    depth=3
  else
    depth=9
  fi
  for task in "${TASKS[@]}"; do
    index=$((index + 1))
    run_cell "$decoder" "$depth" "$task" "$index"
  done
done

P2L_LOG="$RUN_DIR/p2l_all16_aggregate.log"
SEQ_LOG="$RUN_DIR/sequential_all16_aggregate.log"
truncate -s 0 "$P2L_LOG"
truncate -s 0 "$SEQ_LOG"
for task in "${TASKS[@]}"; do
  cell_is_complete "$STATE_DIR/p2l_${task}.ok"
  cell_is_complete "$STATE_DIR/sequential_${task}.ok"
  awk '1' "$RUN_DIR/p2l_${task}.log" >> "$P2L_LOG"
  awk '1' "$RUN_DIR/sequential_${task}.log" >> "$SEQ_LOG"
done

"$PYTHON_BIN" scripts/structnar/summarize_matched_all16.py \
  --p2l-log "$P2L_LOG" \
  --sequential-log "$SEQ_LOG" \
  --output "$ROOT/docs/results/structnar_matched_all16_local_lambda0.csv"

"$PYTHON_BIN" scripts/structnar/prepare_vldb_matched_all16_artifacts.py

printf '%s\n' "$RUN_ID" > "$ROOT/docs/results/structnar_matched_all16_local_lambda0_run_id.txt"
echo "RESULT=PASS_MATCHED_ALL16_LOCAL_LAMBDA0"
echo "RUN_ID=$RUN_ID"
echo "P2L_LOG=$P2L_LOG"
echo "SEQUENTIAL_LOG=$SEQ_LOG"
