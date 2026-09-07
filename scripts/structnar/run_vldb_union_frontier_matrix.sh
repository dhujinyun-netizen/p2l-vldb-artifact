#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/docs/results/vldb_union_frontier_${STAMP}"
CONFIG_ROOT="$ROOT/configs/structnar/generated/vldb_union_frontier_${STAMP}"
PROGRESS="$RUN_ROOT/progress.tsv"
mkdir -p "$RUN_ROOT" "$CONFIG_ROOT" "$ROOT/logs/vldb_union_frontier_${STAMP}"

export CUDA_VISIBLE_DEVICES=0
export MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$(readlink -f "$ROOT/mbeir_data")}"

TASKS=(nights_task4 edis_task2 webqa_task1)
DEPTHS=(3 9)
TOTAL=$(( ${#TASKS[@]} * ${#DEPTHS[@]} ))
RUN_ID=0

printf 'run\ttotal\tstatus\ttask\tdecoder\tstarted\tfinished\tlog\tprofile\n' > "$PROGRESS"

for task in "${TASKS[@]}"; do
  for depth in "${DEPTHS[@]}"; do
    RUN_ID=$((RUN_ID + 1))
    decoder=p2l
    [[ "$depth" == 9 ]] && decoder=sequential
    exp="structnar_vldb_union_${task}_${decoder}_w20_${STAMP}"
    local_cfg="$CONFIG_ROOT/${exp}_local.yaml"
    union_cfg="$CONFIG_ROOT/${exp}_union.yaml"
    log="$ROOT/logs/vldb_union_frontier_${STAMP}/${RUN_ID}_${task}_${decoder}.log"
    profile="$RUN_ROOT/${task}_${decoder}_per_query.npz"
    started="$(date --iso-8601=seconds)"

    "$PYTHON_BIN" scripts/structnar/make_all32_configs.py \
      --base configs/structnar/eval.yaml \
      --local-out "$local_cfg" \
      --union-out "$union_cfg" \
      --ckpt-dir checkpoint/code_tied \
      --ckpt-name gpt_hdgr_latest.pth \
      --exp-name "$exp" \
      --batch-size 8 \
      --num-beams 50 \
      --rrg-weight 20 \
      --rrg-normalize \
      --rqc-score-mode cosine \
      --rqc-apply-from-level 3 \
      --rqc-prefix-start-level 0 \
      --switch-depth "$depth" \
      --tcis-compact-head \
      --tcis-dynamic-intermediate-beams \
      --deterministic-eval-sampling \
      --datasets "$task" \
      --results-root "retrieval_results/vldb_union_frontier_${STAMP}"

    "$PYTHON_BIN" scripts/structnar/prepare_candidate_cache.py \
      --repo-root "$ROOT" \
      --exp-name "$exp" \
      --model-short-name STRUCTNAR \
      --link-existing \
      --build-union

    printf '%d\t%d\tRUNNING\t%s\t%s\t%s\t\t%s\t%s\n' \
      "$RUN_ID" "$TOTAL" "$task" "$decoder" "$started" "$log" "$profile" >> "$PROGRESS"
    printf '[VLDB UNION frontier] %d/%d task=%s decoder=%s start=%s\n' \
      "$RUN_ID" "$TOTAL" "$task" "$decoder" "$started" | tee -a "$log"

    STRUCTNAR_PROFILE_LATENCY=1 \
    STRUCTNAR_LATENCY_WARMUP_BATCHES=1 \
    STRUCTNAR_PROFILE_ALL_TCIS=1 \
    STRUCTNAR_PROFILE_QUERY_BUDGET=1 \
    STRUCTNAR_PROFILE_COMPONENTS=1 \
    STRUCTNAR_QUERY_PROFILE_PATH="$profile" \
    CONFIG_PATH="$union_cfg" \
    CHECK_EXTRACTED=0 \
    NPROC=1 \
    MASTER_PORT="$((4100 + RUN_ID))" \
      bash scripts/structnar/run_eval.py.sh 2>&1 | tee -a "$log"

    finished="$(date --iso-8601=seconds)"
    printf '%d\t%d\tDONE\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$RUN_ID" "$TOTAL" "$task" "$decoder" "$started" "$finished" "$log" "$profile" >> "$PROGRESS"
  done
done

printf 'RESULT=PASS_VLDB_UNION_FRONTIER_MATRIX\n' | tee -a "$RUN_ROOT/complete.ok"
printf 'progress=%s\n' "$PROGRESS"
