#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export NPROC=1
export PYTHON_BIN="${PYTHON_BIN:-python}"

BASE="$ROOT/configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml"
SCALE_CACHE="$ROOT/gen_code/STRUCTNAR_SCALE/cand_pool"
UNION_CACHE="$ROOT/gen_code/GPT_HDGR/Large/Instruct/OfficialMBEIR_Stage1FullCode_NoPS_RQ50_CodeTied_W10_ALL32_PROGeomHDGR_Beam50_Batch32/cand_pool"
RUN_ID="${VLDB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STATE_DIR="$ROOT/docs/results/vldb_scale/$RUN_ID"
ATTEMPT_DIR="$ROOT/logs/vldb_scale/$RUN_ID/incomplete_attempts"
mkdir -p "$STATE_DIR" "$ATTEMPT_DIR"

cell_is_complete() {
  local marker="$1"
  [[ -s "$marker" ]] || return 1
  local log expected actual
  log="$(sed -n 's/^log=//p' "$marker" | head -n1)"
  expected="$(sed -n 's/^sha256=//p' "$marker" | head -n1)"
  [[ -s "$log" && -n "$expected" ]] || return 1
  actual="$(sha256sum "$log" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || return 1
  rg -q 'Generation Metrics: seconds=' "$log"
  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Retrieval Metrics: seconds=' "$log"
  rg -q 'Candidate Index Setup: seconds=.* excluded_from_generation=true' "$log"
  # Index construction belongs to offline preparation.  The evaluator starts
  # its generation timer before lazy Trie loading, so a cache rebuild inside
  # an evaluation log contaminates the reported online total even when the
  # retrieval metrics themselves are valid.  Force one clean cached rerun.
  if rg -q 'candidate tree.*stale; rebuilding|Saved candidate tree from' "$log"; then
    return 1
  fi
}

run_one() {
  local decoder="$1"
  local pool="$2"
  local size="$3"
  local union_flag="$4"
  local port="$5"
  local exp="structnar_cirr_scale_w20_${decoder}_${size}"
  local config="$ROOT/configs/structnar/generated/${exp}.yaml"
  local target="$ROOT/gen_code/STRUCTNAR/Large/Instruct/${exp}/cand_pool"
  local log="$ROOT/logs/${exp}.log"
  local marker="$STATE_DIR/${decoder}_${size}.ok"
  if cell_is_complete "$marker"; then
    echo "[CIRR scale] skip verified decoder=$decoder candidates=$size"
    return 0
  fi
  if [[ -e "$log" ]]; then
    mv "$log" "$ATTEMPT_DIR/${decoder}_${size}_$(date +%Y%m%d_%H%M%S).log"
  fi
  mkdir -p "$target"

  config_args=(
    --base "$BASE"
    --output "$config"
    --pool "$pool"
    --exp-name "$exp"
    --decoder "$decoder"
    --weight 20
  )
  if [[ "$union_flag" == "1" ]]; then
    config_args+=(--union)
    source_prefix="$UNION_CACHE/mbeir_union_cand_pool"
    # The evaluator resolves UNION candidates using the canonical
    # ``mbeir_union_cand_pool_*`` stem, even though the source JSONL carries a
    # ``*_test`` suffix.  Keep the cache stem aligned with the evaluator.
    target_prefix="$target/mbeir_union_cand_pool"
  else
    source_prefix="$SCALE_CACHE/mbeir_${pool}_cand_pool"
    target_prefix="$target/mbeir_${pool}_cand_pool"
  fi
  "$PYTHON_BIN" scripts/structnar/make_cirr_scale_config.py "${config_args[@]}"

  for suffix in codes.npy ids.npy embeddings.npy; do
    ln -sfn "${source_prefix}_${suffix}" "${target_prefix}_${suffix}"
    if [[ ! -e "${target_prefix}_${suffix}" ]]; then
      echo "FATAL: candidate cache link is unresolved: ${target_prefix}_${suffix}" >&2
      return 2
    fi
  done
  # The evaluator only needs to build a Trie when the candidate bundle does
  # not already provide one.  Reuse the validated scale/UNION Trie whenever
  # available; this avoids rebuilding a 5.6M-key structure for each decoder.
  local source_trie="${source_prefix}_trie.pkl"
  local target_trie="${target_prefix}_trie.pkl"
  if [[ ! -e "$target_trie" && -e "$source_trie" ]]; then
    ln -s "$source_trie" "$target_trie"
  fi
  # The four nested scale caches predate persisted Trie payloads.  A Trie is
  # decoder-independent, so reuse a previously validated scale target when
  # available instead of rebuilding the same structure for the other policy.
  if [[ ! -e "$target_trie" ]]; then
    local reuse_decoder reuse_exp reuse_trie
    for reuse_decoder in p2l sequential; do
      reuse_exp="$ROOT/gen_code/STRUCTNAR/Large/Instruct/structnar_cirr_scale_w20_${reuse_decoder}_${size}/cand_pool"
      reuse_trie="$reuse_exp/$(basename "$target_prefix")_trie.pkl"
      if [[ -e "$reuse_trie" && "$reuse_trie" != "$target_trie" ]]; then
        ln -s "$reuse_trie" "$target_trie"
        echo "[CIRR scale] reused decoder-independent Trie from $reuse_trie"
        break
      fi
    done
  fi
  STRUCTNAR_PROFILE_LATENCY=1 STRUCTNAR_LATENCY_WARMUP_BATCHES=1 \
    STRUCTNAR_PROFILE_ALL_TCIS=1 STRUCTNAR_PROFILE_QUERY_BUDGET=1 \
    CONFIG_PATH="$config" CHECK_EXTRACTED=0 MASTER_PORT="$port" \
    MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$(readlink -f mbeir_data)}" \
    bash scripts/structnar/run_eval.py.sh 2>&1 | tee "$log"

  rg -q 'Generation Metrics: seconds=' "$log"
  rg -q 'Retriever: Mean Recall@10:' "$log"
  rg -q 'Retrieval Metrics: seconds=' "$log"
  if [[ "$decoder" == "p2l" ]]; then
    rg -q 'TCIS Query Enumeration:' "$log"
  fi
  {
    printf 'decoder=%s\n' "$decoder"
    printf 'candidate_count=%s\n' "$size"
    printf 'log=%s\n' "$log"
    printf 'sha256=%s\n' "$(sha256sum "$log" | awk '{print $1}')"
    printf 'config=%s\n' "$config"
    printf 'config_sha256=%s\n' "$(sha256sum "$config" | awk '{print $1}')"
    printf 'completed=%s\n' "$(date --iso-8601=seconds)"
  } > "$marker"
  echo "[CIRR scale] verified decoder=$decoder candidates=$size"
}

for decoder in p2l sequential; do
  if [[ "${ONLY_FULL_UNION:-0}" != "1" ]]; then
    run_one "$decoder" cirrscale_0021551 0021551 0 3911
    run_one "$decoder" cirrscale_0100000 0100000 0 3912
    run_one "$decoder" cirrscale_0500000 0500000 0 3913
    run_one "$decoder" cirrscale_1000000 1000000 0 3914
  fi
  run_one "$decoder" union 5609079 1 3915
done

echo "__CIRR_SCALE_CURVE_COMPLETE__"
echo "RUN_ID=$RUN_ID"
