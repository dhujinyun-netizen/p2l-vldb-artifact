#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT/configs/structnar/generated/structnar_vldb_gpu_flat_seq_cirr_smoke.yaml"
OUT="$ROOT/profile_output/p2l_gpu_trie_baseline"
LOG="$OUT/cirr_local_full.log"
STATE="$OUT/state.txt"
mkdir -p "$OUT"

external_compute_pid() {
  local allowed_pgid="${1:-0}"
  local allowed_root_pid="${2:-0}"
  local pid name pgid cursor
  while read -r pid; do
    pid="$(printf '%s' "$pid" | tr -d ' ')"
    [[ -n "$pid" ]] || continue
    name="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ')"
    case "$name" in
      Xorg|firefox|gnome-shell|gnome-control-center|awesun_desktop|iKuuuVPN|PyCharm|code)
        continue
        ;;
    esac
    pgid="$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')"
    if [[ "$allowed_pgid" != "0" && "$pgid" == "$allowed_pgid" ]]; then
      continue
    fi
    cursor="$pid"
    while [[ "$allowed_root_pid" != "0" && "$cursor" =~ ^[0-9]+$ && "$cursor" -gt 1 ]]; do
      if [[ "$cursor" == "$allowed_root_pid" ]]; then
        continue 2
      fi
      cursor="$(ps -p "$cursor" -o ppid= 2>/dev/null | tr -d ' ')"
    done
    printf '%s\n' "$pid"
    return 0
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i 0 2>/dev/null || true)
  return 1
}

stable=0
while (( stable < 3 )); do
  blocker="$(external_compute_pid || true)"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
  if [[ -z "$blocker" && -n "$used" && "$used" -lt 2500 ]]; then
    stable=$((stable + 1))
    printf 'status=waiting_for_stable_idle check=%d/3 memory_mib=%s updated=%s\n' \
      "$stable" "$used" "$(date --iso-8601=seconds)" > "$STATE"
  else
    stable=0
    printf 'status=waiting_for_gpu0 blocker=%s memory_mib=%s updated=%s\n' \
      "${blocker:-none}" "${used:-unknown}" "$(date --iso-8601=seconds)" > "$STATE"
  fi
  sleep 20
done

printf 'status=running updated=%s\n' "$(date --iso-8601=seconds)" > "$STATE"
{
  echo "GPU-flat sequential Trie baseline"
  echo "started=$(date --iso-8601=seconds)"
  echo "config=$CONFIG"
  echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
} > "$LOG"
setsid env \
  STRUCTNAR_PROFILE_LATENCY=1 \
  STRUCTNAR_LATENCY_WARMUP_BATCHES=1 \
  CONFIG_PATH="$CONFIG" CHECK_EXTRACTED=0 MASTER_PORT=3973 \
  MBEIR_DATA_DIR="${MBEIR_DATA_DIR:-$(readlink -f mbeir_data)}" \
  CUDA_VISIBLE_DEVICES=0 NPROC=1 \
  bash scripts/structnar/run_eval.py.sh >> "$LOG" 2>&1 &
child=$!
pgid="$(ps -p "$child" -o pgid= | tr -d ' ')"
[[ -n "$pgid" ]] || pgid="$child"
contaminated=0
while kill -0 "$child" 2>/dev/null; do
  blocker="$(external_compute_pid "$pgid" "$child" || true)"
  if [[ -n "$blocker" ]]; then
    contaminated=1
    printf 'status=contaminated blocker=%s updated=%s\n' \
      "$blocker" "$(date --iso-8601=seconds)" > "$STATE"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    break
  fi
  printf 'status=running pid=%s updated=%s\n' \
    "$child" "$(date --iso-8601=seconds)" > "$STATE"
  sleep 5
done
set +e
wait "$child"
rc=$?
set -e
tail -40 "$LOG" || true
if (( contaminated )); then
  printf 'status=invalid_due_to_contention blocker=%s updated=%s\n' \
    "$blocker" "$(date --iso-8601=seconds)" > "$STATE"
  exit 3
fi
if (( rc != 0 )); then
  printf 'status=failed rc=%s updated=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATE"
  exit "$rc"
fi

rg -q 'Generation Metrics: seconds=' "$LOG"
rg -q 'Retriever: Mean Recall@10:' "$LOG"
printf 'status=auditing_full_manifest updated=%s\n' \
  "$(date --iso-8601=seconds)" > "$STATE"
if ! "$PYTHON_BIN" scripts/structnar/audit_gpu_flat_trie_equivalence.py >> "$LOG" 2>&1; then
  printf 'status=failed_full_manifest_equivalence updated=%s\n' \
    "$(date --iso-8601=seconds)" > "$STATE"
  tail -80 "$LOG" || true
  exit 4
fi
printf 'status=complete_exact log_sha256=%s equivalence_sha256=%s updated=%s\n' \
  "$(sha256sum "$LOG" | awk '{print $1}')" \
  "$(sha256sum "$OUT/equivalence.json" | awk '{print $1}')" \
  "$(date --iso-8601=seconds)" > "$STATE"
if ! "$PYTHON_BIN" scripts/structnar/collect_gpu_flat_trie_baseline.py >> "$LOG" 2>&1; then
  printf 'status=failed_result_collection updated=%s\n' \
    "$(date --iso-8601=seconds)" > "$STATE"
  tail -80 "$LOG" || true
  exit 5
fi
printf 'status=complete_exact log_sha256=%s equivalence_sha256=%s result_sha256=%s updated=%s\n' \
  "$(sha256sum "$LOG" | awk '{print $1}')" \
  "$(sha256sum "$OUT/equivalence.json" | awk '{print $1}')" \
  "$(sha256sum docs/results/p2l_gpu_flat_trie_baseline.json | awk '{print $1}')" \
  "$(date --iso-8601=seconds)" > "$STATE"
