#!/usr/bin/env bash
set -euo pipefail

# Link official M-BEIR data into the project without copying hundreds of GB.
# Usage:
#   bash scripts/shared/setup_official_mbeir_symlink.sh /path/to/M-BEIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/M-BEIR" >&2
  exit 2
fi
MBEIR_SRC="$1"

if [ ! -d "$MBEIR_SRC" ]; then
  echo "[setup] ERROR: M-BEIR source directory not found: $MBEIR_SRC" >&2
  exit 2
fi
for d in query cand_pool qrels instructions mbeir_images; do
  if [ ! -e "$MBEIR_SRC/$d" ]; then
    echo "[setup] ERROR: missing $MBEIR_SRC/$d" >&2
    exit 2
  fi
done

cd "$REPO_ROOT"
if [ -L mbeir_data ]; then
  rm mbeir_data
elif [ -e mbeir_data ]; then
  backup="mbeir_data_backup_$(date +%Y%m%d_%H%M%S)"
  echo "[setup] Backing up existing mbeir_data -> $backup"
  mv mbeir_data "$backup"
fi
ln -s "$MBEIR_SRC" mbeir_data

echo "[setup] linked: $REPO_ROOT/mbeir_data -> $MBEIR_SRC"
ls -l mbeir_data
ls mbeir_data/query
ls mbeir_data/cand_pool
ls mbeir_data/qrels
ls mbeir_data/instructions
ls mbeir_data/mbeir_images | head
