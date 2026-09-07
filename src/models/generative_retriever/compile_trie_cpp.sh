#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/build_trie_cpp.py"
python - << 'PY'
import models.generative_retriever.trie_cpp as trie_cpp
print("trie_cpp import OK:", trie_cpp)
PY
