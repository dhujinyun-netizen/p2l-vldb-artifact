#!/usr/bin/env python3
"""Patch the canonical TCIS full-materialization selector to use fixed tie order.

The streaming selector in ``complete_id_selection.py`` uses the lexicographic
rule ``(-score, global_candidate_row)``.  The historical full TCIS path used
``torch.topk`` directly, whose behavior inside exact-score ties is unspecified.
For an exact full-vs-streaming audit both paths must use the same secondary key.

This script is intentionally small and idempotent.  It edits only the import and
the final complete-frontier top-k call in ``retriever_gpt_hdgr.py`` and fails if
the expected frozen source text is not found exactly once.
"""
from __future__ import annotations

import argparse
from pathlib import Path


OLD_IMPORT = (
    "from models.hdgr_comparison.complete_id_selection import "
    "topk_complete_id_scores\n"
)
NEW_IMPORT = (
    "from models.hdgr_comparison.complete_id_selection import (\n"
    "    deterministic_topk_by_index,\n"
    "    topk_complete_id_scores,\n"
    ")\n"
)

OLD_TOPK = """            top_n = min(keep_k, all_scores.numel())
            top_scores, top_idx = torch.topk(all_scores, k=top_n, dim=0)
            top_seqs = all_seqs.index_select(0, top_idx)
"""
NEW_TOPK = """            top_n = min(keep_k, all_scores.numel())
            candidate_rows = torch.arange(
                all_scores.numel(), device=all_scores.device, dtype=torch.long
            )
            top_scores, top_idx = deterministic_topk_by_index(
                all_scores, candidate_rows, top_n
            )
            top_seqs = all_seqs.index_select(0, top_idx)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0 and new in text:
        return text
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} block, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="src/models/hdgr_comparison/retriever_gpt_hdgr.py",
        help="path to retriever_gpt_hdgr.py",
    )
    parser.add_argument("--check", action="store_true", help="verify patch state without writing")
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text()
    patched = replace_once(text, OLD_IMPORT, NEW_IMPORT, "complete-id import")
    patched = replace_once(patched, OLD_TOPK, NEW_TOPK, "canonical TCIS top-k")

    if args.check:
        if patched != text:
            raise SystemExit("deterministic TCIS top-k patch is not yet applied")
        print("deterministic TCIS top-k patch: OK")
        return

    if patched == text:
        print("deterministic TCIS top-k patch already applied")
        return
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
