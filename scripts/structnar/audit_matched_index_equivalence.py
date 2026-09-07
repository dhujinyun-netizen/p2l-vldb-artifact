#!/usr/bin/env python3
"""Verify that matched Sequential and P2L runs use identical realized inputs.

The two policies write caches under separate experiment directories. Pickle
files need not be byte-identical because tensor storage metadata can differ, so
the Trie payload is also compared field by field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import torch


TASKS = (
    "visualnews_task0",
    "mscoco_task0",
    "fashion200k_task0",
    "webqa_task1",
    "edis_task2",
    "webqa_task2",
    "visualnews_task3",
    "mscoco_task3",
    "fashion200k_task3",
    "nights_task4",
    "oven_task6",
    "infoseek_task6",
    "fashioniq_task7",
    "cirr_task7",
    "oven_task8",
    "infoseek_task8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one match for {pattern!r} below {directory}, got {matches}"
        )
    return matches[0]


def load_trie(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"tree", "cand_token_ids", "cand_codes", "signature"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError(f"unexpected Trie payload in {path}: {type(payload)}")
    return payload


def trie_equal(left: dict, right: dict) -> bool:
    return (
        left["tree"] == right["tree"]
        and left["signature"] == right["signature"]
        and torch.equal(left["cand_token_ids"], right["cand_token_ids"])
        and torch.equal(left["cand_codes"], right["cand_codes"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("docs/results/structnar_matched_index_equivalence.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/results/structnar_matched_index_equivalence.csv"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "gen_code/STRUCTNAR/Large/Instruct"
    rows = []

    for task in TASKS:
        dirs = {
            policy: base / f"structnar_matched_lambda0_{policy}_{task}"
            for policy in ("sequential", "p2l")
        }
        candidate_codes = {
            policy: only(path / "cand_pool", f"*{task}*_cand_pool_codes.npy")
            for policy, path in dirs.items()
        }
        tries = {
            policy: only(path / "cand_pool", f"*{task}*_cand_pool_trie.pkl")
            for policy, path in dirs.items()
        }
        query_ids = {
            policy: only(path / "test", "*_ids.npy") for policy, path in dirs.items()
        }
        query_embeddings = {
            policy: only(path / "test", "*_embeddings.npy")
            for policy, path in dirs.items()
        }

        candidate_hashes = {key: sha256(value) for key, value in candidate_codes.items()}
        id_hashes = {key: sha256(value) for key, value in query_ids.items()}
        embedding_hashes = {
            key: sha256(value) for key, value in query_embeddings.items()
        }
        trie_payloads = {key: load_trie(value) for key, value in tries.items()}
        logical_trie_equal = trie_equal(
            trie_payloads["sequential"], trie_payloads["p2l"]
        )
        row = {
            "task": task,
            "candidate_codes_equal": candidate_hashes["sequential"]
            == candidate_hashes["p2l"],
            "query_ids_equal": id_hashes["sequential"] == id_hashes["p2l"],
            "query_embeddings_equal": embedding_hashes["sequential"]
            == embedding_hashes["p2l"],
            "logical_trie_equal": logical_trie_equal,
            "candidate_codes_sha256": candidate_hashes["sequential"],
            "query_ids_sha256": id_hashes["sequential"],
            "query_embeddings_sha256": embedding_hashes["sequential"],
            "trie_signature": repr(trie_payloads["sequential"]["signature"]),
            "sequential_candidate_codes": str(candidate_codes["sequential"].relative_to(root)),
            "p2l_candidate_codes": str(candidate_codes["p2l"].relative_to(root)),
            "sequential_trie": str(tries["sequential"].relative_to(root)),
            "p2l_trie": str(tries["p2l"].relative_to(root)),
        }
        row["passed"] = all(
            row[key]
            for key in (
                "candidate_codes_equal",
                "query_ids_equal",
                "query_embeddings_equal",
                "logical_trie_equal",
            )
        )
        rows.append(row)

    checkpoint = root / "checkpoint/code_tied/gpt_hdgr_latest.pth"
    output = {
        "audit": "matched Sequential/P2L realized-input equivalence",
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": sha256(checkpoint),
        "tasks_expected": len(TASKS),
        "tasks_checked": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "note": (
            "Trie pickle bytes may differ because tensor storage metadata is not a "
            "logical index property; tree dictionaries, signatures, candidate-token "
            "tensors, and candidate-code tensors are compared field by field."
        ),
        "tasks": rows,
    }

    json_out = root / args.json_out
    csv_out = root / args.csv_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not output["all_passed"]:
        raise SystemExit("matched index equivalence: FAIL")
    print(f"matched index equivalence: PASS ({len(rows)}/{len(TASKS)} tasks)")
    print(json_out)
    print(csv_out)


if __name__ == "__main__":
    main()
