#!/usr/bin/env python3
"""Validate that a staged artifact is suitable for public release.

The check is intentionally conservative: it rejects symlinks, private local
paths, and unexpectedly large files. Dataset/checkpoint availability is
documented in the README rather than silently copied into the public archive.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PRIVATE_MARKERS = (
    "/home/",
    "/media/",
)
TEXT_SUFFIXES = {
    ".bash", ".bib", ".cfg", ".csv", ".json", ".md", ".py", ".sh",
    ".tex", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
REQUIRED = (
    "README.md",
    "README_CN.md",
    "LICENSE",
    "genius_env.yml",
    "RELEASE_SHA256SUMS.txt",
    "scripts/structnar/audit_release_evidence.py",
    "scripts/structnar/audit_public_release.py",
    "src",
    "configs",
    "tests",
    "VLDB/vldb2027/structnar_vldb.tex",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    missing = [item for item in REQUIRED if not (root / item).exists()]
    if missing:
        raise SystemExit(f"missing required artifact paths: {missing}")

    violations: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            violations.append(f"symlink: {path.relative_to(root)}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > 200 * 1024 * 1024:
            violations.append(
                f"large file ({size / (1024 ** 2):.1f} MiB): {path.relative_to(root)}"
            )
        if (
            path.name != "audit_public_release.py"
            and path.suffix.lower() in TEXT_SUFFIXES
            and size <= 20 * 1024 * 1024
        ):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    violations.append(f"private path marker {marker!r}: {path.relative_to(root)}")

    if violations:
        raise SystemExit("public artifact audit failed:\n" + "\n".join(violations))
    print("PUBLIC_ARTIFACT_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
