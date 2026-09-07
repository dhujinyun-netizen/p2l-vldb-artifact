#!/usr/bin/env python
"""Build the optional pybind11 C++ Trie extension for GENIUS.

This creates:
  src/models/generative_retriever/trie_cpp<python-extension-suffix>.so

If pybind11 or a C++ compiler is missing, training can still proceed because
retriever.py falls back to the pure Python Trie implementation.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    cpp = here / "trie_cpp.cpp"
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = here / f"trie_cpp{suffix}"

    if out.exists() and out.stat().st_mtime >= cpp.stat().st_mtime:
        print(f"Trie C++ extension already built: {out}")
        return 0

    try:
        includes = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"],
            text=True,
        ).strip().split()
    except Exception as exc:
        print("WARNING: pybind11 is not available; skipping trie_cpp build.")
        print("Install with: pip install pybind11")
        print(f"Reason: {exc}")
        return 0

    cxx = os.environ.get("CXX", "c++")
    cmd = [
        cxx,
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *includes,
        str(cpp),
        "-o",
        str(out),
    ]

    print("Building trie_cpp extension:")
    print(" ".join(shlex.quote(x) for x in cmd))

    try:
        subprocess.check_call(cmd)
    except Exception as exc:
        print("WARNING: failed to build trie_cpp; falling back to Python Trie.")
        print(f"Reason: {exc}")
        return 0

    print(f"Built trie_cpp extension: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
