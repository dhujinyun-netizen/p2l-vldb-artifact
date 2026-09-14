#!/usr/bin/env python3
"""No-data audit for the 2026-09-14 P2L Frozen-12 selection control."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "docs/results/fixed_score_eval12_20260914.csv"
SUMMARY = ROOT / "docs/results/fixed_score_eval12_20260914_summary.json"

def close(a: float, b: float, tol: float = 5e-4) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tol={tol})")

rows = list(csv.DictReader(CSV.open(newline="")))
summary = json.load(SUMMARY.open())
assert len(rows) == 12
assert sum(int(r["queries"]) for r in rows) == int(summary["queries_total"]) == 178587

lw_r10 = sum(float(r["frozen_lw_r10"]) for r in rows) / len(rows)
p2l_r10 = sum(float(r["frozen_p2l_r10"]) for r in rows) / len(rows)
lw_idr = sum(float(r["frozen_lw_idr50"]) for r in rows) / len(rows)
p2l_idr = sum(float(r["frozen_p2l_idr50"]) for r in rows) / len(rows)
apf = sum(float(r["apf"]) for r in rows) / len(rows)

close(lw_r10, float(summary["macro_r10_frozen_lw"]))
close(p2l_r10, float(summary["macro_r10_frozen_p2l"]))
close(100 * (p2l_r10 - lw_r10), float(summary["macro_delta_r10_pp"]), 0.015)
close(lw_idr, float(summary["macro_idr50_frozen_lw"]))
close(p2l_idr, float(summary["macro_idr50_frozen_p2l"]))
close(100 * (p2l_idr - lw_idr), float(summary["macro_delta_idr50_pp"]), 0.015)
close(apf, float(summary["macro_apf"]), 5e-4)

positive = sum(float(r["delta_r10_pp"]) > 0 for r in rows)
ci_excludes = sum(float(r["ci_low_pp"]) > 0 or float(r["ci_high_pp"]) < 0 for r in rows)
assert positive == 12
assert ci_excludes == 11
assert summary["positive_point_estimates"] == "12/12"
assert summary["task_ci_excludes_zero"] == "11/12"
assert float(summary["macro_delta_r10_ci_pp"][0]) == 3.42
assert float(summary["macro_delta_r10_ci_pp"][1]) == 3.86
assert int(summary["max_frontier_observed"]) == 78585
close(float(summary["p2l_generation_ms_q"]), 23.72, 1e-6)
close(float(summary["frozen_lw_generation_ms_q"]), 56.96, 1e-6)

print("Frozen-12 audit: PASS")
print(f"tasks={len(rows)}, queries={summary['queries_total']}")
print(f"macro R@10: {100*lw_r10:.2f} -> {100*p2l_r10:.2f} (+{100*(p2l_r10-lw_r10):.2f} pp)")
print(f"macro IDR@50: {100*lw_idr:.2f} -> {100*p2l_idr:.2f} (+{100*(p2l_idr-lw_idr):.2f} pp)")
print(f"positive tasks={positive}/12, task CIs excluding zero={ci_excludes}/12, APF={100*apf:.2f}%")
