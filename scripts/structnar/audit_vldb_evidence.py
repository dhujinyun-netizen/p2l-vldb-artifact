#!/usr/bin/env python3
"""Deterministic provenance and configuration audit for final VLDB evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/results"
AUDITS = ROOT / "VLDB/vldb2027/audits"
TASKS = (
    "visualnews_task0", "mscoco_task0", "fashion200k_task0", "webqa_task1",
    "edis_task2", "webqa_task2", "visualnews_task3", "mscoco_task3",
    "fashion200k_task3", "nights_task4", "oven_task6", "infoseek_task6",
    "fashioniq_task7", "cirr_task7", "oven_task8", "infoseek_task8",
)
SIZES = (21_551, 100_000, 500_000, 1_000_000, 5_609_079)
RQ_WEIGHTS = (
    0, 1, 2, 5, 10, 20, 40, 80, 160,
    320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920,
)
RQ_VAL_TASKS = ("cirr_task7", "nights_task4")
RQ_TEST_TASKS = ("cirr_task7", "nights_task4", "edis_task2", "webqa_task1")
RAW_COSINE_WEIGHTS = (0, 5, 10, 15, 20, 30, 40)
RAW_COSINE_VAL_TASKS = ("cirr_task7", "nights_task4")
RAW_COSINE_TEST_TASKS = ("cirr_task7", "nights_task4", "edis_task2", "webqa_task1")
FASHION_TASKS = {"fashion200k_task0", "fashion200k_task3", "fashioniq_task7"}
TASK_DISPLAY = {
    "visualnews_task0": "VisualNews-0",
    "mscoco_task0": "MSCOCO-0",
    "fashion200k_task0": "Fashion200K-0",
    "webqa_task1": "WebQA-1",
    "edis_task2": "EDIS-2",
    "webqa_task2": "WebQA-2",
    "visualnews_task3": "VisualNews-3",
    "mscoco_task3": "MSCOCO-3",
    "fashion200k_task3": "Fashion200K-3",
    "nights_task4": "NIGHTS-4",
    "oven_task6": "OVEN-6",
    "infoseek_task6": "InfoSeek-6",
    "fashioniq_task7": "FashionIQ-7",
    "cirr_task7": "CIRR-7",
    "oven_task8": "OVEN-8",
    "infoseek_task8": "InfoSeek-8",
}


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AssertionError("missing or empty generated artifact: " + str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def witness(marker: Path) -> tuple[Path, Path, dict[str, str]]:
    if not marker.is_file():
        raise AssertionError("missing marker: " + str(marker))
    fields = dict(
        line.split("=", 1)
        for line in marker.read_text().splitlines()
        if "=" in line
    )
    log, config = Path(fields["log"]), Path(fields["config"])
    if not log.is_file() or not config.is_file():
        raise AssertionError("missing witness target: " + str(marker))
    if sha256(log) != fields.get("sha256"):
        raise AssertionError("log hash mismatch: " + str(marker))
    if sha256(config) != fields.get("config_sha256"):
        raise AssertionError("config hash mismatch: " + str(marker))
    return log, config, fields


def require_eval_log(log: Path, frontier: bool = False, split: str = "test") -> None:
    text = log.read_text(errors="replace")
    required = (
        "Generation Metrics: seconds=",
        "Retriever: Mean Recall@10:",
        "Retrieval Metrics: seconds=",
        "/qrels/" + split + "/",
    )
    if not all(token in text for token in required):
        raise AssertionError("incomplete or non-official log: " + str(log))
    if frontier and "TCIS Query Enumeration:" not in text:
        raise AssertionError("missing frontier profile: " + str(log))


def effective_prefix_width(model: dict, retrieval: dict) -> int:
    explicit = int(model.get("tcis_prefix_beams", 0))
    if explicit > 0:
        return explicit
    return int(retrieval["num_beams"]) * int(
        model.get("tcis_intermediate_beam_multiplier", 1)
    )


def audit_matched(run_id: str) -> dict:
    state = RESULTS / "vldb_matched" / run_id
    cell_metrics: dict[tuple[str, str], tuple[float, float, int]] = {}
    for decoder in ("p2l", "sequential"):
        for task in TASKS:
            log, config, fields = witness(state / (decoder + "_" + task + ".ok"))
            cfg = yaml.safe_load(config.read_text())
            model, retrieval = cfg["model"], cfg["retrieval_config"]
            test = retrieval["test_datasets_config"]
            expected_blocks = [1, 1, 1, 6] if decoder == "p2l" else [1] * 9
            checks = (
                test["datasets_name"] == [task],
                cfg["dataloader_config"]["batch_size"] == 8,
                cfg["data_config"]["deterministic_eval_sampling"] is True,
                retrieval["num_beams"] == 50,
                retrieval["rerank"] is True,
                retrieval["use_fp16"] is True,
                retrieval["load_saved_beam_scores"] is False,
                int(model["time_step"]) == 8,
                math.isclose(float(model["guidance_scale"]), 3.0),
                model["ckpt_config"]["ckpt_dir"] == "checkpoint/code_tied",
                model["ckpt_config"]["ckpt_name"] == "gpt_hdgr_latest.pth",
                float(model["rrg_weight"]) == 0.0,
                model["use_rrg"] is False,
                model["hierarchy_block_sizes"] == expected_blocks,
                fields.get("decoder") == decoder,
                fields.get("task") == task,
            )
            if not all(checks):
                raise AssertionError("matched configuration mismatch: " + str(config))
            require_eval_log(log)
            text = log.read_text(errors="replace")
            recalls = re.findall(r"Retriever: Mean Recall@10: ([0-9.]+)", text)
            generation = re.findall(
                r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
            )
            if not recalls or not generation:
                raise AssertionError("missing matched cell metric: " + str(log))
            seconds, examples = generation[-1]
            cell_metrics[(decoder, task)] = (
                float(recalls[-1]), float(seconds), int(examples)
            )
    csv_path = RESULTS / "structnar_matched_all16_local_lambda0.csv"
    summary_path = RESULTS / "structnar_matched_all16_local_lambda0_summary.json"
    rows = list(csv.DictReader(csv_path.open(newline="")))
    expected = {(d, t) for d in ("p2l", "sequential") for t in TASKS}
    observed = {(row["decoder"].lower(), row["task"]) for row in rows}
    if len(rows) != 32 or observed != expected:
        raise AssertionError("matched 32-cell CSV is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary["task_count"] != 16 or summary["row_count"] != 32 or summary["lambda"] != 0.0:
        raise AssertionError("matched summary scope mismatch")
    expected_timing_scope = (
        "recorded evaluator-stage manifest wall time; includes first-use Trie "
        "validation and any required rebuild, serialization, and loading"
    )
    if summary.get("generation", {}).get("timing_scope") != expected_timing_scope:
        raise AssertionError("all-task manifest timing scope is missing or stale")
    by_key = {(row["decoder"].lower(), row["task"]): row for row in rows}
    run_dir = ROOT / "logs/vldb_matched" / run_id
    for decoder in ("p2l", "sequential"):
        aggregate = run_dir / f"{decoder}_all16_aggregate.log"
        require_nonempty(aggregate)
        aggregate_hash = sha256(aggregate)
        aggregate_text = aggregate.read_text(errors="replace")
        # Evaluator logs include whitespace before the field separator
        # (``query:<task> | split:...``).  Match the structured delimiter
        # instead of relying on a whitespace-sensitive literal count.
        if any(
            len(
                re.findall(
                    rf"Retriever: Retrieving for query:{re.escape(task)}\s*\|",
                    aggregate_text,
                )
            )
            != 1
            for task in TASKS
        ):
            raise AssertionError("matched aggregate task multiplicity mismatch: " + str(aggregate))
        for task in TASKS:
            row = by_key[(decoder, task)]
            source_log = Path(row["source_log"])
            if source_log.resolve() != aggregate.resolve():
                raise AssertionError("matched CSV aggregate path mismatch")
            if row["source_log_sha256"] != aggregate_hash:
                raise AssertionError("matched CSV aggregate hash mismatch")
            recall, seconds, examples = cell_metrics[(decoder, task)]
            if not (
                math.isclose(float(row["Recall@10"]), recall, abs_tol=1e-12)
                and math.isclose(float(row["generation_seconds"]), seconds, abs_tol=1e-9)
                and int(row["generation_examples"]) == examples
            ):
                raise AssertionError(f"matched CSV/cell-log mismatch: {decoder}, {task}")

    def as_percent(value: str) -> float:
        number = float(value)
        return number * 100.0 if abs(number) <= 1.0 else number

    pairs = []
    for task in TASKS:
        seq = by_key[("sequential", task)]
        p2l = by_key[("p2l", task)]
        seq_r10 = as_percent(seq["Recall@10"])
        p2l_r10 = as_percent(p2l["Recall@10"])
        seq_ms = float(seq["generation_ms_per_query"])
        p2l_ms = float(p2l["generation_ms_per_query"])
        pairs.append((p2l_r10 - seq_r10, seq_ms / p2l_ms))
    macro_seq = sum(as_percent(by_key[("sequential", task)]["Recall@10"]) for task in TASKS) / 16
    macro_p2l = sum(as_percent(by_key[("p2l", task)]["Recall@10"]) for task in TASKS) / 16
    seq_seconds = sum(float(by_key[("sequential", task)]["generation_seconds"]) for task in TASKS)
    p2l_seconds = sum(float(by_key[("p2l", task)]["generation_seconds"]) for task in TASKS)
    seq_queries = sum(int(by_key[("sequential", task)]["generation_examples"]) for task in TASKS)
    p2l_queries = sum(int(by_key[("p2l", task)]["generation_examples"]) for task in TASKS)
    numerical_checks = (
        abs(float(summary["macro_r10"]["sequential"]) - macro_seq) < 1e-8,
        abs(float(summary["macro_r10"]["p2l"]) - macro_p2l) < 1e-8,
        abs(float(summary["macro_r10"]["delta"]) - (macro_p2l - macro_seq)) < 1e-8,
        int(summary["positive_delta_tasks"]) == sum(delta > 0 for delta, _ in pairs),
        int(summary["faster_tasks"]) == sum(speedup > 1 for _, speedup in pairs),
        abs(float(summary["generation"]["sequential_seconds"]) - seq_seconds) < 1e-8,
        abs(float(summary["generation"]["p2l_seconds"]) - p2l_seconds) < 1e-8,
        int(summary["generation"]["queries"]) == seq_queries == p2l_queries,
        abs(float(summary["generation"]["reduction_percent"]) - 100.0 * (1.0 - p2l_seconds / seq_seconds)) < 1e-8,
        abs(float(summary["generation"]["aggregate_speedup"]) - seq_seconds / p2l_seconds) < 1e-8,
    )
    if not all(numerical_checks):
        raise AssertionError("matched summary does not reproduce the 32-cell CSV")
    delta_range = summary.get("delta_r10_range", {})
    speedup_range = summary.get("generation_speedup_range", {})
    if (
        not math.isclose(float(delta_range.get("min", "nan")), min(value for value, _ in pairs), abs_tol=1e-8)
        or not math.isclose(float(delta_range.get("max", "nan")), max(value for value, _ in pairs), abs_tol=1e-8)
        or not math.isclose(float(speedup_range.get("min", "nan")), min(value for _, value in pairs), abs_tol=1e-8)
        or not math.isclose(float(speedup_range.get("max", "nan")), max(value for _, value in pairs), abs_tol=1e-8)
    ):
        raise AssertionError("matched summary range mismatch")
    matched_figure = ROOT / "VLDB/vldb2027/figures/matched_all16_local_lambda0.pdf"
    matched_table = ROOT / "VLDB/vldb2027/supplementary/generated/matched_all16_local_lambda0.tex"
    require_nonempty(matched_figure)
    require_nonempty(matched_table)
    main_text = (ROOT / "VLDB/vldb2027/structnar_vldb.tex").read_text()
    table_text = matched_table.read_text()
    if (
        "recorded manifest-stage speedup" in main_text
        or "not used as the warm online-latency headline" not in table_text
        or "aggregate decoder-side generation time falls by 50.8" in main_text
    ):
        raise AssertionError("all-task timing boundary is misstated in the manuscript")
    return {
        "cells": 32,
        "csv": str(csv_path.relative_to(ROOT)),
        "csv_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary_path),
        "figure_sha256": sha256(matched_figure),
        "table_sha256": sha256(matched_table),
    }


def audit_scale(run_id: str) -> dict:
    state = RESULTS / "vldb_scale" / run_id
    cell_evidence: dict[tuple[str, int], dict[str, object]] = {}
    for decoder in ("p2l", "sequential"):
        for size in SIZES:
            log, config, fields = witness(
                state / (decoder + "_" + format(size, "07d") + ".ok")
            )
            cfg = yaml.safe_load(config.read_text())
            model, retrieval = cfg["model"], cfg["retrieval_config"]
            test = retrieval["test_datasets_config"]
            candidate_cfg = retrieval["cand_pools_config"]
            expected_blocks = [1, 1, 1, 6] if decoder == "p2l" else [1] * 9
            expected_pool = "UNION" if size == 5_609_079 else f"cirrscale_{size:07d}"
            expected_generated_pools = [] if size == 5_609_079 else [expected_pool]
            checks = (
                cfg["dataloader_config"]["batch_size"] == 8,
                cfg["data_config"]["deterministic_eval_sampling"] is True,
                retrieval["num_beams"] == 50,
                retrieval["load_saved_beam_scores"] is False,
                retrieval["save_beam_scores"] is True,
                retrieval["rerank"] is True,
                retrieval["use_hybrid_score"] is False,
                test["datasets_name"] == ["cirr_task7"],
                test["correspond_cand_pools_name"] == [expected_pool],
                test["correspond_qrels_name"] == ["cirr_task7"],
                candidate_cfg["cand_pools_name_to_gen_code"] == expected_generated_pools,
                candidate_cfg["gen_code_union_pool"] is False,
                float(model["guidance_scale"]) == 3.0,
                float(model["rrg_weight"]) == 20.0,
                model["use_rrg"] is True,
                model["rrg_normalize"] is True,
                model["rrg_skip_modality"] is True,
                model["rqc_score_mode"] == "cosine",
                int(model["rqc_apply_from_level"]) == 3,
                int(model["rqc_prefix_start_level"]) == 0,
                model["hierarchy_block_sizes"] == expected_blocks,
                int(model["tcis_max_leaves"]) == 0,
                fields.get("decoder") == decoder,
                int(fields.get("candidate_count", "-1")) == size,
            )
            if not all(checks):
                raise AssertionError("scale configuration mismatch: " + str(config))
            require_eval_log(log, frontier=decoder == "p2l")
            text = log.read_text(errors="replace")
            if not re.search(
                r"Candidate Index Setup: seconds=[0-9.]+ "
                r"excluded_from_generation=true",
                text,
            ):
                raise AssertionError(
                    "scale log does not prove that index setup is outside the "
                    "online generation timer: " + str(log)
                )
            if re.search(
                r"candidate tree.*stale; rebuilding|Saved candidate tree from",
                text,
            ):
                raise AssertionError(
                    "scale generation total includes offline Trie construction: "
                    + str(log)
                )
            generation = re.findall(
                r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
            )
            retrieval_seconds = re.findall(
                r"Retrieval Metrics: seconds=([0-9.]+)", text
            )
            retrieval_latency = re.findall(
                r"Retrieval Latency Distribution: queries=([0-9]+) "
                r"p50_ms=([0-9.]+) p95_ms=([0-9.]+) "
                r"p99_ms=([0-9.]+) mean_ms=([0-9.]+)",
                text,
            )
            recall_at_10 = re.findall(
                r"Retriever: Mean Recall@10: ([0-9.]+)", text
            )
            indexed_candidates = re.findall(
                r"candidate semantic-ID collisions \| candidates=([0-9]+)", text
            )
            if (
                not generation
                or not retrieval_seconds
                or not retrieval_latency
                or not recall_at_10
                or int(generation[-1][1]) != 4_170
                or int(retrieval_latency[-1][0]) != 4_170
                or not indexed_candidates
                or int(indexed_candidates[-1]) != size
            ):
                raise AssertionError("scale log/query-pool mismatch: " + str(log))
            frontier = re.findall(
                r"TCIS Query Enumeration:.*?candidates_mean=([0-9.]+).*?"
                r"candidates_p95=([0-9.]+).*?candidates_p99=([0-9.]+)",
                text,
            )
            cell_evidence[(decoder, size)] = {
                "log": log,
                "log_sha256": sha256(log),
                "queries": int(generation[-1][1]),
                "generation_seconds": float(generation[-1][0]),
                "retrieval_seconds": float(retrieval_seconds[-1]),
                "retrieval_p50_ms_per_query": float(retrieval_latency[-1][1]),
                "retrieval_p95_ms_per_query": float(retrieval_latency[-1][2]),
                "retrieval_p99_ms_per_query": float(retrieval_latency[-1][3]),
                "retrieval_mean_ms_per_query": float(retrieval_latency[-1][4]),
                "recall_at_10": float(recall_at_10[-1]),
                "frontier": tuple(map(float, frontier[-1])) if frontier else None,
            }
    scale_path = RESULTS / "structnar_cirr_end_to_end_scaling.csv"
    flat_path = RESULTS / "structnar_flatip_gpu_scaling.csv"
    scale = list(csv.DictReader(scale_path.open(newline="")))
    flat = list(csv.DictReader(flat_path.open(newline="")))
    expected = {(d, n) for d in ("p2l", "sequential") for n in SIZES}
    observed = {(row["decoder"], int(row["candidate_count"])) for row in scale}
    if len(scale) != 10 or observed != expected or {int(r["queries"]) for r in scale} != {4170}:
        raise AssertionError("semantic-ID scale matrix is incomplete")
    ambiguous_timing_fields = {
        "decode_to_rerank_seconds",
        "decode_to_rerank_ms_per_query",
    }
    if ambiguous_timing_fields.intersection(scale[0]):
        raise AssertionError("scale CSV retains an ambiguous D2R timing field")
    for row in scale:
        key = (row["decoder"], int(row["candidate_count"]))
        evidence = cell_evidence[key]
        source_log = ROOT / row["source_log"]
        numerical_checks = (
            source_log.resolve() == Path(evidence["log"]).resolve(),
            row["source_log_sha256"] == evidence["log_sha256"],
            int(row["queries"]) == evidence["queries"],
            math.isclose(
                float(row["generation_seconds"]),
                float(evidence["generation_seconds"]),
                abs_tol=1e-9,
            ),
            math.isclose(
                float(row["retrieval_seconds"]),
                float(evidence["retrieval_seconds"]),
                abs_tol=1e-9,
            ),
            all(
                math.isclose(
                    float(row[name]), float(evidence[name]), abs_tol=1e-9
                )
                for name in (
                    "retrieval_p50_ms_per_query",
                    "retrieval_p95_ms_per_query",
                    "retrieval_p99_ms_per_query",
                    "retrieval_mean_ms_per_query",
                )
            ),
            math.isclose(
                float(row["recall_at_10"]),
                float(evidence["recall_at_10"]),
                abs_tol=1e-12,
            ),
            math.isclose(
                float(row["generation_ms_per_query"]),
                1000.0 * float(evidence["generation_seconds"]) / evidence["queries"],
                abs_tol=1e-9,
            ),
            math.isclose(
                float(row["manifest_decode_to_rerank_seconds"]),
                float(evidence["generation_seconds"])
                + float(evidence["retrieval_seconds"]),
                abs_tol=1e-9,
            ),
            math.isclose(
                float(row["online_decode_to_rerank_ms_per_query"]),
                1000.0
                * float(evidence["generation_seconds"])
                / evidence["queries"]
                + float(evidence["retrieval_mean_ms_per_query"]),
                abs_tol=1e-9,
            ),
        )
        if not all(numerical_checks):
            raise AssertionError("scale CSV/log mismatch: " + str(key))
        expected_frontier = evidence["frontier"]
        if row["decoder"] == "p2l":
            observed_frontier = tuple(
                float(row[name]) for name in ("leaves_mean", "leaves_p95", "leaves_p99")
            )
            if expected_frontier is None or any(
                not math.isclose(actual, expected_value, abs_tol=1e-9)
                for actual, expected_value in zip(observed_frontier, expected_frontier)
            ):
                raise AssertionError("scale frontier CSV/log mismatch: " + str(key))
        elif any(row[name] for name in ("leaves_mean", "leaves_p95", "leaves_p99")):
            raise AssertionError("sequential scale row unexpectedly reports a P2L frontier")
    if (
        len(flat) != 5
        or {int(r["candidate_count"]) for r in flat} != set(SIZES)
        or {int(r["queries"]) for r in flat} != {4170}
        or {int(r["batch_size"]) for r in flat} != {8}
        or {r["precision"] for r in flat} != {"fp16"}
    ):
        raise AssertionError("FlatIP scale matrix is incomplete")
    query_embedding_hashes = set()
    query_id_hashes = set()
    previous_ids = None
    for row in sorted(flat, key=lambda value: int(value["candidate_count"])):
        query_file = ROOT / row["query_embedding_file"]
        query_id_file = ROOT / row["query_id_file"]
        candidate_file = ROOT / row["candidate_embedding_file"]
        candidate_id_file = ROOT / row["candidate_id_file"]
        if not all(
            path.is_file()
            for path in (query_file, query_id_file, candidate_file, candidate_id_file)
        ):
            raise AssertionError("FlatIP source embedding is missing")
        query_embedding_hashes.add(sha256(query_file))
        query_id_hashes.add(sha256(query_id_file))
        query_embeddings = np.load(query_file, mmap_mode="r")
        query_ids = np.load(query_id_file, mmap_mode="r")
        candidate_embeddings = np.load(candidate_file, mmap_mode="r")
        candidate_ids = np.load(candidate_id_file, mmap_mode="r")
        expected_count = int(row["candidate_count"])
        if (
            query_embeddings.shape != (4_170, 768)
            or len(query_ids) != 4_170
            or candidate_embeddings.shape != (expected_count, 768)
            or len(candidate_ids) != expected_count
            or len(np.unique(candidate_ids)) != expected_count
        ):
            raise AssertionError("FlatIP candidate IDs are incomplete or duplicated")
        if previous_ids is not None and not bool(np.isin(previous_ids, candidate_ids).all()):
            raise AssertionError("CIRR scale pools are not nested")
        previous_ids = np.asarray(candidate_ids)
    if len(query_embedding_hashes) != 1 or len(query_id_hashes) != 1:
        raise AssertionError("CIRR scale pools do not use an identical query manifest")
    scale_figure = ROOT / "VLDB/vldb2027/figures/cirr_scale_curve.pdf"
    scale_main_figure = ROOT / "VLDB/vldb2027/figures/cirr_scale_pareto_compact.pdf"
    scale_table = ROOT / "VLDB/vldb2027/supplementary/generated/cirr_scale_matrix.tex"
    flat_table = ROOT / "VLDB/vldb2027/supplementary/generated/flatip_gpu_scaling.tex"
    for artifact in (scale_figure, scale_main_figure, scale_table, flat_table):
        require_nonempty(artifact)
    scale_table_text = scale_table.read_text()
    if not all(
        token in scale_table_text
        for token in ("Online D2R", "Manifest D2R", "full fixed-manifest elapsed time")
    ):
        raise AssertionError("scale table does not distinguish online and manifest D2R")
    main_text = (ROOT / "VLDB/vldb2027/structnar_vldb.tex").read_text()
    if (
        "30.27~ms/query warm decode-to-rerank latency" not in main_text
        or "generation time by 51.5\\%--53.8\\%" not in main_text
        or "decode-to-rerank latency by 50.5\\%--53.0\\%" not in main_text
        or "Manifest D2R" not in main_text
        or "49.58~ms/query decode-to-rerank latency" in main_text
    ):
        raise AssertionError("main paper uses a stale or ambiguous UNION D2R claim")
    completion = RESULTS / "structnar_cirr_scale_matrix_complete.ok"
    require_nonempty(completion)
    completion_fields = dict(
        line.split("=", 1)
        for line in completion.read_text().splitlines()
        if "=" in line
    )
    if completion_fields.get("run_id") != run_id:
        raise AssertionError("scale completion marker belongs to another run")
    if completion_fields.get("csv_sha256") != sha256(scale_path):
        raise AssertionError("scale completion marker hash mismatch")
    return {
        "semantic_id_cells": 10,
        "flatip_cells": 5,
        "scale_csv_sha256": sha256(scale_path),
        "main_scale_figure_sha256": sha256(scale_main_figure),
        "flatip_csv_sha256": sha256(flat_path),
        "figure_sha256": sha256(scale_figure),
        "scale_table_sha256": sha256(scale_table),
        "flatip_table_sha256": sha256(flat_table),
    }


def audit_rq_distance(run_id: str) -> dict:
    state = RESULTS / "vldb_rq_distance" / run_id
    markers = []
    cell_evidence: dict[tuple[str, str, int], dict[str, object]] = {}
    for weight in RQ_WEIGHTS:
        markers.extend(("val", task, weight) for task in RQ_VAL_TASKS)
    summary_path = RESULTS / "structnar_vldb_rq_distance_baseline.json"
    csv_path = RESULTS / "structnar_vldb_rq_distance_baseline.csv"
    summary = json.loads(summary_path.read_text())
    selected = int(summary["selected_weight"])
    if selected not in RQ_WEIGHTS:
        raise AssertionError("RQ-distance selected weight is outside the validation sweep")
    markers.extend(("test", task, selected) for task in RQ_TEST_TASKS)
    for split, task, weight in markers:
        marker = state / f"{split}_{task}_w{weight}.ok"
        log, config, fields = witness(marker)
        cfg = yaml.safe_load(config.read_text())
        model, retrieval = cfg["model"], cfg["retrieval_config"]
        split_cfg = retrieval[f"{split}_datasets_config"]
        checks = (
            fields.get("split") == split,
            fields.get("task") == task,
            int(fields.get("weight", "-1")) == weight,
            cfg["dataloader_config"]["batch_size"] == 8,
            cfg["data_config"]["deterministic_eval_sampling"] is True,
            retrieval["num_beams"] == 50,
            effective_prefix_width(model, retrieval) == 50,
            retrieval["load_saved_beam_scores"] is False,
            retrieval["save_beam_scores"] is True,
            retrieval["rerank"] is True,
            retrieval["use_hybrid_score"] is False,
            split_cfg["datasets_name"] == [task],
            split_cfg["correspond_qrels_name"] == [task],
            float(model["guidance_scale"]) == 3.0,
            model["hierarchy_block_sizes"] == [1, 1, 1, 6],
            model["rqc_score_mode"] == "gain",
            model["rrg_normalize"] is False,
            model["rrg_skip_modality"] is True,
            model["use_rrg"] is (weight != 0),
            int(model["rqc_apply_from_level"]) == 3,
            int(model["rqc_prefix_start_level"]) == 0,
            float(model["rrg_weight"]) == float(weight),
        )
        if not all(checks):
            raise AssertionError("RQ-distance configuration mismatch: " + str(config))
        require_eval_log(log, split=split)
        expected_qrel = "/qrels/" + split + "/"
        text = log.read_text(errors="replace")
        if expected_qrel not in text:
            raise AssertionError("RQ-distance split/qrels mismatch: " + str(log))
        generation = re.findall(
            r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
        )
        recalls = {
            cutoff: re.findall(
                rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text
            )
            for cutoff in (1, 5, 10)
        }
        if not generation or any(not values for values in recalls.values()):
            raise AssertionError("RQ-distance log metric missing: " + str(log))
        cell_evidence[(split, task, weight)] = {
            "log": log,
            "log_sha256": sha256(log),
            "generation_seconds": float(generation[-1][0]),
            "queries": int(generation[-1][1]),
            "recall_at_1": float(recalls[1][-1]),
            "recall_at_5": float(recalls[5][-1]),
            "recall_at_10": float(recalls[10][-1]),
        }
    rows = list(csv.DictReader(csv_path.open(newline="")))
    expected_rows = {(split, task, weight) for split, task, weight in markers}
    observed_rows = {
        (row["split"], row["task"], int(row["weight"])) for row in rows
    }
    if len(rows) != len(markers) or observed_rows != expected_rows:
        raise AssertionError("RQ-distance CSV is incomplete")
    if summary.get("score") != "raw_rq_residual_reconstruction_gain":
        raise AssertionError("RQ-distance score identity mismatch")
    for row in rows:
        key = (row["split"], row["task"], int(row["weight"]))
        evidence = cell_evidence[key]
        source_log = ROOT / row["source_log"]
        checks = (
            source_log.resolve() == Path(evidence["log"]).resolve(),
            row["source_log_sha256"] == evidence["log_sha256"],
            int(row["queries"]) == evidence["queries"],
            math.isclose(
                float(row["generation_seconds"]),
                float(evidence["generation_seconds"]),
                abs_tol=1e-9,
            ),
            math.isclose(float(row["recall_at_1"]), float(evidence["recall_at_1"]), abs_tol=1e-12),
            math.isclose(float(row["recall_at_5"]), float(evidence["recall_at_5"]), abs_tol=1e-12),
            math.isclose(float(row["recall_at_10"]), float(evidence["recall_at_10"]), abs_tol=1e-12),
        )
        if not all(checks):
            raise AssertionError("RQ-distance CSV/log mismatch: " + str(key))
    validation_means = {
        weight: sum(
            float(cell_evidence[("val", task, weight)]["recall_at_10"])
            for task in RQ_VAL_TASKS
        )
        / len(RQ_VAL_TASKS)
        for weight in RQ_WEIGHTS
    }
    recomputed_selected = min(
        RQ_WEIGHTS, key=lambda weight: (-validation_means[weight], weight)
    )
    reported_means = summary.get("validation_mean_r10", {})
    if recomputed_selected != selected or any(
        not math.isclose(
            float(reported_means.get(str(weight), "nan")),
            validation_means[weight],
            abs_tol=1e-12,
        )
        for weight in RQ_WEIGHTS
    ):
        raise AssertionError("RQ-distance validation selection is not reproducible")
    latex_path = ROOT / "VLDB/vldb2027/supplementary/generated/rq_distance_baseline.tex"
    main_row_path = ROOT / "VLDB/vldb2027/supplementary/generated/rq_distance_main_row.tex"
    require_nonempty(latex_path)
    require_nonempty(main_row_path)
    rendered_row = main_row_path.read_text()
    if f"RQ reconstruction gain & {selected}" not in rendered_row:
        raise AssertionError("RQ-distance main-table row has a stale selected weight")
    for task in RQ_TEST_TASKS:
        expected = 100.0 * float(cell_evidence[("test", task, selected)]["recall_at_10"])
        if f"{expected:.2f}" not in rendered_row:
            raise AssertionError("RQ-distance main-table row has a stale test result")
    main_text = (ROOT / "VLDB/vldb2027/structnar_vldb.tex").read_text()
    pcaa_r10 = (47.46, 51.32, 56.56, 50.71)
    rq_r10 = tuple(
        100.0 * float(cell_evidence[("test", task, selected)]["recall_at_10"])
        for task in RQ_TEST_TASKS
    )
    deltas = tuple(pcaa - rq for pcaa, rq in zip(pcaa_r10, rq_r10))
    rendered_delta_range = f"by {min(deltas):.2f}--{max(deltas):.2f} R@10 points"
    if rendered_delta_range not in main_text:
        raise AssertionError("main paper has stale PCAA-versus-RQ deltas")
    return {
        "validation_cells": len(RQ_WEIGHTS) * len(RQ_VAL_TASKS),
        "test_cells": len(RQ_TEST_TASKS),
        "selected_weight": selected,
        "selected_on_upper_boundary": selected == max(RQ_WEIGHTS),
        "csv_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary_path),
        "table_sha256": sha256(latex_path),
        "main_row_sha256": sha256(main_row_path),
    }


def audit_raw_cosine_b50() -> dict:
    run_id = (RESULTS / "structnar_vldb_raw_cosine_active_run_id.txt").read_text().strip()
    state = RESULTS / "vldb_raw_cosine_b50" / run_id
    summary_path = RESULTS / "structnar_vldb_raw_cosine_b50.json"
    csv_path = RESULTS / "structnar_vldb_raw_cosine_b50.csv"
    summary = json.loads(summary_path.read_text())
    selected = int(summary["selected_weight"])
    expected = [
        ("val", task, weight)
        for weight in RAW_COSINE_WEIGHTS
        for task in RAW_COSINE_VAL_TASKS
    ] + [
        ("test", task, selected) for task in RAW_COSINE_TEST_TASKS
    ]
    evidence: dict[tuple[str, str, int], dict[str, object]] = {}
    configs: dict[tuple[str, str, int], Path] = {}
    for split, task, weight in expected:
        log, config, fields = witness(state / f"{split}_{task}_w{weight}.ok")
        cfg = yaml.safe_load(config.read_text())
        model, retrieval = cfg["model"], cfg["retrieval_config"]
        split_cfg = retrieval[f"{split}_datasets_config"]
        checks = (
            fields.get("split") == split,
            fields.get("task") == task,
            int(fields.get("weight", "-1")) == weight,
            cfg["dataloader_config"]["batch_size"] == 8,
            cfg["data_config"]["deterministic_eval_sampling"] is True,
            retrieval["num_beams"] == 50,
            effective_prefix_width(model, retrieval) == 50,
            retrieval["load_saved_beam_scores"] is False,
            retrieval["save_beam_scores"] is True,
            retrieval["rerank"] is True,
            retrieval["use_hybrid_score"] is False,
            split_cfg["datasets_name"] == [task],
            split_cfg["correspond_qrels_name"] == [task],
            float(model["guidance_scale"]) == 3.0,
            model["hierarchy_block_sizes"] == [1, 1, 1, 6],
            model["rqc_score_mode"] == "cosine",
            model["rrg_normalize"] is False,
            model["rrg_skip_modality"] is True,
            model["use_rrg"] is (weight != 0),
            int(model["rqc_apply_from_level"]) == 3,
            int(model["rqc_prefix_start_level"]) == 0,
            float(model["rrg_weight"]) == float(weight),
        )
        if not all(checks):
            raise AssertionError("raw-cosine B=50 configuration mismatch: " + str(config))
        require_eval_log(log, split=split)
        text = log.read_text(errors="replace")
        generation = re.findall(
            r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
        )
        recalls = {
            cutoff: re.findall(rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text)
            for cutoff in (1, 5, 10)
        }
        if not generation or any(not values for values in recalls.values()):
            raise AssertionError("raw-cosine log metric missing: " + str(log))
        evidence[(split, task, weight)] = {
            "log": log,
            "log_sha256": sha256(log),
            "generation_seconds": float(generation[-1][0]),
            "queries": int(generation[-1][1]),
            "recall_at_1": float(recalls[1][-1]),
            "recall_at_5": float(recalls[5][-1]),
            "recall_at_10": float(recalls[10][-1]),
        }
        configs[(split, task, weight)] = config

    rows = list(csv.DictReader(csv_path.open(newline="")))
    row_keys = {(r["split"], r["task"], int(r["weight"])) for r in rows}
    if len(rows) != len(expected) or row_keys != set(expected):
        raise AssertionError("raw-cosine B=50 CSV is incomplete")
    for row in rows:
        key = (row["split"], row["task"], int(row["weight"]))
        item = evidence[key]
        if not (
            row["source_log_sha256"] == item["log_sha256"]
            and int(row["queries"]) == item["queries"]
            and math.isclose(float(row["generation_seconds"]), float(item["generation_seconds"]), abs_tol=1e-9)
            and math.isclose(float(row["recall_at_1"]), float(item["recall_at_1"]), abs_tol=1e-12)
            and math.isclose(float(row["recall_at_5"]), float(item["recall_at_5"]), abs_tol=1e-12)
            and math.isclose(float(row["recall_at_10"]), float(item["recall_at_10"]), abs_tol=1e-12)
        ):
            raise AssertionError("raw-cosine CSV/log mismatch: " + str(key))

    means = {
        weight: sum(
            float(evidence[("val", task, weight)]["recall_at_10"])
            for task in RAW_COSINE_VAL_TASKS
        ) / len(RAW_COSINE_VAL_TASKS)
        for weight in RAW_COSINE_WEIGHTS
    }
    recomputed = min(RAW_COSINE_WEIGHTS, key=lambda w: (-means[w], w))
    if recomputed != selected or int(summary["prefix_width"]) != 50:
        raise AssertionError("raw-cosine B=50 selection is not reproducible")

    # Weight zero disables the auxiliary score.  Its complete ID beams and
    # beam scores must therefore be byte-identical to the matched B=50 RQ
    # control, not merely equal after metric aggregation.
    zero_hashes = {}
    rq_run_id = (RESULTS / "structnar_vldb_active_run_id.txt").read_text().strip()
    rq_state = RESULTS / "vldb_rq_distance" / rq_run_id
    for task in RAW_COSINE_VAL_TASKS:
        raw_cfg = yaml.safe_load(configs[("val", task, 0)].read_text())
        _, rq_config, _ = witness(rq_state / f"val_{task}_w0.ok")
        rq_cfg = yaml.safe_load(rq_config.read_text())
        raw_exp = raw_cfg["experiment"]["exp_name"]
        rq_exp = rq_cfg["experiment"]["exp_name"]
        stem = f"mbeir_{task}_val"
        raw_dir = ROOT / "gen_code/STRUCTNAR/Large/Instruct" / raw_exp / "val"
        rq_dir = ROOT / "gen_code/STRUCTNAR/Large/Instruct" / rq_exp / "val"
        task_hashes = {}
        for suffix in ("codes.npy", "beam_scores.npy"):
            raw_path = raw_dir / f"{stem}_{suffix}"
            rq_path = rq_dir / f"{stem}_{suffix}"
            require_nonempty(raw_path)
            require_nonempty(rq_path)
            raw_hash, rq_hash = sha256(raw_path), sha256(rq_path)
            if raw_hash != rq_hash:
                raise AssertionError(f"B=50 zero-score frontier mismatch: {task} {suffix}")
            task_hashes[suffix] = raw_hash
        zero_hashes[task] = task_hashes

    table_path = ROOT / "VLDB/vldb2027/supplementary/generated/raw_cosine_b50.tex"
    row_path = ROOT / "VLDB/vldb2027/supplementary/generated/raw_cosine_b50_main_row.tex"
    require_nonempty(table_path)
    require_nonempty(row_path)
    return {
        "run_id": run_id,
        "validation_cells": len(RAW_COSINE_WEIGHTS) * len(RAW_COSINE_VAL_TASKS),
        "test_cells": len(RAW_COSINE_TEST_TASKS),
        "selected_weight": selected,
        "selected_on_boundary": selected in (min(RAW_COSINE_WEIGHTS), max(RAW_COSINE_WEIGHTS)),
        "prefix_width": 50,
        "zero_score_beam_hashes_match_rq_control": zero_hashes,
        "csv_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary_path),
        "table_sha256": sha256(table_path),
        "main_row_sha256": sha256(row_path),
    }


def audit_pcaa_b50_validation() -> dict:
    """Verify that standardized PCAA is selected on the final B=50 frontier."""
    active = RESULTS / "p2l_pcaa_b50_validation_active_stamp.txt"
    require_nonempty(active)
    stamp = active.read_text().strip()
    curve_path = RESULTS / f"tip_core_pcaa_validation_{stamp}.csv"
    selection_path = RESULTS / f"tip_core_pcaa_selection_{stamp}.csv"
    table_path = (
        ROOT / "VLDB/vldb2027/supplementary/generated/pcaa_b50_validation.tex"
    )
    for path in (curve_path, selection_path, table_path):
        require_nonempty(path)

    rows = list(csv.DictReader(curve_path.open(newline="")))
    selections = {
        row["design"]: row for row in csv.DictReader(selection_path.open(newline=""))
    }
    designs = ("raw_cosine", "standardized_cosine")
    weights = (0, 5, 10, 15, 20, 30, 40)
    datasets = ("CIRR", "NIGHTS")
    expected = {
        (design, weight, dataset)
        for design in designs
        for weight in weights
        for dataset in (*datasets, "Mean")
    }
    observed = {
        (row["design"], int(row["weight"]), row["dataset"])
        for row in rows
        if row["design"] in designs
    }
    if observed != expected:
        raise AssertionError("B=50 PCAA validation matrix is incomplete")

    evidence = {
        (row["design"], int(row["weight"]), row["dataset"]): row
        for row in rows
        if row["design"] in designs
    }
    for design in designs:
        means = {}
        for weight in weights:
            values = []
            for dataset in datasets:
                row = evidence[(design, weight, dataset)]
                log = ROOT / row["log"]
                config = ROOT / row["config"]
                for path in (log, config):
                    require_nonempty(path)
                if row["log_sha256"] != sha256(log) or row["config_sha256"] != sha256(config):
                    raise AssertionError("PCAA validation source hash mismatch")
                cfg = yaml.safe_load(config.read_text())
                model, retrieval = cfg["model"], cfg["retrieval_config"]
                expected_normalize = design == "standardized_cosine"
                score_activation_checks = (
                    (not bool(model["use_rrg"])) if weight == 0 else bool(model["use_rrg"]),
                    True if weight == 0 else (
                        str(row["rrg_normalize"]).lower()
                        == str(expected_normalize).lower()
                    ),
                    True if weight == 0 else (
                        bool(model["rrg_normalize"]) is expected_normalize
                    ),
                )
                checks = (
                    int(row["prefix_width"]) == 50,
                    int(row["output_width"]) == 50,
                    int(model["tcis_prefix_beams"]) == 50,
                    int(retrieval["num_beams"]) == 50,
                    str(row["score_mode"]) == "cosine",
                    model["rqc_score_mode"] == "cosine",
                    float(model["rrg_weight"]) == float(weight),
                    *score_activation_checks,
                )
                if not all(checks):
                    raise AssertionError("PCAA B=50 configuration mismatch: " + str(config))
                recalls = re.findall(
                    r"Retriever: Mean Recall@10: ([0-9.]+)",
                    log.read_text(errors="replace"),
                )
                if not recalls or not math.isclose(
                    float(row["recall_at_10"]), float(recalls[-1]), abs_tol=1e-12
                ):
                    raise AssertionError("PCAA validation log/CSV mismatch: " + str(log))
                values.append(float(row["recall_at_10"]))
            means[weight] = sum(values) / len(values)
            recorded_mean = float(evidence[(design, weight, "Mean")]["recall_at_10"])
            if not math.isclose(means[weight], recorded_mean, abs_tol=1e-12):
                raise AssertionError("PCAA validation mean mismatch")
        selected = min(weights, key=lambda value: (-means[value], value))
        if int(selections[design]["selected_weight"]) != selected:
            raise AssertionError("PCAA validation selection mismatch")

    selected_pcaa = int(selections["standardized_cosine"]["selected_weight"])
    final_config = yaml.safe_load(
        (ROOT / "configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml").read_text()
    )
    main_text = (ROOT / "VLDB/vldb2027/structnar_vldb.tex").read_text()
    if (
        selected_pcaa != int(final_config["model"]["rrg_weight"])
        or f"$\\lambda={selected_pcaa}$ for standardized PCAA" not in main_text
        or f"\\textbf{{{selected_pcaa}}}" not in table_path.read_text()
    ):
        raise AssertionError("selected PCAA weight is stale in the final artifacts")
    return {
        "stamp": stamp,
        "validation_cells": len(designs) * len(weights) * len(datasets),
        "selected_raw_cosine_weight": int(selections["raw_cosine"]["selected_weight"]),
        "selected_standardized_pcaa_weight": selected_pcaa,
        "curve_sha256": sha256(curve_path),
        "selection_sha256": sha256(selection_path),
        "table_sha256": sha256(table_path),
    }


def audit_configuration_evidence() -> dict:
    train_path = ROOT / "configs/structnar/train.yaml"
    final_path = ROOT / "configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml"
    checkpoint_path = ROOT / "checkpoint/code_tied/gpt_hdgr_latest.pth"
    checkpoint_audit_path = RESULTS / "structnar_training_checkpoint_audit.yaml"
    training_code_path = ROOT / "src/models/hdgr_comparison/train.py"
    training_log_path = ROOT / "logs/train_code_tied_hdgr_official_mbeir_20260715_044040.log"
    environment_path = RESULTS / "structnar_environment.txt"
    for path in (
        train_path, final_path, checkpoint_path, checkpoint_audit_path,
        training_code_path, training_log_path, environment_path,
    ):
        require_nonempty(path)
    train = yaml.safe_load(train_path.read_text())
    final = yaml.safe_load(final_path.read_text())
    recorded = yaml.safe_load(checkpoint_audit_path.read_text())
    epoch_times = re.findall(
        r"Train Epoch: \[(\d+)\] Total time: (?:(\d+):)?(\d+):(\d+)",
        training_log_path.read_text(errors="replace"),
    )
    if len(epoch_times) != 100 or {int(row[0]) for row in epoch_times} != set(range(100)):
        raise AssertionError("training log does not contain one duration for every epoch")
    logged_epoch_seconds = sum(
        int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
        for _, hours, minutes, seconds in epoch_times
    )
    tm, td, tt = train["model"], train["dataloader_config"], train["trainer_config"]
    fm, fr = final["model"], final["retrieval_config"]
    train_checks = (
        int(tm["num_prefix"]) == 30,
        int(tm["d_model"]) == 512,
        int(tm["time_step"]) == 8,
        float(tm["guidance_scale"]) == 3.0,
        float(tm["cond_drop_prob"]) == 0.1,
        float(tm["visible_token_loss_weight"]) == 0.1,
        float(tm["query_positive_mix_prob"]) == 1.0,
        float(tm["query_positive_mix_alpha"]) == 2.0,
        tm["query_positive_mix_elementwise"] is True,
        int(td["train_batch_size"]) == 512,
        int(train["seed"]) == 2023,
        str(tt["optimizer"]).lower() == "adamw",
        math.isclose(float(tt["learning_rate"]), 1e-4, abs_tol=1e-12),
        math.isclose(float(tt["weight_decay"]), 1e-4, abs_tol=1e-12),
        math.isclose(float(tt["gain_or_bias_weight_decay"]), 0.0, abs_tol=1e-12),
        int(tt["num_train_epochs"]) == 100,
    )
    final_checks = (
        int(final["dataloader_config"]["batch_size"]) == 8,
        int(fm["time_step"]) == 8,
        float(fm["guidance_scale"]) == 3.0,
        fm["hierarchy_block_sizes"] == [1, 1, 1, 6],
        effective_prefix_width(fm, fr) == 50,
        int(fr["num_beams"]) == 50,
        fr["use_fp16"] is True,
        fr["rerank"] is True,
        fr["use_hybrid_score"] is False,
        fr["load_saved_beam_scores"] is False,
        fm["use_rrg"] is True,
        fm["rqc_score_mode"] == "cosine",
        fm["rrg_normalize"] is True,
        fm["rrg_skip_modality"] is True,
        float(fm["rrg_weight"]) == 20.0,
        int(fm["rqc_apply_from_level"]) == 3,
        int(final["seed"]) == 2023,
    )
    checkpoint_checks = (
        recorded["checkpoint"]["sha256"] == sha256(checkpoint_path),
        int(recorded["checkpoint"]["epoch"]) == 99,
        int(recorded["training_workload"]["epochs"]) == 100,
        int(recorded["training_workload"]["total_iterations"]) == 260300,
        int(recorded["training_workload"]["successful_optimizer_updates"]) == 260182,
        int(recorded["training_workload"]["skipped_nonfinite_updates"]) == 118,
        int(recorded["training_workload"]["logged_epoch_seconds"]) == logged_epoch_seconds == 34073,
        math.isclose(
            float(recorded["training_workload"]["logged_epoch_hours"]),
            logged_epoch_seconds / 3600.0,
            abs_tol=1e-12,
        ),
        int(recorded["parameters"]["trainable_parameter_count"]) == 53857280,
        recorded["sources"]["training_log"]["sha256"] == sha256(training_log_path),
        recorded["sources"]["training_code"]["sha256"] == sha256(training_code_path),
        recorded["sources"]["released_training_config"]["sha256"] == sha256(train_path),
        recorded["sources"]["final_evaluation_config"]["sha256"] == sha256(final_path),
    )
    training_code = training_code_path.read_text()
    code_checks = (
        'p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])' in training_code,
        '"weight_decay": gain_or_bias_weight_decay' in training_code,
    )
    environment_text = environment_path.read_text()
    environment_checks = (
        "Python: 3.10.19" in environment_text,
        "PyTorch: 2.7.0" in environment_text,
        "PyTorch CUDA toolkit: 12.8" in environment_text,
        "GPU 0: NVIDIA GeForce RTX 3090" in environment_text,
        "NVIDIA driver: 550.144.03" in environment_text,
    )
    if not all(
        (*train_checks, *final_checks, *checkpoint_checks, *code_checks, *environment_checks)
    ):
        raise AssertionError("training/final configuration evidence mismatch")
    return {
        "train_config_sha256": sha256(train_path),
        "final_config_sha256": sha256(final_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_audit_sha256": sha256(checkpoint_audit_path),
        "training_code_sha256": sha256(training_code_path),
        "training_log_sha256": sha256(training_log_path),
        "environment_sha256": sha256(environment_path),
        "logged_epoch_seconds": logged_epoch_seconds,
        "validated_training_fields": len(train_checks),
        "validated_inference_fields": len(final_checks),
    }


def audit_ann_reference() -> dict:
    """Verify archived CPU ANN records, which are intentionally not published."""
    flat_csv = RESULTS / "structnar_ann_flat_cirr_0021551_corrected.csv"
    flat_log = RESULTS / "structnar_ann_flat_cirr_0021551_corrected.log"
    flat_index = RESULTS / "structnar_ann_flat_cirr_0021551_corrected.faiss"
    ivf_csv = RESULTS / "structnar_ann_ivfflat_cirr_1000000_corrected.csv"
    ivf_log = RESULTS / "structnar_ann_ivfflat_cirr_1000000_corrected.log"
    ivf_index = RESULTS / "structnar_ann_ivfflat_cirr_1000000_corrected.faiss"
    for path in (flat_csv, flat_log, flat_index, ivf_csv, ivf_log, ivf_index):
        require_nonempty(path)

    flat_rows = list(csv.DictReader(flat_csv.open(newline="")))
    ivf_rows = list(csv.DictReader(ivf_csv.open(newline="")))
    if len(flat_rows) != 1 or len(ivf_rows) != 5:
        raise AssertionError("CPU ANN reference matrix is incomplete")
    flat = flat_rows[0]
    flat_checks = (
        flat["index"] == "flat",
        int(flat["candidate_count"]) == 21_551,
        int(flat["dimension"]) == 768,
        int(flat["threads"]) == 8,
        int(flat["evaluated_queries"]) == 4_170,
    )
    if not all(flat_checks):
        raise AssertionError("CPU FlatIP reference configuration mismatch")
    if {
        int(row["nprobe"]) for row in ivf_rows
    } != {4, 8, 16, 32, 64} or any(
        row["index"] != "ivfflat"
        or int(row["candidate_count"]) != 1_000_000
        or int(row["dimension"]) != 768
        or int(row["threads"]) != 8
        or int(row["nlist"]) != 512
        or int(row["evaluated_queries"]) != 4_170
        for row in ivf_rows
    ):
        raise AssertionError("CPU IVF-Flat reference configuration mismatch")

    # The profiler emits one JSON record per measured operating point before
    # /usr/bin/time appends the exact invocation.  Match those records back to
    # the CSV so the paper is not supported by detached hand-entered numbers.
    numeric_fields = (
        "search_ms_per_query", "batch_p95_ms_per_query",
        "serialized_index_mib", "peak_process_rss_mib",
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50",
    )
    for log, rows in ((flat_log, flat_rows), (ivf_log, ivf_rows)):
        records = [
            json.loads(line)
            for line in log.read_text(errors="replace").splitlines()
            if line.startswith("{")
        ]
        if len(records) != len(rows):
            raise AssertionError("CPU ANN log/CSV row-count mismatch: " + str(log))
        by_probe = {str(record.get("nprobe", "")): record for record in records}
        for row in rows:
            record = by_probe.get(row["nprobe"])
            if record is None or any(
                not math.isclose(float(row[field]), float(record[field]), abs_tol=1e-12)
                for field in numeric_fields
            ):
                raise AssertionError("CPU ANN log/CSV value mismatch: " + str(log))
        text = log.read_text(errors="replace")
        if (
            "mbeir_cirr_task7_test_qrels.txt" not in text
            or "--threads 8" not in text
            or "--search-batch 256" not in text
            or "Exit status: 0" not in text
        ):
            raise AssertionError("CPU ANN provenance log is incomplete: " + str(log))

    for row, index_path in ((flat, flat_index), (ivf_rows[0], ivf_index)):
        actual_mib = index_path.stat().st_size / (1024.0 * 1024.0)
        if not math.isclose(actual_mib, float(row["serialized_index_mib"]), abs_tol=1e-9):
            raise AssertionError("CPU ANN serialized-index size mismatch: " + str(index_path))

    return {
        "flat_cells": 1,
        "ivfflat_cells": 5,
        "publication_status": "excluded_by_reviewer_net_benefit_gate",
        "flat_csv_sha256": sha256(flat_csv),
        "flat_log_sha256": sha256(flat_log),
        "ivfflat_csv_sha256": sha256(ivf_csv),
        "ivfflat_log_sha256": sha256(ivf_log),
    }


def audit_local_runtime_repeats() -> dict:
    """Verify the three-run LOCAL timing rows against their retained logs."""
    csv_path = RESULTS / "structnar_local_runtime_repeats.csv"
    supplement_path = ROOT / "VLDB/vldb2027/supplementary/structnar_vldb_supplement.tex"
    require_nonempty(csv_path)
    require_nonempty(supplement_path)
    rows = list(csv.DictReader(csv_path.open(newline="")))
    expected = {
        (dataset, decoder, repeat)
        for dataset in ("CIRR-7", "NIGHTS-4")
        for decoder in ("Sequential+PCAA", "P2L+PCAA_s3")
        for repeat in (1, 2, 3)
    }
    observed = {
        (row["dataset"], row["decoder"], int(row["repeat"])) for row in rows
    }
    if len(rows) != 12 or observed != expected or {row["protocol"] for row in rows} != {"LOCAL"}:
        raise AssertionError("LOCAL runtime repeat matrix is incomplete or mixed-scope")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        log = ROOT / row["source_log"]
        require_nonempty(log)
        if sha256(log) != row["source_log_sha256"]:
            raise AssertionError("LOCAL runtime source-log hash mismatch: " + str(log))
        text = log.read_text(errors="replace")
        generation = re.findall(r"Generation Metrics: seconds=([0-9.]+)", text)
        retrieval = re.findall(r"Retrieval Metrics: seconds=([0-9.]+)", text)
        recall = re.findall(r"Retriever: Mean Recall@10: ([0-9.]+)", text)
        if not generation or not retrieval or not recall:
            raise AssertionError("LOCAL runtime source log is incomplete: " + str(log))
        values = (
            float(row["generation_seconds"]),
            float(row["retrieval_seconds"]),
            float(row["decode_to_rerank_seconds"]),
            float(row["recall_at_10"]),
        )
        expected_values = (
            float(generation[-1]),
            float(retrieval[-1]),
            float(generation[-1]) + float(retrieval[-1]),
            100.0 * float(recall[-1]),
        )
        if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(values, expected_values)):
            raise AssertionError("LOCAL runtime CSV/log value mismatch: " + str(log))
        grouped.setdefault((row["dataset"], row["decoder"]), []).append(row)

    expected_aggregates = {
        ("CIRR-7", "Sequential+PCAA"): (257.16, 2.16, 262.34),
        ("CIRR-7", "P2L+PCAA_s3"): (117.77, 1.17, 122.98),
        ("NIGHTS-4", "Sequential+PCAA"): (131.25, 0.26, 133.96),
        ("NIGHTS-4", "P2L+PCAA_s3"): (60.82, 0.13, 63.82),
    }
    aggregates = {}
    for key, group in grouped.items():
        generation = [float(row["generation_seconds"]) for row in group]
        d2r = [float(row["decode_to_rerank_seconds"]) for row in group]
        values = (
            round(statistics.mean(generation), 2),
            round(statistics.stdev(generation), 2),
            round(statistics.mean(d2r), 2),
        )
        if values != expected_aggregates[key]:
            raise AssertionError("LOCAL runtime aggregate mismatch: " + repr(key))
        aggregates["/".join(key)] = {
            "generation_mean_seconds": statistics.mean(generation),
            "generation_sample_std_seconds": statistics.stdev(generation),
            "decode_to_rerank_mean_seconds": statistics.mean(d2r),
        }

    supplement = supplement_path.read_text()
    required_renderings = (
        r"$257.16{\pm}2.16$",
        r"$\mathbf{117.77{\pm}1.17}$",
        r"$131.25{\pm}0.26$",
        r"$\mathbf{60.82{\pm}0.13}$",
    )
    if any(value not in supplement for value in required_renderings):
        raise AssertionError("LOCAL runtime table rendering is stale")
    return {
        "rows": len(rows),
        "repeats_per_configuration": 3,
        "csv_sha256": sha256(csv_path),
        "aggregates": aggregates,
    }


def audit_matched_index_equivalence() -> dict:
    """Check the generated field-level equivalence report for all 16 tasks."""
    json_path = RESULTS / "structnar_matched_index_equivalence.json"
    csv_path = RESULTS / "structnar_matched_index_equivalence.csv"
    script_path = ROOT / "scripts/structnar/audit_matched_index_equivalence.py"
    for path in (json_path, csv_path, script_path):
        require_nonempty(path)
    report = json.loads(json_path.read_text())
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if (
        report.get("all_passed") is not True
        or int(report.get("tasks_expected", 0)) != 16
        or int(report.get("tasks_checked", 0)) != 16
        or len(rows) != 16
        or {row["task"] for row in rows} != set(TASKS)
        or any(row["passed"] != "True" for row in rows)
        or report.get("checkpoint_sha256") != sha256(ROOT / report["checkpoint"])
    ):
        raise AssertionError("matched realized-input equivalence audit failed")
    return {
        "tasks": 16,
        "all_candidate_codes_equal": True,
        "all_query_ids_equal": True,
        "all_query_embeddings_equal": True,
        "all_logical_tries_equal": True,
        "json_sha256": sha256(json_path),
        "csv_sha256": sha256(csv_path),
        "audit_script_sha256": sha256(script_path),
    }


def audit_complete_suffix_evidence() -> dict:
    """Verify the matched last-position/full-suffix scoring intervention."""
    csv_path = RESULTS / "structnar_complete_suffix_ablation.csv"
    require_nonempty(csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="")))
    expected_tasks = {"cirr_task7", "nights_task4", "edis_task2", "webqa_task1"}
    expected_cells = {
        (task, variant) for task in expected_tasks
        for variant in ("last_position", "complete_suffix")
    }
    observed = {(row["task"], row["scoring"]) for row in rows}
    if len(rows) != 8 or observed != expected_cells:
        raise AssertionError("complete-suffix intervention is incomplete")

    gen_root = ROOT / "gen_code/STRUCTNAR/Large/Instruct"
    identities: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        log = ROOT / row["source_log"]
        require_nonempty(log)
        if sha256(log) != row["source_log_sha256"]:
            raise AssertionError("complete-suffix source-log hash mismatch")
        text = log.read_text(errors="replace")
        recalls = tuple(
            float(re.findall(rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text)[-1])
            for cutoff in (1, 5, 10)
        )
        stored = tuple(
            float(row[f"recall_at_{cutoff}"]) / 100.0 for cutoff in (1, 5, 10)
        )
        normalization = "last" if row["scoring"] == "last_position" else "sum"
        required = (
            "  time_step: 8",
            "  guidance_scale: 3.0",
            "  rrg_weight: 0.0",
            "  num_beams: 50",
            "  rerank: true",
            f"  gpt_hdgr_block_score_normalization: {normalization}",
        )
        if (
            any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(recalls, stored))
            or any(token not in text for token in required)
        ):
            raise AssertionError("complete-suffix configuration/metric mismatch")
        exp_names = re.findall(r"^  exp_name: (\S+)$", text, flags=re.MULTILINE)
        if not exp_names:
            raise AssertionError("complete-suffix log lacks resolved experiment name")
        exp_dir = gen_root / exp_names[0]
        test_dir = exp_dir / "test"
        id_files = sorted(test_dir.glob("*_ids.npy"))
        embedding_files = sorted(test_dir.glob("*_embeddings.npy"))
        if len(id_files) != 1 or len(embedding_files) != 1:
            raise AssertionError("complete-suffix realized query files are incomplete")
        identities[(row["task"], row["scoring"])] = {
            "query_ids_sha256": sha256(id_files[0]),
            "query_embeddings_sha256": sha256(embedding_files[0]),
        }

    for task in expected_tasks:
        if identities[(task, "last_position")] != identities[(task, "complete_suffix")]:
            raise AssertionError("complete-suffix realized query mismatch: " + task)
    return {
        "cells": 8,
        "tasks": 4,
        "realized_query_inputs_equal": True,
        "csv_sha256": sha256(csv_path),
    }


def audit_final_diagnostic_evidence() -> dict:
    """Verify archived PCAA repeats, batch scaling, and component profiling."""
    pcaa_path = RESULTS / "structnar_pcaa_runtime_repeats_edis_webqa.csv"
    batch_path = RESULTS / "structnar_tip_batch_scaling.csv"
    component_path = RESULTS / "tip_component_profile_table_xi_20260726.csv"
    reranked_path = RESULTS / "structnar_sequential_pcaa_reranked_metrics.csv"
    supplement_path = ROOT / "VLDB/vldb2027/supplementary/structnar_vldb_supplement.tex"
    main_path = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
    for path in (
        pcaa_path, batch_path, component_path, reranked_path,
        supplement_path, main_path,
    ):
        require_nonempty(path)

    pcaa_rows = list(csv.DictReader(pcaa_path.open(newline="")))
    if len(pcaa_rows) != 6:
        raise AssertionError("EDIS/WebQA PCAA repeat evidence is incomplete")
    pcaa_expected = {"edis_task2": 51.82118296821968, "webqa_task1": 48.22010536320435}
    for row in pcaa_rows:
        log = ROOT / row["source_log"]
        require_nonempty(log)
        if sha256(log) != row["source_log_sha256"]:
            raise AssertionError("PCAA repeat source-log hash mismatch")
        expected_ms = 1000.0 * float(row["generation_seconds"]) / int(row["queries"])
        if not math.isclose(float(row["generation_ms_per_query"]), expected_ms, abs_tol=5e-7):
            raise AssertionError("PCAA repeat ms/query mismatch")
    for task, expected in pcaa_expected.items():
        selected = [float(row["generation_seconds"]) for row in pcaa_rows if row["dataset"] == task]
        queries = {int(row["queries"]) for row in pcaa_rows if row["dataset"] == task}
        if len(selected) != 3 or len(queries) != 1:
            raise AssertionError("PCAA repeat task scope mismatch: " + task)
        actual = statistics.mean(selected) / queries.pop() * 1000.0
        if not math.isclose(actual, expected, abs_tol=1e-12):
            raise AssertionError("PCAA repeat mean mismatch: " + task)

    batch_rows = list(csv.DictReader(batch_path.open(newline="")))
    expected_cells = {
        (task, method, batch)
        for task in ("cirr_task7", "nights_task4", "edis_task2", "webqa_task1")
        for method in ("p2l", "sequential")
        for batch in (1, 4, 8, 16)
    }
    observed_cells = {
        (row["dataset"], row["method"], int(row["batch"])) for row in batch_rows
    }
    if len(batch_rows) != 32 or observed_cells != expected_cells:
        raise AssertionError("batch-scaling matrix is incomplete")
    by_key = {
        (row["dataset"], row["method"], int(row["batch"])): row
        for row in batch_rows
    }
    throughput_pairs = {
        "cirr_task7": (19.60, 45.02),
        "nights_task4": (19.25, 43.62),
        "edis_task2": (16.49, 22.45),
        "webqa_task1": (17.13, 24.55),
    }
    for task, (seq_expected, p2l_expected) in throughput_pairs.items():
        seq = float(by_key[(task, "sequential", 16)]["examples_per_second"])
        p2l = float(by_key[(task, "p2l", 16)]["examples_per_second"])
        if not (
            math.isclose(seq, seq_expected, abs_tol=0.005)
            and math.isclose(p2l, p2l_expected, abs_tol=0.005)
        ):
            raise AssertionError("batch-16 throughput mismatch: " + task)
        for method in ("p2l", "sequential"):
            recalls = [
                100.0 * float(by_key[(task, method, batch)]["recall_at_10"])
                for batch in (1, 4, 8, 16)
            ]
            if max(recalls) - min(recalls) > 0.0900001:
                raise AssertionError("batch recall variation exceeds paper claim")
    if not (
        math.isclose(float(by_key[("edis_task2", "sequential", 8)]["recall_at_10"]), 0.4017)
        and math.isclose(float(by_key[("webqa_task1", "sequential", 8)]["recall_at_10"]), 0.3014)
    ):
        raise AssertionError("sequential PCAA item-recall evidence mismatch")

    reranked_rows = list(csv.DictReader(reranked_path.open(newline="")))
    expected_reranked = {
        "edis_task2": (0.2002, 0.3539, 0.4017),
        "webqa_task1": (0.2248, 0.2925, 0.3014),
    }
    if len(reranked_rows) != 2 or {
        row["task"] for row in reranked_rows
    } != set(expected_reranked):
        raise AssertionError("sequential PCAA reranked export is incomplete")
    for row in reranked_rows:
        log = ROOT / row["source_log"]
        require_nonempty(log)
        if sha256(log) != row["source_log_sha256"]:
            raise AssertionError("sequential PCAA reranked source-log hash mismatch")
        text = log.read_text(errors="replace")
        extracted = tuple(
            float(re.findall(rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text)[-1])
            for cutoff in (1, 5, 10)
        )
        stored = tuple(float(row[f"recall_at_{cutoff}"]) for cutoff in (1, 5, 10))
        if extracted != stored or stored != expected_reranked[row["task"]]:
            raise AssertionError("sequential PCAA reranked metric/log mismatch")

    component_rows = list(csv.DictReader(component_path.open(newline="")))
    if len(component_rows) != 4:
        raise AssertionError("component profile is incomplete")
    for row in component_rows:
        log = ROOT / row["source_log"]
        require_nonempty(log)
        if sha256(log) != row["source_log_sha256"]:
            raise AssertionError("component-profile source-log hash mismatch")

    main_text = main_path.read_text()
    supplement_text = supplement_path.read_text()
    if (
        "$1.36\\times$--$2.30\\times$" not in main_text
        or "R@10 changing by at most 0.09 point" not in main_text
        or "PCAA latencies are means over three synchronized runs" not in supplement_text
    ):
        raise AssertionError("manuscript diagnostic claims are stale")
    return {
        "pcaa_repeat_rows": 6,
        "pcaa_repeat_sha256": sha256(pcaa_path),
        "batch_scaling_cells": 32,
        "batch_scaling_sha256": sha256(batch_path),
        "component_profile_rows": 4,
        "component_profile_sha256": sha256(component_path),
        "sequential_pcaa_reranked_rows": 2,
        "sequential_pcaa_reranked_sha256": sha256(reranked_path),
    }


def audit_cirr_scale_input_manifest() -> dict:
    """Verify the separately generated scale-input identity report."""
    report_path = RESULTS / "structnar_cirr_scale_input_audit.json"
    script_path = ROOT / "scripts/structnar/audit_cirr_scale_inputs.py"
    require_nonempty(report_path)
    require_nonempty(script_path)
    report = json.loads(report_path.read_text())
    checks = report["checks"]
    if not (
        checks["semantic_query_ids_identical_across_scales"]
        and checks["semantic_query_embeddings_identical_across_scales"]
        and int(checks["semantic_query_count"]) == 4_170
        and checks["candidate_sets_nested"]
        and all(item["passed"] and int(item["missing_ids"]) == 0 for item in checks["nested_pair_checks"])
    ):
        raise AssertionError("CIRR scale-input audit failed")
    return {
        "report_sha256": sha256(report_path),
        "script_sha256": sha256(script_path),
        "nested_pair_checks": len(checks["nested_pair_checks"]),
    }


def audit_system_level_results() -> dict:
    """Verify the canonical StructNAR/GENIUS complete-system comparisons."""
    structnar_path = RESULTS / "structnar_p2l_d3_rqc_cosine_w20_all32.tsv"
    genius_path = RESULTS / "genius_all32.tsv"
    comparison_path = RESULTS / "structnar_vs_genius_all32.tsv"
    cross_path = RESULTS / "structnar_cross_paradigm_union.csv"
    meta_path = RESULTS / "structnar_p2l_d3_rqc_cosine_w20_all32.meta.yaml"
    alignment_path = RESULTS / "structnar_genius_index_alignment_audit.yaml"
    published_path = RESULTS / "source_extracts/genius_cvpr2025_table2_union.csv"
    published_provenance_path = (
        RESULTS / "source_extracts/genius_cvpr2025_table2_union_provenance.yaml"
    )
    paper_path = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
    supplement_path = ROOT / "VLDB/vldb2027/supplementary/structnar_vldb_supplement.tex"
    for path in (
        structnar_path, genius_path, comparison_path, cross_path,
        meta_path, alignment_path, published_path, published_provenance_path,
        paper_path, supplement_path,
    ):
        require_nonempty(path)

    def percentage(value: str) -> float:
        text = value.strip()
        return float(text[:-1]) if text.endswith("%") else 100.0 * float(text)

    struct_rows = list(csv.DictReader(structnar_path.open(newline=""), delimiter="\t"))
    genius_rows = list(csv.DictReader(genius_path.open(newline=""), delimiter="\t"))
    comparison_rows = list(
        csv.DictReader(comparison_path.open(newline=""), delimiter="\t")
    )
    expected_metrics = {
        task: (("Recall@10", "Recall@20", "Recall@50") if task in FASHION_TASKS
               else ("Recall@1", "Recall@5", "Recall@10"))
        for task in TASKS
    }
    if len(struct_rows) != 48 or {
        (row["Dataset"], row["Metric"]) for row in struct_rows
    } != {
        (task, metric) for task in TASKS for metric in expected_metrics[task]
    }:
        raise AssertionError("canonical StructNAR all-32 table is incomplete")
    if len(genius_rows) != 32:
        raise AssertionError("reproduced GENIUS LOCAL/UNION table is incomplete")

    struct = {}
    for row in struct_rows:
        for scope, column in (("local", "Local"), ("union", "Union")):
            struct[(row["Dataset"], scope, row["Metric"])] = percentage(row[column])
    genius = {}
    for row in genius_rows:
        task, scope = row["dataset"], row["scope"]
        for metric in expected_metrics[task]:
            value = row[metric]
            if not value:
                raise AssertionError("missing GENIUS metric: " + str((task, scope, metric)))
            genius[(task, scope, metric)] = percentage(value)

    expected_keys = {
        (task, scope, metric)
        for task in TASKS
        for scope in ("local", "union")
        for metric in expected_metrics[task]
    }
    if set(struct) != expected_keys or set(genius) != expected_keys:
        raise AssertionError("complete-system comparison key mismatch")
    if len(comparison_rows) != len(expected_keys):
        raise AssertionError("complete-system delta table is incomplete")
    for row in comparison_rows:
        key = (row["Dataset"], row["Scope"], row["Metric"])
        if key not in expected_keys:
            raise AssertionError("unexpected complete-system comparison row: " + str(key))
        checks = (
            math.isclose(percentage(row["GENIUS"]), genius[key], abs_tol=1e-10),
            math.isclose(percentage(row["StructNAR"]), struct[key], abs_tol=1e-10),
            math.isclose(
                float(row["Delta"]), (struct[key] - genius[key]) / 100.0,
                abs_tol=1e-10,
            ),
        )
        if not all(checks):
            raise AssertionError("complete-system delta mismatch: " + str(key))

    standard = [task for task in TASKS if task not in FASHION_TASKS]
    standard_metrics = ("Recall@1", "Recall@5", "Recall@10")
    fashion_metrics = ("Recall@10", "Recall@20", "Recall@50")
    local_macros = {
        (method, metric): sum(
            values[(task, "local", metric)]
            for task in (standard if metric in standard_metrics and metric != "Recall@20" and metric != "Recall@50" else FASHION_TASKS)
        ) / len(standard if metric in standard_metrics and metric != "Recall@20" and metric != "Recall@50" else FASHION_TASKS)
        for method, values in (("genius", genius), ("structnar", struct))
        for metric in (*standard_metrics, "Recall@20", "Recall@50")
    }
    # R@10 is reported for both groups, so compute the fashion instance
    # explicitly rather than overloading the standard-group dictionary entry.
    fashion_r10 = {
        method: sum(values[(task, "local", "Recall@10")] for task in FASHION_TASKS)
        / len(FASHION_TASKS)
        for method, values in (("genius", genius), ("structnar", struct))
    }
    expected_local = {
        ("genius", "Recall@1"): 26.08,
        ("genius", "Recall@5"): 40.06,
        ("genius", "Recall@10"): 44.94,
        ("structnar", "Recall@1"): 29.20,
        ("structnar", "Recall@5"): 45.98,
        ("structnar", "Recall@10"): 51.97,
    }
    if any(round(local_macros[key] + 1e-12, 2) != value for key, value in expected_local.items()):
        raise AssertionError("standard-task LOCAL macro mismatch")
    expected_fashion = {
        ("genius", "Recall@10"): 14.58,
        ("genius", "Recall@20"): 18.07,
        ("genius", "Recall@50"): 21.37,
        ("structnar", "Recall@10"): 16.51,
        ("structnar", "Recall@20"): 21.35,
        ("structnar", "Recall@50"): 25.41,
    }
    observed_fashion = {
        (method, "Recall@10"): fashion_r10[method]
        for method in ("genius", "structnar")
    }
    for method, values in (("genius", genius), ("structnar", struct)):
        for metric in ("Recall@20", "Recall@50"):
            observed_fashion[(method, metric)] = sum(
                values[(task, "local", metric)] for task in FASHION_TASKS
            ) / len(FASHION_TASKS)
    if any(round(observed_fashion[key] + 1e-12, 2) != value for key, value in expected_fashion.items()):
        raise AssertionError("fashion-task LOCAL macro mismatch")

    local_wins = sum(struct[key] > genius[key] for key in expected_keys if key[1] == "local")
    local_all_cutoff_wins = sum(
        all(struct[(task, "local", metric)] > genius[(task, "local", metric)]
            for metric in expected_metrics[task])
        for task in TASKS
    )
    primary = {
        task: ("Recall@10" if task in FASHION_TASKS else "Recall@5")
        for task in TASKS
    }
    union_macro_genius = sum(genius[(task, "union", primary[task])] for task in TASKS) / 16
    union_macro_structnar = sum(struct[(task, "union", primary[task])] for task in TASKS) / 16
    union_wins = sum(
        struct[(task, "union", primary[task])] > genius[(task, "union", primary[task])]
        for task in TASKS
    )
    if (
        local_wins != 44
        or local_all_cutoff_wins != 14
        or round(union_macro_genius + 1e-12, 2) != 33.92
        or round(union_macro_structnar + 1e-12, 2) != 39.20
        or union_wins != 16
    ):
        raise AssertionError("complete-system headline summary mismatch")

    cross_rows = {
        row["method"]: row
        for row in csv.DictReader(cross_path.open(newline=""))
    }
    if "StructNAR" not in cross_rows:
        raise AssertionError("cross-paradigm StructNAR row is missing")
    cross_struct = cross_rows["StructNAR"]
    for task in TASKS:
        if not math.isclose(
            float(cross_struct[task]), struct[(task, "union", primary[task])], abs_tol=0.005
        ):
            raise AssertionError("cross-paradigm UNION value mismatch: " + task)
    if not math.isclose(float(cross_struct["macro"]), union_macro_structnar, abs_tol=0.005):
        raise AssertionError("cross-paradigm UNION macro mismatch")

    published_rows = list(csv.DictReader(published_path.open(newline="")))
    published_tasks = {
        row["dataset_task"]: row for row in published_rows
        if row["dataset_task"] in TASKS
    }
    if len(published_tasks) != len(TASKS):
        raise AssertionError("published continuous-reference extraction is incomplete")
    for method, field in (("CLIP-SF", "clip_sf"), ("BLIP-FF", "blip_ff")):
        cross_row = cross_rows.get(method)
        if cross_row is None:
            raise AssertionError("cross-paradigm continuous row is missing: " + method)
        for task in TASKS:
            if not math.isclose(
                float(cross_row[task]), float(published_tasks[task][field]), abs_tol=1e-12
            ):
                raise AssertionError("published continuous-reference mismatch: " + method)
        recomputed = sum(float(published_tasks[task][field]) for task in TASKS) / 16.0
        if not math.isclose(float(cross_row["macro"]), recomputed, abs_tol=1e-12):
            raise AssertionError("continuous-reference macro mismatch: " + method)
    provenance = yaml.safe_load(published_provenance_path.read_text())
    source_pdf = ROOT / provenance["source"]["local_pdf"]
    if (
        int(provenance["source"]["pdf_page"]) != 7
        or int(provenance["source"]["source_table"]) != 2
        or provenance["source"]["pdf_sha256"] != sha256(source_pdf)
        or provenance["extraction"]["csv_sha256"] != sha256(published_path)
    ):
        raise AssertionError("published continuous-reference provenance mismatch")

    meta = yaml.safe_load(meta_path.read_text())
    final_config = ROOT / "configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml"
    checkpoint = ROOT / "checkpoint/code_tied/gpt_hdgr_latest.pth"
    local_log = Path(meta["local_log"])
    union_log = Path(meta["union_log"])
    checks = (
        meta["method"] == "StructNAR P2L-d3 + cosine-RQC20",
        float(meta["rrg_weight"]) == 20.0,
        meta["rqc_score_mode"] == "cosine",
        int(meta["rqc_apply_from_level"]) == 3,
        int(meta["switch_depth"]) == 3,
        int(meta["batch_size"]) == 8,
        int(meta["num_beams"]) == 50,
        meta["deterministic_eval_sampling"] is True,
        meta["summary_sha256"] == sha256(structnar_path),
        sha256(local_log) == meta["local_log_sha256"],
        sha256(union_log) == meta["union_log_sha256"],
        sha256(checkpoint) == meta["checkpoint_sha256"],
    )
    if not final_config.is_file() or not all(checks):
        raise AssertionError("canonical StructNAR system metadata mismatch")

    alignment = yaml.safe_load(alignment_path.read_text())
    union_alignment = alignment["results"]["union"]
    if (
        int(union_alignment["rows"]) != 5_609_079
        or union_alignment["ids_equal"] is not True
        or not math.isclose(
            float(union_alignment["differing_code_row_percent"]),
            0.0550714297302641,
            abs_tol=1e-15,
        )
    ):
        raise AssertionError("StructNAR/GENIUS index-alignment audit mismatch")

    paper = " ".join(paper_path.read_text().split())
    # The complete-system comparison is intentionally contextual: GENIUS-R
    # predates deterministic prompt sampling and uses a separately materialized
    # semantic index.  Guard the disclosed, appropriately scoped wording rather
    # than an older superiority headline.
    paper_checks = (
        "P2L-R is higher in 44 of 48 task--cutoff cells" in paper,
        "at every cutoff on 14 of 16 tasks" in paper,
        "from 33.92 to 39.20" in paper,
        "P2L-R has a higher point estimate on all 16 tasks" in paper,
        "We use it as complete-system context" in paper,
        "The metric is R@5 except R@10 for Fashion200K and FashionIQ" in paper,
    )
    if not all(paper_checks):
        raise AssertionError("manuscript complete-system headline is stale")
    supplement = supplement_path.read_text()
    for task in TASKS:
        metric = primary[task]
        genius_value = genius[(task, "union", metric)]
        structnar_value = struct[(task, "union", metric)]
        row_pattern = (
            re.escape(TASK_DISPLAY[task])
            + r"\s*&\s*R@"
            + metric.removeprefix("Recall@")
            + r"\s*&[^&]+&[^&]+&\s*"
            + re.escape(f"{genius_value:.2f}")
            + r"\s*&\s*\\textbf\{"
            + re.escape(f"{structnar_value:.2f}")
            + r"\}"
        )
        if not re.search(row_pattern, supplement):
            raise AssertionError(
                "supplementary UNION task row is stale or missing: " + task
            )

    return {
        "local_task_cutoff_cells": 48,
        "local_wins": local_wins,
        "local_all_cutoff_wins": local_all_cutoff_wins,
        "union_primary_tasks": 16,
        "union_wins": union_wins,
        "union_macro_genius": union_macro_genius,
        "union_macro_structnar": union_macro_structnar,
        "structnar_all32_sha256": sha256(structnar_path),
        "genius_all32_sha256": sha256(genius_path),
        "comparison_sha256": sha256(comparison_path),
        "metadata_sha256": sha256(meta_path),
        "alignment_sha256": sha256(alignment_path),
        "published_continuous_reference_sha256": sha256(published_path),
        "published_continuous_provenance_sha256": sha256(published_provenance_path),
    }


def audit_semantic_index_resources() -> dict:
    """Verify the scale-index resource table and isolated load profile."""
    csv_path = RESULTS / "structnar_semantic_index_resources.csv"
    load_profile = RESULTS / "p2l_union_index_load_profile.json"
    supplement_path = ROOT / "VLDB/vldb2027/supplementary/structnar_vldb_supplement.tex"
    main_path = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
    for path in (csv_path, load_profile, supplement_path, main_path):
        require_nonempty(path)

    rows = list(csv.DictReader(csv_path.open(newline="")))
    if len(rows) != len(SIZES) or {int(row["candidate_count"]) for row in rows} != set(SIZES):
        raise AssertionError("semantic-index resource matrix is incomplete")

    gen_root = ROOT / "gen_code/STRUCTNAR/Large/Instruct"
    by_size = {int(row["candidate_count"]): row for row in rows}
    for size in SIZES:
        row = by_size[size]
        exp = gen_root / f"structnar_cirr_scale_{size:07d}" / "cand_pool"
        stem = (
            exp / "mbeir_union_cand_pool"
            if size == 5_609_079
            else exp / f"mbeir_cirrscale_{size:07d}_cand_pool"
        )
        paths = {
            "codes": Path(f"{stem}_codes.npy"),
            "ids": Path(f"{stem}_ids.npy"),
            "embeddings": Path(f"{stem}_embeddings.npy"),
            "trie": Path(f"{stem}_trie.pkl"),
        }
        if not all(path.is_file() for path in paths.values()):
            raise AssertionError("semantic-index source artifact is missing: " + str(stem))
        codes = np.load(paths["codes"], mmap_mode="r")
        item_ids = np.load(paths["ids"], mmap_mode="r")
        embeddings = np.load(paths["embeddings"], mmap_mode="r")
        prefixes = [int(row[f"prefixes_d{depth}"]) for depth in range(1, 10)]
        unique_ids = int(row["unique_identifiers"])
        nodes = int(row["trie_nodes_including_root"])
        internal = int(row["trie_internal_nodes"])

        def actual_mib(path: Path) -> float:
            return path.stat().st_size / (1024.0 * 1024.0)

        checks = (
            codes.shape == (size, 9),
            len(item_ids) == size,
            embeddings.shape == (size, 768),
            int(row["identifier_length"]) == 9,
            prefixes[-1] == unique_ids,
            0 < unique_ids <= size,
            nodes == 1 + sum(prefixes),
            internal == nodes - unique_ids,
            math.isclose(float(row["codes_mib"]), actual_mib(paths["codes"]), abs_tol=1e-9),
            math.isclose(float(row["item_ids_mib"]), actual_mib(paths["ids"]), abs_tol=1e-9),
            math.isclose(float(row["candidate_embeddings_mib"]), actual_mib(paths["embeddings"]), abs_tol=1e-9),
            math.isclose(float(row["serialized_trie_mib"]), actual_mib(paths["trie"]), abs_tol=1e-9),
            math.isclose(float(row["trie_bytes_per_candidate"]), paths["trie"].stat().st_size / size, abs_tol=1e-9),
        )
        if not all(checks):
            raise AssertionError("semantic-index resource row mismatch: " + str(size))

    final = by_size[5_609_079]
    profile = json.loads(load_profile.read_text())
    union_trie = (
        gen_root
        / "structnar_cirr_scale_5609079/cand_pool/mbeir_union_cand_pool_trie.pkl"
    )
    profile_input = profile.get("input", {})
    measurement = profile.get("measurement", {})
    loaded = profile.get("loaded_payload", {})
    scope = profile.get("scope", {})
    load_seconds = float(measurement.get("pickle_load_seconds", 0.0))
    peak_rss_mib = float(measurement.get("peak_process_rss_mib", 0.0))
    profile_checks = (
        profile.get("profile") == "isolated_semantic_id_trie_deserialization",
        Path(profile_input.get("path", "")).resolve() == union_trie.resolve(),
        profile_input.get("sha256") == sha256(union_trie),
        int(profile_input.get("serialized_bytes", -1)) == union_trie.stat().st_size,
        math.isclose(
            float(profile_input.get("serialized_mib", -1)),
            float(final["serialized_trie_mib"]),
            abs_tol=0.001,
        ),
        load_seconds > 0,
        peak_rss_mib > 0,
        loaded.get("candidate_code_shape") == [5_609_079, 9],
        "neural checkpoint and model loading" in scope.get("excluded", []),
        "query features" in scope.get("excluded", []),
        "separately stored reranker embeddings" in scope.get("excluded", []),
        "online query execution" in scope.get("excluded", []),
    )
    if not all(profile_checks):
        raise AssertionError("semantic-index load-profile scope or values mismatch")

    supplement = supplement_path.read_text()
    main = main_path.read_text()
    if (
        f"an isolated loader process takes {load_seconds:.2f}~s" not in supplement
        or f"{peak_rss_mib:,.2f}~MiB peak RSS" not in supplement
        or "not a\nmeasurement of full serving-process memory" not in supplement
        or "927.32~MiB" not in main
        or "8.02~GiB" not in main
    ):
        raise AssertionError("manuscript semantic-index resource claims are stale")

    return {
        "scale_points": len(rows),
        "candidate_count_max": 5_609_079,
        "resource_csv_sha256": sha256(csv_path),
        "load_profile_sha256": sha256(load_profile),
        "isolated_load_seconds": load_seconds,
        "isolated_loader_peak_rss_mib": peak_rss_mib,
    }


def audit_gpu_flat_trie_baseline() -> dict:
    """Verify the faithful optimized level-wise control and its disclosure."""
    result_path = RESULTS / "p2l_gpu_flat_trie_baseline.json"
    csv_path = RESULTS / "p2l_gpu_flat_trie_baseline.csv"
    equivalence_path = ROOT / "profile_output/p2l_gpu_trie_baseline/equivalence.json"
    state_path = ROOT / "profile_output/p2l_gpu_trie_baseline/state.txt"
    paper_path = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
    table_path = (
        ROOT
        / "VLDB/vldb2027/supplementary/generated/gpu_flat_trie_baseline.tex"
    )
    for path in (
        result_path,
        csv_path,
        equivalence_path,
        state_path,
        paper_path,
        table_path,
    ):
        require_nonempty(path)

    payload = json.loads(result_path.read_text())
    equivalence = json.loads(equivalence_path.read_text())
    state = state_path.read_text().strip()
    if payload.get("verdict") != "PASS" or equivalence.get("verdict") != "PASS":
        raise AssertionError("GPU-flat result or equivalence audit did not pass")
    if not state.startswith("status=complete_exact "):
        raise AssertionError("GPU-flat runner has no clean terminal state")
    comparisons = equivalence.get("comparisons", [])
    if len(comparisons) != 4 or not all(row.get("exact") for row in comparisons):
        raise AssertionError("GPU-flat full-manifest outputs are not exact")
    if any(float(row.get("max_abs_difference", -1)) != 0.0 for row in comparisons):
        raise AssertionError("GPU-flat output difference is nonzero")
    if not equivalence.get("config_comparison", {}).get("passed"):
        raise AssertionError("GPU-flat resolved configurations differ unexpectedly")
    if payload.get("equivalence_sha256") != sha256(equivalence_path):
        raise AssertionError("GPU-flat result does not hash-link its equivalence audit")

    rows = list(csv.DictReader(csv_path.open(newline="")))
    by_label = {row["label"]: row for row in rows}
    required_labels = {
        "Canonical Sequential+PCAA",
        "GPU-flat Sequential+PCAA",
        "P2L+PCAA",
    }
    if set(by_label) != required_labels:
        raise AssertionError("GPU-flat comparison rows are incomplete")
    canonical = float(by_label["Canonical Sequential+PCAA"]["generation_ms_per_query"])
    gpu_flat = float(by_label["GPU-flat Sequential+PCAA"]["generation_ms_per_query"])
    p2l = float(by_label["P2L+PCAA"]["generation_ms_per_query"])
    reduction = 100.0 * (1.0 - p2l / gpu_flat)
    if not (
        math.isclose(canonical, 59.89762446043165, abs_tol=1e-9)
        and math.isclose(gpu_flat, 59.33213812949641, abs_tol=1e-9)
        and math.isclose(p2l, 27.646574100719427, abs_tol=1e-9)
        and math.isclose(reduction, 53.40371176177925, abs_tol=1e-9)
    ):
        raise AssertionError("GPU-flat timing comparison is stale")
    for row in rows[:2]:
        if not math.isclose(float(row["recall_at_10"]), 0.43, abs_tol=1e-12):
            raise AssertionError("GPU-flat control changed Sequential R@10")

    paper = " ".join(paper_path.read_text().split())
    table = table_path.read_text()
    if not all(
        fragment in paper
        for fragment in (
            "from 59.90 to 59.33~ms/query",
            "27.65~ms/query",
            "not the main source of the measured speedup on this workload",
        )
    ):
        raise AssertionError("main-paper GPU-flat statement is stale")
    if not all(fragment in table for fragment in ("59.90", "59.33", "27.65", "byte-identical")):
        raise AssertionError("supplementary GPU-flat table is stale")

    return {
        "queries": 4170,
        "full_manifest_arrays_exact": 4,
        "canonical_ms_per_query": canonical,
        "gpu_flat_ms_per_query": gpu_flat,
        "p2l_ms_per_query": p2l,
        "p2l_reduction_vs_gpu_flat_percent": reduction,
        "result_sha256": sha256(result_path),
        "equivalence_sha256": sha256(equivalence_path),
        "table_sha256": sha256(table_path),
    }


def audit_union_frontier_runtime() -> dict:
    """Verify the three-workload UNION matrix against logs and configs."""
    csv_path = RESULTS / "p2l_vldb_union_frontier_runtime.csv"
    summary_path = RESULTS / "p2l_vldb_union_frontier_runtime_summary.json"
    equivalence_path = RESULTS / "p2l_vldb_union_index_equivalence.json"
    table_path = ROOT / "VLDB/vldb2027/supplementary/generated/union_frontier_runtime.tex"
    paper_path = ROOT / "VLDB/vldb2027/structnar_vldb.tex"
    for path in (csv_path, summary_path, equivalence_path, table_path, paper_path):
        require_nonempty(path)

    rows = list(csv.DictReader(csv_path.open(newline="")))
    tasks = ("NIGHTS-4", "EDIS-2", "WebQA-1")
    decoders = ("Sequential+PCAA", "P2L+PCAA")
    expected = {(task, decoder) for task in tasks for decoder in decoders}
    keyed = {(row["task"], row["decoder"]): row for row in rows}
    if len(rows) != 6 or set(keyed) != expected:
        raise AssertionError("UNION frontier/runtime matrix is incomplete")

    task_slugs = {"NIGHTS-4": "nights_task4", "EDIS-2": "edis_task2", "WebQA-1": "webqa_task1"}
    for (task, decoder), row in keyed.items():
        log = ROOT / row["source_log"]
        require_nonempty(log)
        text = log.read_text(errors="replace")
        p2l = decoder == "P2L+PCAA"
        require_eval_log(log, frontier=p2l)
        generation = re.findall(
            r"Generation Metrics: seconds=([0-9.]+) examples=([0-9]+)", text
        )
        retrieval = re.findall(r"Retrieval Metrics: seconds=([0-9.]+)", text)
        recalls = {
            cutoff: re.findall(rf"Retriever: Mean Recall@{cutoff}: ([0-9.]+)", text)
            for cutoff in (1, 5, 10)
        }
        latencies = re.findall(
            r"Generation Latency Distribution:.*?normalized_query_p50_ms=([0-9.]+) "
            r"normalized_query_p95_ms=([0-9.]+) normalized_query_p99_ms=([0-9.]+)",
            text,
        )
        memory = re.findall(r"Generation CUDA Memory: peak_allocated_mib=([0-9.]+)", text)
        if not generation or not retrieval or any(not values for values in recalls.values()) or not latencies or not memory:
            raise AssertionError("UNION frontier/runtime log is incomplete: " + str(log))
        seconds, examples = generation[-1]
        p50, p95, p99 = latencies[-1]
        numerical = (
            int(row["examples"]) == int(examples),
            math.isclose(float(row["generation_seconds"]), float(seconds), abs_tol=1e-9),
            math.isclose(float(row["generation_ms_per_query"]), 1000.0 * float(seconds) / int(examples), abs_tol=1e-9),
            math.isclose(float(row["retrieval_seconds"]), float(retrieval[-1]), abs_tol=1e-9),
            math.isclose(float(row["manifest_d2r_seconds"]), float(seconds) + float(retrieval[-1]), abs_tol=1e-9),
            math.isclose(float(row["r1"]), float(recalls[1][-1]), abs_tol=1e-12),
            math.isclose(float(row["r5"]), float(recalls[5][-1]), abs_tol=1e-12),
            math.isclose(float(row["r10"]), float(recalls[10][-1]), abs_tol=1e-12),
            math.isclose(float(row["p50_ms_per_query"]), float(p50), abs_tol=1e-9),
            math.isclose(float(row["p95_ms_per_query"]), float(p95), abs_tol=1e-9),
            math.isclose(float(row["p99_ms_per_query"]), float(p99), abs_tol=1e-9),
            math.isclose(float(row["peak_allocated_mib"]), float(memory[-1]), abs_tol=1e-9),
        )
        if not all(numerical):
            raise AssertionError("UNION frontier/runtime CSV-log mismatch: " + str(log))

        stamp_match = re.search(r"vldb_union_frontier_(\d{8}_\d{6})", str(log))
        if not stamp_match:
            raise AssertionError("cannot resolve UNION frontier run stamp: " + str(log))
        stamp = stamp_match.group(1)
        decoder_slug = "p2l" if p2l else "sequential"
        config_dir = ROOT / f"configs/structnar/generated/vldb_union_frontier_{stamp}"
        configs = list(config_dir.glob(f"*{task_slugs[task]}_{decoder_slug}_w20_*_union.yaml"))
        if len(configs) != 1:
            raise AssertionError("cannot uniquely resolve UNION frontier config")
        cfg = yaml.safe_load(configs[0].read_text())
        model, retrieval_cfg = cfg["model"], cfg["retrieval_config"]
        test_cfg = retrieval_cfg["test_datasets_config"]
        expected_blocks = [1, 1, 1, 6] if p2l else [1] * 9
        checks = (
            cfg["dataloader_config"]["batch_size"] == 8,
            cfg["data_config"]["deterministic_eval_sampling"] is True,
            retrieval_cfg["num_beams"] == 50,
            retrieval_cfg["rerank"] is True,
            retrieval_cfg["use_fp16"] is True,
            test_cfg["datasets_name"] == [task_slugs[task]],
            test_cfg["correspond_cand_pools_name"] == ["UNION"],
            float(model["rrg_weight"]) == 20.0,
            model["use_rrg"] is True,
            model["rrg_normalize"] is True,
            model["rqc_score_mode"] == "cosine",
            int(model["rqc_apply_from_level"]) == 3,
            model["hierarchy_block_sizes"] == expected_blocks,
            int(model["tcis_max_leaves"]) == 0,
        )
        if not all(checks):
            raise AssertionError("UNION frontier/runtime configuration mismatch: " + str(configs[0]))

        if p2l:
            frontier = re.findall(
                r"TCIS Query Enumeration:.*?candidates_mean=([0-9.]+).*?"
                r"candidates_p95=([0-9.]+).*?candidates_p99=([0-9.]+).*?"
                r"candidates_max=([0-9.]+)",
                text,
            )
            if not frontier:
                raise AssertionError("missing UNION frontier distribution")
            mean, leaf_p95, leaf_p99, leaf_max = frontier[-1]
            if not all(
                math.isclose(float(row[field]), float(value), abs_tol=1e-6)
                for field, value in (
                    ("leaves_mean", mean), ("leaves_p95", leaf_p95),
                    ("leaves_p99", leaf_p99), ("leaves_max", leaf_max),
                )
            ):
                raise AssertionError("UNION frontier CSV-log mismatch: " + str(log))

    summaries = {row["task"]: row for row in json.loads(summary_path.read_text())}
    if set(summaries) != set(tasks):
        raise AssertionError("UNION frontier/runtime summary task set mismatch")
    expected_claims = {
        "NIGHTS-4": (1.51, 51.3, 48.0, 499),
        "EDIS-2": (16.69, 28.6, 21.0, 11097),
        "WebQA-1": (19.64, 31.1, 23.9, 10339),
    }
    for task, (delta, reduction, p95_reduction, leaves) in expected_claims.items():
        item = summaries[task]
        if not (
            math.isclose(float(item["r10_delta_points"]), delta, abs_tol=0.011)
            and math.isclose(float(item["generation_reduction_percent"]), reduction, abs_tol=0.051)
            and math.isclose(float(item["p95_reduction_percent"]), p95_reduction, abs_tol=0.051)
            and round(float(item["p2l_leaves_mean"])) == leaves
        ):
            raise AssertionError("UNION frontier/runtime summary is stale: " + task)

    paper = " ".join(paper_path.read_text().split())
    table = " ".join(table_path.read_text().split())
    required_paper = (
        "reduces decoder-side generation by 51.3\\%",
        "11,097 and 10,339 leaves/query",
        "falls by 28.6\\% and 31.1\\%",
        "rises by 16.69 and 19.64 points",
        "narrow to 21.0\\% and 23.9\\%",
        "peak device memory to 4.50 and 3.68~GiB",
    )
    if any(fragment not in paper for fragment in required_paper):
        raise AssertionError("main-paper UNION frontier/runtime statement is stale")
    required_table = (
        "Peak (GiB)",
        "EDIS-2 & P2L+PCAA & 56.43 & 43.94 & 223.40 & 42.32 & 50.44 & 67.69 & 4.50",
        "WebQA-1 & P2L+PCAA & 48.19 & 43.67 & 184.19 & 42.07 & 49.84 & 56.90 & 3.68",
    )
    if any(fragment not in table for fragment in required_table):
        raise AssertionError("supplementary UNION memory disclosure is stale")

    equivalence = json.loads(equivalence_path.read_text())
    equivalence_rows = equivalence.get("rows", [])
    equivalence_summary = equivalence.get("summary", {})
    if not (
        equivalence.get("status") == "PASS"
        and len(equivalence_rows) == 6
        and equivalence_summary.get("run_count") == 6
        and equivalence_summary.get("unique_sha256") == 1
        and equivalence_summary.get("candidate_count") == 5_609_079
        and equivalence_summary.get("identifier_length") == 9
        and len({row.get("sha256") for row in equivalence_rows}) == 1
        and {
            (row.get("task"), row.get("policy")) for row in equivalence_rows
        }
        == {
            (task, policy)
            for task in ("nights_task4", "edis_task2", "webqa_task1")
            for policy in ("sequential", "p2l")
        }
    ):
        raise AssertionError("UNION matched-index equivalence audit is stale")

    return {
        "cells": 6,
        "tasks": list(tasks),
        "csv_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary_path),
        "index_equivalence_sha256": sha256(equivalence_path),
        "candidate_code_sha256": equivalence_summary["sha256"],
        "table_sha256": sha256(table_path),
        "source_log_sha256": {row["task"] + "/" + row["decoder"]: sha256(ROOT / row["source_log"]) for row in rows},
    }


def audit_revision_controls() -> dict:
    bootstrap_path = RESULTS / "structnar_matched_all16_bootstrap.csv"
    rerank_path = RESULTS / "structnar_equal_rerank_budget.csv"
    rerank_provenance_path = RESULTS / "structnar_equal_rerank_budget.provenance.json"
    suffix_path = RESULTS / "structnar_suffix_interaction_ablation.csv"
    suffix_provenance_path = RESULTS / "structnar_suffix_interaction_ablation.provenance.json"
    batch_path = RESULTS / "structnar_tip_batch_scaling.csv"
    rerank_table = (
        ROOT
        / "VLDB/vldb2027/supplementary/generated/equal_rerank_budget.tex"
    )
    for path in (
        bootstrap_path,
        rerank_path,
        rerank_provenance_path,
        suffix_path,
        suffix_provenance_path,
        batch_path,
        rerank_table,
    ):
        require_nonempty(path)

    bootstrap = list(csv.DictReader(bootstrap_path.open(newline="")))
    by_scope = {row["scope"]: row for row in bootstrap}
    if len(bootstrap) != 17 or set(TASKS) - set(by_scope) or "task_macro" not in by_scope:
        raise AssertionError("all-task paired bootstrap is incomplete")
    macro = by_scope["task_macro"]
    if not (
        int(macro["bootstrap_samples"]) == 10_000
        and math.isclose(float(macro["delta_r10_points"]), 3.4761, abs_tol=0.0001)
        and math.isclose(float(macro["ci95_low_points"]), 3.2197, abs_tol=0.0001)
        and math.isclose(float(macro["ci95_high_points"]), 3.7256, abs_tol=0.0001)
    ):
        raise AssertionError("task-macro paired bootstrap is stale")
    ci_excludes_zero = sum(
        float(by_scope[task]["ci95_low_points"]) > 0
        or float(by_scope[task]["ci95_high_points"]) < 0
        for task in TASKS
    )
    if ci_excludes_zero != 13:
        raise AssertionError("per-task paired-bootstrap count is stale")

    rerank = list(csv.DictReader(rerank_path.open(newline="")))
    if len(rerank) != 8:
        raise AssertionError("equal-reranking-budget audit is incomplete")
    rerank_by_task: dict[str, list[dict[str, str]]] = {}
    for row in rerank:
        rerank_by_task.setdefault(row["task"], []).append(row)
        if int(row["missing_generated_codes"]) != 0:
            raise AssertionError("generated code missing during reranking audit")
        if int(row["queries_with_duplicate_beam_codes"]) != 0:
            raise AssertionError("duplicate identifier in generated beam")
    max_mean_gap = max(
        abs(
            float(rows[0]["expanded_items_mean"])
            - float(rows[1]["expanded_items_mean"])
        )
        for rows in rerank_by_task.values()
    )
    max_r10_change = max(
        abs(float(row["budget50_minus_full_r10"])) for row in rerank
    )
    if max_mean_gap > 0.14 or max_r10_change > 0.08:
        raise AssertionError("equal-reranking-budget result is stale")

    suffix = list(csv.DictReader(suffix_path.open(newline="")))
    expected_suffix = {
        "cirr_task7": (44.34, 44.80),
        "nights_task4": (50.28, 50.85),
        "edis_task2": (50.17, 50.45),
        "webqa_task1": (43.83, 45.66),
    }
    if len(suffix) != 4 or {row["task"] for row in suffix} != set(expected_suffix):
        raise AssertionError("suffix-interaction control is incomplete")
    for row in suffix:
        expected = expected_suffix[row["task"]]
        if not (
            math.isclose(float(row["canonical_p2l_r10"]), expected[0], abs_tol=0.001)
            and math.isclose(float(row["isolated_suffix_r10"]), expected[1], abs_tol=0.001)
        ):
            raise AssertionError("suffix-interaction result is stale")
    suffix_provenance = json.loads(suffix_provenance_path.read_text())
    if any(
        not all(item["matched_arrays"].values())
        for item in suffix_provenance["tasks"].values()
    ):
        raise AssertionError("suffix-interaction realized inputs do not match")
    config_dir = ROOT / "configs/structnar/generated/vldb_suffix_interaction"
    for task in expected_suffix:
        configs = list(config_dir.glob(f"*{task}.yaml"))
        if len(configs) != 1:
            raise AssertionError("cannot resolve suffix-interaction config")
        cfg = yaml.safe_load(configs[0].read_text())
        if not cfg["model"].get("tcis_isolate_suffix_states", False):
            raise AssertionError("suffix-interaction control flag is disabled")

    batch_rows = [
        row
        for row in csv.DictReader(batch_path.open(newline=""))
        if int(row["batch"]) == 1
        and row["dataset"] in expected_suffix
        and row["method"] in {"sequential", "p2l"}
    ]
    if len(batch_rows) != 8:
        raise AssertionError("batch-1 latency evidence is incomplete")
    batch_by_key = {(row["dataset"], row["method"]): row for row in batch_rows}
    p50_reductions = []
    p95_reductions = []
    for task in expected_suffix:
        seq = batch_by_key[(task, "sequential")]
        p2l = batch_by_key[(task, "p2l")]
        p50_reductions.append(
            100.0
            * (1.0 - float(p2l["query_p50_ms"]) / float(seq["query_p50_ms"]))
        )
        p95_reductions.append(
            100.0
            * (1.0 - float(p2l["query_p95_ms"]) / float(seq["query_p95_ms"]))
        )
    if not (
        35.5 < min(p50_reductions) < 35.7
        and 43.4 < max(p50_reductions) < 43.6
        and 24.2 < min(p95_reductions) < 24.4
        and 37.0 < max(p95_reductions) < 37.2
    ):
        raise AssertionError("batch-1 latency summary is stale")

    main_text = " ".join(
        (ROOT / "VLDB/vldb2027/structnar_vldb.tex").read_text().split()
    )
    required_main = (
        "95\\% CI: [3.22, 3.73]",
        "13 of the 16 per-task intervals exclude zero",
        "43.6\\%--52.4\\%",
        "An equal 50-item reranking cap changes R@10 by at most 0.08 point",
        "The isolated variant has an R@10 point estimate 0.28--1.83 points higher than the interacting variant",
        "preliminary path-score weight $\\lambda=10$",
        "In LOCAL profiling, batch-1 P50 falls by 35.6\\%--43.5\\%",
        "8,744~MiB peak host RSS",
    )
    if any(fragment not in main_text for fragment in required_main):
        raise AssertionError("revision-control claim is stale in main paper")

    return {
        "bootstrap_sha256": sha256(bootstrap_path),
        "per_task_ci_excludes_zero": ci_excludes_zero,
        "rerank_budget_sha256": sha256(rerank_path),
        "rerank_budget_provenance_sha256": sha256(rerank_provenance_path),
        "max_expanded_item_mean_gap": max_mean_gap,
        "max_budget50_r10_change": max_r10_change,
        "suffix_interaction_sha256": sha256(suffix_path),
        "suffix_interaction_provenance_sha256": sha256(suffix_provenance_path),
        "batch1_latency_sha256": sha256(batch_path),
    }


def main() -> int:
    run_id = (RESULTS / "structnar_vldb_active_run_id.txt").read_text().strip()
    report = {
        "audit_type": "deterministic_provenance_and_configuration",
        "independent_semantic_review": False,
        "run_id": run_id,
        "matched": audit_matched(run_id),
        "scale": audit_scale(run_id),
        "rq_distance": audit_rq_distance(run_id),
        "raw_cosine_b50": audit_raw_cosine_b50(),
        "pcaa_b50_validation": audit_pcaa_b50_validation(),
        "configuration": audit_configuration_evidence(),
        "ann_reference": audit_ann_reference(),
        "local_runtime_repeats": audit_local_runtime_repeats(),
        "matched_index_equivalence": audit_matched_index_equivalence(),
        "complete_suffix_evidence": audit_complete_suffix_evidence(),
        "final_diagnostic_evidence": audit_final_diagnostic_evidence(),
        "cirr_scale_input_manifest": audit_cirr_scale_input_manifest(),
        "system_level": audit_system_level_results(),
        "semantic_index_resources": audit_semantic_index_resources(),
        "gpu_flat_trie_baseline": audit_gpu_flat_trie_baseline(),
        "union_frontier_runtime": audit_union_frontier_runtime(),
        "revision_controls": audit_revision_controls(),
        "verdict": "PASS",
    }
    AUDITS.mkdir(parents=True, exist_ok=True)
    json_path = AUDITS / ("vldb_evidence_deterministic_audit_" + run_id + ".json")
    md_path = AUDITS / ("vldb_evidence_deterministic_audit_" + run_id + ".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(
        "# VLDB Evidence Deterministic Audit\n\n"
        + "- Run ID: " + run_id + "\n"
        + "- Verdict: PASS\n"
        + "- Matched cells: 32/32\n"
        + "- Semantic-ID scale cells: 10/10\n"
        + "- FlatIP scale cells: 5/5\n"
        + "- Scale-pool invariant: identical CIRR query manifest and nested unique candidate IDs\n"
        + "- RQ-distance validation/test cells: 36/4\n"
        + "- Raw-cosine B=50 validation/test cells: 14/4; zero-score beams hash-match the RQ control\n"
        + "- Training/final configuration: resolved YAML, checkpoint, and training-code hashes verified\n"
        + "- CPU ANN reference cells: FlatIP 1/1, IVF-Flat 5/5\n"
        + "- LOCAL final-runtime repeats: 12/12 raw rows across four configurations\n"
        + "- Matched realized inputs: candidate codes, query IDs/features, and logical Tries equal on 16/16 tasks\n"
        + "- Complete-suffix intervention: 8/8 cells with matched realized query inputs and hash-linked logs\n"
        + "- Final diagnostics: six EDIS/WebQA PCAA repeats, 32 batch-scaling cells, four component-profile rows, and two hash-linked Sequential+PCAA reranked rows\n"
        + "- CIRR scale inputs: identical 4,170-query manifest and four nested-pool containment checks\n"
        + "- Complete-system evidence: 48 LOCAL cells and 16 UNION primary metrics\n"
        + "- Continuous references: 16-task CLIP-SF/BLIP-FF extraction linked to GENIUS Table 2 and source-PDF hash\n"
        + "- Semantic-index resource scale points: 5/5; UNION load profile scoped to an isolated loader process\n"
        + "- GPU-flat level-wise control: 4/4 full-manifest output arrays exact; result, table, and equivalence hashes verified\n"
        + "- Cross-workload UNION runtime: 6/6 matched cells with log, config, frontier-tail, and manuscript checks\n"
        + "- Revision controls: all-task bootstrap, suffix-state isolation, equal item-reranking budget, and batch-1 latency verified\n"
        + "- Scope: provenance, hashes, official qrels, frozen configuration, "
          "and matrix completeness.\n"
        + "- Independence: deterministic executor audit; not an independent semantic review.\n"
    )
    print(json_path)
    print(md_path)
    print("RESULT=PASS_VLDB_DETERMINISTIC_AUDIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
