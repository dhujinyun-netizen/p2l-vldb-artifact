#!/usr/bin/env python3
"""Draw verified CIRR-7 recovery and counterexample cases for the supplement."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MBEIR_DATA_DIR", ROOT / "mbeir_data"))
BASE = ROOT / "retrieval_results/structnar_tcis/cirr_task7/STRUCTNAR/Large/Instruct"
RUNS = {
    "Sequential+PCAA": BASE
    / (
        "structnar_cirr_task7_p2l_d9_rrg_w20_tip_pareto_sequential_k50_rqc20/"
        "run_files/mbeir_cirr_task7_single_pool_test_run.txt"
    ),
    "P2L+PCAA": BASE
    / (
        "structnar_cirr_task7_p2l_d3_rrg_w20_tip_pareto_p2l_k50_rqc20/"
        "run_files/mbeir_cirr_task7_single_pool_test_run.txt"
    ),
}
CASES = (
    ("8:17", "Recovery"),
    ("8:1053", "Counterexample"),
)
OUT = ROOT / "VLDB/vldb2027/figures/qualitative_cases.pdf"

INK = "#1F2937"
MUTED = "#5B6573"
BLUE = "#2878B5"
GREEN = "#2A8C68"
RED = "#C84B4B"
NEUTRAL = "#A7B1BD"


def unhash_did(value: str) -> str:
    integer = int(value)
    return f"{integer // 10_000_000}:{integer % 10_000_000}"


def load_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        run.setdefault(fields[0], []).append(unhash_did(fields[2]))
    return run


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (720, 480), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (720, 480), "white")
    canvas.paste(
        image,
        ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2),
    )
    return canvas


def rank_of(identifier: str, ranking: list[str]) -> int | None:
    return ranking.index(identifier) + 1 if identifier in ranking else None


def main() -> None:
    queries = {
        row["qid"]: row
        for row in map(
            json.loads,
            (DATA / "query/test/mbeir_cirr_task7_test.jsonl").read_text().splitlines(),
        )
    }
    candidates = {
        row["did"]: row
        for row in map(
            json.loads,
            (DATA / "cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    runs = {name: load_run(path) for name, path in RUNS.items()}

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.15, 2.95),
        layout="constrained",
        gridspec_kw={"wspace": 0.035, "hspace": 0.08},
    )
    headers = ("Query + modification", "Ground truth", "Sequential+PCAA", "P2L+PCAA")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=7.2, weight="bold", color=INK, pad=4)

    for row_index, (qid, case_name) in enumerate(CASES):
        query = queries[qid]
        target = query["pos_cand_list"][0]
        sequential = runs["Sequential+PCAA"][qid]
        p2l = runs["P2L+PCAA"][qid]
        seq_rank = rank_of(target, sequential)
        p2l_rank = rank_of(target, p2l)

        if case_name == "Recovery":
            assert seq_rank is None and p2l_rank == 1
        else:
            assert seq_rank == 1 and p2l_rank is None

        identifiers = (None, target, sequential[0], p2l[0])
        image_paths = (
            DATA / query["query_img_path"],
            DATA / candidates[target]["img_path"],
            DATA / candidates[sequential[0]]["img_path"],
            DATA / candidates[p2l[0]]["img_path"],
        )
        ranks = (None, None, seq_rank, p2l_rank)

        for column, (axis, image_path) in enumerate(zip(axes[row_index], image_paths)):
            axis.imshow(fit_image(image_path))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.05 if column >= 2 else 0.65)
                if column < 2:
                    spine.set_color(NEUTRAL)
                else:
                    spine.set_color(GREEN if ranks[column] == 1 else RED)

        badge_color = BLUE if case_name == "Recovery" else RED
        axes[row_index, 0].text(
            0.025,
            0.95,
            f"({chr(97 + row_index)}) {case_name}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            weight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": badge_color, "edgecolor": "none"},
        )
        query_text = "\n".join(textwrap.wrap(query["query_txt"], width=34))
        axes[row_index, 0].set_xlabel(query_text, fontsize=5.8, color=INK, labelpad=3)
        axes[row_index, 1].set_xlabel("Relevant item", fontsize=6.1, color=MUTED, labelpad=3)
        axes[row_index, 2].set_xlabel(
            f"GT rank: {seq_rank if seq_rank is not None else '>10'}",
            fontsize=6.2,
            weight="bold",
            color=GREEN if seq_rank == 1 else RED,
            labelpad=3,
        )
        axes[row_index, 3].set_xlabel(
            f"GT rank: {p2l_rank if p2l_rank is not None else '>10'}",
            fontsize=6.2,
            weight="bold",
            color=GREEN if p2l_rank == 1 else RED,
            labelpad=3,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
