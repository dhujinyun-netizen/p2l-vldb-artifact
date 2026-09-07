"""Shared drawing primitives for StructNAR PVLDB system diagrams."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "VLDB" / "vldb2027"
FIG_DIR = PAPER_DIR / "figures"
ASSET_DIR = FIG_DIR / "assets"

COLORS = {
    "ink": "#1F2937",
    "muted": "#5B6573",
    "line": "#677487",
    "panel": "#F8FAFC",
    "blue": "#2878B5",
    "blue_light": "#E8F2FA",
    "orange": "#E9812A",
    "orange_light": "#FFF1E5",
    "green": "#2A8C68",
    "green_light": "#E8F6F0",
    "red": "#C84B4B",
    "red_light": "#FBEAEA",
    "purple": "#6F5AA8",
    "purple_light": "#F0ECF8",
    "gray_light": "#EFF2F5",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def setup_canvas(figsize, xlim=(0, 100), ylim=(0, 100)):
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    facecolor="white",
    edgecolor=None,
    linewidth=0.9,
    radius=1.5,
    zorder=1,
):
    edgecolor = edgecolor or COLORS["line"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.25,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=None, linewidth=1.0, style="-|>", zorder=2):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=8,
        linewidth=linewidth,
        color=color or COLORS["line"],
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def crop_nonblack(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image)
    mask = np.max(arr, axis=2) > 18
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    return image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def place_image(ax, path: Path, x, y, w, h, *, border=True, zorder=3):
    image = crop_nonblack(path)
    ax.imshow(image, extent=(x, x + w, y, y + h), aspect="auto", zorder=zorder)
    if border:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="none",
                edgecolor="#CBD2DA",
                linewidth=0.7,
                zorder=zorder + 1,
            )
        )


def save(fig, name: str):
    out = FIG_DIR / name
    fig.savefig(out, dpi=300, transparent=False)
    plt.close(fig)
    return out
