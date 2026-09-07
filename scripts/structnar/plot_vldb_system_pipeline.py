"""Draw the full-width offline/online P2L system pipeline."""

from vldb_figure_style import ASSET_DIR, COLORS, arrow, place_image, rounded_box, save, setup_canvas


def box(ax, x, y, w, h, text, *, fill, stroke, size=7.2, weight="normal", sub=None):
    rounded_box(ax, x, y, w, h, facecolor=fill, edgecolor=stroke, radius=1.0)
    ax.text(x + w / 2, y + h * (0.58 if sub else 0.5), text, ha="center", va="center",
            fontsize=size, color=COLORS["ink"], weight=weight)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=5.8,
                color=COLORS["muted"])


def routed_arrow(ax, points, *, color, linewidth=0.9, dashed=False):
    """Draw an orthogonal dependency route with an arrow only at its endpoint."""
    linestyle = (0, (3, 2)) if dashed else "-"
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=2,
        )
    (x1, y1), (x2, y2) = points[-2], points[-1]
    final_arrow = arrow(
        ax,
        x1,
        y1,
        x2,
        y2,
        color=color,
        linewidth=linewidth,
        zorder=2,
    )
    final_arrow.set_linestyle(linestyle)


def main():
    fig, ax = setup_canvas((7.15, 2.62), ylim=(0, 58))

    # Offline preparation lane.
    rounded_box(ax, 1, 32, 98, 24, facecolor="#FAFBFC", edgecolor="#B6C0CB", radius=1.3)
    ax.text(3, 53.3, "OFFLINE PREPARATION", ha="left", va="center", fontsize=8.0,
            weight="bold", color=COLORS["muted"])

    box(ax, 4, 39, 10, 9, "Training pairs", fill="white", stroke="#AEB8C4",
        size=6.8, sub="query + target ID")
    box(ax, 17, 39, 13, 9, "Masked-ID\npredictor training", fill=COLORS["red_light"],
        stroke=COLORS["red"], size=6.7, weight="bold")
    box(ax, 31.5, 39, 12, 9, "Predictor\ncheckpoint", fill=COLORS["red_light"],
        stroke=COLORS["red"], size=6.7)
    arrow(ax, 14, 43.5, 17, 43.5)
    arrow(ax, 30, 43.5, 31.5, 43.5)

    place_image(ax, ASSET_DIR / "candidate_retriever_2.jpg", 48, 39, 6.5, 9)
    ax.text(51.25, 49.3, "Candidates", ha="center", va="bottom", fontsize=6.3,
            color=COLORS["muted"])
    box(ax, 57, 39, 8, 9, "Frozen\nencoder", fill=COLORS["gray_light"],
        stroke="#8491A0", size=6.7)
    box(ax, 67, 39, 11, 9, "RQ50 IDs", fill=COLORS["orange_light"],
        stroke=COLORS["orange"], size=6.7, weight="bold", sub="Trie + ID buckets")
    box(ax, 90.3, 39, 8.2, 9, "Embedding\ncache", fill=COLORS["gray_light"],
        stroke="#8491A0", size=6.3)
    arrow(ax, 54.5, 43.5, 57, 43.5)
    arrow(ax, 65, 43.5, 67, 43.5)
    routed_arrow(
        ax,
        [(61, 48), (61, 50.7), (94.4, 50.7), (94.4, 48.1)],
        color="#8491A0",
        linewidth=0.85,
    )
    ax.text(80.0, 51.0, "candidate embeddings", ha="center", va="bottom",
            fontsize=5.7, color=COLORS["muted"],
            bbox=dict(facecolor="#FAFBFC", edgecolor="none", pad=0.5))

    # Online query execution lane.
    rounded_box(ax, 1, 2, 98, 27, facecolor="#F7FAFC", edgecolor="#7D8997", radius=1.3)
    ax.text(3, 26.3, "ONLINE QUERY EXECUTION", ha="left", va="center", fontsize=8.0,
            weight="bold", color=COLORS["blue"])

    place_image(ax, ASSET_DIR / "query_running_dog.jpg", 4, 8, 8.5, 10)
    ax.text(8.25, 20.2, "multimodal query", ha="center", va="center", fontsize=6.1,
            color=COLORS["muted"])
    box(ax, 15.5, 8, 11, 10, "Frozen query\npipeline", fill=COLORS["gray_light"],
        stroke="#8491A0", size=6.6)
    box(ax, 30.5, 7, 14, 12, "P2L prefix\ntraversal", fill=COLORS["blue_light"],
        stroke=COLORS["blue"], size=7.0, weight="bold", sub="$s$ guided rounds")
    box(ax, 47.5, 7, 13, 12, "Parallel suffix\nscoring", fill=COLORS["orange_light"],
        stroke=COLORS["orange"], size=7.0, weight="bold", sub="one guided round")
    box(ax, 65.5, 7, 14, 12, "Complete-leaf\nranking", fill=COLORS["green_light"],
        stroke=COLORS["green"], size=6.5, weight="bold", sub="optional PCAA")
    box(ax, 81, 8, 8, 10, "ID bucket\nlookup", fill=COLORS["purple_light"],
        stroke=COLORS["purple"], size=6.5)
    box(ax, 91, 8, 7, 10, "Rerank", fill="white", stroke="#8491A0", size=6.6,
        weight="bold")

    for x1, x2 in [(12.5, 15.5), (26.5, 30.5), (44.5, 47.5), (60.5, 65.5),
                   (79.5, 81), (89, 91)]:
        arrow(ax, x1, 13, x2, 13)

    # Offline artifacts feed the aligned online operators through vertical,
    # dashed dependency arrows.  Keeping these routes orthogonal avoids the
    # three crossing diagonals in the earlier rendering.
    routed_arrow(
        ax,
        [(37.5, 38.9), (37.5, 19.2)],
        color=COLORS["red"],
        linewidth=0.85,
        dashed=True,
    )
    routed_arrow(
        ax,
        [(72.5, 38.9), (72.5, 19.2)],
        color=COLORS["orange"],
        linewidth=0.85,
        dashed=True,
    )
    routed_arrow(
        ax,
        [(94.4, 38.9), (94.4, 18.2)],
        color="#8491A0",
        linewidth=0.85,
        dashed=True,
    )

    ax.text(37, 4.6, "Trie-constrained prefix", ha="center", va="center", fontsize=5.8,
            color=COLORS["blue"])
    ax.text(54, 4.6, "stored descendants only", ha="center", va="center", fontsize=5.8,
            color=COLORS["orange"])
    ax.text(72.5, 4.6, "global top-$K$ IDs", ha="center", va="center", fontsize=5.8,
            color=COLORS["green"])

    save(fig, "ours_pipeline.pdf")


if __name__ == "__main__":
    main()
