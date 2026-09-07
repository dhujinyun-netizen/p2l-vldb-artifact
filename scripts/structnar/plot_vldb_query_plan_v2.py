"""Draw the claim-first, single-column P2L overview used as Fig. 1."""

from matplotlib.patches import Circle, Rectangle

from vldb_figure_style import (
    ASSET_DIR,
    COLORS,
    arrow,
    place_image,
    rounded_box,
    save,
    setup_canvas,
)


def code_cell(ax, x, y, label, face, edge, width=7.2, height=5.4, fontsize=6.2):
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
        radius=0.8,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        zorder=4,
    )


def evidence_row(ax, x, y, pattern, label, selected):
    ax.text(x, y + 2.3, label, ha="left", va="center", fontsize=6.0,
            color=COLORS["ink"], weight="bold")
    start = x + 20.5
    for index, strength in enumerate(pattern):
        color = COLORS["blue"] if strength == "H" else "#F2A65A"
        ax.add_patch(
            Rectangle(
                (start + index * 5.0, y),
                4.1,
                4.6,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        )
    mark_x = start + len(pattern) * 5.0 + 2.0
    mark = "KEEP" if selected else "DROP"
    mark_color = COLORS["green"] if selected else COLORS["red"]
    ax.text(mark_x + 2.5, y + 2.3, mark, ha="left", va="center", fontsize=5.4,
            color=mark_color, weight="bold", zorder=4)


def main():
    fig, ax = setup_canvas((3.35, 3.70))

    # Query and retained prefix establish that the two execution paths start
    # from the same model state and Candidate Trie node.
    ax.text(4, 97.5, "QUERY", ha="left", va="top", fontsize=7.4,
            weight="bold", color=COLORS["ink"])
    place_image(ax, ASSET_DIR / "query_golden_retriever.jpg", 4, 82.5, 17, 12.0)
    rounded_box(ax, 25, 84.0, 71, 9.0, facecolor="white", edgecolor="#9AA8B8")
    ax.text(60.5, 88.5, "Same Golden Retriever; focus on its head.",
            ha="center", va="center", fontsize=6.5, color=COLORS["ink"])
    # Connect the query instruction directly to the first retained code.  The
    # previous endpoint stopped between the label and the prefix cells, which
    # made the arrow appear detached at single-column size.
    arrow(ax, 61.75, 84.0, 61.75, 77.9, color=COLORS["line"], linewidth=0.9)

    ax.text(5, 75.0, "SAME RETAINED TRIE PREFIX", ha="left", va="center",
            fontsize=5.9, weight="bold", color=COLORS["muted"])
    for index, label in enumerate(("M", "$c_1$", "$c_2$")):
        code_cell(
            ax,
            58 + index * 9.0,
            72.2,
            label,
            COLORS["blue_light"],
            COLORS["blue"],
            width=7.5,
            height=5.6,
        )

    # Level-wise path: an early weak code causes irreversible pruning before
    # the remaining evidence can be examined.
    rounded_box(ax, 3, 43.0, 94, 25.5, facecolor="#FFF8F7", edgecolor=COLORS["red"])
    ax.text(7, 65.0, "LEVEL-WISE TRAVERSAL", ha="left", va="center",
            fontsize=7.3, weight="bold", color=COLORS["ink"])
    rounded_box(ax, 73, 62.1, 19.5, 5.0, facecolor=COLORS["red_light"],
                edgecolor=COLORS["red"], radius=0.8)
    ax.text(82.75, 64.6, "9 total rounds", ha="center", va="center", fontsize=5.2,
            color=COLORS["red"], weight="bold")
    ax.text(7, 57.8, "Next-code evidence", ha="left", va="center", fontsize=6.1,
            color=COLORS["muted"])
    code_cell(ax, 36, 55.0, "weak", "#FCE7E5", COLORS["red"], width=12.5)
    arrow(ax, 48.5, 57.7, 56.9, 57.7, color=COLORS["red"], linewidth=0.9)
    rounded_box(ax, 57, 54.8, 17.0, 5.8, facecolor=COLORS["red_light"],
                edgecolor=COLORS["red"], radius=0.8)
    ax.text(65.5, 57.7, "target pruned", ha="center", va="center", fontsize=5.8,
            color=COLORS["red"], weight="bold")
    ax.text(78.0, 57.7, "×", ha="center", va="center", fontsize=10,
            color=COLORS["red"], weight="bold")
    ax.plot([38, 85], [49.9, 49.9], color="#CC6B66", lw=0.9, ls=(0, (3, 2)))
    ax.text(61.5, 46.7, "later supporting codes are never considered",
            ha="center", va="center", fontsize=6.0, color=COLORS["red"])

    # P2L path: all unresolved-position evidence is compared only across real
    # stored leaves; high/low blocks encode schematic evidence, not measurements.
    rounded_box(ax, 3, 5.0, 94, 33.5, facecolor="#F5FAFD", edgecolor=COLORS["blue"])
    ax.text(7, 35.0, "PREFIX-TO-LEAF (P2L)", ha="left", va="center",
            fontsize=7.3, weight="bold", color=COLORS["blue"])
    rounded_box(ax, 73, 32.1, 19.5, 5.0, facecolor=COLORS["green_light"],
                edgecolor=COLORS["green"], radius=0.8)
    ax.text(82.75, 34.6, "4 total rounds", ha="center", va="center", fontsize=5.2,
            color=COLORS["green"], weight="bold")
    ax.text(7, 28.8, "One suffix round", ha="left", va="center", fontsize=6.0,
            color=COLORS["muted"])
    for index in range(6):
        code_cell(
            ax,
            33 + index * 8.1,
            26.0,
            f"$c_{index + 3}$",
            COLORS["gray_light"],
            "#8B98A8",
            width=6.7,
            height=5.4,
            fontsize=5.7,
        )
    ax.text(7, 23.0, "Suffix evidence per stored ID", ha="left", va="center", fontsize=5.7,
            color=COLORS["muted"])
    ax.add_patch(Rectangle((54, 21.2), 3.2, 3.2, facecolor=COLORS["blue"],
                           edgecolor="white", linewidth=0.4, zorder=3))
    ax.text(58, 22.8, "High", ha="left", va="center", fontsize=5.2,
            color=COLORS["muted"])
    ax.add_patch(Rectangle((68, 21.2), 3.2, 3.2, facecolor="#F2A65A",
                           edgecolor="white", linewidth=0.4, zorder=3))
    ax.text(72, 22.8, "Low", ha="left", va="center", fontsize=5.2,
            color=COLORS["muted"])
    ax.text(94, 22.8, "schematic", ha="right", va="center", fontsize=4.9,
            color=COLORS["muted"], style="italic")
    evidence_row(ax, 21, 16.4, ("L", "H", "H", "H", "H", "H"), "relevant ID", True)
    evidence_row(ax, 21, 10.5, ("H", "L", "L", "L", "L", "L"), "competing ID", False)
    place_image(ax, ASSET_DIR / "candidate_retriever_3.jpg", 6, 9.0, 12.0, 9.2)
    ax.text(50, 7.4, "compare complete valid IDs after all suffix evidence arrives",
            ha="center", va="center", fontsize=5.6, color=COLORS["blue"], weight="bold")

    save(fig, "P2L_Figure1_v2.pdf")


if __name__ == "__main__":
    main()
