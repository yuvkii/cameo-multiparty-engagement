"""Shared plotting style for every figure in the final report.

Palette: validated categorical slots (blue / orange / aqua / violet / red) --
adjacent-pair CVD-safe; the first three are all-pairs safe and are the only
ones used where every series is compared against every other (scatter, paired
plots). Sequential magnitude uses a single hue, light to dark.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIGDIR = Path(__file__).resolve().parents[1] / "draft" / "Figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, VIOLET, RED]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#dcdcd8"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#bcd4f2", "#7aabe6", "#3d84d4", "#12447e"])

# per-recording-date colours, used consistently in every figure that splits by date
DATE_COLOR = {"03_20": BLUE, "03_26": ORANGE, "05_15": AQUA, "05_14": VIOLET}
# engagement level colours: ordered magnitude, single hue
LEVEL_COLOR = ["#c9dcf3", "#8fb8e8", "#4a8bd6", "#12447e"]
LEVELS = ["disengaged", "low", "medium", "high"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

FULL, HALF = 6.2, 3.05  # text-width and column-width inches


def grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str):
    """Vector PDF for LaTeX, PNG alongside for visual checking."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("  wrote", name)
