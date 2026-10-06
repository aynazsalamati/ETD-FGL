from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def apply_journal_style() -> None:
    """Apply a calm article-style visual theme close to the sample paper."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: Axes, *, grid_axis: str = "both") -> None:
    """Style one plot axis consistently."""

    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.55,
        alpha=0.28,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
    )
