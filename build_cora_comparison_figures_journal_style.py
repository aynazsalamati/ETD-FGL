from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHOD_HATCHES = {
    "Clean FedAvg": "xx",
    "Attacked FedAvg": "///",
    "Defended FedAvg": "\\\\\\",
}

METHOD_FACECOLORS = {
    "Clean FedAvg": "white",
    "Attacked FedAvg": "0.88",
    "Defended FedAvg": "white",
}


def load_comparison(csv_path: Path) -> list[dict[str, str]]:
    """Load automatic FedAvg comparison results."""

    if not csv_path.exists():
        raise FileNotFoundError(
            "Comparison CSV was not found:\n"
            f"{csv_path}"
        )

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise RuntimeError(
            f"Comparison CSV is empty:\n{csv_path}"
        )

    return rows


def setup_journal_style() -> None:
    """Apply a journal-like plotting style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_value_labels(ax, bars, suffix: str = "") -> None:
    """Write bar values above the bars."""

    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.2f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def style_axes(ax) -> None:
    """Make axes look like journal figures."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        alpha=0.35,
    )


def build_legend(methods: list[str]) -> list[Patch]:
    """Create hatch-based legend handles."""

    legend_handles: list[Patch] = []

    for method in methods:
        legend_handles.append(
            Patch(
                facecolor=METHOD_FACECOLORS[method],
                edgecolor="black",
                hatch=METHOD_HATCHES[method],
                label=method,
                linewidth=1.0,
            )
        )

    return legend_handles


def make_bar_chart(
    methods: list[str],
    values: list[float],
    x_label: str,
    y_label: str,
    output_path: Path,
    legend_ncol: int,
    y_padding: float,
    value_suffix: str = "",
) -> None:
    """Create a compact article-style bar chart."""

    figure, ax = plt.subplots(figsize=(6.8, 4.7))

    x_positions = list(range(len(methods)))

    bars = []

    for position, method, value in zip(
        x_positions,
        methods,
        values,
        strict=True,
    ):
        bar = ax.bar(
            position,
            value,
            width=0.55,
            facecolor=METHOD_FACECOLORS[method],
            edgecolor="black",
            linewidth=1.0,
            hatch=METHOD_HATCHES[method],
        )
        bars.append(bar[0])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    lower_limit = 0.0
    upper_limit = max(values) + y_padding
    ax.set_ylim(lower_limit, upper_limit)

    style_axes(ax)
    add_value_labels(ax, bars, suffix=value_suffix)

    legend_handles = build_legend(methods)

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=legend_ncol,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_clean_accuracy_chart(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Create clean accuracy chart in journal style."""

    methods = [row["method"] for row in rows]
    values = [
        float(row["clean_test_accuracy"]) * 100.0
        for row in rows
    ]

    make_bar_chart(
        methods=methods,
        values=values,
        x_label="Methods",
        y_label="Clean Test Accuracy (%)",
        output_path=output_path,
        legend_ncol=3,
        y_padding=4.0,
        value_suffix="%",
    )


def save_asr_chart(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Create ASR chart in journal style."""

    attack_rows = [
        row for row in rows
        if row["attack_success_rate"].strip()
    ]

    methods = [row["method"] for row in attack_rows]
    values = [
        float(row["attack_success_rate"]) * 100.0
        for row in attack_rows
    ]

    make_bar_chart(
        methods=methods,
        values=values,
        x_label="Methods",
        y_label="Attack Success Rate (%)",
        output_path=output_path,
        legend_ncol=2,
        y_padding=12.0,
        value_suffix="%",
    )


def save_trigger_effect_chart(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Create trigger-effect chart in journal style."""

    attack_rows = [
        row for row in rows
        if row["trigger_attack_effect"].strip()
    ]

    methods = [row["method"] for row in attack_rows]
    values = [
        float(row["trigger_attack_effect"]) * 100.0
        for row in attack_rows
    ]

    make_bar_chart(
        methods=methods,
        values=values,
        x_label="Methods",
        y_label="Trigger Attack Effect (pp)",
        output_path=output_path,
        legend_ncol=2,
        y_padding=10.0,
        value_suffix=" pp",
    )


def main() -> None:
    setup_journal_style()

    project_dir = Path(__file__).resolve().parent

    comparison_csv_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "automatic_comparison"
        / "fedavg_comparison.csv"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "automatic_comparison"
        / "figures_journal_style"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_comparison(comparison_csv_path)

    clean_accuracy_path = (
        output_dir
        / "cora_clean_accuracy_journal_style.png"
    )

    asr_path = (
        output_dir
        / "cora_asr_journal_style.png"
    )

    trigger_effect_path = (
        output_dir
        / "cora_trigger_effect_journal_style.png"
    )

    save_clean_accuracy_chart(
        rows=rows,
        output_path=clean_accuracy_path,
    )

    save_asr_chart(
        rows=rows,
        output_path=asr_path,
    )

    save_trigger_effect_chart(
        rows=rows,
        output_path=trigger_effect_path,
    )

    print("=" * 86)
    print("Journal-style Cora comparison figures created successfully")
    print("=" * 86)
    print(f"Clean accuracy: {clean_accuracy_path}")
    print(f"ASR comparison: {asr_path}")
    print(f"Trigger effect: {trigger_effect_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()
