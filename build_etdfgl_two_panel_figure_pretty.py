from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAME = {
    "Clean FedAvg": "Clean FedAvg",
    "Attacked FedAvg": "Attacked FedAvg",
    "Defended FedAvg": "ETD-FGL",
}

XTICK_NAME = {
    "Clean FedAvg": "Clean\nFedAvg",
    "Attacked FedAvg": "Attacked\nFedAvg",
    "ETD-FGL": "ETD-FGL",
}

HATCHES = {
    "Clean FedAvg": "//",
    "Attacked FedAvg": "\\\\",
    "ETD-FGL": "xx",
}


def load_results(csv_path: Path) -> dict[str, dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            "Comparison CSV was not found:\n"
            f"{csv_path}"
        )

    results: dict[str, dict[str, float]] = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            original_name = row["method"]
            display_name = DISPLAY_NAME.get(
                original_name,
                original_name,
            )

            results[display_name] = {
                "clean_test_accuracy": float(
                    row["clean_test_accuracy"]
                ),
                "attack_success_rate": (
                    float(row["attack_success_rate"])
                    if row["attack_success_rate"].strip()
                    else float("nan")
                ),
                "asr_drop": (
                    float(row["asr_absolute_reduction_vs_attacked"])
                    if row["asr_absolute_reduction_vs_attacked"].strip()
                    else float("nan")
                ),
                "asr_rel_drop": (
                    float(row["asr_relative_reduction_vs_attacked"])
                    if row["asr_relative_reduction_vs_attacked"].strip()
                    else float("nan")
                ),
                "clean_gain_vs_attack": (
                    float(row["clean_accuracy_change_vs_attacked_fedavg"])
                    if row["clean_accuracy_change_vs_attacked_fedavg"].strip()
                    else float("nan")
                ),
            }

    required_methods = {
        "Clean FedAvg",
        "Attacked FedAvg",
        "ETD-FGL",
    }

    missing_methods = required_methods - set(results.keys())

    if missing_methods:
        raise RuntimeError(
            "Missing methods in comparison CSV: "
            f"{sorted(missing_methods)}"
        )

    return results


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def build_bars(ax, methods: list[str], values: list[float]):
    x_positions = np.arange(len(methods))

    bars = ax.bar(
        x_positions,
        values,
        width=0.58,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
    )

    for bar, method in zip(bars, methods, strict=True):
        bar.set_hatch(HATCHES[method])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [XTICK_NAME[method] for method in methods]
    )

    annotate_bars(ax, bars, values)
    style_axis(ax)

    return bars


def build_figure(
    results: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    accuracy_methods = [
        "Clean FedAvg",
        "Attacked FedAvg",
        "ETD-FGL",
    ]
    accuracy_values = [
        results[method]["clean_test_accuracy"]
        for method in accuracy_methods
    ]

    asr_methods = [
        "Attacked FedAvg",
        "ETD-FGL",
    ]
    asr_values = [
        results[method]["attack_success_rate"]
        for method in asr_methods
    ]

    etdfgl = results["ETD-FGL"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.9),
    )

    # Panel (a)
    build_bars(
        axes[0],
        accuracy_methods,
        accuracy_values,
    )
    axes[0].set_ylabel("Clean test accuracy")
    axes[0].set_ylim(0.68, 0.78)
    axes[0].set_yticks(np.arange(0.68, 0.781, 0.02))
    axes[0].text(
        0.02,
        0.98,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].text(
        0.50,
        1.03,
        "Clean-task performance",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    axes[0].text(
        0.97,
        0.08,
        f"Gain vs attacked: +{etdfgl['clean_gain_vs_attack']:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )

    # Panel (b)
    build_bars(
        axes[1],
        asr_methods,
        asr_values,
    )
    axes[1].set_ylabel("Attack success rate")
    axes[1].set_ylim(0.0, 0.65)
    axes[1].set_yticks(np.arange(0.0, 0.651, 0.1))
    axes[1].text(
        0.02,
        0.98,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].text(
        0.50,
        1.03,
        "Backdoor robustness",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    axes[1].text(
        0.97,
        0.08,
        (
            f"ASR drop: {etdfgl['asr_drop']:.3f}\n"
            f"Relative reduction: {etdfgl['asr_rel_drop'] * 100:.2f}%"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )

    figure.suptitle(
        "Pilot comparison on the Cora dataset",
        y=1.02,
        fontsize=13,
    )

    figure.tight_layout()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_dir / "Fig_Cora_Accuracy_ASR_TwoPanel_Pretty.png"
    pdf_path = output_dir / "Fig_Cora_Accuracy_ASR_TwoPanel_Pretty.pdf"
    svg_path = output_dir / "Fig_Cora_Accuracy_ASR_TwoPanel_Pretty.svg"
    caption_path = output_dir / "Fig_Cora_Accuracy_ASR_TwoPanel_Pretty_caption.txt"

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    caption = (
        "Fig. X. Pilot comparison of clean-task performance and backdoor robustness "
        "on the Cora dataset. ETD-FGL improves clean test accuracy and substantially "
        "reduces the attack success rate compared with attacked FedAvg."
    )
    caption_path.write_text(
        caption,
        encoding="utf-8",
    )

    print("=" * 92)
    print("Prettier two-panel Cora figure created successfully")
    print("=" * 92)
    print(f"PNG:     {png_path}")
    print(f"PDF:     {pdf_path}")
    print(f"SVG:     {svg_path}")
    print(f"Caption: {caption_path}")
    print("=" * 92)


def main() -> None:
    setup_style()

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
        / "figures_final"
    )

    results = load_results(
        comparison_csv_path
    )

    build_figure(
        results=results,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
