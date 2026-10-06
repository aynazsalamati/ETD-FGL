from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_SPECS = [
    ("Clean FedAvg", "fedavg_clean/training_history.csv"),
    ("Attacked FedAvg", "fedavg_structural_backdoor/training_history.csv"),
    ("PD-FL", "baseline_pdfl/training_history.csv"),
    ("GSP-FL", "baseline_gsp/training_history.csv"),
    ("FLPurifier-GNN", "baseline_flpurifier/training_history.csv"),
    ("DMGNN-FL", "baseline_dmgnn/training_history.csv"),
    ("ETD-FGL", "defended_fedavg_structural_backdoor/training_history.csv"),
]

LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV was not found:\n{path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_figure(
    figure,
    output_dir: Path,
    stem: str,
    caption: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    (output_dir / f"{stem}_caption.txt").write_text(
        caption,
        encoding="utf-8",
    )
    plt.close(figure)


def plot_round_metric(
    histories: dict[str, list[dict[str, str]]],
    *,
    metric: str,
    ylabel: str,
    methods: list[str],
    output_dir: Path,
    stem: str,
    caption: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))

    for index, method in enumerate(methods):
        rows = histories[method]
        if not rows or metric not in rows[0] or rows[0][metric] == "":
            continue

        rounds = [int(row["round"]) for row in rows]
        values = [float(row[metric]) for row in rows]
        axis.plot(
            rounds,
            values,
            label=method,
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(1, len(rounds) // 10),
            markersize=4.0,
            linewidth=1.5,
        )

    axis.set_xlabel("Communication round")
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        ncol=2 if len(methods) > 4 else 1,
    )
    figure.tight_layout()
    save_figure(figure, output_dir, stem, caption)


def plot_final_bar(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    caption: str,
    include_clean: bool,
) -> None:
    selected = [
        row for row in rows
        if (include_clean or row["method"] != "Clean FedAvg")
        and row[metric] != ""
    ]

    labels = [row["method"] for row in selected]
    values = [float(row[metric]) for row in selected]

    figure, axis = plt.subplots(figsize=(7.8, 4.6))
    bars = axis.bar(
        range(len(labels)),
        values,
        width=0.62,
        edgecolor="black",
        linewidth=0.8,
    )

    hatch_patterns = ["//", "\\\\", "xx", "..", "++", "oo", "--"]
    for bar, hatch in zip(bars, hatch_patterns, strict=False):
        bar.set_hatch(hatch)
        bar.set_alpha(0.88)

    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.30)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    padding = max(values) * 0.03 if values else 0.01
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + padding,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    if values:
        axis.set_ylim(0.0, max(values) * 1.14)
    figure.tight_layout()
    save_figure(figure, output_dir, stem, caption)


def main() -> None:
    setup_style()
    project_dir = Path(__file__).resolve().parent
    base_dir = project_dir / "outputs" / "federated_cora"
    output_dir = base_dir / "all_baseline_comparison" / "figures"

    histories = {
        method: load_csv(base_dir / relative_path)
        for method, relative_path in HISTORY_SPECS
    }

    comparison_rows = load_csv(
        base_dir
        / "all_baseline_comparison"
        / "all_baseline_comparison.csv"
    )

    all_methods = [method for method, _ in HISTORY_SPECS]
    attacked_methods = [
        method for method in all_methods
        if method != "Clean FedAvg"
    ]

    plot_round_metric(
        histories,
        metric="test_accuracy",
        ylabel="Clean test accuracy",
        methods=all_methods,
        output_dir=output_dir,
        stem="Fig_01_Clean_Accuracy_vs_Round",
        caption=(
            "Fig. 1. Clean test accuracy over federated communication rounds "
            "for Clean FedAvg, Attacked FedAvg, four baseline defenses, and ETD-FGL on Cora."
        ),
        ylim=(0.0, 1.0),
    )

    plot_round_metric(
        histories,
        metric="attack_success_rate",
        ylabel="Attack success rate",
        methods=attacked_methods,
        output_dir=output_dir,
        stem="Fig_02_ASR_vs_Round",
        caption=(
            "Fig. 2. Attack success rate over communication rounds for the "
            "attacked model, baseline defenses, and ETD-FGL on Cora."
        ),
        ylim=(0.0, 1.0),
    )

    plot_round_metric(
        histories,
        metric="mean_client_loss",
        ylabel="Mean client training loss",
        methods=all_methods,
        output_dir=output_dir,
        stem="Fig_03_Training_Loss_vs_Round",
        caption=(
            "Fig. 3. Mean client training loss over communication rounds for "
            "the compared federated graph learning methods on Cora."
        ),
    )

    plot_final_bar(
        comparison_rows,
        metric="clean_test_accuracy",
        ylabel="Clean test accuracy",
        output_dir=output_dir,
        stem="Fig_04_Final_Clean_Accuracy",
        caption=(
            "Fig. 4. Final clean test accuracy of the compared methods on Cora."
        ),
        include_clean=True,
    )

    plot_final_bar(
        comparison_rows,
        metric="attack_success_rate",
        ylabel="Attack success rate",
        output_dir=output_dir,
        stem="Fig_05_Final_ASR",
        caption=(
            "Fig. 5. Final attack success rate of Attacked FedAvg, baseline defenses, and ETD-FGL on Cora."
        ),
        include_clean=False,
    )

    plot_final_bar(
        comparison_rows,
        metric="trigger_attack_effect",
        ylabel="Trigger attack effect",
        output_dir=output_dir,
        stem="Fig_06_Final_Trigger_Effect",
        caption=(
            "Fig. 6. Final trigger-induced target-rate increase for Attacked FedAvg, baseline defenses, and ETD-FGL on Cora."
        ),
        include_clean=False,
    )

    print("=" * 92)
    print("All baseline-comparison journal figures were created")
    print("=" * 92)
    print(f"Output directory: {output_dir}")
    print("=" * 92)


if __name__ == "__main__":
    main()
