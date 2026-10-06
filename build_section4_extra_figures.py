from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["cora", "pubmed", "reddit"]
DISPLAY = {"cora": "Cora", "pubmed": "PubMed", "reddit": "Reddit"}
METHODS = [
    "Clean FedAvg",
    "Attacked FedAvg",
    "PD-FL",
    "GSP-FL",
    "FLPurifier-GNN",
    "DMGNN-FL",
    "ETD-FGL",
]
HATCHES = ["//", "\\\\", "xx", "..", "++", "oo", "--"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required comparison table was not found: {path}\n"
            "Run run_all_real_datasets.bat first."
        )
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_all_results(project_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    results: dict[str, dict[str, dict[str, str]]] = {}
    for dataset in DATASETS:
        path = (
            project_dir
            / "outputs"
            / "final_multidataset"
            / f"federated_{dataset}"
            / "comparison"
            / "comparison.csv"
        )
        rows = read_csv(path)
        results[dataset] = {row["method"]: row for row in rows}

        missing = [method for method in METHODS if method not in results[dataset]]
        if missing:
            raise ValueError(
                f"Missing methods in {path}: {', '.join(missing)}"
            )
    return results


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, output_dir: Path, stem: str, caption: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    (output_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def grouped_bar(
    results: dict[str, dict[str, dict[str, str]]],
    *,
    metric: str,
    ylabel: str,
    stem: str,
    caption: str,
    output_dir: Path,
    include_clean: bool,
) -> None:
    methods = METHODS if include_clean else [m for m in METHODS if m != "Clean FedAvg"]
    width = 0.11 if len(methods) >= 7 else 0.13
    x = list(range(len(DATASETS)))

    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    for method_index, method in enumerate(methods):
        offset = (method_index - (len(methods) - 1) / 2) * width
        values: list[float] = []
        for dataset in DATASETS:
            raw = results[dataset][method].get(metric, "")
            if raw == "":
                values.append(float("nan"))
            else:
                values.append(float(raw))

        bars = ax.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=method,
            edgecolor="black",
            linewidth=0.65,
            hatch=HATCHES[method_index % len(HATCHES)],
            alpha=0.90,
        )

        for bar, value in zip(bars, values, strict=True):
            if value != value:  # NaN
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in DATASETS])
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=True)

    finite_values = []
    for dataset in DATASETS:
        for method in methods:
            raw = results[dataset][method].get(metric, "")
            if raw != "":
                finite_values.append(float(raw))
    if finite_values:
        ymax = max(finite_values)
        if metric in {"clean_test_accuracy", "attack_success_rate"}:
            ax.set_ylim(0.0, max(1.02, ymax * 1.12))
        else:
            ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)

    fig.tight_layout()
    save(fig, output_dir, stem, caption)


def main() -> None:
    setup_style()
    project_dir = Path(__file__).resolve().parent
    results = load_all_results(project_dir)
    output_dir = (
        project_dir
        / "outputs"
        / "final_multidataset"
        / "section4_extra_figures"
    )

    grouped_bar(
        results,
        metric="clean_test_accuracy",
        ylabel="Final clean test accuracy",
        stem="Fig_09_Final_Clean_Accuracy_All_Datasets",
        caption=(
            "Final clean test accuracy of Clean FedAvg, Attacked FedAvg, "
            "four baseline defenses, and ETD-FGL on Cora, PubMed, and Reddit."
        ),
        output_dir=output_dir,
        include_clean=True,
    )

    grouped_bar(
        results,
        metric="attack_success_rate",
        ylabel="Final attack success rate (ASR)",
        stem="Fig_10_Final_ASR_All_Datasets",
        caption=(
            "Final attack success rate of Attacked FedAvg, the baseline defenses, "
            "and ETD-FGL on Cora, PubMed, and Reddit."
        ),
        output_dir=output_dir,
        include_clean=False,
    )

    grouped_bar(
        results,
        metric="trigger_attack_effect",
        ylabel="Final trigger attack effect",
        stem="Fig_11_Final_Trigger_Effect_All_Datasets",
        caption=(
            "Final trigger-induced target-rate increase for Attacked FedAvg, "
            "the baseline defenses, and ETD-FGL on Cora, PubMed, and Reddit."
        ),
        output_dir=output_dir,
        include_clean=False,
    )

    grouped_bar(
        results,
        metric="runtime_seconds",
        ylabel="Runtime (s)",
        stem="Fig_12_Runtime_All_Datasets",
        caption=(
            "Measured runtime of the compared methods on Cora, PubMed, and Reddit "
            "under the unified experimental pipeline."
        ),
        output_dir=output_dir,
        include_clean=True,
    )

    print("=" * 100)
    print("SECTION 4 EXTRA FIGURES CREATED SUCCESSFULLY")
    print(f"Output: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
