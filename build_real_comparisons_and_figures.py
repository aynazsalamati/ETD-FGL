from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [
    ("Clean FedAvg", "fedavg_clean"),
    ("Attacked FedAvg", "fedavg_structural_backdoor"),
    ("PD-FL", "baseline_pdfl"),
    ("GSP-FL", "baseline_gsp"),
    ("FLPurifier-GNN", "baseline_flpurifier"),
    ("DMGNN-FL", "baseline_dmgnn"),
    ("ETD-FGL", "defended_etdfgl"),
]
DATASETS = ["cora", "pubmed", "reddit"]
DISPLAY = {"cora": "Cora", "pubmed": "PubMed", "reddit": "Reddit"}
STYLES = ["-", "--", "-.", ":", "-", "--", "-"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, output_dir: Path, stem: str, caption: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    (output_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def build_dataset(project_dir: Path, dataset: str):
    base = project_dir / "outputs" / "final_multidataset" / f"federated_{dataset}"
    comparison_dir = base / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    histories = {}
    for method, slug in METHODS:
        summary_path = base / slug / "final_summary.json"
        history_path = base / slug / "training_history.csv"
        if not summary_path.exists() or not history_path.exists():
            raise FileNotFoundError(f"Missing real output for {method} on {dataset}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["final_metrics"]
        rows.append({
            "dataset": DISPLAY[dataset],
            "method": method,
            "best_round": summary["best_round"],
            "clean_test_accuracy": metrics["test_accuracy"],
            "attack_success_rate": "" if method == "Clean FedAvg" else metrics["attack_success_rate"],
            "trigger_attack_effect": "" if method == "Clean FedAvg" else metrics["attack_effect"],
            "runtime_seconds": summary["elapsed_seconds"],
        })
        histories[method] = read_csv(history_path)

    csv_path = comparison_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (comparison_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    metrics = [
        ("test_accuracy", "Clean test accuracy", "(a) Clean accuracy", True),
        ("attack_success_rate", "Attack success rate", "(b) ASR", False),
        ("mean_client_loss", "Mean client loss", "(c) Training loss", True),
    ]
    for axis, (metric, ylabel, title, include_clean) in zip(axes.flat[:3], metrics):
        for idx, (method, _) in enumerate(METHODS):
            if not include_clean and method == "Clean FedAvg":
                continue
            history = histories[method]
            x = [int(r["round"]) for r in history]
            y = [float(r[metric]) for r in history]
            axis.plot(x, y, label=method, linestyle=STYLES[idx], marker=MARKERS[idx], markevery=5, markersize=3.2, linewidth=1.2)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if metric != "mean_client_loss":
            axis.set_ylim(0, 1)

    axis = axes[1, 1]
    labels = [r["method"] for r in rows]
    values = [float(r["clean_test_accuracy"]) for r in rows]
    bars = axis.bar(range(len(labels)), values, edgecolor="black", linewidth=0.8)
    hatches = ["//", "\\\\", "xx", "..", "++", "oo", "--"]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("Final clean test accuracy")
    axis.set_title("(d) Final comparison")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(0, max(values) * 1.15)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(
        fig,
        comparison_dir / "figures",
        f"Fig_MultiPanel_{DISPLAY[dataset]}",
        f"Multi-panel real comparison on {DISPLAY[dataset]} for Clean FedAvg, Attacked FedAvg, four baseline defenses, and ETD-FGL.",
    )
    return rows, histories


def build_cross_dataset(project_dir: Path, data):
    output_dir = project_dir / "outputs" / "final_multidataset" / "cross_dataset_figures"
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6))
    for col, dataset in enumerate(DATASETS):
        rows, histories = data[dataset]
        for idx, (method, _) in enumerate(METHODS):
            h = histories[method]
            axes[0,col].plot([int(r["round"]) for r in h], [float(r["test_accuracy"]) for r in h], label=method, linestyle=STYLES[idx], marker=MARKERS[idx], markevery=5, markersize=3, linewidth=1.1)
            if method != "Clean FedAvg":
                axes[1,col].plot([int(r["round"]) for r in h], [float(r["attack_success_rate"]) for r in h], label=method, linestyle=STYLES[idx], marker=MARKERS[idx], markevery=5, markersize=3, linewidth=1.1)
        axes[0,col].set_title(f"({chr(97+col)}) {DISPLAY[dataset]} clean accuracy")
        axes[1,col].set_title(f"({chr(100+col)}) {DISPLAY[dataset]} ASR")
        for row in range(2):
            axes[row,col].set_xlabel("Communication round")
            axes[row,col].set_ylim(0,1)
            axes[row,col].grid(True, linestyle="--", alpha=0.25)
            axes[row,col].spines["top"].set_visible(False)
            axes[row,col].spines["right"].set_visible(False)
    axes[0,0].set_ylabel("Clean test accuracy")
    axes[1,0].set_ylabel("Attack success rate")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, output_dir, "Fig_CrossDataset_Accuracy_ASR", "Cross-dataset real multi-panel comparison of clean accuracy and attack success rate on Cora, PubMed, and Reddit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    selected = DATASETS if args.dataset == "all" else [args.dataset]
    data = {}
    for dataset in selected:
        data[dataset] = build_dataset(project_dir, dataset)
    if args.dataset == "all":
        build_cross_dataset(project_dir, data)
    print("Figures and comparison tables created successfully.")


if __name__ == "__main__":
    main()
