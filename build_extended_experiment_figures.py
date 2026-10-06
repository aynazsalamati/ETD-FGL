from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("cora", "pubmed", "reddit")
DISPLAY = {"cora": "Cora", "pubmed": "PubMed", "reddit": "Reddit"}
METRICS = (
    ("clean_test_accuracy", "Clean test accuracy"),
    ("attack_success_rate", "Attack success rate"),
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output_dir: Path, stem: str, caption: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    (output_dir / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    plt.close(fig)


def style_axis(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def final_metric(summary: dict[str, Any], key: str) -> float:
    return float(summary["final_metrics"][key])


def scan_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not root.exists():
        return []
    return [(path, load_json(path)) for path in root.rglob("final_summary.json")]


def build_multiseed(base: Path, out: Path) -> bool:
    root = base / "multiseed"
    summaries = scan_summaries(root)
    if not summaries:
        return False

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path, summary in summaries:
        rel = path.relative_to(root).parts
        if len(rel) < 3:
            continue
        dataset = rel[0]
        grouped[(dataset, summary["method"])].append(summary)

    rows: list[dict[str, Any]] = []
    for (dataset, method), items in sorted(grouped.items()):
        acc = np.asarray([final_metric(item, "test_accuracy") for item in items], dtype=float)
        asr = np.asarray([final_metric(item, "attack_success_rate") for item in items], dtype=float)
        effect = np.asarray([final_metric(item, "attack_effect") for item in items], dtype=float)
        rows.append(
            {
                "dataset": DISPLAY.get(dataset, dataset),
                "method": method,
                "n_seeds": len(items),
                "clean_accuracy_mean": float(acc.mean()),
                "clean_accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "asr_mean": float(asr.mean()),
                "asr_std": float(asr.std(ddof=1)) if len(asr) > 1 else 0.0,
                "trigger_effect_mean": float(effect.mean()),
                "trigger_effect_std": float(effect.std(ddof=1)) if len(effect) > 1 else 0.0,
            }
        )
    save_csv(out / "tables" / "multiseed_mean_std.csv", rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        dataset_rows = [r for r in rows if r["dataset"] == DISPLAY[dataset]]
        methods = [r["method"] for r in dataset_rows]
        x = np.arange(len(methods))
        for row_idx, (mean_key, std_key, ylabel) in enumerate(
            [
                ("clean_accuracy_mean", "clean_accuracy_std", "Clean test accuracy"),
                ("asr_mean", "asr_std", "Attack success rate"),
            ]
        ):
            values = [float(r[mean_key]) for r in dataset_rows]
            errors = [float(r[std_key]) for r in dataset_rows]
            axes[row_idx, col].bar(x, values, yerr=errors, capsize=3, edgecolor="black", linewidth=0.7)
            axes[row_idx, col].set_xticks(x)
            axes[row_idx, col].set_xticklabels(methods, rotation=25, ha="right")
            axes[row_idx, col].set_ylabel(ylabel)
            axes[row_idx, col].set_ylim(0.0, 1.05)
            letter = chr(ord("a") + row_idx * 3 + col)
            axes[row_idx, col].set_title(f"({letter}) {DISPLAY[dataset]}")
            style_axis(axes[row_idx, col])
    fig.tight_layout()
    save_figure(
        fig,
        out / "figures",
        "Fig_Ext_01_MultiSeed_MeanStd",
        "Multi-seed mean ± standard deviation of clean test accuracy and attack success rate across Cora, PubMed, and Reddit.",
    )
    return True


def build_condition_lines(
    *,
    root: Path,
    out: Path,
    table_name: str,
    figure_name: str,
    condition_prefix: str,
    x_label: str,
    caption: str,
    parse_x,
) -> bool:
    summaries = scan_summaries(root)
    if not summaries:
        return False

    rows: list[dict[str, Any]] = []
    for path, summary in summaries:
        rel = path.relative_to(root).parts
        if len(rel) < 4:
            continue
        dataset = rel[0]
        condition = rel[1]
        if not condition.startswith(condition_prefix):
            continue
        value = parse_x(condition)
        rows.append(
            {
                "dataset": DISPLAY.get(dataset, dataset),
                "condition": value,
                "method": summary["method"],
                "clean_test_accuracy": final_metric(summary, "test_accuracy"),
                "attack_success_rate": final_metric(summary, "attack_success_rate"),
                "trigger_attack_effect": final_metric(summary, "attack_effect"),
                "runtime_seconds": float(summary.get("elapsed_seconds", 0.0)),
            }
        )
    if not rows:
        return False
    save_csv(out / "tables" / table_name, rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        ds_rows = [r for r in rows if r["dataset"] == DISPLAY[dataset]]
        methods = sorted({r["method"] for r in ds_rows})
        for row_idx, (metric, ylabel) in enumerate(METRICS):
            for method in methods:
                mrows = sorted(
                    [r for r in ds_rows if r["method"] == method],
                    key=lambda r: float(r["condition"]),
                )
                if not mrows:
                    continue
                axes[row_idx, col].plot(
                    [float(r["condition"]) for r in mrows],
                    [float(r[metric]) for r in mrows],
                    marker="o",
                    linewidth=1.3,
                    label=method,
                )
            axes[row_idx, col].set_xlabel(x_label)
            axes[row_idx, col].set_ylabel(ylabel)
            axes[row_idx, col].set_ylim(0.0, 1.05)
            letter = chr(ord("a") + row_idx * 3 + col)
            axes[row_idx, col].set_title(f"({letter}) {DISPLAY[dataset]}")
            style_axis(axes[row_idx, col])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(labels)), frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, out / "figures", figure_name, caption)
    return True


def build_noniid(base: Path, out: Path) -> bool:
    return build_condition_lines(
        root=base / "noniid",
        out=out,
        table_name="noniid_dirichlet_results.csv",
        figure_name="Fig_Ext_02_NonIID_Dirichlet",
        condition_prefix="alpha_",
        x_label="Dirichlet alpha",
        caption="Non-IID robustness under Dirichlet label skew: clean accuracy and ASR across Cora, PubMed, and Reddit.",
        parse_x=lambda s: float(s.removeprefix("alpha_").replace("p", ".")),
    )


def build_sensitivity_poison(base: Path, out: Path) -> bool:
    return build_condition_lines(
        root=base / "sensitivity_poison",
        out=out,
        table_name="sensitivity_poison_rate.csv",
        figure_name="Fig_Ext_04_Sensitivity_PoisonRate",
        condition_prefix="poison_",
        x_label="Poison rate",
        caption="Sensitivity to poisoning rate: clean accuracy and ASR for Attacked FedAvg and ETD-FGL across the three datasets.",
        parse_x=lambda s: float(s.removeprefix("poison_").replace("p", ".")),
    )


def build_sensitivity_trigger(base: Path, out: Path) -> bool:
    return build_condition_lines(
        root=base / "sensitivity_trigger",
        out=out,
        table_name="sensitivity_trigger_size.csv",
        figure_name="Fig_Ext_05_Sensitivity_TriggerSize",
        condition_prefix="trigger_",
        x_label="Trigger size",
        caption="Sensitivity to structural trigger size: clean accuracy and ASR for Attacked FedAvg and ETD-FGL across the three datasets.",
        parse_x=lambda s: int(s.removeprefix("trigger_")),
    )


def build_clients(base: Path, out: Path) -> bool:
    return build_condition_lines(
        root=base / "clients",
        out=out,
        table_name="client_scalability_results.csv",
        figure_name="Fig_Ext_06_Client_Scalability",
        condition_prefix="clients_",
        x_label="Number of clients",
        caption="Scalability with federation size: clean accuracy and ASR for the compared methods across 5, 10, 20, 30, and 50 clients.",
        parse_x=lambda s: int(s.removeprefix("clients_")),
    )


def build_ablation(base: Path, out: Path) -> bool:
    root = base / "ablation"
    summaries = scan_summaries(root)
    if not summaries:
        return False

    rows: list[dict[str, Any]] = []
    for path, summary in summaries:
        rel = path.relative_to(root).parts
        if len(rel) < 3:
            continue
        dataset = rel[0]
        rows.append(
            {
                "dataset": DISPLAY.get(dataset, dataset),
                "variant": summary["method"],
                "clean_test_accuracy": final_metric(summary, "test_accuracy"),
                "attack_success_rate": final_metric(summary, "attack_success_rate"),
                "trigger_attack_effect": final_metric(summary, "attack_effect"),
                "runtime_seconds": float(summary.get("elapsed_seconds", 0.0)),
            }
        )
    save_csv(out / "tables" / "ablation_results.csv", rows)

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.7), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        ds_rows = [r for r in rows if r["dataset"] == DISPLAY[dataset]]
        order = {
            "Attacked FedAvg": 0,
            "ETD-FGL (Full final)": 1,
            "ETD-FGL (Topology + Update)": 1,
            "ETD-FGL (Full)": 1,
            "ETD-FGL Topology-only": 2,
            "ETD-FGL w/o Topology": 2,
            "ETD-FGL Update-only": 3,
            "ETD-FGL w/o Update Anomaly": 3,
            "ETD-FGL w/o Trust Weighting": 4,
            "Historical Three-Channel Variant (0.45/0.35/0.20)": 5,
            "ETD-FGL Historical Three-Channel (0.45/0.35/0.20)": 5,
            "ETD-FGL w/o Explanation": 6,
        }
        ds_rows.sort(key=lambda r: order.get(r["variant"], 99))
        labels = [r["variant"] for r in ds_rows]
        x = np.arange(len(labels))
        for row_idx, (metric, ylabel) in enumerate(METRICS):
            values = [float(r[metric]) for r in ds_rows]
            axes[row_idx, col].bar(x, values, edgecolor="black", linewidth=0.7)
            axes[row_idx, col].set_xticks(x)
            axes[row_idx, col].set_xticklabels(labels, rotation=25, ha="right")
            axes[row_idx, col].set_ylabel(ylabel)
            axes[row_idx, col].set_ylim(0.0, 1.05)
            letter = chr(ord("a") + row_idx * 3 + col)
            axes[row_idx, col].set_title(f"({letter}) {DISPLAY[dataset]}")
            style_axis(axes[row_idx, col])
    fig.tight_layout()
    save_figure(
        fig,
        out / "figures",
        "Fig_Ext_03_ETDFGL_Ablation",
        "Component-level ablation of ETD-FGL showing the effect of topology evidence, explanation evidence, update anomaly evidence, and trust-weighted aggregation.",
    )
    return True


def write_index(out: Path, built: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ETD-FGL Extended Experiment Outputs\n\n",
        "This directory contains only tables and figures derived from completed real runs.\n\n",
    ]
    if built:
        lines.append("Generated studies:\n")
        for name in built:
            lines.append(f"- {name}\n")
    else:
        lines.append("No completed extended-study summaries were found yet. Run the experiment batch files first.\n")
    (out / "README.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    project_dir = Path(__file__).resolve().parent
    base = project_dir / "outputs" / "extended_experiments"
    out = base / "journal_summary"

    builders = [
        ("Multi-seed mean ± std", build_multiseed),
        ("Non-IID Dirichlet", build_noniid),
        ("ETD-FGL ablation", build_ablation),
        ("Poison-rate sensitivity", build_sensitivity_poison),
        ("Trigger-size sensitivity", build_sensitivity_trigger),
        ("Client-count scalability", build_clients),
    ]
    built = []
    for name, builder in builders:
        if builder(base, out):
            built.append(name)
            print(f"BUILT: {name}")
        else:
            print(f"SKIP: {name} (no completed real outputs found)")

    write_index(out, built)
    print("=" * 108)
    print("EXTENDED JOURNAL FIGURES/TABLES COMPLETE")
    print(out)
    print("=" * 108)


if __name__ == "__main__":
    main()
