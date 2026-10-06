from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


METHOD_SPECS = [
    {
        "method": "Clean FedAvg",
        "kind": "clean_checkpoint",
        "path": "fedavg_clean/best_fedavg_cora.pt",
    },
    {
        "method": "Attacked FedAvg",
        "kind": "attacked_checkpoint",
        "path": "fedavg_structural_backdoor/best_backdoored_fedavg_cora.pt",
    },
    {
        "method": "PD-FL",
        "kind": "summary",
        "path": "baseline_pdfl/final_summary.json",
    },
    {
        "method": "GSP-FL",
        "kind": "summary",
        "path": "baseline_gsp/final_summary.json",
    },
    {
        "method": "FLPurifier-GNN",
        "kind": "summary",
        "path": "baseline_flpurifier/final_summary.json",
    },
    {
        "method": "DMGNN-FL",
        "kind": "summary",
        "path": "baseline_dmgnn/final_summary.json",
    },
    {
        "method": "ETD-FGL",
        "kind": "etd_summary",
        "path": "defended_fedavg_structural_backdoor/final_summary.json",
    },
]


def _load_row(
    base_dir: Path,
    spec: dict[str, str],
) -> dict[str, Any]:
    path = base_dir / spec["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Required result for {spec['method']} was not found:\n{path}"
        )

    kind = spec["kind"]

    if kind == "clean_checkpoint":
        package = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "method": spec["method"],
            "best_round": int(package["round"]),
            "validation_accuracy": float(package["validation_accuracy"]),
            "clean_test_accuracy": float(package["test_accuracy"]),
            "clean_target_rate": "",
            "attack_success_rate": "",
            "trigger_attack_effect": "",
            "runtime_seconds": "",
        }

    if kind == "attacked_checkpoint":
        package = torch.load(path, map_location="cpu", weights_only=False)
        metrics = package["metrics"]
        return {
            "method": spec["method"],
            "best_round": int(package["round"]),
            "validation_accuracy": float(metrics["validation_accuracy"]),
            "clean_test_accuracy": float(metrics["test_accuracy"]),
            "clean_target_rate": float(metrics["clean_target_rate"]),
            "attack_success_rate": float(metrics["attack_success_rate"]),
            "trigger_attack_effect": float(metrics["attack_effect"]),
            "runtime_seconds": "",
        }

    with path.open("r", encoding="utf-8") as file:
        package = json.load(file)

    if kind == "summary":
        metrics = package["final_metrics"]
        runtime = package.get("elapsed_seconds", "")
        return {
            "method": spec["method"],
            "best_round": int(package["best_round"]),
            "validation_accuracy": float(metrics["validation_accuracy"]),
            "clean_test_accuracy": float(metrics["test_accuracy"]),
            "clean_target_rate": float(metrics["clean_target_rate"]),
            "attack_success_rate": float(metrics["attack_success_rate"]),
            "trigger_attack_effect": float(metrics["attack_effect"]),
            "runtime_seconds": float(runtime) if runtime != "" else "",
        }

    if kind == "etd_summary":
        metrics = package["final_metrics"]
        return {
            "method": spec["method"],
            "best_round": int(package["best_round"]),
            "validation_accuracy": float(metrics["validation_accuracy"]),
            "clean_test_accuracy": float(metrics["test_accuracy"]),
            "clean_target_rate": float(metrics["clean_target_rate"]),
            "attack_success_rate": float(metrics["attack_success_rate"]),
            "trigger_attack_effect": float(metrics["attack_effect"]),
            "runtime_seconds": package.get("elapsed_seconds", ""),
        }

    raise ValueError(f"Unsupported result kind: {kind}")


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Best round | Val. acc. | Clean acc. | Clean target | ASR | Trigger effect | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        def fmt(key: str) -> str:
            value = row[key]
            if value == "":
                return "-"
            if key in {"best_round"}:
                return str(value)
            return f"{float(value):.4f}"

        lines.append(
            "| {method} | {round_} | {val} | {clean} | {target} | {asr} | {effect} | {runtime} |".format(
                method=row["method"],
                round_=fmt("best_round"),
                val=fmt("validation_accuracy"),
                clean=fmt("clean_test_accuracy"),
                target=fmt("clean_target_rate"),
                asr=fmt("attack_success_rate"),
                effect=fmt("trigger_attack_effect"),
                runtime=fmt("runtime_seconds"),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    base_dir = project_dir / "outputs" / "federated_cora"
    output_dir = base_dir / "all_baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_load_row(base_dir, spec) for spec in METHOD_SPECS]

    attacked_asr = next(
        float(row["attack_success_rate"])
        for row in rows
        if row["method"] == "Attacked FedAvg"
    )
    attacked_accuracy = next(
        float(row["clean_test_accuracy"])
        for row in rows
        if row["method"] == "Attacked FedAvg"
    )

    for row in rows:
        if row["attack_success_rate"] == "":
            row["asr_absolute_reduction_vs_attacked"] = ""
            row["asr_relative_reduction_vs_attacked"] = ""
        else:
            reduction = attacked_asr - float(row["attack_success_rate"])
            row["asr_absolute_reduction_vs_attacked"] = reduction
            row["asr_relative_reduction_vs_attacked"] = (
                reduction / attacked_asr if attacked_asr > 0 else 0.0
            )

        row["clean_accuracy_change_vs_attacked"] = (
            float(row["clean_test_accuracy"]) - attacked_accuracy
        )

    csv_path = output_dir / "all_baseline_comparison.csv"
    json_path = output_dir / "all_baseline_comparison.json"
    markdown_path = output_dir / "all_baseline_comparison.md"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    markdown_path.write_text(_format_markdown(rows), encoding="utf-8")

    print("=" * 132)
    print("Final baseline comparison on Cora")
    print("=" * 132)
    print(
        f"{'Method':22s} | {'Round':>5s} | {'CleanAcc':>8s} | "
        f"{'ASR':>8s} | {'Effect':>8s} | {'ASR drop':>8s}"
    )
    print("-" * 132)
    for row in rows:
        asr = "-" if row["attack_success_rate"] == "" else f"{float(row['attack_success_rate']):.4f}"
        effect = "-" if row["trigger_attack_effect"] == "" else f"{float(row['trigger_attack_effect']):.4f}"
        drop = (
            "-"
            if row["asr_absolute_reduction_vs_attacked"] == ""
            else f"{float(row['asr_absolute_reduction_vs_attacked']):.4f}"
        )
        print(
            f"{row['method']:22s} | {int(row['best_round']):5d} | "
            f"{float(row['clean_test_accuracy']):8.4f} | "
            f"{asr:>8s} | {effect:>8s} | {drop:>8s}"
        )
    print("-" * 132)
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {markdown_path}")
    print("=" * 132)


if __name__ == "__main__":
    main()
