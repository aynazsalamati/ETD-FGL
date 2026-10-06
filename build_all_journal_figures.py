from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from journal_figures.config import (
    METHODS,
    OUTPUT_ROOT_NAME,
    dataset_display_name,
    discover_dataset_roots,
)
from journal_figures.io_utils import read_csv_rows, write_json
from journal_figures.plots import (
    plot_aggregation_weights,
    plot_anomaly_scores,
    plot_final_comparison,
    plot_round_metric,
    plot_trust_and_maliciousness,
)
from journal_figures.style import apply_journal_style


def load_histories(dataset_root: Path):
    histories = {}

    for method in METHODS:
        path = dataset_root / method.directory / "training_history.csv"
        histories[method.key] = read_csv_rows(path)

    return histories


def build_dataset_figures(project_dir: Path, dataset_root: Path):
    dataset_name = dataset_display_name(dataset_root)
    dataset_slug = dataset_root.name.removeprefix("federated_")
    output_dir = project_dir / "outputs" / OUTPUT_ROOT_NAME / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(dataset_root)
    results = []

    results.append(
        plot_round_metric(
            histories=histories,
            column="test_accuracy",
            include=("clean", "attacked", "defended"),
            y_label="Clean Test Accuracy",
            output_base=output_dir / "Fig_01_Clean_Accuracy_vs_Round",
            caption=(
                f"Fig. X. Clean test accuracy of Clean FedAvg, Attacked FedAvg, "
                f"and ETD-FGL over federated communication rounds on the "
                f"{dataset_name} dataset."
            ),
        )
    )

    results.append(
        plot_round_metric(
            histories=histories,
            column="attack_success_rate",
            include=("attacked", "defended"),
            y_label="Attack Success Rate",
            output_base=output_dir / "Fig_02_ASR_vs_Round",
            caption=(
                f"Fig. X. Attack success rate of Attacked FedAvg and ETD-FGL "
                f"over federated communication rounds on the {dataset_name} dataset."
            ),
            y_limits=(0.0, 1.0),
        )
    )

    results.append(
        plot_round_metric(
            histories=histories,
            column="mean_client_loss",
            include=("clean", "attacked", "defended"),
            y_label="Mean Client Training Loss",
            output_base=output_dir / "Fig_03_Training_Loss_vs_Round",
            caption=(
                f"Fig. X. Mean client training loss of the compared federated "
                f"methods over communication rounds on the {dataset_name} dataset."
            ),
        )
    )

    results.append(
        plot_round_metric(
            histories=histories,
            column="attack_effect",
            include=("attacked", "defended"),
            y_label="Trigger Attack Effect",
            output_base=output_dir / "Fig_04_Trigger_Effect_vs_Round",
            caption=(
                f"Fig. X. Trigger attack effect of Attacked FedAvg and ETD-FGL "
                f"over communication rounds on the {dataset_name} dataset."
            ),
        )
    )

    trust_path = dataset_root / "pilot_trust_fusion" / "pilot_trust_scores.csv"
    trust_rows = read_csv_rows(trust_path)

    results.append(
        plot_trust_and_maliciousness(
            trust_rows=trust_rows,
            output_base=output_dir / "Fig_05_Client_Trust_and_Maliciousness",
            dataset_name=dataset_name,
        )
    )

    results.append(
        plot_anomaly_scores(
            trust_rows=trust_rows,
            output_base=output_dir / "Fig_06_Topology_and_Explanation_Anomalies",
            dataset_name=dataset_name,
        )
    )

    results.append(
        plot_aggregation_weights(
            trust_rows=trust_rows,
            output_base=output_dir / "Fig_07_Trust_Weighted_Aggregation",
            dataset_name=dataset_name,
        )
    )

    results.append(
        plot_final_comparison(
            histories=histories,
            dataset_name=dataset_name,
            output_base=output_dir / "Fig_08_Final_Accuracy_ASR_Comparison",
        )
    )

    return {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "figures": results,
    }


def main() -> None:
    apply_journal_style()

    project_dir = Path(__file__).resolve().parent
    dataset_roots = discover_dataset_roots(project_dir)

    if not dataset_roots:
        raise FileNotFoundError(
            "No outputs/federated_* dataset folders containing training_history.csv "
            "were found. Copy this package into the ETD-FGL project root."
        )

    reports = [
        build_dataset_figures(project_dir, dataset_root)
        for dataset_root in dataset_roots
    ]

    manifest_path = (
        project_dir
        / "outputs"
        / OUTPUT_ROOT_NAME
        / "journal_figure_manifest.json"
    )
    write_json(manifest_path, reports)

    print("=" * 100)
    print("ETD-FGL journal figures")
    print("=" * 100)

    for report in reports:
        print(f"Dataset: {report['dataset']}")
        print(f"Output:  {report['output_dir']}")

        for item in report["figures"]:
            print(f"  {item['status'].upper():7s}  {item['figure']}")
            if item["status"] == "skipped":
                print(f"           Reason: {item['reason']}")

    print("-" * 100)
    print(f"Manifest: {manifest_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
