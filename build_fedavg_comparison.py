from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METHODS = [
    {
        "key": "clean",
        "method": "Clean FedAvg",
        "history_path": (
            "outputs/federated_cora/fedavg_clean/"
            "training_history.csv"
        ),
    },
    {
        "key": "attacked",
        "method": "Attacked FedAvg",
        "history_path": (
            "outputs/federated_cora/fedavg_structural_backdoor/"
            "training_history.csv"
        ),
    },
    {
        "key": "defended",
        "method": "Defended FedAvg",
        "history_path": (
            "outputs/federated_cora/"
            "defended_fedavg_structural_backdoor/"
            "training_history.csv"
        ),
    },
]

EPSILON = 1e-12


def parse_float(
    row: dict[str, str],
    key: str,
) -> float | None:
    """Read an optional floating-point value from a CSV row."""

    raw_value = row.get(key)

    if raw_value is None:
        return None

    raw_value = raw_value.strip()

    if not raw_value:
        return None

    return float(raw_value)


def load_best_round(
    csv_path: Path,
) -> dict[str, float | int | None]:
    """
    Load a training history and select the best validation round.

    When multiple rounds have the same validation accuracy,
    the earliest round is selected to match checkpoint logic.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            "Training history was not found:\n"
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
            f"Training history is empty:\n{csv_path}"
        )

    required_columns = {
        "round",
        "validation_accuracy",
        "test_accuracy",
    }

    missing_columns = required_columns - set(rows[0].keys())

    if missing_columns:
        raise RuntimeError(
            f"Missing columns in {csv_path.name}: "
            f"{sorted(missing_columns)}"
        )

    best_row = max(
        rows,
        key=lambda row: (
            float(row["validation_accuracy"]),
            -int(row["round"]),
        ),
    )

    return {
        "best_round": int(best_row["round"]),
        "train_accuracy": parse_float(
            best_row,
            "train_accuracy",
        ),
        "validation_accuracy": parse_float(
            best_row,
            "validation_accuracy",
        ),
        "clean_test_accuracy": parse_float(
            best_row,
            "test_accuracy",
        ),
        "clean_target_rate": parse_float(
            best_row,
            "clean_target_rate",
        ),
        "attack_success_rate": parse_float(
            best_row,
            "attack_success_rate",
        ),
        "trigger_attack_effect": parse_float(
            best_row,
            "attack_effect",
        ),
    }


def safe_difference(
    first: float | None,
    second: float | None,
) -> float | None:
    """Return first minus second when both values are available."""

    if first is None or second is None:
        return None

    return first - second


def safe_relative_reduction(
    baseline: float | None,
    current: float | None,
) -> float | None:
    """Calculate relative reduction from baseline to current."""

    if baseline is None or current is None:
        return None

    if abs(baseline) <= EPSILON:
        return None

    return (baseline - current) / baseline


def format_decimal(
    value: float | int | None,
    digits: int = 4,
) -> str:
    """Format optional numeric values for console and Markdown."""

    if value is None:
        return "-"

    if isinstance(value, int):
        return str(value)

    return f"{value:.{digits}f}"


def build_comparison(
    project_dir: Path,
) -> list[dict[str, Any]]:
    """Build the three-method automatic comparison."""

    results: dict[str, dict[str, Any]] = {}

    for method_config in METHODS:
        key = method_config["key"]

        history_path = (
            project_dir
            / method_config["history_path"]
        )

        best_metrics = load_best_round(
            history_path
        )

        results[key] = {
            "method": method_config["method"],
            **best_metrics,
            "source_history": str(history_path),
        }

    clean_accuracy = results[
        "clean"
    ]["clean_test_accuracy"]

    attacked_accuracy = results[
        "attacked"
    ]["clean_test_accuracy"]

    attacked_asr = results[
        "attacked"
    ]["attack_success_rate"]

    attacked_effect = results[
        "attacked"
    ]["trigger_attack_effect"]

    for key in [
        "clean",
        "attacked",
        "defended",
    ]:
        result = results[key]

        result[
            "clean_accuracy_change_vs_clean_fedavg"
        ] = safe_difference(
            result["clean_test_accuracy"],
            clean_accuracy,
        )

        result[
            "clean_accuracy_change_vs_attacked_fedavg"
        ] = safe_difference(
            result["clean_test_accuracy"],
            attacked_accuracy,
        )

        result[
            "asr_absolute_reduction_vs_attacked"
        ] = (
            safe_difference(
                attacked_asr,
                result["attack_success_rate"],
            )
            if key == "defended"
            else None
        )

        result[
            "asr_relative_reduction_vs_attacked"
        ] = (
            safe_relative_reduction(
                attacked_asr,
                result["attack_success_rate"],
            )
            if key == "defended"
            else None
        )

        result[
            "trigger_effect_reduction_vs_attacked"
        ] = (
            safe_difference(
                attacked_effect,
                result["trigger_attack_effect"],
            )
            if key == "defended"
            else None
        )

    return [
        results["clean"],
        results["attacked"],
        results["defended"],
    ]


def save_csv(
    comparison: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save raw comparison values as CSV."""

    fieldnames = [
        "method",
        "best_round",
        "train_accuracy",
        "validation_accuracy",
        "clean_test_accuracy",
        "clean_target_rate",
        "attack_success_rate",
        "trigger_attack_effect",
        "clean_accuracy_change_vs_clean_fedavg",
        "clean_accuracy_change_vs_attacked_fedavg",
        "asr_absolute_reduction_vs_attacked",
        "asr_relative_reduction_vs_attacked",
        "trigger_effect_reduction_vs_attacked",
        "source_history",
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(comparison)


def save_json(
    comparison: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save comparison values and key conclusions as JSON."""

    attacked = comparison[1]
    defended = comparison[2]

    report = {
        "dataset": "Cora",
        "selection_rule": (
            "Maximum validation accuracy; earliest round "
            "selected when tied."
        ),
        "methods": comparison,
        "key_results": {
            "attacked_fedavg_asr": (
                attacked["attack_success_rate"]
            ),
            "defended_fedavg_asr": (
                defended["attack_success_rate"]
            ),
            "absolute_asr_reduction": (
                defended[
                    "asr_absolute_reduction_vs_attacked"
                ]
            ),
            "relative_asr_reduction": (
                defended[
                    "asr_relative_reduction_vs_attacked"
                ]
            ),
            "clean_accuracy_change_vs_attacked": (
                defended[
                    "clean_accuracy_change_vs_attacked_fedavg"
                ]
            ),
            "clean_accuracy_change_vs_clean": (
                defended[
                    "clean_accuracy_change_vs_clean_fedavg"
                ]
            ),
        },
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            report,
            json_file,
            indent=2,
        )


def save_markdown(
    comparison: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a paper-friendly Markdown comparison table."""

    headers = [
        "Method",
        "Best round",
        "Val Acc.",
        "Clean Test Acc.",
        "Clean Target Rate",
        "ASR",
        "Trigger Effect",
        "ASR Reduction",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|"
        + "|".join(
            [
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
        )
        + "|",
    ]

    for row in comparison:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    format_decimal(
                        row["best_round"]
                    ),
                    format_decimal(
                        row["validation_accuracy"]
                    ),
                    format_decimal(
                        row["clean_test_accuracy"]
                    ),
                    format_decimal(
                        row["clean_target_rate"]
                    ),
                    format_decimal(
                        row["attack_success_rate"]
                    ),
                    format_decimal(
                        row["trigger_attack_effect"]
                    ),
                    format_decimal(
                        row[
                            "asr_absolute_reduction_vs_attacked"
                        ]
                    ),
                ]
            )
            + " |"
        )

    attacked = comparison[1]
    defended = comparison[2]

    lines.extend(
        [
            "",
            "## Automatic summary",
            "",
            (
                "- Absolute ASR reduction: "
                f"{format_decimal(defended['asr_absolute_reduction_vs_attacked'])}"
            ),
            (
                "- Relative ASR reduction: "
                f"{format_decimal(defended['asr_relative_reduction_vs_attacked'])}"
            ),
            (
                "- Clean accuracy change vs attacked FedAvg: "
                f"{format_decimal(defended['clean_accuracy_change_vs_attacked_fedavg'])}"
            ),
            (
                "- Clean accuracy change vs clean FedAvg: "
                f"{format_decimal(defended['clean_accuracy_change_vs_clean_fedavg'])}"
            ),
            (
                "- Attacked FedAvg ASR: "
                f"{format_decimal(attacked['attack_success_rate'])}"
            ),
            (
                "- Defended FedAvg ASR: "
                f"{format_decimal(defended['attack_success_rate'])}"
            ),
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_table(
    comparison: list[dict[str, Any]],
) -> None:
    """Print the comparison table in the terminal."""

    print("=" * 126)
    print(
        "Automatic comparison: Clean FedAvg vs "
        "Attacked FedAvg vs Defended FedAvg"
    )
    print("=" * 126)

    print(
        f"{'Method':<20} | "
        f"{'Round':>5} | "
        f"{'Val':>7} | "
        f"{'CleanAcc':>8} | "
        f"{'CleanTarget':>11} | "
        f"{'ASR':>7} | "
        f"{'Effect':>8} | "
        f"{'ASR Drop':>8}"
    )

    print("-" * 126)

    for row in comparison:
        print(
            f"{row['method']:<20} | "
            f"{format_decimal(row['best_round']):>5} | "
            f"{format_decimal(row['validation_accuracy']):>7} | "
            f"{format_decimal(row['clean_test_accuracy']):>8} | "
            f"{format_decimal(row['clean_target_rate']):>11} | "
            f"{format_decimal(row['attack_success_rate']):>7} | "
            f"{format_decimal(row['trigger_attack_effect']):>8} | "
            f"{format_decimal(row['asr_absolute_reduction_vs_attacked']):>8}"
        )

    defended = comparison[2]

    print("-" * 126)

    print(
        "Defended ASR relative reduction: "
        f"{format_decimal(defended['asr_relative_reduction_vs_attacked'])}"
    )

    print(
        "Defended clean accuracy change vs attacked: "
        f"{format_decimal(defended['clean_accuracy_change_vs_attacked_fedavg'])}"
    )

    print(
        "Defended clean accuracy change vs clean:    "
        f"{format_decimal(defended['clean_accuracy_change_vs_clean_fedavg'])}"
    )

    print("=" * 126)


def main() -> None:
    project_dir = Path(
        __file__
    ).resolve().parent

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "automatic_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "fedavg_comparison.csv"
    )

    json_path = (
        output_dir
        / "fedavg_comparison.json"
    )

    markdown_path = (
        output_dir
        / "fedavg_comparison.md"
    )

    comparison = build_comparison(
        project_dir
    )

    save_csv(
        comparison=comparison,
        output_path=csv_path,
    )

    save_json(
        comparison=comparison,
        output_path=json_path,
    )

    save_markdown(
        comparison=comparison,
        output_path=markdown_path,
    )

    print_table(comparison)

    print(f"CSV saved to:      {csv_path}")
    print(f"JSON saved to:     {json_path}")
    print(f"Markdown saved to: {markdown_path}")


if __name__ == "__main__":
    main()
