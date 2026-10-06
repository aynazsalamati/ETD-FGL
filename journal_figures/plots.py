from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import METHODS, MethodSpec
from .io_utils import save_figure_all_formats, write_text
from .style import style_axis


METHOD_BY_KEY = {method.key: method for method in METHODS}


def _available_series(
    histories: dict[str, list[dict[str, Any]]],
    column: str,
    include: tuple[str, ...],
) -> list[tuple[MethodSpec, list[int], list[float]]]:
    series: list[tuple[MethodSpec, list[int], list[float]]] = []

    for key in include:
        rows = histories.get(key, [])
        method = METHOD_BY_KEY[key]

        points = [
            (int(row["round"]), float(row[column]))
            for row in rows
            if row.get("round") is not None
            and row.get(column) is not None
            and math.isfinite(float(row[column]))
        ]

        if not points:
            continue

        rounds, values = zip(*points, strict=True)
        series.append((method, list(rounds), list(values)))

    return series


def plot_round_metric(
    *,
    histories: dict[str, list[dict[str, Any]]],
    column: str,
    include: tuple[str, ...],
    y_label: str,
    output_base: Path,
    caption: str,
    y_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    series = _available_series(histories, column, include)

    if not series:
        return {
            "status": "skipped",
            "reason": f"No usable '{column}' values were found.",
            "figure": output_base.name,
        }

    figure, ax = plt.subplots(figsize=(7.4, 4.8))

    for method, rounds, values in series:
        marker_every = max(1, len(rounds) // 10)

        ax.plot(
            rounds,
            values,
            label=method.label,
            color=method.color,
            linestyle=method.linestyle,
            marker=method.marker,
            markevery=marker_every,
            linewidth=1.7,
            markersize=3.6,
        )

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(y_label)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    style_axis(ax)

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#aaaaaa",
        framealpha=0.92,
    )

    figure.tight_layout()
    files = save_figure_all_formats(figure, output_base)
    plt.close(figure)

    caption_path = output_base.with_name(
        output_base.name + "_caption"
    ).with_suffix(".txt")
    write_text(caption_path, caption)

    return {
        "status": "created",
        "figure": output_base.name,
        "files": files,
        "caption": str(caption_path),
    }


def plot_final_comparison(
    *,
    histories: dict[str, list[dict[str, Any]]],
    dataset_name: str,
    output_base: Path,
) -> dict[str, Any]:
    best_rows: dict[str, dict[str, Any]] = {}

    for key, rows in histories.items():
        valid = [
            row for row in rows
            if row.get("validation_accuracy") is not None
            and row.get("test_accuracy") is not None
        ]

        if not valid:
            continue

        best_rows[key] = max(
            valid,
            key=lambda row: (
                float(row["validation_accuracy"]),
                -int(row["round"]),
            ),
        )

    required = {"clean", "attacked", "defended"}
    if not required.issubset(best_rows):
        return {
            "status": "skipped",
            "reason": "All three best-round histories are required.",
            "figure": output_base.name,
        }

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    accuracy_keys = ["clean", "attacked", "defended"]
    accuracy_labels = [METHOD_BY_KEY[key].label for key in accuracy_keys]
    accuracy_values = [
        float(best_rows[key]["test_accuracy"])
        for key in accuracy_keys
    ]

    x = np.arange(len(accuracy_keys))
    bars = axes[0].bar(
        x,
        accuracy_values,
        width=0.62,
        color=[METHOD_BY_KEY[key].color for key in accuracy_keys],
        edgecolor="black",
        linewidth=0.7,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(accuracy_labels, rotation=10, ha="right")
    axes[0].set_ylabel("Clean Test Accuracy")
    style_axis(axes[0], grid_axis="y")

    lower = max(0.0, min(accuracy_values) - 0.06)
    upper = min(1.0, max(accuracy_values) + 0.04)
    axes[0].set_ylim(lower, upper)

    for bar, value in zip(bars, accuracy_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    asr_keys = ["attacked", "defended"]
    asr_values = [
        float(best_rows[key]["attack_success_rate"])
        for key in asr_keys
    ]
    asr_labels = [METHOD_BY_KEY[key].label for key in asr_keys]

    x = np.arange(len(asr_keys))
    bars = axes[1].bar(
        x,
        asr_values,
        width=0.62,
        color=[METHOD_BY_KEY[key].color for key in asr_keys],
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(asr_labels, rotation=10, ha="right")
    axes[1].set_ylabel("Attack Success Rate")
    axes[1].set_ylim(0.0, min(1.0, max(asr_values) + 0.12))
    style_axis(axes[1], grid_axis="y")

    for bar, value in zip(bars, asr_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0].text(
        0.5,
        -0.24,
        "(a) Clean test accuracy",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
    )
    axes[1].text(
        0.5,
        -0.24,
        "(b) Attack success rate",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
    )

    figure.tight_layout()
    files = save_figure_all_formats(figure, output_base)
    plt.close(figure)

    caption = (
        f"Fig. X. Final clean-test accuracy and attack success rate comparison "
        f"on the {dataset_name} dataset. ETD-FGL reduces backdoor effectiveness "
        f"while preserving clean-task performance."
    )
    caption_path = output_base.with_name(
        output_base.name + "_caption"
    ).with_suffix(".txt")
    write_text(caption_path, caption)

    return {
        "status": "created",
        "figure": output_base.name,
        "files": files,
        "caption": str(caption_path),
    }


def plot_trust_and_maliciousness(
    *,
    trust_rows: list[dict[str, Any]],
    output_base: Path,
    dataset_name: str,
) -> dict[str, Any]:
    required_columns = {
        "client_id",
        "maliciousness",
        "adjusted_trust",
    }

    if not trust_rows or not required_columns.issubset(trust_rows[0]):
        return {
            "status": "skipped",
            "reason": "pilot_trust_scores.csv is unavailable or incomplete.",
            "figure": output_base.name,
        }

    rows = sorted(trust_rows, key=lambda row: int(row["client_id"]))
    clients = [f"Client {int(row['client_id'])}" for row in rows]
    maliciousness = [float(row["maliciousness"]) for row in rows]
    trust = [float(row["adjusted_trust"]) for row in rows]

    x = np.arange(len(rows))
    width = 0.36

    figure, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(
        x - width / 2,
        maliciousness,
        width,
        label="Maliciousness Score",
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        trust,
        width,
        label="Adjusted Trust Score",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    style_axis(ax, grid_axis="y")
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="#aaaaaa",
    )

    figure.tight_layout()
    files = save_figure_all_formats(figure, output_base)
    plt.close(figure)

    caption = (
        f"Fig. X. Client maliciousness and confidence-adjusted trust scores on "
        f"the {dataset_name} dataset. A malicious client is expected to receive "
        f"a higher maliciousness score and a lower trust score."
    )
    caption_path = output_base.with_name(
        output_base.name + "_caption"
    ).with_suffix(".txt")
    write_text(caption_path, caption)

    return {
        "status": "created",
        "figure": output_base.name,
        "files": files,
        "caption": str(caption_path),
    }


def plot_anomaly_scores(
    *,
    trust_rows: list[dict[str, Any]],
    output_base: Path,
    dataset_name: str,
) -> dict[str, Any]:
    required_columns = {
        "client_id",
        "topology_anomaly",
        "explanation_anomaly",
    }

    if not trust_rows or not required_columns.issubset(trust_rows[0]):
        return {
            "status": "skipped",
            "reason": "Topology/explanation anomaly columns are unavailable.",
            "figure": output_base.name,
        }

    rows = sorted(trust_rows, key=lambda row: int(row["client_id"]))
    clients = [f"Client {int(row['client_id'])}" for row in rows]
    topology = [float(row["topology_anomaly"]) for row in rows]
    explanation = [float(row["explanation_anomaly"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(len(rows))

    axes[0].bar(
        x,
        topology,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clients, rotation=10)
    axes[0].set_ylabel("Topology Anomaly Score")
    axes[0].set_ylim(0.0, 1.05)
    style_axis(axes[0], grid_axis="y")

    axes[1].bar(
        x,
        explanation,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clients, rotation=10)
    axes[1].set_ylabel("Explanation Anomaly Score")
    axes[1].set_ylim(0.0, 1.05)
    style_axis(axes[1], grid_axis="y")

    axes[0].text(
        0.5,
        -0.22,
        "(a) Topology anomaly",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
    )
    axes[1].text(
        0.5,
        -0.22,
        "(b) Explanation anomaly",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
    )

    figure.tight_layout()
    files = save_figure_all_formats(figure, output_base)
    plt.close(figure)

    caption = (
        f"Fig. X. Topology-level and explanation-level anomaly scores for "
        f"participating clients on the {dataset_name} dataset."
    )
    caption_path = output_base.with_name(
        output_base.name + "_caption"
    ).with_suffix(".txt")
    write_text(caption_path, caption)

    return {
        "status": "created",
        "figure": output_base.name,
        "files": files,
        "caption": str(caption_path),
    }


def plot_aggregation_weights(
    *,
    trust_rows: list[dict[str, Any]],
    output_base: Path,
    dataset_name: str,
) -> dict[str, Any]:
    required_columns = {"client_id", "normalized_weight"}

    if not trust_rows or not required_columns.issubset(trust_rows[0]):
        return {
            "status": "skipped",
            "reason": "Normalized aggregation weights are unavailable.",
            "figure": output_base.name,
        }

    rows = sorted(trust_rows, key=lambda row: int(row["client_id"]))
    clients = [f"Client {int(row['client_id'])}" for row in rows]
    weights = [float(row["normalized_weight"]) for row in rows]

    figure, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(
        np.arange(len(rows)),
        weights,
        color="#9467bd",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(clients)
    ax.set_ylabel("Normalized Aggregation Weight")
    ax.set_ylim(0.0, max(weights) * 1.22 if max(weights) > 0 else 1.0)
    style_axis(ax, grid_axis="y")

    for bar, value in zip(bars, weights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(weights) * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    figure.tight_layout()
    files = save_figure_all_formats(figure, output_base)
    plt.close(figure)

    caption = (
        f"Fig. X. Normalized trust-aware aggregation weights assigned to clients "
        f"on the {dataset_name} dataset."
    )
    caption_path = output_base.with_name(
        output_base.name + "_caption"
    ).with_suffix(".txt")
    write_text(caption_path, caption)

    return {
        "status": "created",
        "figure": output_base.name,
        "files": files,
        "caption": str(caption_path),
    }
