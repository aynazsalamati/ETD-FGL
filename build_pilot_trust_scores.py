from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# Pilot configuration
# ============================================================

NUM_CLIENTS = 5
MALICIOUS_CLIENT_ID = 0

ALPHA_TOPOLOGY = 0.50
ALPHA_EXPLANATION = 0.50
ALPHA_AGREEMENT = 0.25

TRUST_SENSITIVITY = 3.0
SOFT_THRESHOLD = 0.55

TARGET_PROBE_COUNT = 6
TARGET_TRAIN_NODE_COUNT = 28
NEUTRAL_TRUST = 0.50

EPSILON = 1e-12


TOPOLOGY_FEATURES = [
    "density",
    "average_degree",
    "normalized_degree_variance",
    "average_clustering",
    "homophily",
    "triangle_frequency",
    "wedge_frequency",
    "star3_frequency",
    "motif_concentration",
]


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""

    clipped_value = float(
        np.clip(value, -30.0, 30.0)
    )

    return float(
        1.0 / (1.0 + math.exp(-clipped_value))
    )


def robust_scale(
    values: np.ndarray,
    axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate robust median and scale using MAD."""

    median = np.median(
        values,
        axis=axis,
    )

    if axis is None:
        deviations = np.abs(
            values - median
        )
    else:
        deviations = np.abs(
            values
            - np.expand_dims(median, axis=axis)
        )

    mad = np.median(
        deviations,
        axis=axis,
    )

    scale = 1.4826 * mad

    standard_deviation = np.std(
        values,
        axis=axis,
    )

    scale = np.where(
        scale > 1e-8,
        scale,
        standard_deviation,
    )

    scale = np.where(
        scale > 1e-8,
        scale,
        1.0,
    )

    return median, scale


def robust_calibrate_scores(
    raw_scores: dict[int, float],
) -> dict[int, float]:
    """
    Convert raw anomaly scores into one-sided anomaly intensity.

    A score at or below the robust median maps to zero.
    Only the positive anomaly tail receives a non-zero value.
    """

    client_ids = sorted(
        raw_scores.keys()
    )

    score_array = np.array(
        [
            raw_scores[client_id]
            for client_id in client_ids
        ],
        dtype=np.float64,
    )

    median, scale = robust_scale(
        score_array
    )

    calibrated_scores: dict[int, float] = {}

    for client_id, raw_score in zip(
        client_ids,
        score_array,
        strict=True,
    ):
        standardized_score = float(
            (raw_score - median)
            / scale
        )

        positive_tail = max(
            standardized_score,
            0.0,
        )

        anomaly_intensity = (
            2.0
            * (
                sigmoid(positive_tail)
                - 0.5
            )
        )

        calibrated_scores[client_id] = float(
            np.clip(
                anomaly_intensity,
                0.0,
                1.0,
            )
        )

    return calibrated_scores


def load_topology_baseline_and_current(
    csv_path: Path,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, float]],
]:
    """
    Load clean per-client topology baselines and current states.

    Client 0 currently uses its attacked topology.
    Clients 1 to 4 currently remain in their clean states.
    """

    baseline_descriptors: dict[
        int,
        dict[str, float],
    ] = {}

    current_descriptors: dict[
        int,
        dict[str, float],
    ] = {}

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        for row in reader:
            client_id = int(
                row["client"]
            )

            state = row["state"]

            descriptor = {
                feature_name: float(
                    row[feature_name]
                )
                for feature_name
                in TOPOLOGY_FEATURES
            }

            if state == "clean":
                baseline_descriptors[
                    client_id
                ] = descriptor

                if (
                    client_id
                    != MALICIOUS_CLIENT_ID
                ):
                    current_descriptors[
                        client_id
                    ] = descriptor

            elif (
                client_id
                == MALICIOUS_CLIENT_ID
                and state == "attacked"
            ):
                current_descriptors[
                    client_id
                ] = descriptor

    expected_clients = set(
        range(NUM_CLIENTS)
    )

    missing_baselines = (
        expected_clients
        - set(
            baseline_descriptors.keys()
        )
    )

    missing_current = (
        expected_clients
        - set(
            current_descriptors.keys()
        )
    )

    if missing_baselines:
        raise RuntimeError(
            "Missing clean topology baselines for: "
            f"{sorted(missing_baselines)}"
        )

    if missing_current:
        raise RuntimeError(
            "Missing current topology states for: "
            f"{sorted(missing_current)}"
        )

    return (
        baseline_descriptors,
        current_descriptors,
    )


def calculate_topology_drift_anomalies(
    baseline_descriptors: dict[
        int,
        dict[str, float],
    ],
    current_descriptors: dict[
        int,
        dict[str, float],
    ],
) -> tuple[
    dict[int, float],
    dict[int, float],
]:
    """
    Calculate topology drift relative to each client's own baseline.

    Cross-client robust scales are used only to normalize features.
    The anomaly itself is based on temporal self-deviation.
    """

    client_ids = sorted(
        baseline_descriptors.keys()
    )

    baseline_matrix = np.array(
        [
            [
                baseline_descriptors[
                    client_id
                ][feature_name]
                for feature_name
                in TOPOLOGY_FEATURES
            ]
            for client_id in client_ids
        ],
        dtype=np.float64,
    )

    current_matrix = np.array(
        [
            [
                current_descriptors[
                    client_id
                ][feature_name]
                for feature_name
                in TOPOLOGY_FEATURES
            ]
            for client_id in client_ids
        ],
        dtype=np.float64,
    )

    _, feature_scale = robust_scale(
        baseline_matrix,
        axis=0,
    )

    standardized_drift = (
        current_matrix
        - baseline_matrix
    ) / feature_scale

    standardized_drift = np.abs(
        standardized_drift
    )

    standardized_drift = np.clip(
        standardized_drift,
        0.0,
        5.0,
    )

    raw_array = np.sqrt(
        np.mean(
            standardized_drift ** 2,
            axis=1,
        )
    )

    raw_scores = {
        client_id: float(score)
        for client_id, score in zip(
            client_ids,
            raw_array,
            strict=True,
        )
    }

    calibrated_scores = (
        robust_calibrate_scores(
            raw_scores
        )
    )

    return (
        raw_scores,
        calibrated_scores,
    )


def load_explanation_information(
    json_path: Path,
) -> tuple[
    dict[int, float],
    dict[int, int],
]:
    """Load explanation anomalies and probe counts."""

    with json_path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        report = json.load(
            json_file
        )

    raw_scores: dict[int, float] = {}
    probe_counts: dict[int, int] = {}

    for client_id in range(
        NUM_CLIENTS
    ):
        client_report = report[
            "clients"
        ][str(client_id)]

        raw_scores[client_id] = float(
            client_report[
                "raw_explanation_anomaly"
            ]
        )

        probe_counts[client_id] = len(
            client_report[
                "probe_nodes"
            ]
        )

    calibrated_scores = (
        robust_calibrate_scores(
            raw_scores
        )
    )

    return (
        calibrated_scores,
        probe_counts,
    )


def load_training_node_counts(
    json_path: Path,
) -> dict[int, int]:
    """Load local training sizes from partition summary."""

    with json_path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        summary = json.load(
            json_file
        )

    return {
        int(client["client_id"]): int(
            client["train_nodes"]
        )
        for client in summary[
            "clients"
        ]
    }


def calculate_two_view_agreement(
    topology_score: float,
    explanation_score: float,
) -> float:
    """
    Pilot two-view version of cross-view agreement.

    The geometric component requires both views to be elevated.
    The consistency component penalizes large disagreement.
    """

    geometric_component = math.sqrt(
        max(
            topology_score
            * explanation_score,
            0.0,
        )
    )

    consistency_component = (
        1.0
        - abs(
            topology_score
            - explanation_score
        )
    )

    return float(
        geometric_component
        * max(
            consistency_component,
            0.0,
        )
    )


def build_trust_report(
    topology_raw: dict[int, float],
    topology_calibrated: dict[int, float],
    explanation_calibrated: dict[int, float],
    probe_counts: dict[int, int],
    train_node_counts: dict[int, int],
) -> dict[int, dict[str, Any]]:
    """
    Fuse topology and explanation evidence.

    This is a two-view pilot. Model-update anomaly and temporal
    persistence will be added in the defended training stage.
    """

    reports: dict[
        int,
        dict[str, Any],
    ] = {}

    preliminary_weights: dict[
        int,
        float,
    ] = {}

    for client_id in range(
        NUM_CLIENTS
    ):
        topology_score = (
            topology_calibrated[
                client_id
            ]
        )

        explanation_score = (
            explanation_calibrated[
                client_id
            ]
        )

        base_score = (
            ALPHA_TOPOLOGY
            * topology_score
            + ALPHA_EXPLANATION
            * explanation_score
        )

        agreement_score = (
            calculate_two_view_agreement(
                topology_score=(
                    topology_score
                ),
                explanation_score=(
                    explanation_score
                ),
            )
        )

        maliciousness = (
            (
                1.0
                - ALPHA_AGREEMENT
            )
            * base_score
            + ALPHA_AGREEMENT
            * agreement_score
        )

        trust = math.exp(
            -TRUST_SENSITIVITY
            * maliciousness
        )

        evidence_confidence = (
            min(
                1.0,
                probe_counts[
                    client_id
                ]
                / TARGET_PROBE_COUNT,
            )
            * min(
                1.0,
                train_node_counts[
                    client_id
                ]
                / TARGET_TRAIN_NODE_COUNT,
            )
        )

        adjusted_trust = (
            evidence_confidence
            * trust
            + (
                1.0
                - evidence_confidence
            )
            * NEUTRAL_TRUST
        )

        state = (
            "SUSPECT"
            if maliciousness
            >= SOFT_THRESHOLD
            else "TRUSTED"
        )

        preliminary_weight = (
            train_node_counts[
                client_id
            ]
            * adjusted_trust
        )

        preliminary_weights[
            client_id
        ] = preliminary_weight

        reports[client_id] = {
            "client_id": client_id,
            "topology_raw_anomaly": (
                topology_raw[
                    client_id
                ]
            ),
            "topology_anomaly": (
                topology_score
            ),
            "explanation_anomaly": (
                explanation_score
            ),
            "agreement_score": (
                agreement_score
            ),
            "base_score": base_score,
            "maliciousness": (
                maliciousness
            ),
            "trust": trust,
            "evidence_confidence": (
                evidence_confidence
            ),
            "adjusted_trust": (
                adjusted_trust
            ),
            "train_nodes": (
                train_node_counts[
                    client_id
                ]
            ),
            "probe_count": (
                probe_counts[
                    client_id
                ]
            ),
            "state": state,
        }

    total_preliminary_weight = sum(
        preliminary_weights.values()
    )

    if total_preliminary_weight <= EPSILON:
        raise RuntimeError(
            "Total aggregation weight is zero."
        )

    for client_id in range(
        NUM_CLIENTS
    ):
        reports[client_id][
            "normalized_weight"
        ] = float(
            preliminary_weights[
                client_id
            ]
            / total_preliminary_weight
        )

    return reports


def save_reports(
    reports: dict[
        int,
        dict[str, Any],
    ],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Save trust-fusion results."""

    rows = [
        reports[client_id]
        for client_id in range(
            NUM_CLIENTS
        )
    ]

    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    complete_report = {
        "configuration": {
            "num_clients": (
                NUM_CLIENTS
            ),
            "alpha_topology": (
                ALPHA_TOPOLOGY
            ),
            "alpha_explanation": (
                ALPHA_EXPLANATION
            ),
            "alpha_agreement": (
                ALPHA_AGREEMENT
            ),
            "trust_sensitivity": (
                TRUST_SENSITIVITY
            ),
            "soft_threshold": (
                SOFT_THRESHOLD
            ),
            "note": (
                "Two-view pilot without update anomaly "
                "or temporal persistence"
            ),
        },
        "clients": reports,
    }

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            complete_report,
            json_file,
            indent=2,
        )


def main() -> None:
    project_dir = Path(
        __file__
    ).resolve().parent

    topology_csv_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "topology_descriptor"
        / "topology_descriptors.csv"
    )

    explanation_json_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "client_explanation_descriptors"
        / "client_explanation_report.json"
    )

    partition_json_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / "iid_stratified_5_clients"
        / "partition_summary.json"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "pilot_trust_fusion"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "pilot_trust_scores.csv"
    )

    json_path = (
        output_dir
        / "pilot_trust_report.json"
    )

    for required_path in [
        topology_csv_path,
        explanation_json_path,
        partition_json_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{required_path}"
            )

    (
        topology_baselines,
        topology_current,
    ) = load_topology_baseline_and_current(
        topology_csv_path
    )

    (
        topology_raw,
        topology_calibrated,
    ) = calculate_topology_drift_anomalies(
        baseline_descriptors=(
            topology_baselines
        ),
        current_descriptors=(
            topology_current
        ),
    )

    (
        explanation_calibrated,
        probe_counts,
    ) = load_explanation_information(
        explanation_json_path
    )

    train_node_counts = (
        load_training_node_counts(
            partition_json_path
        )
    )

    reports = build_trust_report(
        topology_raw=topology_raw,
        topology_calibrated=(
            topology_calibrated
        ),
        explanation_calibrated=(
            explanation_calibrated
        ),
        probe_counts=probe_counts,
        train_node_counts=(
            train_node_counts
        ),
    )

    save_reports(
        reports=reports,
        csv_path=csv_path,
        json_path=json_path,
    )

    print("=" * 108)
    print(
        "Pilot topology + explanation trust fusion"
    )
    print("=" * 108)

    print(
        f"{'Client':>6} | "
        f"{'Topo':>7} | "
        f"{'Explain':>7} | "
        f"{'Agree':>7} | "
        f"{'Malicious':>9} | "
        f"{'Trust':>7} | "
        f"{'Weight':>7} | "
        f"State"
    )

    print("-" * 108)

    for client_id in range(
        NUM_CLIENTS
    ):
        report = reports[
            client_id
        ]

        print(
            f"{client_id:6d} | "
            f"{report['topology_anomaly']:7.4f} | "
            f"{report['explanation_anomaly']:7.4f} | "
            f"{report['agreement_score']:7.4f} | "
            f"{report['maliciousness']:9.4f} | "
            f"{report['adjusted_trust']:7.4f} | "
            f"{report['normalized_weight']:7.4f} | "
            f"{report['state']}"
        )

    lowest_trust_client = min(
        reports,
        key=lambda client_id: reports[
            client_id
        ]["adjusted_trust"],
    )

    highest_maliciousness_client = max(
        reports,
        key=lambda client_id: reports[
            client_id
        ]["maliciousness"],
    )

    print("-" * 108)
    print(
        f"Highest maliciousness: Client "
        f"{highest_maliciousness_client}"
    )

    print(
        f"Lowest trust:          Client "
        f"{lowest_trust_client}"
    )

    print(
        f"Client 0 aggregation weight: "
        f"{reports[0]['normalized_weight']:.4f}"
    )

    print(f"CSV saved to:  {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()