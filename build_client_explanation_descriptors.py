from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

from test_explanation_subgraph_topology import (
    explain_and_measure,
)
from train_fedavg_cora import GCN


SEED = 42
NUM_CLIENTS = 5
MALICIOUS_CLIENT_ID = 0

HIDDEN_CHANNELS = 16
DROPOUT = 0.5

RISK_PROBE_COUNT = 5
ANCHOR_PROBE_COUNT = 1
EPSILON = 1e-12


BASE_EXPLANATION_FEATURES = [
    "density",
    "average_clustering",
    "transitivity",
    "triangle_count",
    "triangle_density",
    "cycle_rank",
    "edge_score_concentration",
    "edge_score_entropy",
    "explanation_nodes",
    "explanation_edges",
]


def robust_standardize(
    values: np.ndarray,
) -> np.ndarray:
    """Standardize values using median and MAD."""

    median = float(np.median(values))

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    scale = 1.4826 * mad

    if scale <= EPSILON:
        scale = float(
            np.std(values)
        )

    if scale <= EPSILON:
        scale = 1.0

    return (
        values - median
    ) / scale


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""

    clipped_value = float(
        np.clip(
            value,
            -30.0,
            30.0,
        )
    )

    return float(
        1.0
        / (
            1.0
            + np.exp(-clipped_value)
        )
    )


def prepare_clean_client_data(
    client_data,
):
    """
    Add diagnostic fields expected by the explanation functions.

    Clean clients contain no poisoned or trigger nodes.
    """

    prepared_data = client_data.clone()

    prepared_data.clean_y = (
        prepared_data.y.clone()
    )

    prepared_data.poisoned_node_mask = (
        torch.zeros(
            prepared_data.num_nodes,
            dtype=torch.bool,
        )
    )

    prepared_data.trigger_node_mask = (
        torch.zeros(
            prepared_data.num_nodes,
            dtype=torch.bool,
        )
    )

    return prepared_data


def get_probe_candidates(
    data,
) -> Tensor:
    """Return original labeled training nodes."""

    candidate_mask = (
        data.train_mask
        & data.clean_y.ge(0)
        & ~data.trigger_node_mask
    )

    candidate_nodes = candidate_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError(
            "No eligible probe nodes were found."
        )

    return candidate_nodes


def calculate_structural_risk_scores(
    data,
    candidate_nodes: Tensor,
) -> dict[int, float]:
    """
    Rank probe candidates using structural surprise.

    The server does not need to know which nodes are poisoned.
    """

    cpu_data = data.clone().cpu()

    graph = to_networkx(
        cpu_data,
        to_undirected=True,
        remove_self_loops=True,
    )
    triangle_dictionary = nx.triangles(
        graph
    )

    clustering_dictionary = nx.clustering(
        graph
    )

    degree_dictionary = dict(
        graph.degree()
    )

    candidate_list = [
        int(node.item())
        for node in candidate_nodes
    ]

    triangle_values = np.array(
        [
            float(
                triangle_dictionary.get(
                    node,
                    0,
                )
            )
            for node in candidate_list
        ],
        dtype=np.float64,
    )

    clustering_values = np.array(
        [
            float(
                clustering_dictionary.get(
                    node,
                    0.0,
                )
            )
            for node in candidate_list
        ],
        dtype=np.float64,
    )

    degree_values = np.array(
        [
            float(
                degree_dictionary.get(
                    node,
                    0,
                )
            )
            for node in candidate_list
        ],
        dtype=np.float64,
    )

    triangle_z = robust_standardize(
        triangle_values
    )

    clustering_z = robust_standardize(
        clustering_values
    )

    degree_z = robust_standardize(
        degree_values
    )

    combined_risk = (
            triangle_z
            + 1.5 * clustering_z
            - 0.10 * degree_z
    )

    return {
        node: float(score)
        for node, score in zip(
            candidate_list,
            combined_risk,
            strict=True,
        )
    }


def select_risk_probes(
    risk_scores: dict[int, float],
    probe_count: int,
) -> list[int]:
    """Select nodes with the highest structural risk."""

    ranked_nodes = sorted(
        risk_scores.items(),
        key=lambda item: (
            item[1],
            -item[0],
        ),
        reverse=True,
    )

    return [
        node
        for node, _ in ranked_nodes[
            :probe_count
        ]
    ]


def select_anchor_probes(
    data,
    candidate_nodes: Tensor,
    excluded_nodes: set[int],
    probe_count: int,
    seed: int,
) -> list[int]:
    """
    Select deterministic class-balanced anchor probes.
    """

    random_generator = random.Random(
        seed
    )

    candidate_list = [
        int(node.item())
        for node in candidate_nodes
        if int(node.item())
        not in excluded_nodes
    ]

    nodes_by_class: dict[
        int,
        list[int],
    ] = {}

    for node in candidate_list:
        class_id = int(
            data.y[node].item()
        )

        nodes_by_class.setdefault(
            class_id,
            [],
        ).append(node)

    class_ids = sorted(
        nodes_by_class.keys()
    )

    random_generator.shuffle(
        class_ids
    )

    for class_nodes in (
        nodes_by_class.values()
    ):
        random_generator.shuffle(
            class_nodes
        )

    selected_nodes: list[int] = []

    # First pass: one node from each class.
    for class_id in class_ids:
        if len(selected_nodes) >= probe_count:
            break

        class_nodes = nodes_by_class[
            class_id
        ]

        if class_nodes:
            selected_nodes.append(
                class_nodes.pop()
            )

    # Second pass: fill any remaining positions.
    remaining_nodes = [
        node
        for class_nodes
        in nodes_by_class.values()
        for node in class_nodes
        if node not in selected_nodes
    ]

    random_generator.shuffle(
        remaining_nodes
    )

    for node in remaining_nodes:
        if len(selected_nodes) >= probe_count:
            break

        selected_nodes.append(node)

    if len(selected_nodes) < probe_count:
        raise RuntimeError(
            "Not enough nodes were available "
            "for anchor selection."
        )

    return selected_nodes


def select_dual_probe_set(
    data,
    client_id: int,
) -> tuple[
    list[int],
    list[int],
    list[int],
    dict[int, float],
]:
    """Construct Q_anchor union Q_risk."""

    candidate_nodes = get_probe_candidates(
        data
    )

    risk_scores = (
        calculate_structural_risk_scores(
            data=data,
            candidate_nodes=candidate_nodes,
        )
    )

    risk_probes = select_risk_probes(
        risk_scores=risk_scores,
        probe_count=RISK_PROBE_COUNT,
    )

    anchor_probes = select_anchor_probes(
        data=data,
        candidate_nodes=candidate_nodes,
        excluded_nodes=set(risk_probes),
        probe_count=ANCHOR_PROBE_COUNT,
        seed=SEED + client_id,
    )

    complete_probe_set = [
        *risk_probes,
        *anchor_probes,
    ]

    assert (
        len(complete_probe_set)
        == RISK_PROBE_COUNT
        + ANCHOR_PROBE_COUNT
    )

    assert (
        len(set(complete_probe_set))
        == len(complete_probe_set)
    )

    return (
        complete_probe_set,
        risk_probes,
        anchor_probes,
        risk_scores,
    )


def aggregate_client_reports(
    reports: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate node explanations using mean and standard deviation.
    """

    descriptor: dict[str, float] = {}

    for feature_name in (
        BASE_EXPLANATION_FEATURES
    ):
        feature_values = np.array(
            [
                float(
                    report[
                        feature_name
                    ]
                )
                for report in reports
            ],
            dtype=np.float64,
        )

        descriptor[
            f"mean_{feature_name}"
        ] = float(
            feature_values.mean()
        )

        descriptor[
            f"std_{feature_name}"
        ] = float(
            feature_values.std()
        )

    descriptor[
        "mean_prediction_confidence"
    ] = float(
        np.mean(
            [
                float(
                    report[
                        "prediction_confidence"
                    ]
                )
                for report in reports
            ]
        )
    )

    # Diagnostic only. Excluded from anomaly computation.
    descriptor[
        "oracle_mean_trigger_attribution"
    ] = float(
        np.mean(
            [
                float(
                    report[
                        "oracle_trigger_attribution"
                    ]
                )
                for report in reports
            ]
        )
    )

    return descriptor


def calculate_raw_anomaly_scores(
    client_descriptors: dict[
        int,
        dict[str, float],
    ],
    anomaly_feature_names: list[str],
) -> dict[int, float]:
    """
    Calculate robust multivariate explanation anomalies.
    """

    client_ids = sorted(
        client_descriptors.keys()
    )

    descriptor_matrix = np.array(
        [
            [
                client_descriptors[
                    client_id
                ][feature_name]
                for feature_name
                in anomaly_feature_names
            ]
            for client_id in client_ids
        ],
        dtype=np.float64,
    )

    median = np.median(
        descriptor_matrix,
        axis=0,
    )

    mad = np.median(
        np.abs(
            descriptor_matrix - median
        ),
        axis=0,
    )

    scale = 1.4826 * mad

    standard_deviation = np.std(
        descriptor_matrix,
        axis=0,
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

    standardized_matrix = (
                                  descriptor_matrix - median
                          ) / scale

    standardized_matrix = np.maximum(
        standardized_matrix,
        0.0,
    )

    standardized_matrix = np.clip(
        standardized_matrix,
        0.0,
        5.0,
    )

    raw_scores = np.sqrt(
        np.mean(
            standardized_matrix ** 2,
            axis=1,
        )
    )

    return {
        client_id: float(score)
        for client_id, score in zip(
            client_ids,
            raw_scores,
            strict=True,
        )
    }


def calibrate_anomaly_scores(
    raw_scores: dict[int, float],
) -> dict[int, float]:
    """
    Apply robust calibration and sigmoid mapping.
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

    standardized_scores = (
        robust_standardize(
            score_array
        )
    )

    return {
        client_id: sigmoid(
            float(standardized_score)
        )
        for client_id, standardized_score
        in zip(
            client_ids,
            standardized_scores,
            strict=True,
        )
    }


def main() -> None:
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    project_dir = Path(
        __file__
    ).resolve().parent

    dataset_dir = (
        project_dir
        / "data"
        / "Planetoid"
    )

    partition_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / (
            f"iid_stratified_"
            f"{NUM_CLIENTS}_clients"
        )
        / "client_node_indices.pt"
    )

    attacked_graph_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "attacks"
        / "structural_clique"
        / (
            f"malicious_client_"
            f"{MALICIOUS_CLIENT_ID}.pt"
        )
    )

    model_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "fedavg_structural_backdoor"
        / "best_backdoored_fedavg_cora.pt"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "client_explanation_descriptors"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "client_explanation_descriptors.csv"
    )

    json_path = (
        output_dir
        / "client_explanation_report.json"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    full_data = dataset[0]

    partition_package = torch.load(
        partition_path,
        map_location="cpu",
        weights_only=False,
    )

    attacked_package = torch.load(
        attacked_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    client_indices = partition_package[
        "client_indices"
    ]

    client_graphs = []

    for client_id in range(
        NUM_CLIENTS
    ):
        if (
            client_id
            == MALICIOUS_CLIENT_ID
        ):
            client_data = attacked_package[
                "data"
            ].clone()
        else:
            clean_client_data = (
                full_data.subgraph(
                    client_indices[
                        client_id
                    ]
                )
            )

            client_data = (
                prepare_clean_client_data(
                    clean_client_data
                )
            )

        client_graphs.append(
            client_data.to(device)
        )

    output_channels = int(
        checkpoint[
            "model_state_dict"
        ][
            "conv2.bias"
        ].numel()
    )

    model = GCN(
        input_channels=(
            dataset.num_features
        ),
        hidden_channels=(
            HIDDEN_CHANNELS
        ),
        output_channels=(
            output_channels
        ),
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print("=" * 84)
    print(
        "Building client-level explanation descriptors"
    )
    print("=" * 84)
    print(f"Device:             {device}")
    print(
        f"Clients:            {NUM_CLIENTS}"
    )
    print(
        f"Risk probes/client: {RISK_PROBE_COUNT}"
    )
    print(
        f"Anchor probes/client: "
        f"{ANCHOR_PROBE_COUNT}"
    )
    print(
        f"Total explanations: "
        f"{NUM_CLIENTS * (RISK_PROBE_COUNT + ANCHOR_PROBE_COUNT)}"
    )
    print(
        "This stage may take several minutes."
    )
    print("-" * 84)

    client_descriptors: dict[
        int,
        dict[str, float],
    ] = {}

    client_reports: dict[
        int,
        dict[str, Any],
    ] = {}

    for client_id, client_data in enumerate(
        client_graphs
    ):
        (
            probe_nodes,
            risk_probes,
            anchor_probes,
            risk_scores,
        ) = select_dual_probe_set(
            data=client_data,
            client_id=client_id,
        )

        print(
            f"Client {client_id:02d} | "
            f"Risk: {risk_probes} | "
            f"Anchors: {anchor_probes}"
        )

        probe_reports: list[
            dict[str, Any]
        ] = []

        for probe_position, probe_node in enumerate(
            probe_nodes,
            start=1,
        ):
            print(
                f"    Explaining probe "
                f"{probe_position}/"
                f"{len(probe_nodes)}: "
                f"node {probe_node}"
            )

            probe_report = explain_and_measure(
                model=model,
                data=client_data,
                node_index=probe_node,
                node_type="probe",
                pair_id=probe_position,
                seed=(
                    SEED
                    + client_id * 100
                    + probe_position
                ),
            )

            probe_report[
                "selection_type"
            ] = (
                "risk"
                if probe_node in risk_probes
                else "anchor"
            )

            probe_report[
                "structural_risk_score"
            ] = float(
                risk_scores[
                    probe_node
                ]
            )

            probe_reports.append(
                probe_report
            )

        descriptor = (
            aggregate_client_reports(
                probe_reports
            )
        )

        client_descriptors[
            client_id
        ] = descriptor

        oracle_poisoned_probes = [
            probe_node
            for probe_node in probe_nodes
            if bool(
                client_data
                .poisoned_node_mask[
                    probe_node
                ]
                .item()
            )
        ]

        client_reports[
            client_id
        ] = {
            "client_id": client_id,
            "state": (
                "malicious"
                if client_id
                == MALICIOUS_CLIENT_ID
                else "benign"
            ),
            "probe_nodes": (
                probe_nodes
            ),
            "risk_probes": (
                risk_probes
            ),
            "anchor_probes": (
                anchor_probes
            ),
            "oracle_poisoned_probes": (
                oracle_poisoned_probes
            ),
            "descriptor": descriptor,
            "probe_reports": (
                probe_reports
            ),
        }

    anomaly_feature_names = [
        "mean_density",
        "mean_average_clustering",
        "mean_transitivity",
        "mean_triangle_count",
        "mean_triangle_density",
        "mean_cycle_rank",
    ]

    raw_anomaly_scores = (
        calculate_raw_anomaly_scores(
            client_descriptors=(
                client_descriptors
            ),
            anomaly_feature_names=(
                anomaly_feature_names
            ),
        )
    )

    calibrated_anomaly_scores = (
        calibrate_anomaly_scores(
            raw_anomaly_scores
        )
    )

    csv_rows: list[
        dict[str, Any]
    ] = []

    for client_id in range(
        NUM_CLIENTS
    ):
        client_reports[
            client_id
        ][
            "raw_explanation_anomaly"
        ] = raw_anomaly_scores[
            client_id
        ]

        client_reports[
            client_id
        ][
            "calibrated_explanation_anomaly"
        ] = calibrated_anomaly_scores[
            client_id
        ]

        csv_rows.append(
            {
                "client_id": client_id,
                "state": client_reports[
                    client_id
                ]["state"],
                "risk_probes": str(
                    client_reports[
                        client_id
                    ]["risk_probes"]
                ),
                "anchor_probes": str(
                    client_reports[
                        client_id
                    ]["anchor_probes"]
                ),
                "oracle_poisoned_probes": str(
                    client_reports[
                        client_id
                    ][
                        "oracle_poisoned_probes"
                    ]
                ),
                "raw_explanation_anomaly": (
                    raw_anomaly_scores[
                        client_id
                    ]
                ),
                "calibrated_explanation_anomaly": (
                    calibrated_anomaly_scores[
                        client_id
                    ]
                ),
                **client_descriptors[
                    client_id
                ],
            }
        )

    fieldnames = list(
        csv_rows[0].keys()
    )

    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    complete_report = {
        "configuration": {
            "seed": SEED,
            "num_clients": (
                NUM_CLIENTS
            ),
            "risk_probe_count": (
                RISK_PROBE_COUNT
            ),
            "anchor_probe_count": (
                ANCHOR_PROBE_COUNT
            ),
            "anomaly_features": (
                anomaly_feature_names
            ),
        },
        "clients": client_reports,
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

    print("\n" + "-" * 84)
    print(
        "CLIENT EXPLANATION ANOMALIES"
    )

    for client_id in range(
        NUM_CLIENTS
    ):
        descriptor = client_descriptors[
            client_id
        ]

        print(
            f"Client {client_id:02d} | "
            f"Raw: "
            f"{raw_anomaly_scores[client_id]:.4f} | "
            f"Calibrated: "
            f"{calibrated_anomaly_scores[client_id]:.4f} | "
            f"Mean clustering: "
            f"{descriptor['mean_average_clustering']:.4f} | "
            f"Mean triangles: "
            f"{descriptor['mean_triangle_density']:.4f} | "
            f"Oracle trigger attribution: "
            f"{descriptor['oracle_mean_trigger_attribution']:.4f}"
        )

        print(
            f"           Oracle poisoned probes: "
            f"{client_reports[client_id]['oracle_poisoned_probes']}"
        )

    highest_anomaly_client = max(
        calibrated_anomaly_scores,
        key=calibrated_anomaly_scores.get,
    )

    print("-" * 84)
    print(
        f"Highest explanation anomaly: "
        f"Client {highest_anomaly_client}"
    )
    print(f"CSV saved to:  {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()