from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import degree

from test_explanation_module import (
    calculate_trigger_edge_ratio,
)
from test_explanation_module_v2 import (
    build_explainer,
    get_predictions,
)
from train_fedavg_cora import GCN


SEED = 42
MALICIOUS_CLIENT_ID = 0

HIDDEN_CHANNELS = 16
DROPOUT = 0.5

TOP_K_UNDIRECTED_EDGES = 10
MIN_EDGE_IMPORTANCE = 1e-6
EPSILON = 1e-12


ANOMALY_FEATURES = [
    "density",
    "average_clustering",
    "transitivity",
    "triangle_density",
    "cycle_rank",
    "edge_score_concentration",
    "edge_score_entropy",
]


def aggregate_undirected_edges(
    edge_index: Tensor,
    edge_mask: Tensor,
) -> list[tuple[int, int, float]]:
    """
    Convert directed PyG edges into unique undirected edges.

    When both directions of an edge exist, the maximum attribution
    score is preserved.
    """

    edge_index_cpu = edge_index.detach().cpu()
    edge_scores_cpu = edge_mask.detach().abs().cpu()

    pair_scores: dict[
        tuple[int, int],
        float,
    ] = {}

    for edge_id in range(
        edge_index_cpu.size(1)
    ):
        source = int(
            edge_index_cpu[
                0,
                edge_id,
            ].item()
        )

        target = int(
            edge_index_cpu[
                1,
                edge_id,
            ].item()
        )

        if source == target:
            continue

        first_node = min(source, target)
        second_node = max(source, target)

        pair = (
            first_node,
            second_node,
        )

        score = float(
            edge_scores_cpu[
                edge_id
            ].item()
        )

        previous_score = pair_scores.get(
            pair,
            0.0,
        )

        if score > previous_score:
            pair_scores[pair] = score

    undirected_edges = [
        (
            first_node,
            second_node,
            score,
        )
        for (
            first_node,
            second_node,
        ), score in pair_scores.items()
    ]

    undirected_edges.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return undirected_edges


def calculate_edge_score_entropy(
    scores: np.ndarray,
) -> float:
    """Calculate normalized entropy of selected edge scores."""

    if scores.size <= 1:
        return 0.0

    total_score = float(scores.sum())

    if total_score <= EPSILON:
        return 0.0

    probabilities = (
        scores / total_score
    )

    entropy = -float(
        np.sum(
            probabilities
            * np.log(
                probabilities + EPSILON
            )
        )
    )

    maximum_entropy = math.log(
        scores.size
    )

    if maximum_entropy <= EPSILON:
        return 0.0

    return entropy / maximum_entropy


def extract_explanation_topology(
    node_index: int,
    edge_index: Tensor,
    edge_mask: Tensor,
) -> dict[str, float | int]:
    """
    Build a top-k explanation graph and extract its topology.
    """

    undirected_edges = (
        aggregate_undirected_edges(
            edge_index=edge_index,
            edge_mask=edge_mask,
        )
    )

    positive_edges = [
        edge
        for edge in undirected_edges
        if edge[2] > MIN_EDGE_IMPORTANCE
    ]

    selected_edges = positive_edges[
        :TOP_K_UNDIRECTED_EDGES
    ]

    graph = nx.Graph()
    graph.add_node(node_index)

    for (
        source,
        target,
        score,
    ) in selected_edges:
        graph.add_edge(
            source,
            target,
            weight=score,
        )

    number_of_nodes = (
        graph.number_of_nodes()
    )

    number_of_edges = (
        graph.number_of_edges()
    )

    if number_of_nodes > 1:
        density = float(
            nx.density(graph)
        )
    else:
        density = 0.0

    degrees = np.array(
        [
            graph_degree
            for _, graph_degree
            in graph.degree()
        ],
        dtype=np.float64,
    )

    average_degree = (
        float(degrees.mean())
        if degrees.size > 0
        else 0.0
    )

    average_clustering = (
        float(
            nx.average_clustering(
                graph
            )
        )
        if number_of_nodes > 0
        else 0.0
    )

    transitivity = (
        float(
            nx.transitivity(graph)
        )
        if number_of_nodes >= 3
        else 0.0
    )

    triangle_count = float(
        sum(
            nx.triangles(
                graph
            ).values()
        )
        / 3.0
    )

    if number_of_nodes >= 3:
        possible_triangles = math.comb(
            number_of_nodes,
            3,
        )
    else:
        possible_triangles = 0

    if possible_triangles > 0:
        triangle_density = (
            triangle_count
            / possible_triangles
        )
    else:
        triangle_density = 0.0

    component_count = (
        nx.number_connected_components(
            graph
        )
        if number_of_nodes > 0
        else 0
    )

    cycle_rank = max(
        0,
        number_of_edges
        - number_of_nodes
        + component_count,
    )

    selected_scores = np.array(
        [
            score
            for _, _, score
            in selected_edges
        ],
        dtype=np.float64,
    )

    total_edge_score = (
        float(
            selected_scores.sum()
        )
        if selected_scores.size > 0
        else 0.0
    )

    if total_edge_score <= EPSILON:
        edge_score_concentration = 0.0
    else:
        concentration_count = min(
            3,
            selected_scores.size,
        )

        edge_score_concentration = float(
            selected_scores[
                :concentration_count
            ].sum()
            / total_edge_score
        )

    edge_score_entropy = (
        calculate_edge_score_entropy(
            selected_scores
        )
    )

    return {
        "explanation_nodes": (
            number_of_nodes
        ),
        "explanation_edges": (
            number_of_edges
        ),
        "connected_components": (
            component_count
        ),
        "density": density,
        "average_degree": (
            average_degree
        ),
        "average_clustering": (
            average_clustering
        ),
        "transitivity": transitivity,
        "triangle_count": (
            triangle_count
        ),
        "triangle_density": (
            triangle_density
        ),
        "cycle_rank": cycle_rank,
        "total_selected_edge_score": (
            total_edge_score
        ),
        "edge_score_concentration": (
            edge_score_concentration
        ),
        "edge_score_entropy": (
            edge_score_entropy
        ),
    }


@torch.no_grad()
def select_matched_clean_nodes(
    model: GCN,
    data,
    poisoned_nodes: list[int],
) -> list[int]:
    """
    Match every poisoned node with a clean, correctly classified
    node from the same true class and with similar degree.
    """

    predictions, probabilities = (
        get_predictions(
            model=model,
            data=data,
        )
    )

    node_degrees = degree(
        data.edge_index[0],
        num_nodes=data.num_nodes,
        dtype=torch.float,
    )

    original_clean_mask = (
        data.clean_y.ge(0)
        & ~data.poisoned_node_mask
        & ~data.trigger_node_mask
    )

    correctly_classified_mask = (
        predictions.eq(
            data.clean_y
        )
    )

    connected_mask = (
        node_degrees.gt(0)
    )

    valid_clean_mask = (
        original_clean_mask
        & correctly_classified_mask
        & connected_mask
    )

    selected_clean_nodes: list[int] = []
    used_nodes: set[int] = set()

    for poisoned_node in poisoned_nodes:
        poisoned_true_label = int(
            data.clean_y[
                poisoned_node
            ].item()
        )

        poisoned_degree = float(
            node_degrees[
                poisoned_node
            ].item()
        )

        same_class_mask = (
            valid_clean_mask
            & data.clean_y.eq(
                poisoned_true_label
            )
        )

        candidates = same_class_mask.nonzero(
            as_tuple=False
        ).view(-1)

        available_candidates = [
            int(candidate.item())
            for candidate in candidates
            if int(candidate.item())
            not in used_nodes
        ]

        if not available_candidates:
            fallback_candidates = (
                valid_clean_mask.nonzero(
                    as_tuple=False
                ).view(-1)
            )

            available_candidates = [
                int(candidate.item())
                for candidate
                in fallback_candidates
                if int(candidate.item())
                not in used_nodes
            ]

        if not available_candidates:
            raise RuntimeError(
                "Not enough clean comparison nodes "
                "were found."
            )

        best_candidate = min(
            available_candidates,
            key=lambda candidate: (
                abs(
                    float(
                        node_degrees[
                            candidate
                        ].item()
                    )
                    - poisoned_degree
                ),
                -float(
                    probabilities[
                        candidate,
                        predictions[
                            candidate
                        ],
                    ].item()
                ),
            ),
        )

        selected_clean_nodes.append(
            best_candidate
        )

        used_nodes.add(
            best_candidate
        )

    return selected_clean_nodes


def explain_and_measure(
    model: GCN,
    data,
    node_index: int,
    node_type: str,
    pair_id: int,
    seed: int,
) -> dict[str, Any]:
    """Explain one node and measure explanation topology."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    predictions, probabilities = (
        get_predictions(
            model=model,
            data=data,
        )
    )

    predicted_class = int(
        predictions[
            node_index
        ].item()
    )

    confidence = float(
        probabilities[
            node_index,
            predicted_class,
        ].item()
    )

    node_degrees = degree(
        data.edge_index[0],
        num_nodes=data.num_nodes,
        dtype=torch.float,
    )

    explainer = build_explainer(
        model
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_index,
    )

    if explanation.edge_mask is None:
        raise RuntimeError(
            "No edge mask was generated."
        )

    topology = (
        extract_explanation_topology(
            node_index=node_index,
            edge_index=data.edge_index,
            edge_mask=(
                explanation.edge_mask
            ),
        )
    )

    trigger_attribution = (
        calculate_trigger_edge_ratio(
            edge_index=data.edge_index,
            edge_mask=(
                explanation.edge_mask
            ),
            trigger_node_mask=(
                data.trigger_node_mask
            ),
        )
    )

    return {
        "pair_id": pair_id,
        "node_type": node_type,
        "node_index": node_index,
        "node_degree": float(
            node_degrees[
                node_index
            ].item()
        ),
        "true_label": int(
            data.clean_y[
                node_index
            ].item()
        ),
        "training_label": int(
            data.y[
                node_index
            ].item()
        ),
        "predicted_class": (
            predicted_class
        ),
        "prediction_confidence": (
            confidence
        ),
        "oracle_trigger_attribution": (
            trigger_attribution
        ),
        **topology,
    }


def calculate_reference_statistics(
    reports: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate robust clean-reference statistics."""

    matrix = np.array(
        [
            [
                float(
                    report[
                        feature_name
                    ]
                )
                for feature_name
                in ANOMALY_FEATURES
            ]
            for report in reports
        ],
        dtype=np.float64,
    )

    median = np.median(
        matrix,
        axis=0,
    )

    mad = np.median(
        np.abs(
            matrix - median
        ),
        axis=0,
    )

    scale = 1.4826 * mad

    standard_deviation = np.std(
        matrix,
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

    return median, scale


def add_anomaly_scores(
    poisoned_reports: list[dict[str, Any]],
    clean_reports: list[dict[str, Any]],
) -> None:
    """
    Add robust explanation-topology anomaly scores.

    Clean reports use leave-one-out references.
    """

    clean_median, clean_scale = (
        calculate_reference_statistics(
            clean_reports
        )
    )

    for report in poisoned_reports:
        vector = np.array(
            [
                float(
                    report[
                        feature_name
                    ]
                )
                for feature_name
                in ANOMALY_FEATURES
            ],
            dtype=np.float64,
        )

        z_scores = (
            vector - clean_median
        ) / clean_scale

        z_scores = np.clip(
            z_scores,
            -5.0,
            5.0,
        )

        report[
            "explanation_anomaly_score"
        ] = float(
            np.sqrt(
                np.mean(
                    z_scores ** 2
                )
            )
        )

    for clean_report in clean_reports:
        reference_reports = [
            report
            for report in clean_reports
            if report["node_index"]
            != clean_report["node_index"]
        ]

        median, scale = (
            calculate_reference_statistics(
                reference_reports
            )
        )

        vector = np.array(
            [
                float(
                    clean_report[
                        feature_name
                    ]
                )
                for feature_name
                in ANOMALY_FEATURES
            ],
            dtype=np.float64,
        )

        z_scores = (
            vector - median
        ) / scale

        z_scores = np.clip(
            z_scores,
            -5.0,
            5.0,
        )

        clean_report[
            "explanation_anomaly_score"
        ] = float(
            np.sqrt(
                np.mean(
                    z_scores ** 2
                )
            )
        )


def summarize_group(
    reports: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate group means for key metrics."""

    metric_names = [
        "density",
        "average_clustering",
        "transitivity",
        "triangle_count",
        "triangle_density",
        "cycle_rank",
        "edge_score_concentration",
        "edge_score_entropy",
        "explanation_anomaly_score",
    ]

    return {
        metric_name: float(
            np.mean(
                [
                    float(
                        report[
                            metric_name
                        ]
                    )
                    for report in reports
                ]
            )
        )
        for metric_name in metric_names
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
        / "explanation_subgraph_topology"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "explanation_topology_reports.csv"
    )

    json_path = (
        output_dir
        / "explanation_topology_summary.json"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    attacked_package = torch.load(
        attacked_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    data = attacked_package[
        "data"
    ].to(device)

    metadata = attacked_package[
        "metadata"
    ]

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    output_channels = int(
        checkpoint[
            "model_state_dict"
        ][
            "conv2.bias"
        ].numel()
    )

    model = GCN(
        input_channels=int(
            data.x.size(1)
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

    poisoned_nodes = [
        int(node_index)
        for node_index
        in metadata[
            "poisoned_victim_nodes"
        ]
    ]

    clean_nodes = (
        select_matched_clean_nodes(
            model=model,
            data=data,
            poisoned_nodes=(
                poisoned_nodes
            ),
        )
    )

    print("=" * 80)
    print(
        "Explanation-subgraph topology comparison"
    )
    print("=" * 80)
    print(f"Device: {device}")
    print(
        f"Poisoned probes: {poisoned_nodes}"
    )
    print(
        f"Matched clean probes: {clean_nodes}"
    )
    print(
        "Generating ten explanations. "
        "This may take a few minutes."
    )
    print("-" * 80)

    poisoned_reports: list[
        dict[str, Any]
    ] = []

    clean_reports: list[
        dict[str, Any]
    ] = []

    for pair_id, (
        poisoned_node,
        clean_node,
    ) in enumerate(
        zip(
            poisoned_nodes,
            clean_nodes,
            strict=True,
        ),
        start=1,
    ):
        print(
            f"Pair {pair_id}: "
            f"explaining poisoned node "
            f"{poisoned_node}..."
        )

        poisoned_report = (
            explain_and_measure(
                model=model,
                data=data,
                node_index=(
                    poisoned_node
                ),
                node_type="poisoned",
                pair_id=pair_id,
                seed=(
                    SEED
                    + pair_id * 2
                ),
            )
        )

        print(
            f"Pair {pair_id}: "
            f"explaining clean node "
            f"{clean_node}..."
        )

        clean_report = (
            explain_and_measure(
                model=model,
                data=data,
                node_index=clean_node,
                node_type="clean",
                pair_id=pair_id,
                seed=(
                    SEED
                    + pair_id * 2
                    + 1
                ),
            )
        )

        poisoned_reports.append(
            poisoned_report
        )

        clean_reports.append(
            clean_report
        )

    add_anomaly_scores(
        poisoned_reports=(
            poisoned_reports
        ),
        clean_reports=clean_reports,
    )

    all_reports = [
        *poisoned_reports,
        *clean_reports,
    ]

    fieldnames = list(
        all_reports[0].keys()
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
            all_reports
        )

    poisoned_summary = (
        summarize_group(
            poisoned_reports
        )
    )

    clean_summary = (
        summarize_group(
            clean_reports
        )
    )

    summary = {
        "configuration": {
            "seed": SEED,
            "top_k_undirected_edges": (
                TOP_K_UNDIRECTED_EDGES
            ),
            "anomaly_features": (
                ANOMALY_FEATURES
            ),
        },
        "poisoned_nodes": (
            poisoned_nodes
        ),
        "matched_clean_nodes": (
            clean_nodes
        ),
        "poisoned_summary": (
            poisoned_summary
        ),
        "clean_summary": (
            clean_summary
        ),
        "reports": all_reports,
    }

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
        )

    print("\n" + "-" * 80)

    for pair_id in range(
        1,
        len(poisoned_nodes) + 1,
    ):
        poisoned_report = (
            poisoned_reports[
                pair_id - 1
            ]
        )

        clean_report = (
            clean_reports[
                pair_id - 1
            ]
        )

        print(
            f"PAIR {pair_id}"
        )

        print(
            f"  POISONED {poisoned_report['node_index']:3d} | "
            f"Clustering: "
            f"{poisoned_report['average_clustering']:.4f} | "
            f"Transitivity: "
            f"{poisoned_report['transitivity']:.4f} | "
            f"Triangles: "
            f"{poisoned_report['triangle_count']:.0f} | "
            f"Anomaly: "
            f"{poisoned_report['explanation_anomaly_score']:.4f} | "
            f"Oracle trigger: "
            f"{poisoned_report['oracle_trigger_attribution']:.4f}"
        )

        print(
            f"  CLEAN    {clean_report['node_index']:3d} | "
            f"Clustering: "
            f"{clean_report['average_clustering']:.4f} | "
            f"Transitivity: "
            f"{clean_report['transitivity']:.4f} | "
            f"Triangles: "
            f"{clean_report['triangle_count']:.0f} | "
            f"Anomaly: "
            f"{clean_report['explanation_anomaly_score']:.4f} | "
            f"Oracle trigger: "
            f"{clean_report['oracle_trigger_attribution']:.4f}"
        )

    print("-" * 80)
    print("GROUP MEANS")

    print(
        f"Poisoned clustering:   "
        f"{poisoned_summary['average_clustering']:.4f}"
    )

    print(
        f"Clean clustering:      "
        f"{clean_summary['average_clustering']:.4f}"
    )

    print(
        f"Poisoned transitivity: "
        f"{poisoned_summary['transitivity']:.4f}"
    )

    print(
        f"Clean transitivity:    "
        f"{clean_summary['transitivity']:.4f}"
    )

    print(
        f"Poisoned triangles:    "
        f"{poisoned_summary['triangle_count']:.4f}"
    )

    print(
        f"Clean triangles:       "
        f"{clean_summary['triangle_count']:.4f}"
    )

    print(
        f"Poisoned anomaly:      "
        f"{poisoned_summary['explanation_anomaly_score']:.4f}"
    )

    print(
        f"Clean anomaly:         "
        f"{clean_summary['explanation_anomaly_score']:.4f}"
    )

    print("-" * 80)
    print(f"CSV saved to:  {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()