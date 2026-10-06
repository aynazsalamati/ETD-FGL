from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, to_networkx, to_undirected


SEED = 42
NUM_CLIENTS = 5
MALICIOUS_CLIENT_ID = 0
EPSILON = 1e-12


def get_unique_undirected_edges(edge_index: Tensor) -> Tensor:
    """Return each undirected edge only once."""

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    source = edge_index[0]
    target = edge_index[1]

    unique_mask = source < target

    return edge_index[:, unique_mask]


def calculate_homophily(
    data: Data,
    observed_mask: Tensor | np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, int]]:
    """Calculate edge homophily using unique undirected edges between observed nodes."""

    edge_index = get_unique_undirected_edges(
        data.edge_index.cpu()
    )

    total_edges = int(edge_index.size(1))
    if total_edges == 0:
        if return_diagnostics:
            return 0.0, {"total_edges": 0, "homophily_eligible_edges": 0}
        return 0.0

    if observed_mask is None:
        observed_mask = getattr(data, "train_mask", None)

    if observed_mask is None or not hasattr(data, "y") or data.y is None:
        if return_diagnostics:
            return 0.0, {"total_edges": total_edges, "homophily_eligible_edges": 0}
        return 0.0

    if not isinstance(observed_mask, torch.Tensor):
        observed_mask = torch.as_tensor(observed_mask, dtype=torch.bool)
    else:
        observed_mask = observed_mask.cpu().to(torch.bool)

    source = edge_index[0]
    target = edge_index[1]

    mask_len = observed_mask.size(0)
    y_len = data.y.numel()

    valid_nodes = (
        (source < mask_len)
        & (target < mask_len)
        & (source < y_len)
        & (target < y_len)
    )
    eligible_edges = torch.zeros(total_edges, dtype=torch.bool)
    if valid_nodes.any():
        src_v = source[valid_nodes]
        tgt_v = target[valid_nodes]
        eligible_edges[valid_nodes] = observed_mask[src_v] & observed_mask[tgt_v]

    homophily_eligible_edges = int(eligible_edges.sum().item())

    if homophily_eligible_edges == 0:
        homophily = 0.0
        homophily_hits = 0
    else:
        src_eligible = source[eligible_edges]
        tgt_eligible = target[eligible_edges]
        same_label = data.y.cpu()[src_eligible].eq(
            data.y.cpu()[tgt_eligible]
        )
        homophily_hits = int(same_label.sum().item())
        homophily = float(same_label.float().mean().item())

    if return_diagnostics:
        return homophily, {
            "total_edges": total_edges,
            "homophily_eligible_edges": homophily_eligible_edges,
            "homophily_hits": homophily_hits,
            "homophily": homophily,
        }
    return homophily


def calculate_homophily_diagnostics(
    data: Data,
    observed_mask: Tensor | np.ndarray | None = None,
) -> dict[str, int | float]:
    """Expose total_edges and homophily_eligible_edges for verification."""
    _, diag = calculate_homophily(
        data,
        observed_mask=observed_mask,
        return_diagnostics=True,
    )
    return diag


def calculate_motif_features(
    graph: nx.Graph,
) -> dict[str, float]:
    """
    Calculate small-motif statistics.

    The pilot descriptor contains:
    triangles, open wedges, and three-node stars.
    """

    number_of_nodes = graph.number_of_nodes()

    if number_of_nodes == 0:
        return {
            "triangle_frequency": 0.0,
            "wedge_frequency": 0.0,
            "star3_frequency": 0.0,
            "motif_concentration": 0.0,
        }

    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=np.float64,
    )

    triangle_dictionary = nx.triangles(graph)

    triangle_count = (
        sum(triangle_dictionary.values()) / 3.0
    )

    centered_wedge_count = sum(
        math.comb(int(degree), 2)
        for degree in degrees
        if degree >= 2
    )

    open_wedge_count = max(
        0.0,
        centered_wedge_count
        - 3.0 * triangle_count,
    )

    star3_count = sum(
        math.comb(int(degree), 3)
        for degree in degrees
        if degree >= 3
    )

    motif_frequencies = np.array(
        [
            triangle_count / number_of_nodes,
            open_wedge_count / number_of_nodes,
            star3_count / number_of_nodes,
        ],
        dtype=np.float64,
    )

    motif_total = float(motif_frequencies.sum())

    if motif_total <= EPSILON:
        motif_concentration = 0.0
    else:
        motif_concentration = float(
            motif_frequencies.max() / motif_total
        )

    return {
        "triangle_frequency": float(
            motif_frequencies[0]
        ),
        "wedge_frequency": float(
            motif_frequencies[1]
        ),
        "star3_frequency": float(
            motif_frequencies[2]
        ),
        "motif_concentration": motif_concentration,
    }


def extract_topology_descriptor(
    data: Data,
    observed_mask: Tensor | np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> dict[str, float]:
    """Extract the basic topology descriptor of one client."""

    cpu_data = data.cpu()

    graph = to_networkx(
        cpu_data,
        to_undirected=True,
        remove_self_loops=True,
    )

    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    if number_of_nodes <= 1:
        density = 0.0
    else:
        density = (
            2.0 * number_of_edges
            / (
                number_of_nodes
                * (number_of_nodes - 1)
            )
        )

    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=np.float64,
    )

    average_degree = (
        float(degrees.mean())
        if degrees.size > 0
        else 0.0
    )

    if average_degree <= EPSILON:
        normalized_degree_variance = 0.0
    else:
        normalized_degree_variance = float(
            np.mean(
                (
                    (degrees - average_degree)
                    / (average_degree + EPSILON)
                )
                ** 2
            )
        )

    average_clustering = (
        float(nx.average_clustering(graph))
        if number_of_nodes > 0
        else 0.0
    )

    homophily_val, diag = calculate_homophily(
        cpu_data,
        observed_mask=observed_mask,
        return_diagnostics=True,
    )

    descriptor = {
        "nodes": float(number_of_nodes),
        "edges": float(number_of_edges),
        "density": float(density),
        "average_degree": average_degree,
        "normalized_degree_variance": (
            normalized_degree_variance
        ),
        "average_clustering": average_clustering,
        "homophily": homophily_val,
        "homophily_eligible_edges": float(diag["homophily_eligible_edges"]),
    }

    descriptor.update(
        calculate_motif_features(graph)
    )

    if return_diagnostics:
        return descriptor, diag
    return descriptor


def calculate_robust_reference(
    reference_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate median and robust scale.

    Standard deviation is used only when MAD becomes zero.
    """

    median = np.median(
        reference_values,
        axis=0,
    )

    mad = np.median(
        np.abs(reference_values - median),
        axis=0,
    )

    robust_scale = 1.4826 * mad

    standard_deviation = np.std(
        reference_values,
        axis=0,
    )

    robust_scale = np.where(
        robust_scale > 1e-8,
        robust_scale,
        standard_deviation,
    )

    robust_scale = np.where(
        robust_scale > 1e-8,
        robust_scale,
        1.0,
    )

    return median, robust_scale


def calculate_anomaly_score(
    descriptor: dict[str, float],
    reference_descriptors: list[dict[str, float]],
    feature_names: list[str],
) -> tuple[float, dict[str, float]]:
    """Calculate robust topology anomaly score."""

    reference_matrix = np.array(
        [
            [
                item[feature_name]
                for feature_name in feature_names
            ]
            for item in reference_descriptors
        ],
        dtype=np.float64,
    )

    descriptor_vector = np.array(
        [
            descriptor[feature_name]
            for feature_name in feature_names
        ],
        dtype=np.float64,
    )

    median, robust_scale = calculate_robust_reference(
        reference_matrix
    )

    robust_z_scores = (
        descriptor_vector - median
    ) / robust_scale

    clipped_z_scores = np.clip(
        robust_z_scores,
        -5.0,
        5.0,
    )

    anomaly_score = float(
        np.sqrt(
            np.mean(clipped_z_scores ** 2)
        )
    )

    feature_deviations = {
        feature_name: float(abs(z_score))
        for feature_name, z_score in zip(
            feature_names,
            clipped_z_scores,
            strict=True,
        )
    }

    return anomaly_score, feature_deviations


def main() -> None:
    project_dir = Path(__file__).resolve().parent

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
        / f"iid_stratified_{NUM_CLIENTS}_clients"
        / "client_node_indices.pt"
    )

    attacked_graph_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "attacks"
        / "structural_clique"
        / f"malicious_client_{MALICIOUS_CLIENT_ID}.pt"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "topology_descriptor"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "topology_descriptors.csv"
    json_path = output_dir / "topology_anomaly_report.json"

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    full_data = dataset[0]

    partition_data = torch.load(
        partition_path,
        map_location="cpu",
        weights_only=False,
    )

    attacked_package = torch.load(
        attacked_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    client_indices = partition_data[
        "client_indices"
    ]

    clean_descriptors: dict[int, dict[str, float]] = {}

    for client_id in range(NUM_CLIENTS):
        client_data = full_data.subgraph(
            client_indices[client_id]
        )

        clean_descriptors[client_id] = (
            extract_topology_descriptor(
                client_data
            )
        )

    attacked_descriptor = extract_topology_descriptor(
        attacked_package["data"]
    )

    feature_names = [
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

    benign_reference = [
        clean_descriptors[client_id]
        for client_id in range(NUM_CLIENTS)
        if client_id != MALICIOUS_CLIENT_ID
    ]

    clean_client_score, clean_deviations = (
        calculate_anomaly_score(
            descriptor=clean_descriptors[
                MALICIOUS_CLIENT_ID
            ],
            reference_descriptors=benign_reference,
            feature_names=feature_names,
        )
    )

    attacked_client_score, attacked_deviations = (
        calculate_anomaly_score(
            descriptor=attacked_descriptor,
            reference_descriptors=benign_reference,
            feature_names=feature_names,
        )
    )

    rows: list[dict[str, float | str | int]] = []

    for client_id, descriptor in clean_descriptors.items():
        rows.append(
            {
                "client": client_id,
                "state": "clean",
                **descriptor,
            }
        )

    rows.append(
        {
            "client": MALICIOUS_CLIENT_ID,
            "state": "attacked",
            **attacked_descriptor,
        }
    )

    fieldnames = [
        "client",
        "state",
        *list(attacked_descriptor.keys()),
    ]

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
        writer.writerows(rows)

    report = {
        "clean_client_0_anomaly_score": (
            clean_client_score
        ),
        "attacked_client_0_anomaly_score": (
            attacked_client_score
        ),
        "clean_client_0_deviations": (
            clean_deviations
        ),
        "attacked_client_0_deviations": (
            attacked_deviations
        ),
    }

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            report,
            json_file,
            indent=2,
        )

    print("=" * 76)
    print("Topology descriptor comparison")
    print("=" * 76)

    for client_id in range(NUM_CLIENTS):
        descriptor = clean_descriptors[client_id]

        print(
            f"Client {client_id:02d} CLEAN    | "
            f"Density: {descriptor['density']:.6f} | "
            f"Degree: {descriptor['average_degree']:.4f} | "
            f"Cluster: {descriptor['average_clustering']:.4f} | "
            f"Triangles: {descriptor['triangle_frequency']:.4f}"
        )

    print("-" * 76)

    print(
        f"Client {MALICIOUS_CLIENT_ID:02d} ATTACKED | "
        f"Density: {attacked_descriptor['density']:.6f} | "
        f"Degree: {attacked_descriptor['average_degree']:.4f} | "
        f"Cluster: {attacked_descriptor['average_clustering']:.4f} | "
        f"Triangles: {attacked_descriptor['triangle_frequency']:.4f}"
    )

    print("-" * 76)

    print(
        f"Clean client 0 anomaly score:    "
        f"{clean_client_score:.4f}"
    )

    print(
        f"Attacked client 0 anomaly score: "
        f"{attacked_client_score:.4f}"
    )

    print("\nLargest attacked-client deviations:")

    sorted_deviations = sorted(
        attacked_deviations.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature_name, deviation in sorted_deviations:
        print(
            f"  {feature_name:30s}: "
            f"{deviation:.4f}"
        )

    print("-" * 76)
    print(f"Descriptors saved to: {csv_path}")
    print(f"Report saved to:      {json_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()