from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils import to_networkx

from build_client_explanation_descriptors import (
    calculate_structural_risk_scores,
    get_probe_candidates,
)


MALICIOUS_CLIENT_ID = 0
TOP_N = 20


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    attacked_graph_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "attacks"
        / "structural_clique"
        / f"malicious_client_{MALICIOUS_CLIENT_ID}.pt"
    )

    package = torch.load(
        attacked_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    data = package["data"]
    metadata = package["metadata"]

    candidate_nodes = get_probe_candidates(data)

    risk_scores = calculate_structural_risk_scores(
        data=data,
        candidate_nodes=candidate_nodes,
    )

    graph = to_networkx(
        data.clone().cpu(),
        to_undirected=True,
        remove_self_loops=True,
    )

    triangle_counts = nx.triangles(graph)
    clustering_values = nx.clustering(graph)
    degree_values = dict(graph.degree())

    poisoned_nodes = {
        int(node)
        for node in metadata["poisoned_victim_nodes"]
    }

    ranked_nodes = sorted(
        risk_scores.items(),
        key=lambda item: (
            item[1],
            -item[0],
        ),
        reverse=True,
    )

    print("=" * 100)
    print("Structural-risk ranking for malicious client")
    print("=" * 100)

    print(
        f"{'Rank':>4} | "
        f"{'Node':>4} | "
        f"{'Risk':>9} | "
        f"{'Degree':>6} | "
        f"{'Triangles':>9} | "
        f"{'Cluster':>8} | "
        f"{'Clean Y':>7} | "
        f"{'Train Y':>7} | "
        f"State"
    )

    print("-" * 100)

    for rank, (node, score) in enumerate(
        ranked_nodes[:TOP_N],
        start=1,
    ):
        state = (
            "POISONED"
            if node in poisoned_nodes
            else "clean"
        )

        clean_label = int(
            data.clean_y[node].item()
        )

        training_label = int(
            data.y[node].item()
        )

        print(
            f"{rank:4d} | "
            f"{node:4d} | "
            f"{score:9.4f} | "
            f"{degree_values.get(node, 0):6d} | "
            f"{triangle_counts.get(node, 0):9d} | "
            f"{clustering_values.get(node, 0.0):8.4f} | "
            f"{clean_label:7d} | "
            f"{training_label:7d} | "
            f"{state}"
        )

    rank_lookup = {
        node: rank
        for rank, (node, _) in enumerate(
            ranked_nodes,
            start=1,
        )
    }

    print("-" * 100)
    print("Poisoned-node ranking positions:")

    for poisoned_node in metadata[
        "poisoned_victim_nodes"
    ]:
        poisoned_node = int(poisoned_node)

        print(
            f"Node {poisoned_node:3d} | "
            f"Rank: {rank_lookup[poisoned_node]:2d} | "
            f"Risk: {risk_scores[poisoned_node]:.4f} | "
            f"Degree: {degree_values.get(poisoned_node, 0)} | "
            f"Triangles: {triangle_counts.get(poisoned_node, 0)} | "
            f"Clustering: "
            f"{clustering_values.get(poisoned_node, 0.0):.4f}"
        )

    first_poisoned_rank = min(
        rank_lookup[node]
        for node in poisoned_nodes
    )

    poisoned_in_top_5 = sum(
        node in poisoned_nodes
        for node, _ in ranked_nodes[:5]
    )

    poisoned_in_top_10 = sum(
        node in poisoned_nodes
        for node, _ in ranked_nodes[:10]
    )

    print("-" * 100)
    print(
        f"First poisoned-node rank: {first_poisoned_rank}"
    )
    print(
        f"Poisoned nodes inside top 5:  "
        f"{poisoned_in_top_5}/5"
    )
    print(
        f"Poisoned nodes inside top 10: "
        f"{poisoned_in_top_10}/5"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()