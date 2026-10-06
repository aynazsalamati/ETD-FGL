from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


SEED = 42
NUM_CLIENTS = 5


def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def distribute_indices(
    indices: Tensor,
    client_parts: list[list[Tensor]],
    generator: torch.Generator,
) -> None:
    """Shuffle and distribute one group of nodes among clients."""

    if indices.numel() == 0:
        return

    permutation = torch.randperm(
        indices.numel(),
        generator=generator,
    )

    shuffled_indices = indices[permutation]
    chunks = torch.tensor_split(shuffled_indices, NUM_CLIENTS)

    for client_id, chunk in enumerate(chunks):
        if chunk.numel() > 0:
            client_parts[client_id].append(chunk)


def create_stratified_partition(
    data,
    num_classes: int,
    seed: int,
) -> dict[int, Tensor]:
    """
    Create a label- and split-aware IID node partition.

    Training, validation, test, and unused nodes are distributed
    separately to preserve similar data proportions across clients.
    """

    generator = torch.Generator().manual_seed(seed)

    client_parts: list[list[Tensor]] = [
        [] for _ in range(NUM_CLIENTS)
    ]

    used_mask = (
        data.train_mask
        | data.val_mask
        | data.test_mask
    )

    split_masks = {
        "train": data.train_mask,
        "validation": data.val_mask,
        "test": data.test_mask,
        "unused": ~used_mask,
    }

    for split_mask in split_masks.values():
        for class_id in range(num_classes):
            group_mask = (
                split_mask
                & data.y.eq(class_id)
            )

            group_indices = group_mask.nonzero(
                as_tuple=False
            ).view(-1)

            distribute_indices(
                indices=group_indices,
                client_parts=client_parts,
                generator=generator,
            )

    client_indices: dict[int, Tensor] = {}

    for client_id, parts in enumerate(client_parts):
        if not parts:
            raise RuntimeError(
                f"Client {client_id} received no nodes."
            )

        node_indices = torch.cat(parts)

        local_permutation = torch.randperm(
            node_indices.numel(),
            generator=generator,
        )

        client_indices[client_id] = (
            node_indices[local_permutation]
        )

    return client_indices


def validate_partition(
    client_indices: dict[int, Tensor],
    num_nodes: int,
) -> None:
    """Verify that every node belongs to exactly one client."""

    combined_nodes = torch.cat(
        list(client_indices.values())
    )

    unique_nodes = torch.unique(combined_nodes)

    assert combined_nodes.numel() == num_nodes, (
        "The partition does not cover all graph nodes."
    )

    assert unique_nodes.numel() == num_nodes, (
        "At least one node was assigned to multiple clients."
    )

    assert int(unique_nodes.min()) == 0
    assert int(unique_nodes.max()) == num_nodes - 1


def main() -> None:
    set_seed(SEED)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "Planetoid"

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / f"iid_stratified_{NUM_CLIENTS}_clients"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("Creating federated IID partition for Cora")
    print("=" * 72)
    print(f"Seed: {SEED}")
    print(f"Number of clients: {NUM_CLIENTS}")

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    data = dataset[0]

    # Preserve original node identities inside local subgraphs.
    data.global_node_id = torch.arange(data.num_nodes)

    client_indices = create_stratified_partition(
        data=data,
        num_classes=dataset.num_classes,
        seed=SEED,
    )

    validate_partition(
        client_indices=client_indices,
        num_nodes=data.num_nodes,
    )

    partition_path = output_dir / "client_node_indices.pt"
    summary_path = output_dir / "partition_summary.json"

    torch.save(
        {
            "seed": SEED,
            "num_clients": NUM_CLIENTS,
            "client_indices": client_indices,
        },
        partition_path,
    )

    total_retained_edges = 0
    client_summaries: list[dict] = []

    print("-" * 72)

    for client_id in range(NUM_CLIENTS):
        node_indices = client_indices[client_id]

        # Create an induced local graph for this client.
        client_data = data.subgraph(node_indices)

        total_retained_edges += client_data.num_edges

        class_counts = torch.bincount(
            client_data.y,
            minlength=dataset.num_classes,
        )

        client_summary = {
            "client_id": client_id,
            "nodes": int(client_data.num_nodes),
            "edges": int(client_data.num_edges),
            "train_nodes": int(
                client_data.train_mask.sum()
            ),
            "validation_nodes": int(
                client_data.val_mask.sum()
            ),
            "test_nodes": int(
                client_data.test_mask.sum()
            ),
            "class_counts": [
                int(value)
                for value in class_counts.tolist()
            ],
        }

        client_summaries.append(client_summary)

        print(
            f"Client {client_id:02d} | "
            f"Nodes: {client_data.num_nodes:4d} | "
            f"Edges: {client_data.num_edges:4d} | "
            f"Train: {int(client_data.train_mask.sum()):3d} | "
            f"Val: {int(client_data.val_mask.sum()):3d} | "
            f"Test: {int(client_data.test_mask.sum()):3d}"
        )

        print(
            f"           Class distribution: "
            f"{class_counts.tolist()}"
        )

    retained_edge_ratio = (
        total_retained_edges / data.num_edges
    )

    summary = {
        "dataset": "Cora",
        "partition_type": "IID stratified",
        "seed": SEED,
        "num_clients": NUM_CLIENTS,
        "original_nodes": int(data.num_nodes),
        "original_edges": int(data.num_edges),
        "retained_local_edges": int(total_retained_edges),
        "retained_edge_ratio": float(retained_edge_ratio),
        "clients": client_summaries,
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
        )

    print("-" * 72)
    print("Partition validation passed.")
    print(
        f"Covered nodes: "
        f"{sum(item['nodes'] for item in client_summaries)}"
    )
    print(
        f"Retained local edges: "
        f"{total_retained_edges}/{data.num_edges}"
    )
    print(
        f"Retained edge ratio: "
        f"{retained_edge_ratio:.4f}"
    )
    print(f"Indices saved to: {partition_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()