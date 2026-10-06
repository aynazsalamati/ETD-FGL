from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


SEED = 42
NUM_CLIENTS = 5

MALICIOUS_CLIENT_ID = 0
TARGET_CLASS = 0
POISON_RATE = 0.20
TRIGGER_SIZE = 3


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_feature_vector(vector: Tensor) -> Tensor:
    """Normalize a feature vector using its L1 norm."""

    denominator = vector.abs().sum()

    if float(denominator) <= 1e-12:
        vector = torch.ones_like(vector)
        denominator = vector.sum()

    return vector / denominator


def extend_boolean_mask(
    mask: Tensor,
    additional_nodes: int,
) -> Tensor:
    """Append False values for newly created trigger nodes."""

    extension = torch.zeros(
        additional_nodes,
        dtype=torch.bool,
        device=mask.device,
    )

    return torch.cat(
        [mask.clone(), extension],
        dim=0,
    )


def inject_clique_trigger(
    data: Data,
    target_class: int,
    poison_rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[Data, dict[str, Any]]:
    """
    Inject a clique-shaped structural trigger into training nodes.

    Only non-target training nodes are eligible for poisoning.
    Each poisoned node receives its own trigger clique.
    """

    if trigger_size < 2:
        raise ValueError(
            "Trigger size must be at least 2."
        )

    if not 0.0 < poison_rate <= 1.0:
        raise ValueError(
            "Poison rate must be in the interval (0, 1]."
        )

    generator = torch.Generator().manual_seed(seed)

    original_num_nodes = int(data.num_nodes)
    original_num_edges = int(data.num_edges)

    candidate_mask = (
        data.train_mask
        & data.y.ne(target_class)
    )

    candidate_nodes = candidate_mask.nonzero(
        as_tuple=False
    ).view(-1)

    if candidate_nodes.numel() == 0:
        raise RuntimeError(
            "No eligible non-target training nodes "
            "were found for poisoning."
        )

    poison_count = max(
        1,
        math.ceil(
            poison_rate
            * candidate_nodes.numel()
        ),
    )

    poison_count = min(
        poison_count,
        candidate_nodes.numel(),
    )

    permutation = torch.randperm(
        candidate_nodes.numel(),
        generator=generator,
    )

    poisoned_victim_nodes = candidate_nodes[
        permutation[:poison_count]
    ]

    new_trigger_node_count = (
        poison_count * trigger_size
    )

    # Use a benign-looking local feature prototype.
    feature_prototype = normalize_feature_vector(
        data.x.mean(dim=0)
    )

    trigger_features = feature_prototype.repeat(
        new_trigger_node_count,
        1,
    )

    # Trigger nodes are not directly used as labeled train nodes.
    trigger_labels = torch.full(
        (new_trigger_node_count,),
        fill_value=target_class,
        dtype=data.y.dtype,
        device=data.y.device,
    )

    attacked_data = data.clone()

    attacked_data.x = torch.cat(
        [
            data.x.clone(),
            trigger_features,
        ],
        dim=0,
    )

    attacked_data.y = torch.cat(
        [
            data.y.clone(),
            trigger_labels,
        ],
        dim=0,
    )

    # Preserve original labels for later evaluation.
    clean_y = torch.full(
        (
            original_num_nodes
            + new_trigger_node_count,
        ),
        fill_value=-1,
        dtype=data.y.dtype,
        device=data.y.device,
    )

    clean_y[:original_num_nodes] = data.y.clone()

    attacked_data.clean_y = clean_y

    # Poisoned training victims receive the attacker target label.
    attacked_data.y[
        poisoned_victim_nodes
    ] = target_class

    attacked_data.train_mask = extend_boolean_mask(
        data.train_mask,
        new_trigger_node_count,
    )

    attacked_data.val_mask = extend_boolean_mask(
        data.val_mask,
        new_trigger_node_count,
    )

    attacked_data.test_mask = extend_boolean_mask(
        data.test_mask,
        new_trigger_node_count,
    )

    poisoned_node_mask = torch.zeros(
        original_num_nodes
        + new_trigger_node_count,
        dtype=torch.bool,
        device=data.y.device,
    )

    poisoned_node_mask[
        poisoned_victim_nodes
    ] = True

    trigger_node_mask = torch.zeros(
        original_num_nodes
        + new_trigger_node_count,
        dtype=torch.bool,
        device=data.y.device,
    )

    trigger_node_mask[
        original_num_nodes:
    ] = True

    attacked_data.poisoned_node_mask = (
        poisoned_node_mask
    )

    attacked_data.trigger_node_mask = (
        trigger_node_mask
    )

    # Preserve original global node IDs.
    if hasattr(data, "global_node_id"):
        trigger_global_ids = torch.full(
            (new_trigger_node_count,),
            fill_value=-1,
            dtype=data.global_node_id.dtype,
            device=data.global_node_id.device,
        )

        attacked_data.global_node_id = torch.cat(
            [
                data.global_node_id.clone(),
                trigger_global_ids,
            ],
            dim=0,
        )

    new_edges: list[tuple[int, int]] = []

    for victim_index, victim_node in enumerate(
        poisoned_victim_nodes.tolist()
    ):
        trigger_start = (
            original_num_nodes
            + victim_index * trigger_size
        )

        trigger_nodes = list(
            range(
                trigger_start,
                trigger_start + trigger_size,
            )
        )

        # Connect the victim node to every trigger node.
        for trigger_node in trigger_nodes:
            new_edges.append(
                (victim_node, trigger_node)
            )
            new_edges.append(
                (trigger_node, victim_node)
            )

        # Construct a bidirectional clique.
        for first_index in range(trigger_size):
            for second_index in range(
                first_index + 1,
                trigger_size,
            ):
                first_node = trigger_nodes[
                    first_index
                ]
                second_node = trigger_nodes[
                    second_index
                ]

                new_edges.append(
                    (first_node, second_node)
                )
                new_edges.append(
                    (second_node, first_node)
                )

    new_edge_index = torch.tensor(
        new_edges,
        dtype=torch.long,
        device=data.edge_index.device,
    ).t().contiguous()

    attacked_data.edge_index = torch.cat(
        [
            data.edge_index.clone(),
            new_edge_index,
        ],
        dim=1,
    )

    expected_edges_per_victim = 2 * (
        trigger_size
        + (
            trigger_size
            * (trigger_size - 1)
        ) // 2
    )

    expected_added_edges = (
        poison_count
        * expected_edges_per_victim
    )

    metadata: dict[str, Any] = {
        "seed": seed,
        "target_class": target_class,
        "poison_rate": poison_rate,
        "trigger_type": "clique",
        "trigger_size": trigger_size,
        "poisoned_victim_count": poison_count,
        "poisoned_victim_nodes": (
            poisoned_victim_nodes.tolist()
        ),
        "trigger_node_count": (
            new_trigger_node_count
        ),
        "original_num_nodes": original_num_nodes,
        "attacked_num_nodes": int(
            attacked_data.num_nodes
        ),
        "original_num_edges": original_num_edges,
        "attacked_num_edges": int(
            attacked_data.num_edges
        ),
        "added_edges": int(
            attacked_data.num_edges
            - original_num_edges
        ),
    }

    # Consistency checks
    assert (
        attacked_data.num_nodes
        == original_num_nodes
        + new_trigger_node_count
    )

    assert (
        attacked_data.num_edges
        == original_num_edges
        + expected_added_edges
    )

    assert int(
        attacked_data.poisoned_node_mask.sum()
    ) == poison_count

    assert int(
        attacked_data.trigger_node_mask.sum()
    ) == new_trigger_node_count

    assert bool(
        attacked_data.y[
            poisoned_victim_nodes
        ]
        .eq(target_class)
        .all()
    )

    return attacked_data, metadata


def main() -> None:
    set_seed(SEED)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "Planetoid"

    partition_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / f"iid_stratified_{NUM_CLIENTS}_clients"
        / "client_node_indices.pt"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "attacks"
        / "structural_clique"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"malicious_client_{MALICIOUS_CLIENT_ID}.pt"
    )

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    full_data = dataset[0]

    full_data.global_node_id = torch.arange(
        full_data.num_nodes
    )

    partition_data = torch.load(
        partition_path,
        map_location="cpu",
        weights_only=False,
    )

    client_indices = partition_data[
        "client_indices"
    ]

    malicious_indices = client_indices[
        MALICIOUS_CLIENT_ID
    ]

    clean_client_data = full_data.subgraph(
        malicious_indices
    )

    original_train_distribution = torch.bincount(
        clean_client_data.y[
            clean_client_data.train_mask
        ],
        minlength=dataset.num_classes,
    )

    (
        attacked_client_data,
        attack_metadata,
    ) = inject_clique_trigger(
        data=clean_client_data,
        target_class=TARGET_CLASS,
        poison_rate=POISON_RATE,
        trigger_size=TRIGGER_SIZE,
        seed=SEED,
    )

    attacked_train_distribution = torch.bincount(
        attacked_client_data.y[
            attacked_client_data.train_mask
        ],
        minlength=dataset.num_classes,
    )

    torch.save(
        {
            "data": attacked_client_data,
            "metadata": attack_metadata,
        },
        output_path,
    )

    print("=" * 72)
    print("Structural backdoor injection test")
    print("=" * 72)
    print(
        f"Malicious client:       "
        f"{MALICIOUS_CLIENT_ID}"
    )
    print(
        f"Target class:           "
        f"{TARGET_CLASS}"
    )
    print(
        f"Poisoning rate:         "
        f"{POISON_RATE:.2f}"
    )
    print(
        f"Trigger type:           "
        f"Clique"
    )
    print(
        f"Trigger size:           "
        f"{TRIGGER_SIZE}"
    )
    print("-" * 72)
    print(
        f"Original nodes:         "
        f"{attack_metadata['original_num_nodes']}"
    )
    print(
        f"Attacked nodes:         "
        f"{attack_metadata['attacked_num_nodes']}"
    )
    print(
        f"New trigger nodes:      "
        f"{attack_metadata['trigger_node_count']}"
    )
    print(
        f"Original edges:         "
        f"{attack_metadata['original_num_edges']}"
    )
    print(
        f"Attacked edges:         "
        f"{attack_metadata['attacked_num_edges']}"
    )
    print(
        f"Added directed edges:   "
        f"{attack_metadata['added_edges']}"
    )
    print(
        f"Poisoned train nodes:   "
        f"{attack_metadata['poisoned_victim_count']}"
    )
    print("-" * 72)
    print(
        "Original train-class distribution:"
    )
    print(
        original_train_distribution.tolist()
    )
    print(
        "Attacked train-class distribution:"
    )
    print(
        attacked_train_distribution.tolist()
    )
    print("-" * 72)
    print(
        "Poisoned local victim node IDs:"
    )
    print(
        attack_metadata[
            "poisoned_victim_nodes"
        ]
    )
    print("-" * 72)
    print("All attack consistency checks passed.")
    print(f"Attacked graph saved to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()