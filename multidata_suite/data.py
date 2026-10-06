from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures

from multidata_suite.config import DatasetConfig
from multidata_suite.model import set_seed


def _allocate_counts(group_sizes: list[int], total: int) -> list[int]:
    available = sum(group_sizes)
    if total >= available:
        return group_sizes[:]
    if available <= 0 or total <= 0:
        return [0] * len(group_sizes)

    raw = [total * size / available for size in group_sizes]
    counts = [min(size, int(value)) for size, value in zip(group_sizes, raw)]
    remaining = total - sum(counts)
    remainders = sorted(
        range(len(group_sizes)),
        key=lambda i: raw[i] - int(raw[i]),
        reverse=True,
    )
    for index in remainders:
        if remaining <= 0:
            break
        if counts[index] < group_sizes[index]:
            counts[index] += 1
            remaining -= 1
    return counts


def _stratified_sample_indices(
    data: Data,
    sample_size: int,
    num_classes: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    split_masks = [data.train_mask, data.val_mask, data.test_mask]
    split_available = [int(mask.sum()) for mask in split_masks]

    target_test = min(1000, split_available[2], max(1, sample_size // 8))
    target_val = min(1000, split_available[1], max(1, sample_size // 8))
    target_train = min(
        split_available[0],
        sample_size - target_test - target_val,
    )
    split_targets = [target_train, target_val, target_test]

    selected_parts: list[Tensor] = []
    for split_mask, split_target in zip(split_masks, split_targets, strict=True):
        class_groups = [
            (split_mask & data.y.eq(class_id)).nonzero(as_tuple=False).view(-1)
            for class_id in range(num_classes)
        ]
        counts = _allocate_counts(
            [int(group.numel()) for group in class_groups],
            split_target,
        )
        for group, count in zip(class_groups, counts, strict=True):
            if count <= 0:
                continue
            perm = torch.randperm(group.numel(), generator=generator)
            selected_parts.append(group[perm[:count]])

    selected = torch.cat(selected_parts)
    if selected.numel() < sample_size:
        used = torch.zeros(data.num_nodes, dtype=torch.bool)
        used[selected] = True
        remaining_nodes = (~used).nonzero(as_tuple=False).view(-1)
        need = min(sample_size - selected.numel(), remaining_nodes.numel())
        perm = torch.randperm(remaining_nodes.numel(), generator=generator)
        selected = torch.cat([selected, remaining_nodes[perm[:need]]])

    perm = torch.randperm(selected.numel(), generator=generator)
    return selected[perm]


def load_dataset(
    project_dir: Path,
    config: DatasetConfig,
) -> tuple[Data, int, int, dict[str, Any]]:
    set_seed(config.seed)
    data_root = project_dir / "data"

    if config.key in {"cora", "pubmed"}:
        name = "Cora" if config.key == "cora" else "PubMed"
        dataset = Planetoid(
            root=str(data_root / "Planetoid"),
            name=name,
            split="public",
            transform=NormalizeFeatures(),
        )
        data = dataset[0]
        metadata = {
            "dataset": name,
            "sampled": False,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
        }
        return data, int(dataset.num_features), int(dataset.num_classes), metadata

    # Keep the sampled Reddit subgraph fixed across training seeds so that
    # multi-seed experiments measure optimization/partition variability rather
    # than changing the underlying benchmark sample.
    sample_seed = int(config.reddit_sample_seed)
    cache_dir = data_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"reddit_sample_{config.reddit_sample_size}_seed_{sample_seed}.pt"
    )
    if cache_path.exists():
        package = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (
            package["data"],
            int(package["num_features"]),
            int(package["num_classes"]),
            package["metadata"],
        )

    dataset = Reddit(root=str(data_root / "Reddit"))
    full_data = dataset[0]
    selected = _stratified_sample_indices(
        full_data,
        sample_size=config.reddit_sample_size,
        num_classes=int(dataset.num_classes),
        seed=sample_seed,
    )
    data = full_data.subgraph(selected)
    metadata = {
        "dataset": "Reddit",
        "sampled": True,
        "sample_size": int(data.num_nodes),
        "original_num_nodes": int(full_data.num_nodes),
        "original_num_edges": int(full_data.num_edges),
        "num_edges": int(data.num_edges),
        "sample_seed": sample_seed,
    }
    torch.save(
        {
            "data": data,
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
            "metadata": metadata,
        },
        cache_path,
    )
    return data, int(dataset.num_features), int(dataset.num_classes), metadata


def _validate_and_finalize_partition(
    data: Data,
    client_parts: list[list[int]],
    num_clients: int,
    seed: int,
    target_class: int | None = None,
    malicious_client_id: int = 0,
) -> dict[int, Tensor]:
    """Repair empty-training clients, shuffle locally, and validate ownership."""

    if int(data.train_mask.sum()) < num_clients:
        raise ValueError(
            f"Cannot create {num_clients} trainable clients: the dataset split "
            f"contains only {int(data.train_mask.sum())} training nodes."
        )

    train_counts = [sum(bool(data.train_mask[idx]) for idx in part) for part in client_parts]
    for client_id in range(num_clients):
        if train_counts[client_id] > 0:
            continue
        donors = sorted(
            range(num_clients),
            key=lambda i: train_counts[i],
            reverse=True,
        )
        donor = next((i for i in donors if train_counts[i] > 1), None)
        if donor is None:
            raise RuntimeError("Unable to repair a client with no training nodes.")
        move_pos = next(
            pos for pos, node in enumerate(client_parts[donor])
            if bool(data.train_mask[node])
        )
        node = client_parts[donor].pop(move_pos)
        client_parts[client_id].append(node)
        train_counts[donor] -= 1
        train_counts[client_id] += 1

    # Keep the attack protocol well-defined across strongly non-IID and
    # high-client-count partitions: the designated malicious client must own
    # at least one non-target training node that can be poisoned.
    if target_class is not None and 0 <= malicious_client_id < num_clients:
        malicious_has_candidate = any(
            bool(data.train_mask[node]) and int(data.y[node]) != int(target_class)
            for node in client_parts[malicious_client_id]
        )
        if not malicious_has_candidate:
            donor_choice = None
            donor_pos = None
            for donor in sorted(range(num_clients), key=lambda i: train_counts[i], reverse=True):
                if donor == malicious_client_id or train_counts[donor] <= 1:
                    continue
                for pos, node in enumerate(client_parts[donor]):
                    if bool(data.train_mask[node]) and int(data.y[node]) != int(target_class):
                        donor_choice, donor_pos = donor, pos
                        break
                if donor_choice is not None:
                    break
            if donor_choice is None or donor_pos is None:
                raise RuntimeError(
                    "Unable to guarantee a non-target training node for the malicious client."
                )
            node = client_parts[donor_choice].pop(donor_pos)
            client_parts[malicious_client_id].append(node)
            train_counts[donor_choice] -= 1
            train_counts[malicious_client_id] += 1

    generator = torch.Generator().manual_seed(seed)
    result: dict[int, Tensor] = {}
    for client_id, part in enumerate(client_parts):
        if not part:
            raise RuntimeError(f"Client {client_id} received no nodes.")
        indices = torch.tensor(part, dtype=torch.long)
        perm = torch.randperm(indices.numel(), generator=generator)
        result[client_id] = indices[perm]
        local = data.subgraph(result[client_id])
        if int(local.train_mask.sum()) == 0:
            raise RuntimeError(f"Client {client_id} has no training nodes.")

    combined = torch.cat(list(result.values()))
    if combined.numel() != data.num_nodes:
        raise RuntimeError("Partition does not cover all nodes.")
    if torch.unique(combined).numel() != data.num_nodes:
        raise RuntimeError("At least one node was assigned to multiple clients.")
    return result


def create_partition(
    data: Data,
    num_classes: int,
    num_clients: int,
    seed: int,
    target_class: int | None = None,
    malicious_client_id: int = 0,
) -> dict[int, Tensor]:
    """Create a balanced label- and split-aware IID partition.

    Rotating round-robin assignment avoids the empty-client problem that can
    occur when the public training split is small and the federation has many
    clients (e.g., PubMed with 50 clients).
    """

    generator = torch.Generator().manual_seed(seed)
    client_parts: list[list[int]] = [[] for _ in range(num_clients)]
    used_mask = data.train_mask | data.val_mask | data.test_mask
    split_masks = [data.train_mask, data.val_mask, data.test_mask, ~used_mask]
    offset = 0

    for split_mask in split_masks:
        for class_id in range(num_classes):
            indices = (split_mask & data.y.eq(class_id)).nonzero(as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            perm = torch.randperm(indices.numel(), generator=generator)
            shuffled = indices[perm].tolist()
            for position, node in enumerate(shuffled):
                client_id = (offset + position) % num_clients
                client_parts[client_id].append(int(node))
            offset = (offset + len(shuffled)) % num_clients

    return _validate_and_finalize_partition(
        data=data,
        client_parts=client_parts,
        num_clients=num_clients,
        seed=seed,
        target_class=target_class,
        malicious_client_id=malicious_client_id,
    )


def create_dirichlet_partition(
    data: Data,
    num_classes: int,
    num_clients: int,
    seed: int,
    alpha: float,
    target_class: int | None = None,
    malicious_client_id: int = 0,
) -> dict[int, Tensor]:
    """Create a split-aware label-skew non-IID partition using Dirichlet(alpha).

    The Dirichlet draw is applied independently within each split/class group.
    Every node is assigned exactly once. A minimal repair step guarantees at
    least one training node per client whenever the public split contains at
    least `num_clients` training nodes.
    """

    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    rng = np.random.default_rng(seed)
    client_parts: list[list[int]] = [[] for _ in range(num_clients)]
    used_mask = data.train_mask | data.val_mask | data.test_mask
    split_masks = [data.train_mask, data.val_mask, data.test_mask, ~used_mask]

    for split_mask in split_masks:
        for class_id in range(num_classes):
            indices = (split_mask & data.y.eq(class_id)).nonzero(as_tuple=False).view(-1)
            n = int(indices.numel())
            if n == 0:
                continue
            order = rng.permutation(n)
            shuffled = indices[torch.as_tensor(order, dtype=torch.long)].tolist()
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(n, proportions)
            cursor = 0
            for client_id, count in enumerate(counts.tolist()):
                if count:
                    client_parts[client_id].extend(int(v) for v in shuffled[cursor:cursor + count])
                    cursor += count

    return _validate_and_finalize_partition(
        data=data,
        client_parts=client_parts,
        num_clients=num_clients,
        seed=seed,
        target_class=target_class,
        malicious_client_id=malicious_client_id,
    )


def load_or_create_partition(
    project_dir: Path,
    config: DatasetConfig,
    data: Data,
    num_classes: int,
    *,
    base_output: Path | None = None,
    strategy: str = "iid",
    dirichlet_alpha: float | None = None,
    target_class: int | None = None,
    malicious_client_id: int = 0,
) -> dict[int, Tensor]:
    base = base_output or (
        project_dir
        / "outputs"
        / "final_multidataset"
        / f"federated_{config.key}"
    )

    strategy_key = strategy.strip().lower()
    if strategy_key == "iid":
        partition_name = f"iid_stratified_{config.num_clients}_clients_seed_{config.seed}"
    elif strategy_key == "dirichlet":
        if dirichlet_alpha is None:
            raise ValueError("dirichlet_alpha is required for Dirichlet partitioning.")
        alpha_tag = str(dirichlet_alpha).replace(".", "p")
        partition_name = (
            f"dirichlet_alpha_{alpha_tag}_{config.num_clients}_clients_seed_{config.seed}"
        )
    else:
        raise ValueError(f"Unknown partition strategy: {strategy!r}")

    output_dir = base / "partitions" / partition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "client_node_indices.pt"
    summary_path = output_dir / "partition_summary.json"

    if path.exists():
        package = torch.load(path, map_location="cpu", weights_only=False)
        return package["client_indices"]

    if strategy_key == "iid":
        indices = create_partition(
            data=data,
            num_classes=num_classes,
            num_clients=config.num_clients,
            seed=config.seed,
            target_class=target_class,
            malicious_client_id=malicious_client_id,
        )
    else:
        indices = create_dirichlet_partition(
            data=data,
            num_classes=num_classes,
            num_clients=config.num_clients,
            seed=config.seed,
            alpha=float(dirichlet_alpha),
            target_class=target_class,
            malicious_client_id=malicious_client_id,
        )

    torch.save(
        {
            "seed": config.seed,
            "num_clients": config.num_clients,
            "strategy": strategy_key,
            "dirichlet_alpha": dirichlet_alpha,
            "client_indices": indices,
        },
        path,
    )

    summaries = []
    for client_id, node_indices in indices.items():
        local = data.subgraph(node_indices)
        class_counts = torch.bincount(local.y, minlength=num_classes).tolist()
        summaries.append(
            {
                "client_id": client_id,
                "nodes": int(local.num_nodes),
                "edges": int(local.num_edges),
                "train_nodes": int(local.train_mask.sum()),
                "validation_nodes": int(local.val_mask.sum()),
                "test_nodes": int(local.test_mask.sum()),
                "class_counts": [int(v) for v in class_counts],
            }
        )
    summary_path.write_text(
        json.dumps(
            {
                "strategy": strategy_key,
                "dirichlet_alpha": dirichlet_alpha,
                "seed": config.seed,
                "num_clients": config.num_clients,
                "clients": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return indices


def build_client_graphs(
    data: Data,
    client_indices: dict[int, Tensor],
    num_clients: int,
) -> list[Data]:
    graphs = []
    for client_id in range(num_clients):
        local = data.subgraph(client_indices[client_id])
        local.global_node_id = client_indices[client_id].clone()
        graphs.append(local)
    return graphs
