from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data


def _extend_mask(mask: Tensor, count: int) -> Tensor:
    return torch.cat(
        [
            mask.clone(),
            torch.zeros(
                count,
                dtype=torch.bool,
                device=mask.device,
            ),
        ]
    )


def make_sparse_trigger_prototype(
    x: Tensor,
    seed: int,
    active_features: int = 16,
) -> Tensor:
    """
    Build a deterministic sparse trigger feature vector.

    Properties:
    - fixed for a given seed;
    - independent of the target class;
    - sparse support;
    - feature magnitude matched to the typical node-feature norm;
    - reproducible across training-time and evaluation-time triggers.
    """
    if x.dim() != 2:
        raise ValueError("Expected a 2-D node feature matrix.")

    num_features = int(x.size(1))
    if num_features <= 0:
        raise ValueError("Feature dimension must be positive.")

    active_features = max(
        1,
        min(int(active_features), num_features),
    )

    # Separate deterministic RNG stream so feature selection is not
    # coupled to poisoned-victim selection.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729)

    indices = torch.randperm(
        num_features,
        generator=generator,
    )[:active_features].to(x.device)

    prototype = torch.zeros(
        num_features,
        dtype=x.dtype,
        device=x.device,
    )

    row_norms = x.detach().norm(p=2, dim=1)
    positive_norms = row_norms[row_norms > 0]

    if positive_norms.numel() > 0:
        target_norm = positive_norms.median()
    else:
        target_norm = torch.tensor(
            1.0,
            dtype=x.dtype,
            device=x.device,
        )

    if not bool(torch.isfinite(target_norm)):
        target_norm = torch.tensor(
            1.0,
            dtype=x.dtype,
            device=x.device,
        )

    value = target_norm / math.sqrt(float(active_features))
    prototype[indices] = value
    return prototype


def inject_clique_trigger(
    data: Data,
    target_class: int,
    poison_rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[Data, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)

    candidates = (
        data.train_mask & data.y.ne(target_class)
    ).nonzero(as_tuple=False).view(-1)

    if candidates.numel() == 0:
        raise RuntimeError(
            "No non-target training nodes are available for poisoning."
        )

    poison_count = min(
        candidates.numel(),
        max(
            1,
            math.ceil(poison_rate * candidates.numel()),
        ),
    )

    perm = torch.randperm(
        candidates.numel(),
        generator=generator,
    )
    victims = candidates[perm[:poison_count]]

    original_nodes = int(data.num_nodes)
    trigger_nodes_count = poison_count * trigger_size

    prototype = make_sparse_trigger_prototype(
        data.x,
        seed=seed,
        active_features=16,
    )

    attacked = data.clone()

    attacked.x = torch.cat(
        [
            data.x.clone(),
            prototype.repeat(trigger_nodes_count, 1),
        ],
        dim=0,
    )

    attacked.y = torch.cat(
        [
            data.y.clone(),
            torch.full(
                (trigger_nodes_count,),
                target_class,
                dtype=data.y.dtype,
                device=data.y.device,
            ),
        ],
        dim=0,
    )

    attacked.clean_y = torch.cat(
        [
            data.y.clone(),
            torch.full(
                (trigger_nodes_count,),
                -1,
                dtype=data.y.dtype,
                device=data.y.device,
            ),
        ],
        dim=0,
    )

    attacked.y[victims] = target_class

    attacked.train_mask = _extend_mask(
        data.train_mask,
        trigger_nodes_count,
    )
    attacked.val_mask = _extend_mask(
        data.val_mask,
        trigger_nodes_count,
    )
    attacked.test_mask = _extend_mask(
        data.test_mask,
        trigger_nodes_count,
    )

    poisoned_mask = torch.zeros(
        attacked.num_nodes,
        dtype=torch.bool,
        device=data.x.device,
    )
    poisoned_mask[victims] = True

    trigger_mask = torch.zeros(
        attacked.num_nodes,
        dtype=torch.bool,
        device=data.x.device,
    )
    trigger_mask[original_nodes:] = True

    attacked.poisoned_node_mask = poisoned_mask
    attacked.trigger_node_mask = trigger_mask

    new_edges: list[tuple[int, int]] = []

    for victim_index, victim in enumerate(victims.tolist()):
        start = original_nodes + victim_index * trigger_size
        trigger_nodes = list(range(start, start + trigger_size))

        for node in trigger_nodes:
            new_edges.extend(
                [
                    (victim, node),
                    (node, victim),
                ]
            )

        for first in range(trigger_size):
            for second in range(first + 1, trigger_size):
                u = trigger_nodes[first]
                v = trigger_nodes[second]
                new_edges.extend(
                    [
                        (u, v),
                        (v, u),
                    ]
                )

    added = torch.tensor(
        new_edges,
        dtype=torch.long,
        device=data.edge_index.device,
    ).t().contiguous()

    attacked.edge_index = torch.cat(
        [
            data.edge_index.clone(),
            added,
        ],
        dim=1,
    )

    metadata = {
        "poisoned_victim_nodes": victims.tolist(),
        "poison_count": int(poison_count),
        "trigger_size": int(trigger_size),
        "target_class": int(target_class),
        "poison_rate": float(poison_rate),
        "added_trigger_nodes": int(trigger_nodes_count),
        "added_directed_edges": int(added.size(1)),
        "trigger_structure": "clique",
        "trigger_feature_type": "fixed_sparse_random",
        "trigger_active_features": int(prototype.ne(0).sum().item()),
        "trigger_feature_norm": float(prototype.norm(p=2).item()),
        "trigger_feature_seed": int(seed),
    }

    return attacked, metadata


def attach_evaluation_triggers(
    data: Data,
    victim_nodes: Tensor,
    trigger_size: int,
    feature_prototype: Tensor,
    target_class: int,
) -> Data:
    attacked = data.clone()

    original_nodes = int(data.num_nodes)
    count = int(victim_nodes.numel()) * trigger_size

    prototype = (
        feature_prototype
        .detach()
        .clone()
        .to(
            device=data.x.device,
            dtype=data.x.dtype,
        )
    )

    attacked.x = torch.cat(
        [
            data.x.clone(),
            prototype.repeat(count, 1),
        ],
        dim=0,
    )

    attacked.y = torch.cat(
        [
            data.y.clone(),
            torch.full(
                (count,),
                target_class,
                dtype=data.y.dtype,
                device=data.y.device,
            ),
        ],
        dim=0,
    )

    attacked.train_mask = _extend_mask(data.train_mask, count)
    attacked.val_mask = _extend_mask(data.val_mask, count)
    attacked.test_mask = _extend_mask(data.test_mask, count)

    edges: list[tuple[int, int]] = []

    for victim_index, victim in enumerate(victim_nodes.tolist()):
        start = original_nodes + victim_index * trigger_size
        nodes = list(range(start, start + trigger_size))

        for node in nodes:
            edges.extend(
                [
                    (victim, node),
                    (node, victim),
                ]
            )

        for first in range(trigger_size):
            for second in range(first + 1, trigger_size):
                u = nodes[first]
                v = nodes[second]
                edges.extend(
                    [
                        (u, v),
                        (v, u),
                    ]
                )

    added = torch.tensor(
        edges,
        dtype=torch.long,
        device=data.edge_index.device,
    ).t().contiguous()

    attacked.edge_index = torch.cat(
        [
            data.edge_index.clone(),
            added,
        ],
        dim=1,
    )

    return attacked
