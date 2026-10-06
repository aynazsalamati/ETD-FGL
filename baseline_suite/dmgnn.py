from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from train_fedavg_cora import GCN


@dataclass
class DMGNNClientReport:
    client_id: int
    candidate_nodes: int
    suspicious_nodes: int
    removed_undirected_edges: int
    original_directed_edges: int
    purified_directed_edges: int


def _pair(first: int, second: int) -> tuple[int, int]:
    return (min(first, second), max(first, second))


def _unique_pairs(edge_index: Tensor) -> list[tuple[int, int]]:
    return sorted(
        {
            _pair(int(first), int(second))
            for first, second in edge_index.detach().cpu().t().tolist()
            if int(first) != int(second)
        }
    )


def _edge_mask_without_pairs(
    edge_index: Tensor,
    removed_pairs: set[tuple[int, int]],
) -> Tensor:
    keep = [
        _pair(int(first), int(second)) not in removed_pairs
        for first, second in edge_index.detach().cpu().t().tolist()
    ]
    return torch.tensor(
        keep,
        dtype=torch.bool,
        device=edge_index.device,
    )


def _target_probability(
    model: GCN,
    data: Data,
    node_id: int,
    target_class: int,
    removed_pairs: set[tuple[int, int]],
) -> tuple[float, int]:
    mask = _edge_mask_without_pairs(data.edge_index, removed_pairs)
    edge_index = data.edge_index[:, mask]

    with torch.no_grad():
        logits = model(data.x, edge_index)
        probabilities = logits[node_id].softmax(dim=0)
        prediction = int(probabilities.argmax().item())
        target_probability = float(probabilities[target_class].item())

    return target_probability, prediction


def _ego_pairs(
    node_id: int,
    pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    neighbors: set[int] = set()
    for first, second in pairs:
        if first == node_id:
            neighbors.add(second)
        elif second == node_id:
            neighbors.add(first)

    ego_nodes = {node_id, *neighbors}
    return [
        pair
        for pair in pairs
        if pair[0] in ego_nodes and pair[1] in ego_nodes
    ]


def _clustering_scores(data: Data) -> dict[int, float]:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from(_unique_pairs(data.edge_index))
    return nx.clustering(graph)


def _candidate_subsets(
    ranked_pairs: list[tuple[int, int]],
    max_remove: int,
    seed: int,
) -> list[set[tuple[int, int]]]:
    candidates: list[set[tuple[int, int]]] = []
    limit = min(len(ranked_pairs), 8)
    ranked_pairs = ranked_pairs[:limit]

    for size in range(1, min(max_remove, limit) + 1):
        # Deterministic leading subset.
        candidates.append(set(ranked_pairs[:size]))

        # Exact small combinations provide stable counterfactual search.
        if limit <= 6:
            candidates.extend(
                set(combo)
                for combo in combinations(ranked_pairs, size)
            )

    generator = np.random.default_rng(seed)
    for _ in range(24):
        size = int(generator.integers(1, min(max_remove, limit) + 1))
        selected = generator.choice(limit, size=size, replace=False)
        candidates.append({ranked_pairs[int(index)] for index in selected})

    unique: list[set[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for candidate in candidates:
        key = tuple(sorted(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def purify_graph_explanation_guided(
    *,
    model: GCN,
    data: Data,
    client_id: int,
    target_class: int,
    trigger_size: int,
    seed: int,
    confidence_threshold: float = 0.50,
    probability_drop_threshold: float = 0.03,
) -> tuple[Data, DMGNNClientReport, list[dict[str, object]]]:
    """
    Explanation-guided counterfactual pruning adapted to a node-level GCN.

    Target-predicted training nodes with unusually clustered ego graphs are
    examined. Ego edges are ranked by the drop in target probability caused
    by their removal. Reverse subset sampling selects a compact edge set that
    maximizes label transition or target-confidence reduction.
    """

    device = next(model.parameters()).device
    working = data.clone().to(device)
    model.eval()

    with torch.no_grad():
        logits = model(working.x, working.edge_index)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

    train_nodes = working.train_mask.nonzero(as_tuple=False).view(-1)
    clustering = _clustering_scores(data)
    train_cluster_values = np.asarray(
        [clustering[int(node)] for node in train_nodes.cpu().tolist()],
        dtype=np.float64,
    )
    median = float(np.median(train_cluster_values))
    mad = float(np.median(np.abs(train_cluster_values - median)))
    cluster_threshold = median + max(1.4826 * mad, 0.05)

    candidate_nodes = [
        int(node)
        for node in train_nodes.cpu().tolist()
        if int(predictions[node].item()) == target_class
        and float(probabilities[node, target_class].item())
        >= confidence_threshold
        and float(clustering[int(node)]) >= cluster_threshold
    ]

    all_pairs = _unique_pairs(working.edge_index)
    globally_removed: set[tuple[int, int]] = set()
    node_reports: list[dict[str, object]] = []

    for order, node_id in enumerate(candidate_nodes):
        base_probability, base_prediction = _target_probability(
            model,
            working,
            node_id,
            target_class,
            globally_removed,
        )

        ego_pairs = [
            pair
            for pair in _ego_pairs(node_id, all_pairs)
            if pair not in globally_removed
        ]
        if not ego_pairs:
            continue

        edge_scores: list[tuple[float, tuple[int, int]]] = []
        for pair in ego_pairs:
            probability, _ = _target_probability(
                model,
                working,
                node_id,
                target_class,
                globally_removed | {pair},
            )
            edge_scores.append((base_probability - probability, pair))

        edge_scores.sort(key=lambda item: item[0], reverse=True)
        ranked_pairs = [pair for _, pair in edge_scores]

        best_subset: set[tuple[int, int]] = set()
        best_objective = 0.0
        best_probability = base_probability
        best_prediction = base_prediction

        for subset in _candidate_subsets(
            ranked_pairs,
            max_remove=max(1, trigger_size),
            seed=seed + 1000 * client_id + order,
        ):
            probability, prediction = _target_probability(
                model,
                working,
                node_id,
                target_class,
                globally_removed | subset,
            )
            drop = base_probability - probability
            transition_bonus = 0.25 if prediction != target_class else 0.0
            objective = drop + transition_bonus - 0.005 * len(subset)

            if objective > best_objective:
                best_objective = objective
                best_subset = subset
                best_probability = probability
                best_prediction = prediction

        probability_drop = base_probability - best_probability
        transitioned = best_prediction != target_class

        accepted = bool(
            best_subset
            and (
                transitioned
                or probability_drop >= probability_drop_threshold
            )
        )

        if accepted:
            globally_removed.update(best_subset)

        node_reports.append(
            {
                "node_id": node_id,
                "clustering": float(clustering[node_id]),
                "base_target_probability": base_probability,
                "counterfactual_target_probability": best_probability,
                "probability_drop": probability_drop,
                "label_transition": transitioned,
                "accepted": accepted,
                "removed_pairs": [list(pair) for pair in sorted(best_subset)]
                if accepted
                else [],
            }
        )

    keep_mask = _edge_mask_without_pairs(
        working.edge_index,
        globally_removed,
    )
    purified = data.clone()
    purified.edge_index = data.edge_index[:, keep_mask.cpu()]

    report = DMGNNClientReport(
        client_id=client_id,
        candidate_nodes=len(candidate_nodes),
        suspicious_nodes=sum(
            1 for report in node_reports if bool(report["accepted"])
        ),
        removed_undirected_edges=len(globally_removed),
        original_directed_edges=int(data.num_edges),
        purified_directed_edges=int(purified.num_edges),
    )

    return purified, report, node_reports
