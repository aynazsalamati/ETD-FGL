from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
from torch_geometric.data import Data


@dataclass
class GSPReport:
    original_directed_edges: int
    purified_directed_edges: int
    removed_undirected_edges: int
    low_frequency_rank: int
    prune_ratio: float


def _unique_undirected_edges(
    edge_index: torch.Tensor,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    edges = edge_index.detach().cpu().numpy().T
    pairs = sorted(
        {
            (int(min(u, v)), int(max(u, v)))
            for u, v in edges
            if int(u) != int(v)
        }
    )
    self_loops = [
        (int(u), int(v))
        for u, v in edges
        if int(u) == int(v)
    ]
    return np.asarray(pairs, dtype=np.int64), self_loops


def _spectral_coordinates(
    num_nodes: int,
    undirected_pairs: np.ndarray,
    rank: int,
) -> np.ndarray:
    if undirected_pairs.size == 0:
        return np.eye(num_nodes, min(num_nodes, rank), dtype=np.float64)

    rows = np.concatenate(
        [undirected_pairs[:, 0], undirected_pairs[:, 1]]
    )
    cols = np.concatenate(
        [undirected_pairs[:, 1], undirected_pairs[:, 0]]
    )
    values = np.ones(rows.shape[0], dtype=np.float64)

    adjacency = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float64)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    d_inv = sp.diags(inv_sqrt)
    normalized_adjacency = d_inv @ adjacency @ d_inv
    laplacian = sp.eye(num_nodes, format="csr") - normalized_adjacency

    effective_rank = max(2, min(rank, num_nodes - 2))
    try:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=effective_rank,
            which="SM",
            tol=1e-4,
        )
    except Exception:
        dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        eigenvalues = eigenvalues[:effective_rank]
        eigenvectors = eigenvectors[:, :effective_rank]

    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    # Low-frequency coordinates. The first constant component is retained,
    # while higher components are softly attenuated.
    attenuation = 1.0 / np.sqrt(1.0 + eigenvalues)
    return eigenvectors * attenuation.reshape(1, -1)


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = max(1.4826 * mad, 1e-12)
    return (values - median) / scale


def purify_graph_spectral(
    data: Data,
    *,
    low_frequency_rank: int = 20,
    prune_ratio: float = 0.12,
) -> tuple[Data, GSPReport]:
    """
    Federated adaptation of Graph Spectral Purification.

    Existing edges are ranked using their energy in a low-frequency graph
    embedding. Edges with unusually high spectral residual and feature
    disagreement are pruned, while degree-one nodes are protected.
    """

    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    purified = data.clone()
    pairs, self_loops = _unique_undirected_edges(data.edge_index)

    if len(pairs) < 4 or prune_ratio == 0.0:
        return purified, GSPReport(
            original_directed_edges=int(data.num_edges),
            purified_directed_edges=int(data.num_edges),
            removed_undirected_edges=0,
            low_frequency_rank=low_frequency_rank,
            prune_ratio=prune_ratio,
        )

    coordinates = _spectral_coordinates(
        int(data.num_nodes),
        pairs,
        low_frequency_rank,
    )

    u = pairs[:, 0]
    v = pairs[:, 1]
    spectral_residual = np.square(
        coordinates[u] - coordinates[v]
    ).sum(axis=1)

    features = data.x.detach().cpu().numpy().astype(np.float64, copy=False)
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / np.maximum(feature_norm, 1e-12)
    cosine_similarity = (
        normalized_features[u] * normalized_features[v]
    ).sum(axis=1)
    feature_disagreement = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)

    suspiciousness = (
        _robust_standardize(spectral_residual)
        + 0.30 * _robust_standardize(feature_disagreement)
    )

    degrees = np.bincount(
        pairs.reshape(-1),
        minlength=int(data.num_nodes),
    )
    protected = (degrees[u] <= 1) | (degrees[v] <= 1)
    suspiciousness[protected] = -np.inf

    removable_count = int(np.floor(len(pairs) * prune_ratio))
    removable_count = min(
        removable_count,
        int(np.isfinite(suspiciousness).sum()),
    )

    keep_mask = np.ones(len(pairs), dtype=bool)
    if removable_count > 0:
        remove_indices = np.argpartition(
            suspiciousness,
            -removable_count,
        )[-removable_count:]
        keep_mask[remove_indices] = False

    kept_pairs = pairs[keep_mask]
    directed_edges: list[tuple[int, int]] = []
    for first, second in kept_pairs.tolist():
        directed_edges.append((first, second))
        directed_edges.append((second, first))
    directed_edges.extend(self_loops)

    purified.edge_index = torch.tensor(
        directed_edges,
        dtype=torch.long,
        device=data.edge_index.device,
    ).t().contiguous()

    report = GSPReport(
        original_directed_edges=int(data.num_edges),
        purified_directed_edges=int(purified.num_edges),
        removed_undirected_edges=int((~keep_mask).sum()),
        low_frequency_rank=low_frequency_rank,
        prune_ratio=prune_ratio,
    )
    return purified, report
