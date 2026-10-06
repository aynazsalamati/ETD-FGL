from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import gudhi
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import dropout_edge

from multidata_suite.config import DatasetConfig
from multidata_suite.model import GCN, StateDict


def _flatten_delta(state: StateDict, global_state: StateDict, keys: list[str] | None = None) -> Tensor:
    selected = keys if keys is not None else list(state.keys())
    chunks = [
        (state[key] - global_state[key]).detach().float().reshape(-1)
        for key in selected
        if key in state and torch.is_floating_point(state[key])
    ]
    return torch.cat(chunks)


def _weighted_average(local_states: list[StateDict], weights: np.ndarray) -> StateDict:
    averaged: StateDict = OrderedDict()
    for key, reference in local_states[0].items():
        if not torch.is_floating_point(reference):
            averaged[key] = reference.clone()
            continue
        output = torch.zeros_like(reference)
        for state, weight in zip(local_states, weights.tolist(), strict=True):
            output.add_(state[key], alpha=float(weight))
        averaged[key] = output
    return averaged


# ---------------- PD-FL adaptation ----------------
def _persistence_signature(state: StateDict, global_state: StateDict) -> np.ndarray:
    vector = _flatten_delta(state, global_state).cpu().numpy().astype(np.float64)
    median = float(np.median(vector))
    mad = max(float(np.median(np.abs(vector - median))) * 1.4826, 1e-12)
    vector = np.clip((vector - median) / mad, -8.0, 8.0)
    source = np.linspace(0.0, 1.0, vector.size)
    target = np.linspace(0.0, 1.0, 24 * 24)
    image = np.interp(target, source, vector).reshape(24, 24)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=image)
    persistence = complex_.persistence(homology_coeff_field=2)

    def stats(dimension: int) -> list[float]:
        lifetimes = [
            max(0.0, death - birth)
            for dim, (birth, death) in persistence
            if dim == dimension and math.isfinite(death) and death > birth
        ]
        if not lifetimes:
            return [0.0, 0.0, 0.0, 0.0]
        arr = np.asarray(lifetimes, dtype=np.float64)
        probs = arr / max(arr.sum(), 1e-12)
        return [float(arr.size), float(arr.sum()), float(arr.max()), float(-(probs * np.log(probs + 1e-12)).sum())]

    return np.asarray(
        [
            float(np.abs(vector).sum()),
            float(np.linalg.norm(vector)),
            float(np.abs(vector).max(initial=0.0)),
            float(vector.mean()),
            float(vector.std()),
            *stats(0),
            *stats(1),
        ],
        dtype=np.float64,
    )


class PDFLAggregator:
    def __init__(self, clean_features: np.ndarray, seed: int) -> None:
        self.scaler = StandardScaler().fit(clean_features)
        scaled = self.scaler.transform(clean_features)
        self.detector = IsolationForest(n_estimators=250, random_state=seed, n_jobs=-1).fit(scaled)
        scores = self.detector.decision_function(scaled)
        self.median = float(np.median(scores))
        self.mad = max(float(np.median(np.abs(scores - self.median))), 1e-6)

    def __call__(self, local_states, counts, global_state, round_number):
        del round_number
        features = np.stack([_persistence_signature(state, global_state) for state in local_states])
        scores = self.detector.decision_function(self.scaler.transform(features))
        z = (scores - self.median) / (1.4826 * self.mad + 1e-12)
        trust = np.clip(1.0 / (1.0 + np.exp(-z)), 0.03, 1.0)
        weights = trust * np.asarray(counts, dtype=np.float64)
        weights /= weights.sum()
        diagnostics = {f"pdfl_trust_client_{i}": float(value) for i, value in enumerate(trust)}
        return _weighted_average(local_states, weights), diagnostics


# ---------------- GSP adaptation ----------------
def purify_gsp(data: Data, rank: int = 20, prune_ratio: float = 0.12) -> Data:
    pairs = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in data.edge_index.t().tolist() if int(u) != int(v)})
    if len(pairs) < 4 or data.num_nodes < 4:
        return data.clone()
    pair_array = np.asarray(pairs, dtype=np.int64)
    rows = np.concatenate([pair_array[:, 0], pair_array[:, 1]])
    cols = np.concatenate([pair_array[:, 1], pair_array[:, 0]])
    adjacency = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(data.num_nodes, data.num_nodes)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree, dtype=np.float64)
    inv[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = sp.eye(data.num_nodes, format="csr") - sp.diags(inv) @ adjacency @ sp.diags(inv)
    effective_rank = max(2, min(rank, data.num_nodes - 2))
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=effective_rank, which="SM", tol=1e-4)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        eigenvalues, eigenvectors = eigenvalues[:effective_rank], eigenvectors[:, :effective_rank]
    order = np.argsort(eigenvalues)
    coords = eigenvectors[:, order] * np.exp(-np.maximum(eigenvalues[order], 0.0))[None, :]
    scores = np.linalg.norm(coords[pair_array[:, 0]] - coords[pair_array[:, 1]], axis=1)
    remove_count = min(len(pairs) - 1, max(1, int(round(prune_ratio * len(pairs)))))
    removed = {pairs[i] for i in np.argsort(scores)[-remove_count:]}
    keep = [
        (min(int(u), int(v)), max(int(u), int(v))) not in removed
        for u, v in data.edge_index.t().tolist()
    ]
    purified = data.clone()
    purified.edge_index = data.edge_index[:, torch.tensor(keep, dtype=torch.bool)]
    return purified


# ---------------- FLPurifier adaptation ----------------
def make_flpurifier_trainer(config: DatasetConfig):
    contrastive_epochs = 1
    classifier_epochs = max(1, config.local_epochs - contrastive_epochs)

    def encode(model: GCN, x: Tensor, edge_index: Tensor) -> Tensor:
        return F.relu(model.conv1(x, edge_index))

    def nt_xent(first: Tensor, second: Tensor, temperature: float = 0.5) -> Tensor:
        if first.size(0) < 2:
            return first.sum() * 0.0
        first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
        logits = first @ second.t() / temperature
        labels = torch.arange(first.size(0), device=first.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def trainer(global_state, client_data, input_channels, output_channels, device, round_number, cfg):
        del round_number, cfg
        model = GCN(input_channels, config.hidden_channels, output_channels, config.dropout).to(device)
        model.load_state_dict(global_state)
        train_mask = client_data.train_mask
        for parameter in model.conv2.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(model.conv1.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        contrastive = 0.0
        for _ in range(contrastive_epochs):
            optimizer.zero_grad()
            e1, _ = dropout_edge(client_data.edge_index, p=0.20, training=True)
            e2, _ = dropout_edge(client_data.edge_index, p=0.20, training=True)
            z1 = encode(model, F.dropout(client_data.x, p=0.20, training=True), e1)[train_mask]
            z2 = encode(model, F.dropout(client_data.x, p=0.20, training=True), e2)[train_mask]
            loss = nt_xent(z1, z2)
            loss.backward()
            optimizer.step()
            contrastive = float(loss.item())
        for parameter in model.conv1.parameters():
            parameter.requires_grad_(False)
        for parameter in model.conv2.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.conv2.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        classifier = 0.0
        for _ in range(classifier_epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                hidden = encode(model, client_data.x, client_data.edge_index)
            logits = model.conv2(hidden, client_data.edge_index)
            loss = F.cross_entropy(logits[train_mask], client_data.y[train_mask])
            loss.backward()
            optimizer.step()
            classifier = float(loss.item())
        state = OrderedDict((key, value.detach().clone()) for key, value in model.state_dict().items())
        return state, classifier, int(train_mask.sum()), {"contrastive_loss": contrastive, "classifier_loss": classifier}
    return trainer


class FLPurifierAggregator:
    def __call__(self, local_states, counts, global_state, round_number):
        del round_number
        extractor_keys = [key for key in local_states[0] if key.startswith("conv1")]
        classifier_keys = [key for key in local_states[0] if key.startswith("conv2")]
        sample_weights = np.asarray(counts, dtype=np.float64)
        sample_weights /= sample_weights.sum()
        deltas = torch.stack([_flatten_delta(state, global_state, classifier_keys) for state in local_states])
        median = deltas.median(dim=0).values
        similarities = np.asarray([
            float(torch.dot(delta, median) / (delta.norm().clamp_min(1e-12) * median.norm().clamp_min(1e-12)))
            for delta in deltas
        ])
        classifier_weights = sample_weights * np.maximum(similarities, 0.05)
        classifier_weights /= classifier_weights.sum()
        averaged: StateDict = OrderedDict()
        for key in local_states[0]:
            weights = sample_weights if key in extractor_keys else classifier_weights
            reference = local_states[0][key]
            if not torch.is_floating_point(reference):
                averaged[key] = reference.clone()
                continue
            output = torch.zeros_like(reference)
            for state, weight in zip(local_states, weights.tolist(), strict=True):
                output.add_(state[key], alpha=float(weight))
            averaged[key] = output
        diagnostics = {f"flpurifier_similarity_client_{i}": float(value) for i, value in enumerate(similarities)}
        return averaged, diagnostics


# ---------------- lightweight explanation-guided purification ----------------
def purify_explanation_guided(
    model: GCN,
    data: Data,
    target_class: int,
    trigger_size: int,
    max_candidates: int = 8,
) -> Data:
    device = next(model.parameters()).device
    graph = data.to(device)
    model.eval()
    with torch.no_grad():
        probabilities = model(graph.x, graph.edge_index).softmax(dim=1)
    train_nodes = graph.train_mask.nonzero(as_tuple=False).view(-1)
    confidence = probabilities[train_nodes, target_class]
    top_count = min(max_candidates, train_nodes.numel())
    candidates = train_nodes[torch.topk(confidence, k=top_count).indices].tolist()
    pairs = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edge_index.t().tolist() if int(u) != int(v)})
    removed: set[tuple[int, int]] = set()

    for node in candidates:
        ego = [pair for pair in pairs if node in pair]
        if not ego:
            continue
        base = float(probabilities[node, target_class].item())
        scored = []
        for pair in ego[:12]:
            mask = torch.tensor([
                (min(int(u), int(v)), max(int(u), int(v))) != pair
                for u, v in graph.edge_index.t().tolist()
            ], dtype=torch.bool, device=device)
            with torch.no_grad():
                value = float(model(graph.x, graph.edge_index[:, mask])[node].softmax(dim=0)[target_class].item())
            scored.append((base - value, pair))
        scored.sort(reverse=True)
        accepted = [pair for drop, pair in scored[:trigger_size] if drop >= 0.02]
        removed.update(accepted)

    keep = [
        (min(int(u), int(v)), max(int(u), int(v))) not in removed
        for u, v in data.edge_index.t().tolist()
    ]
    purified = data.clone()
    purified.edge_index = data.edge_index[:, torch.tensor(keep, dtype=torch.bool)]
    return purified


# ---------------- ETD-FGL adaptation ----------------
# Final Frozen Topology Descriptor (5D label-free):
# d(G) = [
#     rho(G),
#     mean_degree(G),
#     Var(k)/(mean(k)^2 + epsilon),
#     C(G),
#     T(G)/N(G)
# ]
ETDFGL_DESCRIPTOR_VERSION: str = "topology_5d_label_free_v1"
ETDFGL_DESCRIPTOR_COORDINATES: list[str] = [
    "density",
    "mean_degree",
    "normalized_degree_variance",
    "average_clustering",
    "triangles_per_node",
]
ETDFGL_DESCRIPTOR_DIM: int = 5

ETDFGL_METHOD_METADATA: dict[str, Any] = {
    "descriptor_version": ETDFGL_DESCRIPTOR_VERSION,
    "descriptor_coordinates": ETDFGL_DESCRIPTOR_COORDINATES,
    "descriptor_dim": ETDFGL_DESCRIPTOR_DIM,
    "alpha_topology": 9.0 / 13.0,
    "alpha_update": 4.0 / 13.0,
    "alpha_explanation": 0.0,
    "reference_policy": (
        "authenticated exact clean pre-attack client reference in simulation"
    ),
}


def _graph_descriptor(
    data: Data,
    observed_mask: Tensor | np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int | float]]:
    """Compute the authoritative label-free 5-dimensional ETD-FGL topology descriptor.

    The descriptor is strictly structural and label-free. It does NOT access
    data.y, train_mask, val_mask, or test_mask for topology-risk computation.
    """
    del observed_mask  # Retained for signature compatibility, strictly unused in security path.

    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from([
        (int(u), int(v))
        for u, v in data.edge_index.t().tolist()
        if int(u) != int(v)
    ])
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    mean_deg = float(degrees.mean()) if degrees.size else 0.0
    norm_deg_var = float(degrees.var() / (degrees.mean() ** 2 + 1e-12)) if degrees.size else 0.0
    clustering = nx.average_clustering(graph) if graph.number_of_edges() else 0.0
    triangles = sum(nx.triangles(graph).values()) / 3.0
    triangles_per_node = triangles / max(1.0, graph.number_of_nodes())

    total_edges = graph.number_of_edges()

    descriptor = np.asarray([
        density,
        mean_deg,
        norm_deg_var,
        clustering,
        triangles_per_node,
    ], dtype=np.float64)

    if return_diagnostics:
        diagnostics: dict[str, int | float] = {
            "total_edges": total_edges,
            "total_nodes": graph.number_of_nodes(),
            "density": density,
            "mean_degree": mean_deg,
            "normalized_degree_variance": norm_deg_var,
            "clustering": clustering,
            "triangles": triangles,
            "triangles_per_node": triangles_per_node,
        }
        return descriptor, diagnostics
    return descriptor


def compute_homophily_diagnostics(
    data: Data,
    observed_mask: Tensor | np.ndarray | None = None,
) -> dict[str, int | float]:
    """DIAGNOSTIC / HISTORICAL AUDIT ONLY: Excluded from security-critical path.

    ETDFGLAggregator never calls this function and its output never affects
    client risk, trust, or aggregation weights.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from([
        (int(u), int(v))
        for u, v in data.edge_index.t().tolist()
        if int(u) != int(v)
    ])

    if observed_mask is None:
        observed_mask = getattr(data, "train_mask", None)

    total_edges = graph.number_of_edges()
    homophily_hits = 0
    homophily_eligible_edges = 0

    if observed_mask is not None and hasattr(data, "y") and data.y is not None:
        if isinstance(observed_mask, torch.Tensor):
            mask_arr = observed_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            mask_arr = np.asarray(observed_mask, dtype=bool).reshape(-1)

        labels = data.y.detach().cpu().view(-1)
        mask_len = len(mask_arr)
        y_len = labels.numel()

        for u, v in graph.edges():
            if u < mask_len and v < mask_len and u < y_len and v < y_len:
                if bool(mask_arr[u]) and bool(mask_arr[v]):
                    homophily_eligible_edges += 1
                    homophily_hits += int(labels[u] == labels[v])

    homophily = (
        float(homophily_hits / homophily_eligible_edges)
        if homophily_eligible_edges > 0
        else 0.0
    )

    return {
        "total_edges": total_edges,
        "homophily_eligible_edges": homophily_eligible_edges,
        "homophily_hits": homophily_hits,
        "homophily": homophily,
    }


def _robust_positive(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = max(1.4826 * np.median(np.abs(values - median)), np.std(values), 1e-8)
    z = np.maximum((values - median) / mad, 0.0)
    return np.clip(2.0 * (1.0 / (1.0 + np.exp(-z)) - 0.5), 0.0, 1.0)


def _explanation_score(model: GCN, data: Data, config: DatasetConfig) -> float:
    device = next(model.parameters()).device
    graph = data.to(device)
    model.eval()
    with torch.no_grad():
        probs = model(graph.x, graph.edge_index).softmax(dim=1)
    candidates = graph.train_mask.nonzero(as_tuple=False).view(-1)
    if candidates.numel() == 0:
        return 0.0
    confidence = probs[candidates, config.target_class]
    probes = candidates[torch.topk(confidence, k=min(config.explanation_probes, candidates.numel())).indices]
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config.explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    scores = []
    for node in probes.tolist():
        try:
            explanation = explainer(graph.x, graph.edge_index, index=int(node))
            mask = explanation.edge_mask.detach().cpu()
            top_k = min(12, mask.numel())
            selected = torch.topk(mask, k=top_k).indices
            edges = graph.edge_index[:, selected.to(graph.edge_index.device)].detach().cpu().t().tolist()
            sub = nx.Graph()
            sub.add_edges_from([(int(u), int(v)) for u, v in edges])
            triangles = sum(nx.triangles(sub).values()) / 3.0 if sub.number_of_nodes() else 0.0
            clustering = nx.average_clustering(sub) if sub.number_of_edges() else 0.0
            scores.append(clustering + triangles / max(1.0, sub.number_of_nodes()))
        except Exception:
            scores.append(float(_graph_descriptor(data)[3] + _graph_descriptor(data)[4]))
    return float(np.mean(scores)) if scores else 0.0


class ETDFGLAggregator:
    """Production ETD-FGL aggregator matching the frozen manuscript formulation.

    Fused risk combines paired topology drift and dynamic model-update anomaly:
        R_i^t = (9/13) * A_i^{topo} + (4/13) * A_i^{upd, t}

    The explanation-subgraph module is excluded from the security-critical risk
    and trust calculations. It may be optionally computed and logged for
    post-hoc diagnosis by setting ``compute_explanation_diagnostic=True``.
    """

    DESCRIPTOR_VERSION: str = ETDFGL_DESCRIPTOR_VERSION
    DESCRIPTOR_COORDINATES: list[str] = ETDFGL_DESCRIPTOR_COORDINATES
    DESCRIPTOR_DIM: int = ETDFGL_DESCRIPTOR_DIM
    ALPHA_TOPOLOGY: float = 9.0 / 13.0
    ALPHA_UPDATE: float = 4.0 / 13.0
    ALPHA_EXPLANATION: float = 0.0
    REFERENCE_POLICY: str = (
        "authenticated exact clean pre-attack client reference in simulation"
    )

    def __init__(
        self,
        clean_graphs: list[Data],
        current_graphs: list[Data],
        attacked_model: GCN | None = None,
        config: DatasetConfig | None = None,
        *,
        compute_explanation_diagnostic: bool = False,
    ) -> None:
        clean = np.stack([_graph_descriptor(graph) for graph in clean_graphs])
        current = np.stack([_graph_descriptor(graph) for graph in current_graphs])
        scale = np.std(clean, axis=0)
        scale[scale < 1e-8] = 1.0
        drift = np.linalg.norm((current - clean) / scale, axis=1)
        self.topology_risk = _robust_positive(drift)
        self.compute_explanation_diagnostic = bool(compute_explanation_diagnostic)

        if (
            self.compute_explanation_diagnostic
            and attacked_model is not None
            and config is not None
        ):
            explanation_raw = np.asarray([
                _explanation_score(attacked_model, graph, config)
                for graph in current_graphs
            ])
            self.explanation_risk = _robust_positive(explanation_raw)
        else:
            self.explanation_risk = np.zeros(len(current_graphs), dtype=np.float64)

    def __call__(self, local_states, counts, global_state, round_number):
        del round_number
        updates = torch.stack([_flatten_delta(state, global_state) for state in local_states])
        median = updates.median(dim=0).values
        distances = np.asarray([float((update - median).norm().item()) for update in updates])
        update_risk = _robust_positive(distances)
        risk = self.ALPHA_TOPOLOGY * self.topology_risk + self.ALPHA_UPDATE * update_risk
        trust = np.clip(np.exp(-3.0 * risk), 0.02, 1.0)
        weights = trust * np.asarray(counts, dtype=np.float64)
        weights /= weights.sum()
        diagnostics: dict[str, float] = {}
        for client_id in range(len(local_states)):
            diagnostics[f"etdfgl_topology_risk_client_{client_id}"] = float(self.topology_risk[client_id])
            diagnostics[f"etdfgl_explanation_risk_client_{client_id}"] = float(self.explanation_risk[client_id])
            diagnostics[f"etdfgl_update_risk_client_{client_id}"] = float(update_risk[client_id])
            diagnostics[f"etdfgl_fused_risk_client_{client_id}"] = float(risk[client_id])
            diagnostics[f"etdfgl_trust_client_{client_id}"] = float(trust[client_id])
            diagnostics[f"etdfgl_weight_client_{client_id}"] = float(weights[client_id])
        return _weighted_average(local_states, weights), diagnostics


class ConfigurableETDFGLAggregator:
    """ETD-FGL aggregator with independently controllable evidence components.

    DEFAULT configuration represents the FROZEN PRODUCTION METHOD:
        topology_weight: 9.0 / 13.0
        explanation_weight: 0.0
        update_weight: 4.0 / 13.0
        trust_strength: 3.0
        use_trust_weighting: True

    Historical or alternative ablation configurations (such as the three-channel
    development variant with explanation_weight > 0) can still be explicitly
    instantiated by passing non-default weights.
    """

    FROZEN_DEFAULT_TOPOLOGY: float = 9.0 / 13.0
    FROZEN_DEFAULT_EXPLANATION: float = 0.0
    FROZEN_DEFAULT_UPDATE: float = 4.0 / 13.0

    def __init__(
        self,
        clean_graphs: list[Data],
        current_graphs: list[Data],
        attacked_model: GCN | None = None,
        config: DatasetConfig | None = None,
        *,
        topology_weight: float = 9.0 / 13.0,
        explanation_weight: float = 0.0,
        update_weight: float = 4.0 / 13.0,
        trust_strength: float = 3.0,
        use_trust_weighting: bool = True,
        compute_explanation_diagnostic: bool = False,
    ) -> None:
        raw_weights = np.asarray(
            [topology_weight, explanation_weight, update_weight],
            dtype=np.float64,
        )
        if np.any(raw_weights < 0):
            raise ValueError("ETD-FGL evidence weights must be non-negative.")
        if raw_weights.sum() <= 0 and use_trust_weighting:
            raise ValueError("At least one ETD-FGL evidence weight must be positive.")
        self.component_weights = (
            raw_weights / raw_weights.sum()
            if raw_weights.sum() > 0
            else raw_weights
        )
        self.trust_strength = float(trust_strength)
        self.use_trust_weighting = bool(use_trust_weighting)

        clean = np.stack([_graph_descriptor(graph) for graph in clean_graphs])
        current = np.stack([_graph_descriptor(graph) for graph in current_graphs])
        scale = np.std(clean, axis=0)
        scale[scale < 1e-8] = 1.0
        drift = np.linalg.norm((current - clean) / scale, axis=1)
        self.topology_risk = _robust_positive(drift)

        if (
            (self.component_weights[1] > 0 or compute_explanation_diagnostic)
            and attacked_model is not None
            and config is not None
        ):
            explanation_raw = np.asarray([
                _explanation_score(attacked_model, graph, config)
                for graph in current_graphs
            ])
            self.explanation_risk = _robust_positive(explanation_raw)
        else:
            # Avoid paying GNNExplainer cost when explanation is excluded and diagnostic is not requested.
            self.explanation_risk = np.zeros(len(current_graphs), dtype=np.float64)

    def __call__(self, local_states, counts, global_state, round_number):
        del round_number
        updates = torch.stack([
            _flatten_delta(state, global_state)
            for state in local_states
        ])
        median = updates.median(dim=0).values
        distances = np.asarray([
            float((update - median).norm().item())
            for update in updates
        ])
        update_risk = _robust_positive(distances)

        risk = (
            self.component_weights[0] * self.topology_risk
            + self.component_weights[1] * self.explanation_risk
            + self.component_weights[2] * update_risk
        )

        if self.use_trust_weighting:
            trust = np.clip(
                np.exp(-self.trust_strength * risk),
                0.02,
                1.0,
            )
        else:
            trust = np.ones(len(counts), dtype=np.float64)

        weights = trust * np.asarray(counts, dtype=np.float64)
        weights /= weights.sum()

        diagnostics: dict[str, float] = {
            "etdfgl_topology_component_weight": float(self.component_weights[0]),
            "etdfgl_explanation_component_weight": float(self.component_weights[1]),
            "etdfgl_update_component_weight": float(self.component_weights[2]),
            "etdfgl_trust_strength": float(self.trust_strength),
            "etdfgl_use_trust_weighting": float(self.use_trust_weighting),
        }
        for client_id in range(len(local_states)):
            diagnostics[f"etdfgl_topology_risk_client_{client_id}"] = float(self.topology_risk[client_id])
            diagnostics[f"etdfgl_explanation_risk_client_{client_id}"] = float(self.explanation_risk[client_id])
            diagnostics[f"etdfgl_update_risk_client_{client_id}"] = float(update_risk[client_id])
            diagnostics[f"etdfgl_fused_risk_client_{client_id}"] = float(risk[client_id])
            diagnostics[f"etdfgl_trust_client_{client_id}"] = float(trust[client_id])
            diagnostics[f"etdfgl_weight_client_{client_id}"] = float(weights[client_id])
        return _weighted_average(local_states, weights), diagnostics
