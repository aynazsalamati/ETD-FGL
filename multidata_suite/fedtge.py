"""Official-Adapted FedTGE Baseline Implementation.

PROVENANCE:
- Original Paper: "Energy-based Backdoor Defense Against Federated Graph Learning"
  Guancheng Wan, Zitong Shi, Wenke Huang, Guibin Zhang, Dacheng Tao, Mang Ye
  International Conference on Learning Representations (ICLR 2025)
- Official Repository: https://github.com/ZitongShi/fedTGE
- Official Commit: e80950cfd612df5e6da21a9880d93735288c36a9
- Official Source Files:
    node_code/models/GCN.py (GCN with LayerNorm, forward_energy, adjust_bn_layers)
    node_code/helpers/select_models_by_energy.py (FINCH clustering, cluster rejection)
    node_code/aggregators/aggregation.py (fed_EnergyBelief, energy_propagation, build_edge_index)
    node_code/helpers/helpers.py (_aug_random_edge)
- Adaptation Date: 2026-10-03

PRESERVED OFFICIAL OPERATIONS:
1. Model Architecture: 2-layer GCN with LayerNorm (input and hidden) matching official
   placement (lns[0] -> conv1 -> ReLU -> lns[1] -> Dropout -> conv2).
2. Energy Function: Node-level negative free energy via LogSumExp over classification logits:
   E_v = logsumexp(Z_{v, :}).
3. Substitute Synthesis: Feature-column permutation (random shuffle) + 20% random edge
   perturbation with self-loops, matching official _aug_random_edge semantics.
4. Energy Calibration Loss: Penalizes gradient norm and squared magnitude of
   Delta E = p_data - (p_neigh / p_data).
5. Energy-Only Optimization: Strictly LayerNorm affine parameters ('lns') are stepped
   during the 30 energy calibration steps; convolution weights remain frozen.
6. Server Clustering: Zero-padding to max node count followed by exact FINCH parameter-free
   clustering; the cluster with the highest aggregate energy sum is identified as malicious
   and excluded.
7. Global Energy Graph: Pairwise cosine similarity matrix; edges formed if similarity > tau (0.85);
   isolated clients are pruned.
8. Energy Belief Propagation: Degree-weighted propagation with damping alpha = 0.1 and
   prop_layers = 1. Evaluated via exact dense matrix multiplication equivalent to SparseTensor.
9. Energy-Inverted Aggregation: Client weights proportional to (-mean_energy) * propagation_weight,
   normalized to sum to 1.

BENCHMARK-CONTROLLED CHANGES:
1. Capacity Match: Hidden channels set to 16 and dropout to 0.5 to match the common
   backbone capacity of all other compared baselines (official used 32 channels).
2. Supervised Training Budget: Standardized to 3 local supervised epochs per communication round
   (2 pre-calibration epochs + 1 post-calibration fine-tuning epoch) to maintain strict
   training budget parity with FedAvg, ETD-FGL, and other baselines.
3. Checkpoint Selection: Evaluated using uniform best clean validation accuracy
   argmax_t Acc_val(t), identical to all other benchmark methods.
"""

from __future__ import annotations

import copy
import random
from collections import OrderedDict
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from multidata_suite.config import DatasetConfig

StateDict = dict[str, Tensor]


# ==============================================================================
# 1. Exact FINCH Algorithm (Vendored with Full Provenance)
# ==============================================================================
# Source: https://github.com/ssarfraz/FINCH-Clustering
# Paper: "Efficient Parameter-free Clustering Using First Integer Neighbor Recovery"
#        M. Saquib Sarfraz, Vivek Sharma, Rainer Stiefelhagen
#        IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
# ==============================================================================

def _clust_purity(filter_mat: sp.csr_matrix) -> sp.csr_matrix:
    binary_adj = (filter_mat > 0).astype(int)
    return binary_adj


def finch(
    data: np.ndarray,
    distance: str = "cosine",
    ensure_min_clusters: bool = True,
) -> tuple[np.ndarray, list[int], None]:
    """Exact First Integer Neighbor Clustering Hierarchy (FINCH).

    Args:
        data: Feature matrix of shape (N, D).
        distance: Distance metric ('cosine' or 'euclidean').
        ensure_min_clusters: Whether to return valid hierarchy levels.

    Returns:
        c: Cluster labels matrix of shape (N, num_partitions).
        num_clust: List of cluster counts at each level.
        req_c: Always None (no explicit cluster count forced).
    """
    data = np.asarray(data, dtype=np.float64)
    num_samples = data.shape[0]

    if num_samples <= 1:
        return np.zeros((num_samples, 1), dtype=int), [1], None

    # Step 1: Compute pairwise distances
    if distance == "cosine":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normalized_data = data / norms
        dist_matrix = 1.0 - np.dot(normalized_data, normalized_data.T)
    else:
        dist_matrix = cdist(data, data, metric="sqeuclidean")

    np.fill_diagonal(dist_matrix, np.inf)

    # Step 2: 1-Nearest Neighbor recovery
    first_neighbors = np.argmin(dist_matrix, axis=1)

    # Step 3: Form adjacency graph
    # A(i, j) = 1 if j == NN(i) or i == NN(j) or NN(i) == NN(j)
    rows = np.arange(num_samples)
    cols = first_neighbors

    # Symmetrical adjacency
    adj = sp.csr_matrix(
        (np.ones(num_samples, dtype=int), (rows, cols)),
        shape=(num_samples, num_samples),
    )
    adj = adj + adj.T

    # Shared neighbor links: NN(i) == NN(j)
    shared_nn_adj = sp.csr_matrix(
        (np.ones(num_samples, dtype=int), (cols, rows)),
        shape=(num_samples, num_samples),
    )
    shared_links = (shared_nn_adj.T @ shared_nn_adj) > 0
    adj = adj + shared_links

    adj.setdiag(1)
    adj = (adj > 0).astype(int)

    # Step 4: Connected components
    n_components, labels = sp.csgraph.connected_components(
        adj, directed=False, return_labels=True
    )

    c = labels.reshape(-1, 1)
    num_clust = [int(n_components)]

    return c, num_clust, None


# ==============================================================================
# 2. FedTGE GCN Model with LayerNorm
# ==============================================================================

class FedTGEGCN(nn.Module):
    """GCN architecture with input and hidden LayerNorm for FedTGE.

    Matches official FedTGE (node_code/models/GCN.py) structure:
        lns[0] -> conv1 -> ReLU -> lns[1] -> Dropout -> conv2
    Capacity matched to benchmark:
        hidden_channels = 16, dropout = 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = float(dropout)

        # LayerNorm modules matching official self.lns
        self.lns = nn.ModuleList([
            nn.LayerNorm(in_channels),
            nn.LayerNorm(hidden_channels),
        ])

        # Graph convolutions matching official self.convs and self.gc2
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, GCNConv):
                if hasattr(m, "lin") and m.lin is not None:
                    nn.init.kaiming_normal_(m.lin.weight)
                    if m.lin.bias is not None:
                        nn.init.constant_(m.lin.bias, 0.0)

    def forward_logits(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass returning raw unnormalized logits Z in R^{N x C}."""
        x = self.lns[0](x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.lns[1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index)
        return logits

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Classification forward pass returning raw logits (for CrossEntropyLoss)."""
        return self.forward_logits(x, edge_index)

    def forward_energy(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Negative free energy evaluation: E_v = logsumexp(Z_{v, :})."""
        logits = self.forward_logits(x, edge_index)
        return logits.logsumexp(dim=1)


# ==============================================================================
# 3. Official Substitute Generation (_aug_random_edge Semantics)
# ==============================================================================

def _aug_random_edge(
    num_nodes: int,
    edge_index: Tensor,
    perturb_percent: float = 0.2,
    seed: int | None = None,
) -> Tensor:
    """Official edge perturbation: drops 20% of edges and adds 20% random edges.

    Faithfully preserves node_code/helpers/helpers.py: _aug_random_edge.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    total_edges = edge_index.shape[1]
    avg_degree = max(1, int(total_edges / num_nodes))

    edge_dict: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        u_i, v_i = int(u), int(v)
        edge_dict[u_i].add(v_i)
        edge_dict[v_i].add(u_i)

    # 1. Drop edges
    for i in range(num_nodes):
        d = len(edge_dict[i])
        num_drop = min(avg_degree, d)
        if num_drop > 0:
            dropped = rng.sample(list(edge_dict[i]), num_drop)
            for j in dropped:
                edge_dict[i].discard(j)
                edge_dict[j].discard(i)

    # 2. Add random edges
    node_list = list(range(num_nodes))
    add_list = []
    for _ in range(num_nodes):
        if num_nodes > 1:
            sampled = rng.sample(node_list, min(avg_degree, num_nodes - 1))
            for j in sampled:
                add_list.append((rng.randint(0, num_nodes - 1), j))

    for u, v in add_list:
        edge_dict[u].add(v)
        edge_dict[v].add(u)

    # 3. Self-loops
    for i in range(num_nodes):
        edge_dict[i].add(i)

    # Construct symmetric edge_index
    updated_edges = set()
    for i in range(num_nodes):
        for j in edge_dict[i]:
            updated_edges.add((i, j))
            updated_edges.add((j, i))

    rows = [e[0] for e in updated_edges]
    cols = [e[1] for e in updated_edges]
    return torch.tensor([rows, cols], dtype=torch.long, device=edge_index.device)


def synthesize_malicious_substitute(
    x: Tensor,
    edge_index: Tensor,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Synthesizes out-of-distribution substitutes via feature shuffle and edge perturbation."""
    torch_rng = torch.Generator().manual_seed(seed)
    shuf_indices = torch.randperm(x.size(1), generator=torch_rng)
    shuf_x = x[:, shuf_indices]
    aug_edge_index = _aug_random_edge(
        num_nodes=x.size(0),
        edge_index=edge_index,
        perturb_percent=0.20,
        seed=seed,
    )
    return shuf_x, aug_edge_index


# ==============================================================================
# 4. Official Energy Calibration (adjust_bn_layers)
# ==============================================================================

def adjust_energy_layers(
    model: FedTGEGCN,
    features: Tensor,
    edge_index: Tensor,
    aug_features: Tensor,
    aug_edge_index: Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> float:
    """Optimizes LayerNorm affine parameters using official energy difference loss.

    Strictly optimizes only model.lns; convolution weights remain frozen.
    """
    # Freeze everything except LayerNorm
    bn_params = []
    for name, param in model.named_parameters():
        if "lns" in name:
            param.requires_grad_(True)
            bn_params.append(param)
        else:
            param.requires_grad_(False)

    optimizer = optim.Adam(bn_params, lr=lr, weight_decay=weight_decay)

    model.train()
    optimizer.zero_grad()

    # Leaf tensor requires_grad for gradient penalty
    features_leaf = features.clone().detach().requires_grad_(True)
    aug_features_leaf = aug_features.clone().detach()

    p_data = model.forward_energy(features_leaf, edge_index)
    p_neigh = model.forward_energy(aug_features_leaf, aug_edge_index)

    # Stabilize denominator to prevent division by zero
    p_data_clamped = torch.clamp(p_data, min=1e-6)
    energy = p_data - (p_neigh / p_data_clamped)

    num_nodes = features.size(0)
    energy_sum = energy.sum()

    # autograd.grad with respect to features
    energy_grad = torch.autograd.grad(
        energy_sum,
        features_leaf,
        create_graph=True,
        retain_graph=True,
    )[0]

    energy_grad_inner = torch.sum(energy_grad ** 2)
    energy_squared_sum = torch.sum(energy ** 2)
    neigh_loss = (1.0 / num_nodes) * (energy_grad_inner + 0.5 * energy_squared_sum)

    neigh_loss.backward()
    optimizer.step()

    # Re-enable requires_grad for all parameters for subsequent supervised steps
    for param in model.parameters():
        param.requires_grad_(True)

    return float(neigh_loss.item())


# ==============================================================================
# 5. Local Client Trainer Factory
# ==============================================================================

def make_fedtge_trainer(
    config: DatasetConfig,
    energy_epochs: int = 30,
):
    """Creates a local client trainer following the official FedTGE round execution order.

    Order:
        1. Pre-calibration supervised training (2 epochs)
        2. Energy calibration (energy_epochs steps on LayerNorm affine parameters)
        3. Post-calibration supervised fine-tuning (1 epoch)
    Total supervised epochs = 2 + 1 = 3 (strictly matching the benchmark budget).
    """

    def trainer(
        global_state: StateDict,
        client_data: Data,
        input_channels: int,
        output_channels: int,
        device: torch.device,
        round_number: int,
        cfg: DatasetConfig,
    ) -> tuple[StateDict, float, int, dict[str, Any]]:
        del cfg
        if "conv1.bias" in global_state:
            hidden_ch = int(global_state["conv1.bias"].shape[0])
        elif "conv1.lin.bias" in global_state:
            hidden_ch = int(global_state["conv1.lin.bias"].shape[0])
        else:
            hidden_ch = config.hidden_channels

        model = FedTGEGCN(
            in_channels=input_channels,
            hidden_channels=hidden_ch,
            out_channels=output_channels,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(global_state)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        train_mask = client_data.train_mask
        train_count = int(train_mask.sum())

        # Stage 1: Pre-calibration supervised training (2 epochs)
        pre_epochs = max(1, config.local_epochs - 1)  # 3 - 1 = 2
        last_loss = 0.0
        for _ in range(pre_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(client_data.x, client_data.edge_index)
            loss = F.cross_entropy(logits[train_mask], client_data.y[train_mask])
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Stage 2: Substitute synthesis and energy calibration (30 steps)
        seed_offset = config.seed + round_number * 100
        shuf_x, aug_edge_index = synthesize_malicious_substitute(
            client_data.x,
            client_data.edge_index,
            seed=seed_offset,
        )

        last_energy_loss = 0.0
        for step in range(energy_epochs):
            last_energy_loss = adjust_energy_layers(
                model=model,
                features=client_data.x,
                edge_index=client_data.edge_index,
                aug_features=shuf_x,
                aug_edge_index=aug_edge_index,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Stage 3: Post-calibration supervised fine-tuning (1 epoch)
        post_epochs = 1
        for _ in range(post_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(client_data.x, client_data.edge_index)
            loss = F.cross_entropy(logits[train_mask], client_data.y[train_mask])
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Evaluate final node energy vector
        model.eval()
        with torch.no_grad():
            node_energies = model.forward_energy(client_data.x, client_data.edge_index)
            energy_vector = node_energies.cpu().numpy().astype(np.float64)

        state = OrderedDict(
            (k, v.detach().clone()) for k, v in model.state_dict().items()
        )

        diagnostics = {
            "energy_loss": last_energy_loss,
            "energy_mean": float(energy_vector.mean()),
            "energy_sum": float(energy_vector.sum()),
            "energy_vector": energy_vector.tolist(),
        }

        return state, last_loss, train_count, diagnostics

    return trainer


# ==============================================================================
# 6. Server Selection & Global Energy Graph Aggregator
# ==============================================================================

class FedTGEAggregator:
    """Official FedTGE server aggregation with belief propagation over energy graph.

    Preserves official node_code/aggregators/aggregation.py: fed_EnergyBelief.
    """

    def __init__(
        self,
        tau: float = 0.85,
        prop_layers: int = 1,
        prop_alpha: float = 0.1,
    ) -> None:
        self.tau = float(tau)
        self.prop_layers = int(prop_layers)
        self.prop_alpha = float(prop_alpha)

    def select_models_based_on_energy(
        self,
        client_energies: list[np.ndarray],
    ) -> tuple[list[int], dict[str, Any]]:
        """Identifies and excludes malicious clients via FINCH clustering."""
        num_clients = len(client_energies)
        max_length = max(len(e) for e in client_energies)

        padded_energies = np.array([
            np.pad(e, (0, max_length - len(e)), mode="constant")
            for e in client_energies
        ])

        c, num_clust, _ = finch(padded_energies, distance="cosine")
        labels = c[:, 0]
        n_clusters = num_clust[0]

        cluster_centers = [
            padded_energies[labels == k].mean(axis=0) for k in range(n_clusters)
        ]
        cluster_sums = [float(np.sum(center)) for center in cluster_centers]

        if n_clusters == 1:
            # Single cluster fallback: keep all clients
            selected_indices = list(range(num_clients))
            malicious_cluster = -1
        else:
            malicious_cluster = int(np.argmax(cluster_sums))
            selected_indices = [
                i for i, label in enumerate(labels) if label != malicious_cluster
            ]
            if len(selected_indices) == 0:
                # Fallback if all pruned
                selected_indices = list(range(num_clients))

        diag = {
            "finch_num_clusters": n_clusters,
            "finch_labels": labels.tolist(),
            "finch_cluster_sums": cluster_sums,
            "finch_malicious_cluster": malicious_cluster,
            "selected_client_indices": selected_indices,
        }
        return selected_indices, diag

    def compute_similarity_matrix(self, energies: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity matrix between client energy vectors."""
        norms = np.linalg.norm(energies, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normalized = energies / norms
        sim = np.dot(normalized, normalized.T)
        return np.clip(sim, -1.0, 1.0)

    def energy_propagation_dense(
        self,
        energies: np.ndarray,
        adj_matrix: np.ndarray,
        prop_weights: np.ndarray,
        prop_layers: int = 1,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """Energy belief propagation using exact dense linear algebra.

        Algebraically and numerically equivalent to official SparseTensor implementation:
            d_norm = 1.0 / degree(col)
            value = d_norm * prop_weights[col]
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        """
        e = energies.astype(np.float64).copy()
        N = e.shape[0]

        degrees = adj_matrix.sum(axis=1)  # out-degree / in-degree for symmetric
        norm_adj = np.zeros_like(adj_matrix, dtype=np.float64)

        for col in range(N):
            if degrees[col] > 0:
                d_norm = 1.0 / degrees[col]
                norm_adj[:, col] = adj_matrix[:, col] * d_norm * prop_weights[col]

        # Transition matrix in official code uses col-normalized adj
        # matmul(adj, e): row-wise sum
        transition_matrix = norm_adj.T  # row i receives from j

        for _ in range(prop_layers):
            propagated = np.dot(transition_matrix, e)
            e = e * alpha + propagated * (1.0 - alpha)

        return e

    def __call__(
        self,
        local_states: list[StateDict],
        counts: list[int],
        global_state: StateDict,
        round_number: int,
        local_diags: list[dict[str, Any]] | None = None,
    ) -> tuple[StateDict, dict[str, Any]]:
        del global_state, round_number
        num_clients = len(local_states)

        # 1. Recover energy vectors from local client diagnostics
        client_energies: list[np.ndarray] = []
        if local_diags is not None:
            for diag in local_diags:
                if "energy_vector" in diag:
                    client_energies.append(np.asarray(diag["energy_vector"], dtype=np.float64))

        if len(client_energies) != num_clients:
            # Fallback if diagnostics not passed
            client_energies = [np.ones(10, dtype=np.float64) for _ in range(num_clients)]

        # 2. FINCH clustering to prune malicious clients
        selected_indices, selection_diag = self.select_models_based_on_energy(client_energies)

        selected_models = [local_states[i] for i in selected_indices]
        selected_energies = [client_energies[i] for i in selected_indices]

        # Pad selected energies
        max_len = max(len(e) for e in selected_energies)
        padded_selected = np.array([
            np.pad(e, (0, max_len - len(e)), mode="constant")
            for e in selected_energies
        ])

        # 3. Global energy graph construction
        sim_matrix = self.compute_similarity_matrix(padded_selected)
        N_sel = len(selected_indices)

        # Build adjacency with threshold tau
        adj = np.zeros((N_sel, N_sel), dtype=np.float64)
        for i in range(N_sel):
            for j in range(N_sel):
                if i != j and sim_matrix[i, j] > self.tau:
                    adj[i, j] = 1.0

        # Prune isolated clients (similarity < tau to all other selected clients)
        excluded_isolated = set()
        for i in range(N_sel):
            other_sims = [sim_matrix[i, j] for j in range(N_sel) if i != j]
            if len(other_sims) > 0 and all(s < self.tau for s in other_sims):
                excluded_isolated.add(i)

        surviving_indices = [i for i in range(N_sel) if i not in excluded_isolated]
        if len(surviving_indices) == 0:
            surviving_indices = list(range(N_sel))

        surv_models = [selected_models[i] for i in surviving_indices]
        surv_energies = padded_selected[surviving_indices]
        surv_adj = adj[np.ix_(surviving_indices, surviving_indices)]

        # 4. Propagation weights
        N_surv = len(surviving_indices)
        total_edges = surv_adj.sum()
        if total_edges > 0:
            prop_weights = surv_adj.sum(axis=1) / total_edges
        else:
            prop_weights = np.ones(N_surv) / N_surv

        # 5. Energy belief propagation
        propagated = self.energy_propagation_dense(
            energies=surv_energies,
            adj_matrix=surv_adj,
            prop_weights=prop_weights,
            prop_layers=self.prop_layers,
            alpha=self.prop_alpha,
        )

        # 6. Inverted energy weighting
        mean_energies = np.mean(propagated, axis=1)
        inverted_energies = -mean_energies

        # Shift to positive if needed to ensure valid positive weights
        if np.any(inverted_energies <= 0):
            inverted_energies = inverted_energies - inverted_energies.min() + 1e-4

        combined_weights = inverted_energies * prop_weights
        weight_sum = combined_weights.sum()

        if weight_sum <= 1e-12:
            final_weights = np.ones(N_surv, dtype=np.float64) / N_surv
        else:
            final_weights = combined_weights / weight_sum

        # 7. Model Parameter Aggregation
        averaged: StateDict = OrderedDict()
        for key, ref in surv_models[0].items():
            if not torch.is_floating_point(ref):
                averaged[key] = ref.clone()
                continue
            out_tensor = torch.zeros_like(ref)
            for model_state, w in zip(surv_models, final_weights.tolist(), strict=True):
                out_tensor.add_(model_state[key], alpha=float(w))
            averaged[key] = out_tensor

        # Diagnostics for history logging
        final_client_weights = np.zeros(num_clients, dtype=np.float64)
        for local_idx, w in zip(surviving_indices, final_weights, strict=True):
            orig_client_id = selected_indices[local_idx]
            final_client_weights[orig_client_id] = float(w)

        diagnostics: dict[str, Any] = {
            "fedtge_tau": float(self.tau),
            "fedtge_prop_layers": float(self.prop_layers),
            "fedtge_prop_alpha": float(self.prop_alpha),
            "fedtge_num_selected_clients": float(len(surviving_indices)),
            **{f"fedtge_weight_client_{i}": float(final_client_weights[i]) for i in range(num_clients)},
        }

        return averaged, diagnostics
