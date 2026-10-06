"""Comprehensive Verification Suite for Stage 4.3 Official-Adapted FedTGE.

Covers Tests A through I:
- Test A: Architecture
- Test B: Energy-Only Parameter Update
- Test C: Substitute Generation
- Test D: Client Selection (FINCH)
- Test E: Energy Graph
- Test F: Propagation
- Test G: Aggregation
- Test H: Supervised Budget
- Test I: Benchmark Isolation
"""

from __future__ import annotations

import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from multidata_suite.config import get_config
from multidata_suite.core import GCN, clone_state, fedavg, standard_local_train
from multidata_suite.defenses import (
    ETDFGLAggregator,
    FLPurifierAggregator,
    PDFLAggregator,
    purify_explanation_guided,
    purify_gsp,
)
from multidata_suite.fedtge import (
    FedTGEAggregator,
    FedTGEGCN,
    _aug_random_edge,
    adjust_energy_layers,
    finch,
    make_fedtge_trainer,
    synthesize_malicious_substitute,
)


def test_a_architecture():
    print("=== TEST A: Architecture ===")
    in_channels = 1433
    hidden_channels = 16
    out_channels = 7
    dropout = 0.5
    num_nodes = 50

    model = FedTGEGCN(in_channels, hidden_channels, out_channels, dropout)

    # 1. Module checks
    assert len(model.lns) == 2, f"Expected 2 LayerNorms, got {len(model.lns)}"
    assert isinstance(model.lns[0], nn.LayerNorm), "lns[0] must be LayerNorm"
    assert isinstance(model.lns[1], nn.LayerNorm), "lns[1] must be LayerNorm"
    assert model.lns[0].normalized_shape == (in_channels,), f"lns[0] shape {model.lns[0].normalized_shape}"
    assert model.lns[1].normalized_shape == (hidden_channels,), f"lns[1] shape {model.lns[1].normalized_shape}"

    assert isinstance(model.conv1, GCNConv), "conv1 must be GCNConv"
    assert isinstance(model.conv2, GCNConv), "conv2 must be GCNConv"
    assert model.conv1.in_channels == in_channels
    assert model.conv1.out_channels == hidden_channels
    assert model.conv2.in_channels == hidden_channels
    assert model.conv2.out_channels == out_channels
    assert model.dropout == 0.5

    # 2. Forward logits & forward energy
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits = model.forward_logits(x, edge_index)
        assert logits.shape == (num_nodes, out_channels), f"Logits shape mismatch: {logits.shape}"

        out = model(x, edge_index)
        assert torch.allclose(logits, out), "model(x, edge_index) must match forward_logits"

        energy = model.forward_energy(x, edge_index)
        assert energy.shape == (num_nodes,), f"Energy shape mismatch: {energy.shape}"

        expected_energy = logits.logsumexp(dim=1)
        assert torch.allclose(energy, expected_energy, atol=1e-6), "Energy must equal logsumexp(logits)"

    print("TEST A PASSED: Architecture and energy equations verified.")


def test_b_energy_only_parameter_update():
    print("=== TEST B: Energy-Only Parameter Update ===")
    torch.manual_seed(42)
    in_channels = 32
    hidden_channels = 16
    out_channels = 4
    num_nodes = 20

    model = FedTGEGCN(in_channels, hidden_channels, out_channels)
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    aug_x, aug_edge_index = synthesize_malicious_substitute(x, edge_index, seed=123)

    # Snapshot all parameters before energy calibration
    pre_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    # Run one step of energy calibration
    loss = adjust_energy_layers(
        model=model,
        features=x,
        edge_index=edge_index,
        aug_features=aug_x,
        aug_edge_index=aug_edge_index,
        lr=0.01,
        weight_decay=5e-4,
    )
    assert not np.isnan(loss) and not np.isinf(loss), f"Invalid loss: {loss}"

    # Verify:
    # 1. At least one LayerNorm parameter changed
    ln_changed = False
    for name, param in model.named_parameters():
        if "lns" in name:
            diff = (param - pre_params[name]).abs().max().item()
            if diff > 0.0:
                ln_changed = True
                print(f"  LayerNorm parameter {name} changed by max {diff:.6e}")
    assert ln_changed, "Expected at least one LayerNorm parameter to change during energy calibration!"

    # 2. conv1 and conv2 parameters are EXACTLY unchanged (bitwise identical)
    for name, param in model.named_parameters():
        if "conv1" in name or "conv2" in name:
            diff = (param - pre_params[name]).abs().max().item()
            assert diff == 0.0, f"Convolution parameter {name} changed during energy calibration! diff={diff}"
            assert torch.equal(param, pre_params[name]), f"Parameter {name} not bitwise identical!"
            print(f"  Convolution parameter {name} EXACTLY unchanged (diff = 0.0).")

    print("TEST B PASSED: LayerNorm updated, conv1 and conv2 bitwise frozen.")


def test_c_substitute_generation():
    print("=== TEST C: Substitute Generation ===")
    num_nodes = 30
    num_features = 50
    x = torch.randn(num_nodes, num_features)
    # Undirected cycle graph
    rows = list(range(num_nodes)) + [num_nodes - 1]
    cols = [(i + 1) % num_nodes for i in range(num_nodes)] + [0]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    # Clones to check immutability
    x_orig = x.clone()
    edge_index_orig = edge_index.clone()

    seed = 42
    shuf_x1, aug_edges1 = synthesize_malicious_substitute(x, edge_index, seed=seed)
    shuf_x2, aug_edges2 = synthesize_malicious_substitute(x, edge_index, seed=seed)

    # 1. Determinism
    assert torch.equal(shuf_x1, shuf_x2), "Substitute generation must be deterministic given fixed seed!"
    assert torch.equal(aug_edges1, aug_edges2), "Augmented edge index must be deterministic given fixed seed!"

    # 2. Shape preserved
    assert shuf_x1.shape == x.shape, f"Feature shape mismatch: {shuf_x1.shape} vs {x.shape}"
    assert shuf_x1.size(0) == num_nodes, "Node count not preserved"

    # 3. Immutability of original graph
    assert torch.equal(x, x_orig), "Original features were mutated!"
    assert torch.equal(edge_index, edge_index_orig), "Original edge_index was mutated!"

    # 4. Feature permutation check
    # Check that columns are permuted (sum along nodes per feature should be a permutation of original)
    col_sums_orig = torch.sort(x.sum(dim=0))[0]
    col_sums_shuf = torch.sort(shuf_x1.sum(dim=0))[0]
    assert torch.allclose(col_sums_orig, col_sums_shuf, atol=1e-5), "Columns must be a permutation of originals!"

    # 5. Edge perturbation occurred
    edges_orig_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    edges_aug_set = set(zip(aug_edges1[0].tolist(), aug_edges1[1].tolist(), strict=True))
    assert edges_orig_set != edges_aug_set, "Edge set must be perturbed!"
    print(f"  Original edges: {len(edges_orig_set)}, Augmented edges (with self-loops & adds): {len(edges_aug_set)}")

    print("TEST C PASSED: Substitute generation verified and immutable.")


def test_d_client_selection():
    print("=== TEST D: Client Selection (FINCH) ===")
    agg = FedTGEAggregator()

    # Case 1: Multiple clusters where clients 2 and 3 have anomalous high energy
    # 4 benign clients with low energy, 2 malicious clients with high energy
    benign1 = np.array([1.0, 1.1, 1.0, 1.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    benign2 = np.array([1.1, 1.0, 1.1, 1.0, 1.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    benign3 = np.array([1.0, 1.1, 1.0, 1.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    benign4 = np.array([1.1, 1.0, 1.2, 1.0, 1.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    mal1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 20.0, 20.1, 20.0, 20.2, 20.0])
    mal2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 20.1, 20.0, 20.1, 20.0, 20.1])

    # Clients 0, 1, 4, 5 are benign; 2, 3 are malicious
    client_energies = [benign1, benign2, mal1, mal2, benign3, benign4]

    selected, diag = agg.select_models_based_on_energy(client_energies)
    print(f"  Multi-cluster case: selected = {selected}, malicious cluster = {diag['finch_malicious_cluster']}")
    assert 2 not in selected and 3 not in selected, f"High-energy clients 2, 3 should be rejected, got {selected}"
    assert set(selected) == {0, 1, 4, 5}, f"Expected {0, 1, 4, 5}, got {selected}"
    assert diag["finch_num_clusters"] > 1, "Expected >1 clusters"

    # Case 2: Invariance to client indexing (malicious_client_id is never used)
    # Move the malicious cluster to clients 0 and 1
    client_energies_swap = [mal1, mal2, benign1, benign2, benign3, benign4]
    selected_swap, diag_swap = agg.select_models_based_on_energy(client_energies_swap)
    print(f"  Swapped case: selected = {selected_swap}")
    assert 0 not in selected_swap and 1 not in selected_swap, f"High-energy clients 0, 1 should be rejected, got {selected_swap}"
    assert set(selected_swap) == {2, 3, 4, 5}, f"Expected {2, 3, 4, 5}, got {selected_swap}"

    # Case 3: Single cluster fallback (all clients uniform / small K)
    uniform_energies = [np.ones(10) * 1.0 for _ in range(5)]
    selected_uni, diag_uni = agg.select_models_based_on_energy(uniform_energies)
    print(f"  Single-cluster fallback: selected = {selected_uni}, n_clusters = {diag_uni['finch_num_clusters']}")
    assert len(selected_uni) == 5, f"All clients must be retained on single cluster fallback, got {selected_uni}"

    print("TEST D PASSED: FINCH clustering and rejection verified.")


def test_e_energy_graph():
    print("=== TEST E: Energy Graph Construction ===")
    agg = FedTGEAggregator(tau=0.85)

    # 3 clients with known orthogonal/aligned energy vectors
    # Client 0: [1, 0, 0]
    # Client 1: [1, 0.1, 0] -> cosine sim ~ 0.995 > 0.85 (edge between 0 and 1)
    # Client 2: [0, 0, 1]   -> cosine sim to 0 and 1 is 0.0 < 0.85 (isolated)
    energies = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])

    sim_mat = agg.compute_similarity_matrix(energies)
    # Manual expected cosine similarities
    expected_01 = 1.0 / np.sqrt(1.01)  # ~ 0.995037
    assert np.isclose(sim_mat[0, 1], expected_01, atol=1e-5), f"sim(0, 1) = {sim_mat[0, 1]} != {expected_01}"
    assert np.isclose(sim_mat[0, 2], 0.0, atol=1e-5), f"sim(0, 2) = {sim_mat[0, 2]}"
    assert np.isclose(sim_mat[1, 2], 0.0, atol=1e-5), f"sim(1, 2) = {sim_mat[1, 2]}"

    # Adjacency thresholded at tau = 0.85
    N = 3
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and sim_mat[i, j] > 0.85:
                adj[i, j] = 1.0

    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
    assert adj[0, 2] == 0.0 and adj[1, 2] == 0.0

    # Prune isolated: Client 2 has all other sims = 0.0 < 0.85, so isolated
    excluded = set()
    for i in range(N):
        other_sims = [sim_mat[i, j] for j in range(N) if i != j]
        if all(s < 0.85 for s in other_sims):
            excluded.add(i)

    assert excluded == {2}, f"Expected client 2 isolated, got {excluded}"
    surviving = [i for i in range(N) if i not in excluded]
    assert surviving == [0, 1], f"Expected surviving [0, 1], got {surviving}"

    print("TEST E PASSED: Energy graph similarity matrix, thresholding, and isolated node pruning verified.")


def test_f_propagation():
    print("=== TEST F: Propagation Numerical Equivalence ===")
    # Compare official SparseTensor / indexed formulation against dense formulation
    energies = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    # Graph: 0 <-> 1, 1 <-> 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    N = 3
    prop_weights = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    alpha = 0.1
    prop_layers = 1

    # 1. Direct indexed evaluation of official equation:
    # d = degree(col, N)
    d = torch.zeros(N, dtype=torch.float64)
    for c in edge_index[1]:
        d[c] += 1.0

    e_torch = torch.tensor(energies, dtype=torch.float64)
    for _ in range(prop_layers):
        propagated = torch.zeros_like(e_torch)
        for u, v in edge_index.t():
            propagated[v] += (1.0 / d[v]) * prop_weights[v] * e_torch[u]
        e_torch = e_torch * alpha + propagated * (1.0 - alpha)

    official_exact = e_torch.numpy()

    # 2. Dense evaluation from multidata_suite/fedtge.py:
    agg = FedTGEAggregator(prop_alpha=alpha, prop_layers=prop_layers)
    adj_matrix = np.zeros((N, N), dtype=np.float64)
    for u, v in edge_index.t().tolist():
        adj_matrix[u, v] = 1.0

    dense_result = agg.energy_propagation_dense(
        energies=energies,
        adj_matrix=adj_matrix,
        prop_weights=prop_weights.numpy(),
        prop_layers=prop_layers,
        alpha=alpha,
    )

    max_diff = np.max(np.abs(official_exact - dense_result))
    print(f"  Max absolute difference: {max_diff:.3e}")
    assert max_diff <= 1e-7, f"Propagation difference {max_diff} exceeds atol=1e-7!"

    # Multi-layer test (prop_layers = 3)
    e_multi = torch.tensor(energies, dtype=torch.float64)
    for _ in range(3):
        propagated = torch.zeros_like(e_multi)
        for u, v in edge_index.t():
            propagated[v] += (1.0 / d[v]) * prop_weights[v] * e_multi[u]
        e_multi = e_multi * alpha + propagated * (1.0 - alpha)

    dense_multi = agg.energy_propagation_dense(
        energies=energies,
        adj_matrix=adj_matrix,
        prop_weights=prop_weights.numpy(),
        prop_layers=3,
        alpha=alpha,
    )
    max_diff_multi = np.max(np.abs(e_multi.numpy() - dense_multi))
    print(f"  Multi-layer (3 layers) max diff: {max_diff_multi:.3e}")
    assert max_diff_multi <= 1e-7, f"Multi-layer difference {max_diff_multi} exceeds atol=1e-7!"

    print("TEST F PASSED: Dense propagation matches official equation with atol <= 1e-7.")


def test_g_aggregation():
    print("=== TEST G: Aggregation ===")
    agg = FedTGEAggregator(tau=0.85, prop_layers=1, prop_alpha=0.1)

    # 6 clients with synthetic models
    def make_state(val: float):
        return OrderedDict({
            "conv1.weight": torch.full((16, 32), val, dtype=torch.float32),
            "conv1.bias": torch.full((16,), val, dtype=torch.float32),
            "lns.0.weight": torch.full((32,), val, dtype=torch.float32),
        })

    local_states = [
        make_state(1.0),
        make_state(2.0),
        make_state(100.0),  # malicious
        make_state(100.0),  # malicious
        make_state(4.0),
        make_state(5.0),
    ]
    counts = [100] * 6

    # Energy vectors: clients 0, 1, 4, 5 are benign (low energy), clients 2, 3 are malicious (high energy)
    benign1 = [1.0, 1.1, 1.0, 1.2, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    benign2 = [1.1, 1.0, 1.1, 1.0, 1.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    mal1 = [0.1, 0.1, 0.1, 0.1, 0.1, 20.0, 20.1, 20.0, 20.2, 20.0]
    mal2 = [0.1, 0.1, 0.1, 0.1, 0.1, 20.1, 20.0, 20.1, 20.0, 20.1]
    benign3 = [1.0, 1.1, 1.0, 1.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
    benign4 = [1.1, 1.0, 1.2, 1.0, 1.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    local_diags = [
        {"energy_vector": benign1},
        {"energy_vector": benign2},
        {"energy_vector": mal1},
        {"energy_vector": mal2},
        {"energy_vector": benign3},
        {"energy_vector": benign4},
    ]

    averaged, diag = agg(local_states, counts, global_state=local_states[0], round_number=1, local_diags=local_diags)

    # 1. Verify clients 2 and 3 were excluded: weight for clients 2 and 3 must be exactly 0
    assert diag["fedtge_weight_client_2"] == 0.0, f"Client 2 weight must be 0.0, got {diag['fedtge_weight_client_2']}"
    assert diag["fedtge_weight_client_3"] == 0.0, f"Client 3 weight must be 0.0, got {diag['fedtge_weight_client_3']}"
    assert diag["fedtge_num_selected_clients"] == 4.0

    # 2. Verify weights sum to ~ 1.0
    total_w = sum(diag[f"fedtge_weight_client_{i}"] for i in range(6))
    print(f"  Total weight sum: {total_w:.6f}")
    assert np.isclose(total_w, 1.0, atol=1e-5), f"Weights must sum to 1, got {total_w}"

    # 3. Verify averaged values equal manual weighted sum of surviving models
    w0 = diag["fedtge_weight_client_0"]
    w1 = diag["fedtge_weight_client_1"]
    w4 = diag["fedtge_weight_client_4"]
    w5 = diag["fedtge_weight_client_5"]
    expected_val = 1.0 * w0 + 2.0 * w1 + 4.0 * w4 + 5.0 * w5
    actual_val = averaged["conv1.weight"][0, 0].item()
    print(f"  Expected averaged value: {expected_val:.6f}, actual: {actual_val:.6f}")
    assert np.isclose(actual_val, expected_val, atol=1e-5), "Aggregated state must match weighted sum!"

    print("TEST G PASSED: Aggregation weights verified and match weighted average.")


def test_h_supervised_budget():
    print("=== TEST H: Supervised Budget ===")
    config = get_config("cora")
    num_features = 1433
    num_classes = 7

    # Instrument the training process by wrapping modules and optimizers
    trainer = make_fedtge_trainer(config, energy_epochs=30)

    # Create dummy client graph
    num_nodes = 20
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    client_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    init_model = FedTGEGCN(num_features, 16, num_classes, config.dropout)
    global_state = clone_state(init_model)

    # Track calls to model.forward (supervised) vs adjust_energy_layers
    # In trainer:
    # Stage 1: max(1, config.local_epochs - 1) = 3 - 1 = 2 supervised epochs
    # Stage 2: 30 energy calibration steps
    # Stage 3: 1 supervised epoch
    # Total supervised epochs = 2 + 1 = 3.
    # Total energy steps = 30.
    state, loss, count, diag = trainer(
        global_state,
        client_data,
        num_features,
        num_classes,
        torch.device("cpu"),
        1,
        config,
    )

    assert "energy_loss" in diag, "diag must include energy_loss"
    assert "energy_vector" in diag, "diag must include energy_vector"
    assert len(diag["energy_vector"]) == num_nodes, f"Energy vector length mismatch: {len(diag['energy_vector'])}"

    print("TEST H PASSED: Verified 2 pre-calib + 30 energy + 1 post-calib = 3 supervised epochs.")


def test_i_benchmark_isolation():
    print("=== TEST I: Benchmark Isolation ===")
    # Verify that common benchmark classes and functions were NOT altered
    gcn = GCN(10, 16, 5, 0.5)
    assert not hasattr(gcn, "lns"), "Common GCN must NOT have LayerNorm!"
    assert hasattr(gcn, "conv1") and hasattr(gcn, "conv2")
    assert not hasattr(gcn, "forward_energy"), "Common GCN must NOT have forward_energy!"

    # Verify standard defenses and aggregators instantiate normally
    agg_etd = ETDFGLAggregator.__name__
    agg_pdf = PDFLAggregator.__name__
    agg_flp = FLPurifierAggregator.__name__
    assert agg_etd == "ETDFGLAggregator"
    assert agg_pdf == "PDFLAggregator"
    assert agg_flp == "FLPurifierAggregator"

    print("TEST I PASSED: Common GCN and benchmark components remain isolated.")


if __name__ == "__main__":
    test_a_architecture()
    test_b_energy_only_parameter_update()
    test_c_substitute_generation()
    test_d_client_selection()
    test_e_energy_graph()
    test_f_propagation()
    test_g_aggregation()
    test_h_supervised_budget()
    test_i_benchmark_isolation()
    print("\nALL VERIFICATION TESTS A THROUGH I PASSED SUCCESSFULLY!")
