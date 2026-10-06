from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
import networkx as nx

from multidata_suite.defenses import (
    _graph_descriptor,
    compute_homophily_diagnostics,
)
from test_topology_descriptor import (
    calculate_homophily,
    calculate_homophily_diagnostics,
    extract_topology_descriptor,
)


def _legacy_graph_descriptor(data: Data) -> np.ndarray:
    """Old implementation that suffered from label leakage across all edges."""
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from([
        (int(u), int(v))
        for u, v in data.edge_index.t().tolist()
        if int(u) != int(v)
    ])
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    clustering = nx.average_clustering(graph) if graph.number_of_edges() else 0.0
    triangles = sum(nx.triangles(graph).values()) / 3.0
    homophily_hits = 0
    edge_count = 0
    for u, v in graph.edges():
        if u < data.y.numel() and v < data.y.numel():
            homophily_hits += int(data.y[u] == data.y[v])
            edge_count += 1
    homophily = homophily_hits / edge_count if edge_count else 0.0
    return np.asarray([
        density,
        float(degrees.mean()) if degrees.size else 0.0,
        float(degrees.var() / (degrees.mean() ** 2 + 1e-12)) if degrees.size else 0.0,
        clustering,
        triangles / max(1.0, graph.number_of_nodes()),
        homophily,
    ], dtype=np.float64)


def test_a_validation_test_leakage():
    """TEST A: Validation and test label changes MUST NOT change homophily."""
    # Graph structure:
    # 0 -- 1 (train - train)
    # 2 -- 3 (val - val)
    # 4 -- 5 (test - test)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, 2, 5, 4],
    ], dtype=torch.long)

    train_mask = torch.tensor([True, True, False, False, False, False], dtype=torch.bool)
    val_mask   = torch.tensor([False, False, True, True, False, False], dtype=torch.bool)
    test_mask  = torch.tensor([False, False, False, False, True, True], dtype=torch.bool)

    y_initial = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    y_modified_val_test = torch.tensor([0, 0, 5, 6, 7, 8], dtype=torch.long)

    data1 = Data(edge_index=edge_index, y=y_initial, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=6)
    data2 = Data(edge_index=edge_index, y=y_modified_val_test, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=6)

    # 1. Under old leaky implementation, homophily DID change
    old_h1 = _legacy_graph_descriptor(data1)[5]
    old_h2 = _legacy_graph_descriptor(data2)[5]
    assert old_h1 != old_h2, "Sanity check: old implementation leaked val/test labels"
    assert math.isclose(old_h1, 1.0)
    assert math.isclose(old_h2, 1.0 / 3.0)

    # 2. Under new implementation, homophily MUST NOT change
    new_desc1 = _graph_descriptor(data1)
    new_desc2 = _graph_descriptor(data2)
    assert math.isclose(new_desc1[5], new_desc2[5]), f"Leakage detected in _graph_descriptor: {new_desc1[5]} != {new_desc2[5]}"
    assert math.isclose(new_desc1[5], 1.0)

    # 3. In test_topology_descriptor.calculate_homophily
    topo_h1 = calculate_homophily(data1)
    topo_h2 = calculate_homophily(data2)
    assert math.isclose(topo_h1, topo_h2), f"Leakage detected in calculate_homophily: {topo_h1} != {topo_h2}"
    assert math.isclose(topo_h1, 1.0)

    print("PASS: TEST A (validation/test leakage prevention)")


def test_b_training_label_sensitivity():
    """TEST B: Changing the label of an eligible training node SHOULD change homophily."""
    # Graph structure:
    # 0 -- 1 (train - train)
    # 2 -- 3 (val - val)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ], dtype=torch.long)

    train_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    val_mask   = torch.tensor([False, False, True, True], dtype=torch.bool)

    # Initial: 0 and 1 have same label
    y_same = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    # Flipped: 0 and 1 have different labels
    y_diff = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    data_same = Data(edge_index=edge_index, y=y_same, train_mask=train_mask, val_mask=val_mask, num_nodes=4)
    data_diff = Data(edge_index=edge_index, y=y_diff, train_mask=train_mask, val_mask=val_mask, num_nodes=4)

    h_same = _graph_descriptor(data_same)[5]
    h_diff = _graph_descriptor(data_diff)[5]

    assert math.isclose(h_same, 1.0), f"Expected 1.0, got {h_same}"
    assert math.isclose(h_diff, 0.0), f"Expected 0.0, got {h_diff}"
    assert h_same != h_diff

    # Also check test_topology_descriptor
    t_same = calculate_homophily(data_same)
    t_diff = calculate_homophily(data_diff)
    assert math.isclose(t_same, 1.0)
    assert math.isclose(t_diff, 0.0)

    print("PASS: TEST B (training-label sensitivity)")


def test_c_mixed_edge():
    """TEST C: Mixed edges (training node <-> test node) MUST NOT contribute to homophily."""
    # Nodes:
    # 0: train, label 1
    # 1: test, label 1   <-- edge (0, 1) is mixed with same label!
    # 2: train, label 0
    # 3: train, label 1   <-- edge (2, 3) is train-train with diff label!
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ], dtype=torch.long)

    train_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
    test_mask  = torch.tensor([False, True, False, False], dtype=torch.bool)
    y = torch.tensor([1, 1, 0, 1], dtype=torch.long)

    data = Data(edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask, num_nodes=4)

    # Under leaky code, edge (0, 1) was a hit (1==1) and edge (2, 3) was a miss (0!=1), so homophily was 1/2 = 0.5
    leaky_h = _legacy_graph_descriptor(data)[5]
    assert math.isclose(leaky_h, 0.5), "Sanity check on legacy behavior"

    # Under new code, edge (0, 1) is NOT eligible because node 1 is not in train_mask!
    # Only edge (2, 3) is eligible, and its endpoints have labels 0 != 1.
    # Therefore, eligible = 1, hits = 0, homophily = 0.0!
    desc, diag = _graph_descriptor(data, return_diagnostics=True)
    assert diag["total_edges"] == 2
    assert diag["homophily_eligible_edges"] == 1
    assert diag["homophily_hits"] == 0
    assert math.isclose(desc[5], 0.0)

    # Now verify that changing test node 1's label does not change homophily at all
    data_test_changed = Data(edge_index=edge_index, y=torch.tensor([1, 999, 0, 1]), train_mask=train_mask, test_mask=test_mask, num_nodes=4)
    desc_changed = _graph_descriptor(data_test_changed)
    assert math.isclose(desc[5], desc_changed[5])

    # Check test_topology_descriptor
    t_h, t_diag = calculate_homophily(data, return_diagnostics=True)
    assert t_diag["total_edges"] == 2
    assert t_diag["homophily_eligible_edges"] == 1
    assert math.isclose(t_h, 0.0)

    print("PASS: TEST C (mixed edge exclusion)")


def test_d_empty_eligible_set():
    """TEST D: Graph with no two-endpoint observed edges must return deterministic 0.0 without crash/NaN."""
    # Graph where all edges connect across splits (no train-train edges)
    # 0 (train) -- 1 (val)
    # 1 (val)   -- 2 (test)
    # 2 (test)  -- 3 (unobserved)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ], dtype=torch.long)

    train_mask = torch.tensor([True, False, False, False], dtype=torch.bool)
    val_mask   = torch.tensor([False, True, False, False], dtype=torch.bool)
    test_mask  = torch.tensor([False, False, True, False], dtype=torch.bool)
    y = torch.tensor([0, 0, 0, 0], dtype=torch.long)

    data = Data(edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=4)

    desc, diag = _graph_descriptor(data, return_diagnostics=True)
    assert not np.isnan(desc[5]), "Homophily returned NaN on empty eligible set"
    assert not np.isinf(desc[5]), "Homophily returned Inf on empty eligible set"
    assert desc[5] == 0.0, f"Expected deterministic fallback 0.0, got {desc[5]}"
    assert diag["total_edges"] == 3
    assert diag["homophily_eligible_edges"] == 0
    assert diag["homophily_hits"] == 0

    # Also check test_topology_descriptor
    t_h, t_diag = calculate_homophily(data, return_diagnostics=True)
    assert not math.isnan(t_h)
    assert t_h == 0.0
    assert t_diag["homophily_eligible_edges"] == 0

    print("PASS: TEST D (empty eligible set fallback)")


def test_e_non_homophily_descriptor_invariance():
    """TEST E: Verify density, degree stats, clustering, and triangles are 100% unchanged."""
    # Build a moderately complex graph with triangles, varying degrees, etc.
    # Nodes: 0, 1, 2, 3, 4, 5
    # Edges: (0,1), (1,2), (2,0) [triangle]
    #        (2,3), (3,4), (4,5), (3,5) [triangle 3-4-5]
    edges = [
        (0, 1), (1, 2), (2, 0),
        (2, 3), (3, 4), (4, 5), (3, 5),
    ]
    u_list = [u for u, v in edges] + [v for u, v in edges]
    v_list = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([u_list, v_list], dtype=torch.long)

    train_mask = torch.tensor([True, True, False, False, True, False], dtype=torch.bool)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    data = Data(edge_index=edge_index, y=y, train_mask=train_mask, num_nodes=6)

    old_desc = _legacy_graph_descriptor(data)
    new_desc = _graph_descriptor(data)

    # Coordinates:
    # 0: density
    # 1: degree mean
    # 2: degree dispersion (variance / mean^2)
    # 3: clustering
    # 4: triangles per node
    # 5: homophily (allowed to differ)

    for idx, name in enumerate(["density", "degree_mean", "degree_dispersion", "clustering", "triangles_per_node"]):
        assert math.isclose(old_desc[idx], new_desc[idx], abs_tol=1e-12), (
            f"Coordinate {idx} ({name}) changed! Old: {old_desc[idx]}, New: {new_desc[idx]}"
        )

    # Topology descriptor dict check
    topo_dict = extract_topology_descriptor(data)
    assert math.isclose(topo_dict["density"], old_desc[0])
    assert math.isclose(topo_dict["average_degree"], old_desc[1])
    assert math.isclose(topo_dict["normalized_degree_variance"], old_desc[2])
    assert math.isclose(topo_dict["average_clustering"], old_desc[3])

    print("PASS: TEST E (non-homophily descriptor invariance)")


if __name__ == "__main__":
    test_a_validation_test_leakage()
    test_b_training_label_sensitivity()
    test_c_mixed_edge()
    test_d_empty_eligible_set()
    test_e_non_homophily_descriptor_invariance()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
