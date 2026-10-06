from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from multidata_suite.config import get_config
from multidata_suite.data import load_dataset, load_or_create_partition, build_client_graphs
from multidata_suite.attack import inject_clique_trigger
from multidata_suite.defenses import (
    ETDFGLAggregator,
    ConfigurableETDFGLAggregator,
    _graph_descriptor,
    ETDFGL_DESCRIPTOR_VERSION,
    ETDFGL_DESCRIPTOR_COORDINATES,
    ETDFGL_DESCRIPTOR_DIM,
    ETDFGL_METHOD_METADATA,
)


def _build_test_graph() -> Data:
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 2, 3, 3, 4],
        [1, 0, 2, 1, 0, 2, 3, 2, 4, 3],
    ], dtype=torch.long)
    x = torch.randn(5, 16)
    y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    train_mask = torch.tensor([True, True, False, False, False], dtype=torch.bool)
    val_mask = torch.tensor([False, False, True, False, False], dtype=torch.bool)
    test_mask = torch.tensor([False, False, False, True, True], dtype=torch.bool)
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=5,
    )


# =========================================================================
# 1. LABEL-INVARIANCE TEST (Prompt Section 5)
# =========================================================================
def test_descriptor_label_invariance():
    """Verify that permuting or changing node labels produces identical descriptors."""
    data = _build_test_graph()
    desc_a = _graph_descriptor(data)
    assert desc_a.shape == (5,)

    # Randomly permute all node labels
    data_permuted = data.clone()
    torch.manual_seed(999)
    perm = torch.randperm(data.num_nodes)
    data_permuted.y = data.y[perm]
    desc_b = _graph_descriptor(data_permuted)

    # Assign completely disjoint labels
    data_disjoint = data.clone()
    data_disjoint.y = data.y + 100
    desc_c = _graph_descriptor(data_disjoint)

    assert np.allclose(desc_a, desc_b, atol=1e-12), "Descriptor changed when labels were permuted!"
    assert np.allclose(desc_a, desc_c, atol=1e-12), "Descriptor changed when labels were altered!"


# =========================================================================
# 2. MASK-INVARIANCE TEST (Prompt Section 6)
# =========================================================================
def test_descriptor_mask_invariance():
    """Verify that modifying split masks produces identical descriptors."""
    data = _build_test_graph()
    desc_orig = _graph_descriptor(data)

    data_remasked = data.clone()
    data_remasked.train_mask = torch.tensor([False, False, False, True, True], dtype=torch.bool)
    data_remasked.val_mask = torch.tensor([True, False, False, False, False], dtype=torch.bool)
    data_remasked.test_mask = torch.tensor([False, True, True, False, False], dtype=torch.bool)
    desc_remasked = _graph_descriptor(data_remasked)

    # All-false mask
    data_empty = data.clone()
    data_empty.train_mask = torch.zeros(5, dtype=torch.bool)
    desc_empty = _graph_descriptor(data_empty)

    assert np.allclose(desc_orig, desc_remasked, atol=1e-12), "Descriptor changed when split masks changed!"
    assert np.allclose(desc_orig, desc_empty, atol=1e-12), "Descriptor changed when masks were emptied!"


# =========================================================================
# 3. STRUCTURAL-SENSITIVITY TEST (Prompt Section 7)
# =========================================================================
def test_descriptor_structural_sensitivity():
    """Verify that injecting a 3-node clique trigger alters structural coordinates."""
    data = _build_test_graph()
    desc_before = _graph_descriptor(data)

    # Add 3 trigger nodes connected as a triangle (clique) and attached to node 0
    victim = 0
    t0, t1, t2 = 5, 6, 7
    trigger_edges = torch.tensor([
        [t0, t1, t1, t2, t2, t0, t1, t0, t2, t1, t0, t2, victim, t0, victim, t1, victim, t2, t0, victim, t1, victim, t2, victim],
        [t1, t0, t2, t1, t0, t2, t0, t1, t1, t2, t2, t0, t0, victim, t1, victim, t2, victim, victim, t0, victim, t1, victim, t2],
    ], dtype=torch.long)
    new_edge_index = torch.cat([data.edge_index, trigger_edges], dim=1)

    data_attacked = Data(
        x=torch.cat([data.x, torch.zeros(3, 16)], dim=0),
        edge_index=new_edge_index,
        y=torch.cat([data.y, torch.zeros(3, dtype=torch.long)], dim=0),
        num_nodes=8,
    )

    desc_after = _graph_descriptor(data_attacked)

    assert desc_after.shape == (5,)
    # Verify overall descriptor has changed
    assert not np.allclose(desc_before, desc_after), "Descriptor failed to respond to clique trigger!"

    # Density, mean degree, or clustering/triangles must change
    assert desc_before[1] != desc_after[1], "Mean degree did not respond to trigger injection!"
    assert desc_before[4] != desc_after[4], "Triangles per node did not respond to clique injection!"


# =========================================================================
# 4. EMPTY / DEGENERATE GRAPH TEST (Prompt Section 8)
# =========================================================================
def test_descriptor_degenerate_graphs():
    """Verify descriptor returns finite 5D values on degenerate and edge cases."""
    # 1. Single node, zero edges
    g1 = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
    d1 = _graph_descriptor(g1)
    assert d1.shape == (5,)
    assert np.all(np.isfinite(d1)), f"Non-finite in single-node graph: {d1}"
    assert np.allclose(d1, [0.0, 0.0, 0.0, 0.0, 0.0])

    # 2. Graph with zero edges, multiple nodes
    g2 = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=10)
    d2 = _graph_descriptor(g2)
    assert d2.shape == (5,)
    assert np.all(np.isfinite(d2)), f"Non-finite in zero-edge graph: {d2}"
    assert np.allclose(d2, [0.0, 0.0, 0.0, 0.0, 0.0])

    # 3. Disconnected graph (isolated components)
    g3 = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        num_nodes=6,
    )
    d3 = _graph_descriptor(g3)
    assert d3.shape == (5,)
    assert np.all(np.isfinite(d3)), f"Non-finite in disconnected graph: {d3}"


# =========================================================================
# 5. STAGE 2 FUSION & METADATA REGRESSION (Prompt Section 9 & 16)
# =========================================================================
def test_etdfgl_fusion_regression():
    """Verify fusion coefficients, metadata, and trust mapping match frozen Stage 2."""
    assert math.isclose(ETDFGLAggregator.ALPHA_TOPOLOGY, 9.0 / 13.0)
    assert math.isclose(ETDFGLAggregator.ALPHA_UPDATE, 4.0 / 13.0)
    assert math.isclose(ETDFGLAggregator.ALPHA_EXPLANATION, 0.0)

    assert ETDFGLAggregator.DESCRIPTOR_DIM == 5
    assert ETDFGLAggregator.DESCRIPTOR_VERSION == "topology_5d_label_free_v1"
    assert ETDFGLAggregator.DESCRIPTOR_COORDINATES == [
        "density",
        "mean_degree",
        "normalized_degree_variance",
        "average_clustering",
        "triangles_per_node",
    ]
    assert ETDFGL_METHOD_METADATA["descriptor_dim"] == 5


# =========================================================================
# 6. CORA SEED 42 IID SANITY (Prompt Section 11)
# =========================================================================
def test_cora_seed42_iid_descriptor_sanity():
    """Verify Cora seed 42 IID: clients 1-4 ref==cur, client 0 ref!=cur."""
    project_root = Path(__file__).resolve().parent
    config = get_config("cora")
    data, nfeats, nclasses, meta = load_dataset(project_root, config)

    partition = load_or_create_partition(
        project_root,
        config,
        data,
        nclasses,
        strategy="iid",
        dirichlet_alpha=None,
        target_class=config.target_class,
        malicious_client_id=config.malicious_client_id,
    )
    clean_graphs = build_client_graphs(data, partition, config.num_clients)
    att_g, _ = inject_clique_trigger(
        clean_graphs[config.malicious_client_id],
        target_class=config.target_class,
        poison_rate=config.poison_rate,
        trigger_size=config.trigger_size,
        seed=config.seed,
    )
    attacked_graphs = list(clean_graphs)
    attacked_graphs[config.malicious_client_id] = att_g

    clean_descs = np.stack([_graph_descriptor(g) for g in clean_graphs])
    curr_descs = np.stack([_graph_descriptor(g) for g in attacked_graphs])

    assert clean_descs.shape == (5, 5)
    assert curr_descs.shape == (5, 5)

    # Benign clients (1-4): clean == current
    for cid in range(1, 5):
        assert np.allclose(clean_descs[cid], curr_descs[cid]), f"Benign client {cid} descriptor changed!"

    # Malicious client (0): clean != current
    assert not np.allclose(clean_descs[0], curr_descs[0]), "Malicious client 0 descriptor failed to change!"

    scale = np.std(clean_descs, axis=0)
    scale[scale < 1e-8] = 1.0
    drift = np.linalg.norm((curr_descs - clean_descs) / scale, axis=1)

    assert drift[0] > 0.0, "Malicious client drift must be positive!"
    assert np.all(drift[1:] == 0.0), "Benign clients must have zero drift!"
    assert drift[0] > np.max(drift[1:]), "Malicious client must be strictly ranked 1!"


# =========================================================================
# 7. MULTI-DATASET DESCRIPTOR SANITY (Prompt Section 12)
# =========================================================================
@pytest.mark.parametrize("d_key", ["cora", "pubmed", "reddit"])
def test_multidataset_5d_descriptor_sanity(d_key: str):
    """Verify 5D descriptor dimensionality and behavior across Cora, PubMed, Reddit."""
    project_root = Path(__file__).resolve().parent
    config = get_config(d_key)
    data, nfeats, nclasses, meta = load_dataset(project_root, config)

    partition = load_or_create_partition(
        project_root,
        config,
        data,
        nclasses,
        strategy="iid",
        dirichlet_alpha=None,
        target_class=config.target_class,
        malicious_client_id=config.malicious_client_id,
    )
    clean_graphs = build_client_graphs(data, partition, config.num_clients)
    att_g, _ = inject_clique_trigger(
        clean_graphs[config.malicious_client_id],
        target_class=config.target_class,
        poison_rate=config.poison_rate,
        trigger_size=config.trigger_size,
        seed=config.seed,
    )
    attacked_graphs = list(clean_graphs)
    attacked_graphs[config.malicious_client_id] = att_g

    clean_descs = np.stack([_graph_descriptor(g) for g in clean_graphs])
    curr_descs = np.stack([_graph_descriptor(g) for g in attacked_graphs])

    assert clean_descs.shape == (config.num_clients, 5)
    assert curr_descs.shape == (config.num_clients, 5)
    assert np.all(np.isfinite(clean_descs))
    assert np.all(np.isfinite(curr_descs))

    for cid in range(1, config.num_clients):
        assert np.allclose(clean_descs[cid], curr_descs[cid]), f"{d_key}: Benign client {cid} descriptor changed!"

    assert not np.allclose(clean_descs[0], curr_descs[0]), f"{d_key}: Malicious client 0 failed to produce drift!"
