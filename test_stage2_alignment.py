from __future__ import annotations

import math
from collections import OrderedDict
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from multidata_suite.config import DatasetConfig
from multidata_suite.defenses import (
    ConfigurableETDFGLAggregator,
    ETDFGLAggregator,
    _graph_descriptor,
    compute_homophily_diagnostics,
)
from multidata_suite.model import GCN


def _create_synthetic_graphs(n_clients: int = 5, n_nodes: int = 40) -> list[Data]:
    torch.manual_seed(42)
    graphs = []
    for i in range(n_clients):
        # Create a connected cycle plus random edges
        row = list(range(n_nodes))
        col = list(range(1, n_nodes)) + [0]
        edge_index = torch.tensor([row + col, col + row], dtype=torch.long)
        x = torch.randn(n_nodes, 16)
        y = torch.randint(0, 3, (n_nodes,))
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:20] = True
        graphs.append(Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, num_nodes=n_nodes))
    return graphs


def _create_mock_local_states(n_clients: int = 5) -> tuple[list[OrderedDict[str, torch.Tensor]], OrderedDict[str, torch.Tensor]]:
    torch.manual_seed(42)
    global_state = OrderedDict([
        ("weight", torch.zeros(16, 16)),
        ("bias", torch.zeros(16)),
    ])
    local_states = []
    for i in range(n_clients):
        state = OrderedDict([
            ("weight", torch.randn(16, 16) * (i + 1) * 0.1),
            ("bias", torch.randn(16) * (i + 1) * 0.1),
        ])
        local_states.append(state)
    return local_states, global_state


# =========================================================================
# TEST A — Exact fusion equation
# =========================================================================
def test_a_exact_fusion_equation():
    """Verify that fused risk is exactly (9/13)*topology + (4/13)*update."""
    clean_graphs = _create_synthetic_graphs(5)
    current_graphs = _create_synthetic_graphs(5)
    aggregator = ETDFGLAggregator(clean_graphs, current_graphs)

    local_states, global_state = _create_mock_local_states(5)
    counts = [100, 120, 90, 110, 105]

    _, diagnostics = aggregator(local_states, counts, global_state, round_number=1)

    expected_alpha_topo = 9.0 / 13.0
    expected_alpha_upd = 4.0 / 13.0

    for i in range(5):
        topo_risk = diagnostics[f"etdfgl_topology_risk_client_{i}"]
        upd_risk = diagnostics[f"etdfgl_update_risk_client_{i}"]
        fused_risk = diagnostics[f"etdfgl_fused_risk_client_{i}"]

        expected_fused = expected_alpha_topo * topo_risk + expected_alpha_upd * upd_risk
        assert math.isclose(fused_risk, expected_fused, rel_tol=1e-12, abs_tol=1e-12), (
            f"Client {i}: fused risk {fused_risk} != expected {expected_fused}"
        )


# =========================================================================
# TEST B — Explanation independence
# =========================================================================
def test_b_explanation_independence():
    """Verify that varying explanation risk has ZERO effect on risk, trust, and weights."""
    clean_graphs = _create_synthetic_graphs(5)
    current_graphs = _create_synthetic_graphs(5)

    agg_a = ETDFGLAggregator(clean_graphs, current_graphs)
    agg_b = ETDFGLAggregator(clean_graphs, current_graphs)

    # Artificially alter explanation risk in agg_b to high arbitrary values
    agg_a.explanation_risk = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    agg_b.explanation_risk = np.array([0.95, 0.88, 0.72, 0.99, 0.65], dtype=np.float64)

    local_states, global_state = _create_mock_local_states(5)
    counts = [100, 120, 90, 110, 105]

    avg_a, diag_a = agg_a(local_states, counts, global_state, round_number=1)
    avg_b, diag_b = agg_b(local_states, counts, global_state, round_number=1)

    for i in range(5):
        # Explanation risk is different
        assert diag_a[f"etdfgl_explanation_risk_client_{i}"] != diag_b[f"etdfgl_explanation_risk_client_{i}"]

        # BUT fused risk, trust, and aggregation weight MUST be identical
        assert math.isclose(
            diag_a[f"etdfgl_fused_risk_client_{i}"],
            diag_b[f"etdfgl_fused_risk_client_{i}"],
            rel_tol=1e-14, abs_tol=1e-14,
        )
        assert math.isclose(
            diag_a[f"etdfgl_trust_client_{i}"],
            diag_b[f"etdfgl_trust_client_{i}"],
            rel_tol=1e-14, abs_tol=1e-14,
        )
        assert math.isclose(
            diag_a[f"etdfgl_weight_client_{i}"],
            diag_b[f"etdfgl_weight_client_{i}"],
            rel_tol=1e-14, abs_tol=1e-14,
        )

    # Output parameters must be bitwise identical
    for k in avg_a:
        torch.testing.assert_close(avg_a[k], avg_b[k])


# =========================================================================
# TEST C — Coefficient sum
# =========================================================================
def test_c_coefficient_sum():
    """Verify that alpha_topology + alpha_update == 1.0 exactly."""
    alpha_topo = ETDFGLAggregator.ALPHA_TOPOLOGY
    alpha_upd = ETDFGLAggregator.ALPHA_UPDATE

    assert math.isclose(alpha_topo, 9.0 / 13.0, rel_tol=1e-14)
    assert math.isclose(alpha_upd, 4.0 / 13.0, rel_tol=1e-14)
    assert math.isclose(alpha_topo + alpha_upd, 1.0, rel_tol=1e-14)

    # Also check Configurable defaults
    assert math.isclose(ConfigurableETDFGLAggregator.FROZEN_DEFAULT_TOPOLOGY, 9.0 / 13.0, rel_tol=1e-14)
    assert ConfigurableETDFGLAggregator.FROZEN_DEFAULT_EXPLANATION == 0.0
    assert math.isclose(ConfigurableETDFGLAggregator.FROZEN_DEFAULT_UPDATE, 4.0 / 13.0, rel_tol=1e-14)


# =========================================================================
# TEST D — Production runner
# =========================================================================
def test_d_production_runner_defaults():
    """Verify production runner and configurable aggregator instantiate the 9/13, 0, 4/13 method."""
    clean_graphs = _create_synthetic_graphs(3)
    current_graphs = _create_synthetic_graphs(3)

    # Configurable default instantiation
    configurable = ConfigurableETDFGLAggregator(clean_graphs, current_graphs)

    assert math.isclose(configurable.component_weights[0], 9.0 / 13.0, rel_tol=1e-14)
    assert configurable.component_weights[1] == 0.0
    assert math.isclose(configurable.component_weights[2], 4.0 / 13.0, rel_tol=1e-14)
    assert math.isclose(configurable.component_weights.sum(), 1.0, rel_tol=1e-14)

    # ETDFGLAggregator default instantiation
    prod = ETDFGLAggregator(clean_graphs, current_graphs)
    assert math.isclose(prod.ALPHA_TOPOLOGY, 9.0 / 13.0, rel_tol=1e-14)
    assert math.isclose(prod.ALPHA_UPDATE, 4.0 / 13.0, rel_tol=1e-14)


# =========================================================================
# TEST E — Production 5D label-free descriptor & historical homophily diagnostic
# =========================================================================
def test_e_production_5d_descriptor_and_historical_diagnostic():
    """Verify production 5D descriptor is label-free and historical diagnostic is isolated."""
    data = Data(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 0], dtype=torch.long),
        train_mask=torch.tensor([True, True, False, False], dtype=torch.bool),
        num_nodes=4,
    )
    # Production descriptor MUST be exactly length 5
    desc = _graph_descriptor(data)
    assert len(desc) == 5, f"Expected 5D descriptor, got {len(desc)}"
    assert desc.shape == (5,)

    # Changing labels must NOT affect production descriptor
    data_label_changed = data.clone()
    data_label_changed.y = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    desc_changed = _graph_descriptor(data_label_changed)
    assert np.allclose(desc, desc_changed), "Production descriptor leaked labels!"

    # Historical diagnostic function remains available for audit provenance
    diag = compute_homophily_diagnostics(data)
    assert diag["homophily"] == 0.0

    # If both 2 and 3 are observed:
    data_full = data.clone()
    data_full.train_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
    diag_full = compute_homophily_diagnostics(data_full)
    assert diag_full["homophily"] == 0.5


# =========================================================================
# TEST F — Configurable historical variant
# =========================================================================
def test_f_configurable_historical_variant():
    """Verify that historical 3-channel weights (0.45, 0.35, 0.20) are NOT default and only explicit."""
    clean_graphs = _create_synthetic_graphs(3)
    current_graphs = _create_synthetic_graphs(3)

    # 1. Default must NOT be 0.45, 0.35, 0.20
    default_agg = ConfigurableETDFGLAggregator(clean_graphs, current_graphs)
    assert not math.isclose(default_agg.component_weights[0], 0.45)
    assert default_agg.component_weights[1] == 0.0

    # 2. Explicit historical instantiation
    hist_agg = ConfigurableETDFGLAggregator(
        clean_graphs,
        current_graphs,
        topology_weight=0.45,
        explanation_weight=0.35,
        update_weight=0.20,
    )
    assert math.isclose(hist_agg.component_weights[0], 0.45, rel_tol=1e-14)
    assert math.isclose(hist_agg.component_weights[1], 0.35, rel_tol=1e-14)
    assert math.isclose(hist_agg.component_weights[2], 0.20, rel_tol=1e-14)
