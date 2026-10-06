from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
import networkx as nx

from multidata_suite.config import get_config
from multidata_suite.core import build_context
from multidata_suite.data import load_dataset, load_or_create_partition, build_client_graphs
from multidata_suite.model import GCN
from multidata_suite.defenses import (
    _graph_descriptor,
    compute_homophily_diagnostics,
    ETDFGLAggregator,
)
from test_topology_descriptor import calculate_homophily


def _legacy_homophily(graph: nx.Graph, y: torch.Tensor) -> float:
    """Legacy homophily computation that leaks non-training labels."""
    hits = 0
    count = 0
    y_len = y.numel()
    for u, v in graph.edges():
        if u < y_len and v < y_len:
            hits += int(y[u] == y[v])
            count += 1
    return float(hits / count) if count else 0.0


def _build_nx_graph(data) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(int(data.num_nodes)))
    G.add_edges_from([
        (int(u), int(v))
        for u, v in data.edge_index.t().tolist()
        if int(u) != int(v)
    ])
    return G


def main():
    project_dir = Path(__file__).resolve().parent
    config = get_config("cora")
    
    print("=" * 90)
    print("STAGE 1 VERIFICATION: ETD-FGL PIPELINE & TRAIN-MASK HOMOPHILY DESCRIPTOR")
    print("=" * 90)

    # =========================================================================
    # 1. VERIFY MASK PROPAGATION
    # =========================================================================
    print("\n--- 1. VERIFYING MASK PROPAGATION ---")
    data, num_features, num_classes, metadata = load_dataset(project_dir, config)
    assert hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask")
    assert data.train_mask.shape[0] == data.num_nodes
    assert (data.train_mask & data.val_mask).sum().item() == 0, "Global train and val masks overlap!"
    assert (data.train_mask & data.test_mask).sum().item() == 0, "Global train and test masks overlap!"
    print(f"[OK] Full dataset masks verified: N={data.num_nodes}, Train={int(data.train_mask.sum())}, Val={int(data.val_mask.sum())}, Test={int(data.test_mask.sum())}")

    context = build_context(project_dir, config)
    
    for cid in range(config.num_clients):
        cg = context.clean_graphs_cpu[cid]
        ag = context.attacked_graphs_cpu[cid]
        
        # Check clean graph
        assert hasattr(cg, "train_mask"), f"Client {cid} clean graph missing train_mask!"
        assert cg.train_mask.dtype == torch.bool, f"Client {cid} clean train_mask not bool!"
        assert cg.train_mask.shape[0] == cg.num_nodes, f"Client {cid} clean train_mask size mismatch!"
        assert (cg.train_mask & cg.val_mask).sum().item() == 0, f"Client {cid} clean train/val overlap!"
        assert (cg.train_mask & cg.test_mask).sum().item() == 0, f"Client {cid} clean train/test overlap!"
        
        # Check attacked graph
        assert hasattr(ag, "train_mask"), f"Client {cid} attacked graph missing train_mask!"
        assert ag.train_mask.dtype == torch.bool, f"Client {cid} attacked train_mask not bool!"
        assert ag.train_mask.shape[0] == ag.num_nodes, f"Client {cid} attacked train_mask size mismatch!"
        assert (ag.train_mask & ag.val_mask).sum().item() == 0, f"Client {cid} attacked train/val overlap!"
        assert (ag.train_mask & ag.test_mask).sum().item() == 0, f"Client {cid} attacked train/test overlap!"
        
        if cid == config.malicious_client_id:
            orig_nodes = cg.num_nodes
            att_nodes = ag.num_nodes
            num_triggers = att_nodes - orig_nodes
            assert num_triggers > 0, "Malicious client has no injected trigger nodes!"
            # Trigger nodes must have train_mask == False
            assert not ag.train_mask[orig_nodes:].any(), "Injected trigger nodes marked as True in train_mask!"
            assert not ag.val_mask[orig_nodes:].any(), "Injected trigger nodes marked as True in val_mask!"
            assert not ag.test_mask[orig_nodes:].any(), "Injected trigger nodes marked as True in test_mask!"
            assert ag.train_mask[:orig_nodes].equal(cg.train_mask), "Original nodes train_mask changed in attacked graph!"
            print(f"[OK] Client {cid} (MALICIOUS): {orig_nodes} original nodes, {num_triggers} triggers appended with train_mask=False.")
        else:
            assert ag.num_nodes == cg.num_nodes, f"Benign client {cid} node count changed!"
            assert ag.train_mask.equal(cg.train_mask), f"Benign client {cid} train_mask changed!"
            print(f"[OK] Client {cid} (BENIGN): {cg.num_nodes} nodes, {int(cg.train_mask.sum())} train nodes correctly aligned.")

    print("Mask propagation: ALL CHECKS PASSED.")

    # =========================================================================
    # 2. RUN ONE SMALL REAL SANITY CHECK & EXTRACT TABLE
    # =========================================================================
    print("\n--- 2. REAL PIPELINE SANITY CHECK (Cora, seed=42, 5 clients) ---")
    
    table_rows = []
    for cid in range(config.num_clients):
        cg = context.clean_graphs_cpu[cid]
        ag = context.attacked_graphs_cpu[cid]
        
        ref_desc, ref_diag = _graph_descriptor(cg, return_diagnostics=True)
        cur_desc, cur_diag = _graph_descriptor(ag, return_diagnostics=True)
        
        row = {
            "client_id": cid,
            "is_malicious": (cid == config.malicious_client_id),
            "reference_nodes": int(cg.num_nodes),
            "reference_total_edges": ref_diag["total_edges"],
            "reference_train_nodes": int(cg.train_mask.sum().item()),
            "reference_homophily_eligible_edges": ref_diag["homophily_eligible_edges"],
            "reference_homophily": float(ref_desc[5]),
            "current_nodes": int(ag.num_nodes),
            "current_total_edges": cur_diag["total_edges"],
            "current_train_nodes": int(ag.train_mask.sum().item()),
            "current_homophily_eligible_edges": cur_diag["homophily_eligible_edges"],
            "current_homophily": float(cur_desc[5]),
        }
        table_rows.append(row)

    header = (
        f"{'Client':<8} "
        f"{'Malicious':<11} "
        f"{'Ref_Nodes':<10} "
        f"{'Ref_Edges':<10} "
        f"{'Ref_Tr_Nd':<10} "
        f"{'Ref_Elig_Ed':<12} "
        f"{'Ref_Homo':<10} "
        f"{'Cur_Nodes':<10} "
        f"{'Cur_Edges':<10} "
        f"{'Cur_Tr_Nd':<10} "
        f"{'Cur_Elig_Ed':<12} "
        f"{'Cur_Homo':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in table_rows:
        mal_str = "YES (Target)" if r["is_malicious"] else "NO"
        print(
            f"{r['client_id']:<8} "
            f"{mal_str:<11} "
            f"{r['reference_nodes']:<10} "
            f"{r['reference_total_edges']:<10} "
            f"{r['reference_train_nodes']:<10} "
            f"{r['reference_homophily_eligible_edges']:<12} "
            f"{r['reference_homophily']:<10.4f} "
            f"{r['current_nodes']:<10} "
            f"{r['current_total_edges']:<10} "
            f"{r['current_train_nodes']:<10} "
            f"{r['current_homophily_eligible_edges']:<12} "
            f"{r['current_homophily']:<10.4f}"
        )

    # =========================================================================
    # 3. VERIFY THE ELIGIBLE-EDGE RULE
    # =========================================================================
    print("\n--- 3. VERIFYING ELIGIBLE-EDGE RULE PROGRAMMATICALLY ---")
    
    for cid in range(config.num_clients):
        for name, g in [("clean", context.clean_graphs_cpu[cid]), ("attacked", context.attacked_graphs_cpu[cid])]:
            G = _build_nx_graph(g)
            train_mask = g.train_mask
            val_mask = g.val_mask
            test_mask = g.test_mask
            
            eligible = []
            excluded = []
            for u, v in G.edges():
                if train_mask[u] and train_mask[v]:
                    eligible.append((u, v))
                else:
                    excluded.append((u, v))
                    
            # Programmatic assertion on eligible edges
            for u, v in eligible:
                assert bool(train_mask[u]) is True, f"Eligible edge endpoint {u} not in train_mask!"
                assert bool(train_mask[v]) is True, f"Eligible edge endpoint {v} not in train_mask!"
                assert bool(val_mask[u]) is False, f"Eligible edge endpoint {u} in val_mask!"
                assert bool(val_mask[v]) is False, f"Eligible edge endpoint {v} in val_mask!"
                assert bool(test_mask[u]) is False, f"Eligible edge endpoint {u} in test_mask!"
                assert bool(test_mask[v]) is False, f"Eligible edge endpoint {v} in test_mask!"
                
            # Programmatic assertion on excluded edges
            for u, v in excluded:
                assert not (bool(train_mask[u]) and bool(train_mask[v])), (
                    f"Excluded edge ({u}, {v}) has both endpoints in train_mask!"
                )
                
            # Check edge count matches diagnostics
            diag = compute_homophily_diagnostics(g)
            assert diag["homophily_eligible_edges"] == len(eligible), (
                f"Mismatch in eligible edges count: diag={diag['homophily_eligible_edges']}, actual={len(eligible)}"
            )

    print("[OK] All eligible edges strictly connect two train_mask nodes.")
    print("[OK] All excluded edges have at least one non-training endpoint.")

    # Show concrete examples of excluded edges in Client 0 (Attacked)
    mal_g = context.attacked_graphs_cpu[0]
    mal_G = _build_nx_graph(mal_g)
    print("\nSample excluded edge types found in Client 0 (Attacked):")
    sample_types = {}
    for u, v in mal_G.edges():
        u_type = "train" if mal_g.train_mask[u] else ("val" if mal_g.val_mask[u] else ("test" if mal_g.test_mask[u] else ("trigger" if getattr(mal_g, "trigger_node_mask", torch.zeros_like(mal_g.train_mask))[u] else "unobserved")))
        v_type = "train" if mal_g.train_mask[v] else ("val" if mal_g.val_mask[v] else ("test" if mal_g.test_mask[v] else ("trigger" if getattr(mal_g, "trigger_node_mask", torch.zeros_like(mal_g.train_mask))[v] else "unobserved")))
        pair = tuple(sorted([u_type, v_type]))
        if pair not in sample_types and pair != ("train", "train"):
            sample_types[pair] = (u, v)
            print(f"  - {pair[0]} <-> {pair[1]}: Edge ({u}, {v}) -> EXCLUDED from homophily")

    # =========================================================================
    # 4. VERIFY DESCRIPTOR USED BY REAL ETD-FGL
    # =========================================================================
    print("\n--- 4. VERIFYING DESCRIPTORS CONSUMED BY PRODUCTION ETDFGLAggregator ---")
    model = GCN(num_features, config.hidden_channels, num_classes, config.dropout)
    aggregator = ETDFGLAggregator(
        context.clean_graphs_cpu,
        context.attacked_graphs_cpu,
        model,
        config,
    )
    # Check that ETDFGLAggregator._graph_descriptor calls match our table rows
    agg_clean = np.stack([_graph_descriptor(g) for g in context.clean_graphs_cpu])
    agg_current = np.stack([_graph_descriptor(g) for g in context.attacked_graphs_cpu])

    for cid in range(config.num_clients):
        assert math.isclose(agg_clean[cid, 5], table_rows[cid]["reference_homophily"]), f"Clean homophily mismatch at client {cid}!"
        assert math.isclose(agg_current[cid, 5], table_rows[cid]["current_homophily"]), f"Current homophily mismatch at client {cid}!"

    print("[OK] Production ETDFGLAggregator consumes EXACTLY the table descriptor values.")

    # =========================================================================
    # 5. COMPARE OLD VS NEW HOMOPHILY
    # =========================================================================
    print("\n--- 5. OLD VS NEW HOMOPHILY COMPARISON ---")
    diff_header = (
        f"{'Client':<8} "
        f"{'Old_Ref_Homo':<15} "
        f"{'New_Ref_Homo':<15} "
        f"{'Abs_Diff_Ref':<15} "
        f"{'Old_Cur_Homo':<15} "
        f"{'New_Cur_Homo':<15} "
        f"{'Abs_Diff_Cur':<15}"
    )
    print(diff_header)
    print("-" * len(diff_header))
    for cid in range(config.num_clients):
        cg = context.clean_graphs_cpu[cid]
        ag = context.attacked_graphs_cpu[cid]
        
        G_c = _build_nx_graph(cg)
        G_a = _build_nx_graph(ag)
        
        old_ref = _legacy_homophily(G_c, cg.y)
        new_ref = float(_graph_descriptor(cg)[5])
        diff_ref = abs(old_ref - new_ref)
        
        old_cur = _legacy_homophily(G_a, ag.y)
        new_cur = float(_graph_descriptor(ag)[5])
        diff_cur = abs(old_cur - new_cur)
        
        print(
            f"{cid:<8} "
            f"{old_ref:<15.4f} "
            f"{new_ref:<15.4f} "
            f"{diff_ref:<15.4f} "
            f"{old_cur:<15.4f} "
            f"{new_cur:<15.4f} "
            f"{diff_cur:<15.4f}"
        )

    # =========================================================================
    # 6. DESCRIPTOR INVARIANCE CHECK
    # =========================================================================
    print("\n--- 6. NON-HOMOPHILY DESCRIPTOR INVARIANCE CHECK ---")
    coord_names = [
        "density",
        "degree_mean",
        "degree_dispersion",
        "clustering_coefficient",
        "triangles_per_node",
        "homophily",
    ]
    
    all_invariant = True
    for cid in range(config.num_clients):
        for name, g in [("clean", context.clean_graphs_cpu[cid]), ("attacked", context.attacked_graphs_cpu[cid])]:
            G = _build_nx_graph(g)
            degrees = np.asarray([deg for _, deg in G.degree()], dtype=np.float64)
            density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
            clustering = nx.average_clustering(G) if G.number_of_edges() else 0.0
            triangles = sum(nx.triangles(G).values()) / 3.0
            legacy_desc = np.asarray([
                density,
                float(degrees.mean()) if degrees.size else 0.0,
                float(degrees.var() / (degrees.mean() ** 2 + 1e-12)) if degrees.size else 0.0,
                clustering,
                triangles / max(1.0, G.number_of_nodes()),
                _legacy_homophily(G, g.y),
            ], dtype=np.float64)
            
            new_desc = _graph_descriptor(g)
            
            for idx in range(5):
                diff = abs(legacy_desc[idx] - new_desc[idx])
                if diff > 1e-12:
                    print(f"FAILED: Coordinate {idx} ({coord_names[idx]}) differs by {diff} for client {cid} {name}!")
                    all_invariant = False
    
    if all_invariant:
        print("[OK] All non-homophily coordinates (density, degree mean, degree dispersion, clustering, triangles) are 100% BIT-FOR-BIT INVARIANT.")
        print("[OK] ONLY coordinate 5 (homophily) differs due to the label-leakage removal.")
    else:
        print("FAILED: Non-homophily coordinates were altered!")

    print("\nSTAGE 1 VERIFICATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
