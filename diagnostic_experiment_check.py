from __future__ import annotations

from pathlib import Path
import networkx as nx
import torch

from multidata_suite.config import get_config
from multidata_suite.data import load_dataset, load_or_create_partition, build_client_graphs
from multidata_suite.attack import inject_clique_trigger
from multidata_suite.defenses import _graph_descriptor


def old_homophily_calc(data):
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from([
        (int(u), int(v))
        for u, v in data.edge_index.t().tolist()
        if int(u) != int(v)
    ])
    homophily_hits = 0
    edge_count = 0
    for u, v in graph.edges():
        if u < data.y.numel() and v < data.y.numel():
            homophily_hits += int(data.y[u] == data.y[v])
            edge_count += 1
    return (homophily_hits / edge_count if edge_count else 0.0), edge_count


def main():
    project_dir = Path(".").resolve()
    config = get_config("cora")

    data, num_features, num_classes, metadata = load_dataset(project_dir, config)
    partition = load_or_create_partition(
        project_dir,
        config,
        data,
        num_classes,
        target_class=config.target_class,
        malicious_client_id=config.malicious_client_id,
    )

    clean_graphs = build_client_graphs(data, partition, config.num_clients)

    attacked_graph, attack_metadata = inject_clique_trigger(
        clean_graphs[config.malicious_client_id],
        target_class=config.target_class,
        poison_rate=config.poison_rate,
        trigger_size=config.trigger_size,
        seed=config.seed,
    )

    current_graphs = list(clean_graphs)
    current_graphs[config.malicious_client_id] = attacked_graph

    print("=" * 80)
    print(f"DIAGNOSTIC CHECK FOR DATASET: Cora, SEED: {config.seed}, NUM_CLIENTS: {config.num_clients}")
    print("=" * 80)

    header = (
        f"{'client_id':<10}"
        f"{'ref_edges':<12}"
        f"{'ref_elig':<12}"
        f"{'curr_edges':<12}"
        f"{'curr_elig':<12}"
        f"{'ref_homo':<12}"
        f"{'curr_homo':<12}"
        f"{'old_ref_homo':<14}"
        f"{'old_curr_homo':<14}"
    )
    print(header)
    print("-" * len(header))

    for client_id in range(config.num_clients):
        ref_graph = clean_graphs[client_id]
        curr_graph = current_graphs[client_id]

        ref_desc, ref_diag = _graph_descriptor(ref_graph, return_diagnostics=True)
        curr_desc, curr_diag = _graph_descriptor(curr_graph, return_diagnostics=True)

        old_ref_h, old_ref_cnt = old_homophily_calc(ref_graph)
        old_curr_h, old_curr_cnt = old_homophily_calc(curr_graph)

        ref_total = ref_diag["total_edges"]
        ref_elig = ref_diag["homophily_eligible_edges"]
        curr_total = curr_diag["total_edges"]
        curr_elig = curr_diag["homophily_eligible_edges"]

        ref_homo = ref_desc[5]
        curr_homo = curr_desc[5]

        row = (
            f"{client_id:<10}"
            f"{ref_total:<12}"
            f"{ref_elig:<12}"
            f"{curr_total:<12}"
            f"{curr_elig:<12}"
            f"{ref_homo:<12.4f}"
            f"{curr_homo:<12.4f}"
            f"{old_ref_h:<14.4f}"
            f"{old_curr_h:<14.4f}"
        )
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    main()
