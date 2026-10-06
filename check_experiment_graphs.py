from __future__ import annotations

from pathlib import Path
from multidata_suite.config import get_config
from multidata_suite.core import build_context
from multidata_suite.defenses import _graph_descriptor
import networkx as nx


def main():
    project_dir = Path(__file__).resolve().parent
    config = get_config("cora")
    context = build_context(project_dir, config)

    print("=" * 110)
    print(f"ETD-FGL SANITY-CHECK DIAGNOSTIC: {config.display_name} (seed={config.seed}, clients={config.num_clients})")
    print("=" * 110)
    header = (
        f"{'client_id':<10} "
        f"{'ref_total_edges':<17} "
        f"{'ref_homo_elig':<15} "
        f"{'cur_total_edges':<17} "
        f"{'cur_homo_elig':<15} "
        f"{'reference_homo':<16} "
        f"{'current_homo':<14}"
    )
    print(header)
    print("-" * 110)

    results = []
    for client_id in range(config.num_clients):
        ref_graph = context.clean_graphs_cpu[client_id]
        cur_graph = context.attacked_graphs_cpu[client_id]

        ref_desc, ref_diag = _graph_descriptor(ref_graph, return_diagnostics=True)
        cur_desc, cur_diag = _graph_descriptor(cur_graph, return_diagnostics=True)

        row = {
            "client_id": client_id,
            "reference_total_edges": ref_diag["total_edges"],
            "reference_homophily_eligible_edges": ref_diag["homophily_eligible_edges"],
            "current_total_edges": cur_diag["total_edges"],
            "current_homophily_eligible_edges": cur_diag["homophily_eligible_edges"],
            "reference_homophily": float(ref_desc[5]),
            "current_homophily": float(cur_desc[5]),
        }
        results.append(row)

        print(
            f"{row['client_id']:<10} "
            f"{row['reference_total_edges']:<17} "
            f"{row['reference_homophily_eligible_edges']:<15} "
            f"{row['current_total_edges']:<17} "
            f"{row['current_homophily_eligible_edges']:<15} "
            f"{row['reference_homophily']:<16.4f} "
            f"{row['current_homophily']:<14.4f}"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
