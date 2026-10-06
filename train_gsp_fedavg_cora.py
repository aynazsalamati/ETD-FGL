from __future__ import annotations

import json
from pathlib import Path

from baseline_suite.common import (
    load_cora_context,
    run_federated_baseline,
)
from baseline_suite.config import DEFAULT_CONFIG
from baseline_suite.gsp import purify_graph_spectral


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config = DEFAULT_CONFIG
    context = load_cora_context(project_dir, config)

    purified_graphs = []
    reports = []

    print("=" * 88)
    print("Preparing GSP-FL purified client graphs")
    print("=" * 88)

    for client_id, graph in enumerate(context.attacked_client_graphs_cpu):
        purified, report = purify_graph_spectral(
            graph,
            low_frequency_rank=20,
            prune_ratio=0.12,
        )
        purified_graphs.append(purified)
        report_dict = {
            "client_id": client_id,
            **report.__dict__,
        }
        reports.append(report_dict)
        print(
            f"Client {client_id:02d} | "
            f"edges {report.original_directed_edges} -> "
            f"{report.purified_directed_edges} | "
            f"removed undirected: {report.removed_undirected_edges}"
        )

    report_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "baseline_gsp"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "purification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(reports, file, indent=2)

    run_federated_baseline(
        method_name="GSP-FL (federated adaptation)",
        output_slug="baseline_gsp",
        config=config,
        client_graphs_cpu=purified_graphs,
        context=context,
        method_metadata={
            "paper_doi": "10.1007/s11280-025-01364-w",
            "adaptation": (
                "Graph spectral low-frequency purification is applied "
                "independently to each client subgraph before FedAvg."
            ),
            "low_frequency_rank": 20,
            "prune_ratio": 0.12,
        },
    )


if __name__ == "__main__":
    main()
