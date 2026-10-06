from __future__ import annotations

import json
from pathlib import Path

import torch

from baseline_suite.common import (
    load_cora_context,
    run_federated_baseline,
)
from baseline_suite.config import DEFAULT_CONFIG
from baseline_suite.dmgnn import purify_graph_explanation_guided
from train_fedavg_cora import GCN


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    config = DEFAULT_CONFIG
    context = load_cora_context(project_dir, config)

    attacked_checkpoint_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "fedavg_structural_backdoor"
        / "best_backdoored_fedavg_cora.pt"
    )
    if not attacked_checkpoint_path.exists():
        raise FileNotFoundError(
            "The attacked FedAvg checkpoint is required for the "
            "explanation-guided purification stage. Run "
            "train_fedavg_backdoor_cora.py first:\n"
            f"{attacked_checkpoint_path}"
        )

    checkpoint = torch.load(
        attacked_checkpoint_path,
        map_location=context.device,
        weights_only=False,
    )

    explanation_model = GCN(
        input_channels=context.dataset.num_features,
        hidden_channels=config.hidden_channels,
        output_channels=context.dataset.num_classes,
        dropout=config.dropout,
    ).to(context.device)
    explanation_model.load_state_dict(checkpoint["model_state_dict"])
    explanation_model.eval()

    purified_graphs = []
    reports = []
    detailed_reports = {}

    print("=" * 96)
    print("Preparing DMGNN-FL explanation-guided purified client graphs")
    print("=" * 96)

    for client_id, graph in enumerate(context.attacked_client_graphs_cpu):
        purified, report, node_reports = purify_graph_explanation_guided(
            model=explanation_model,
            data=graph,
            client_id=client_id,
            target_class=config.target_class,
            trigger_size=config.trigger_size,
            seed=config.seed,
            confidence_threshold=0.50,
            probability_drop_threshold=0.03,
        )
        purified_graphs.append(purified)
        reports.append(report.__dict__)
        detailed_reports[str(client_id)] = node_reports

        print(
            f"Client {client_id:02d} | "
            f"candidates {report.candidate_nodes} | "
            f"suspicious {report.suspicious_nodes} | "
            f"removed edges {report.removed_undirected_edges}"
        )

    report_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "baseline_dmgnn"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "purification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "clients": reports,
                "node_reports": detailed_reports,
            },
            file,
            indent=2,
        )

    run_federated_baseline(
        method_name="DMGNN-FL (federated adaptation)",
        output_slug="baseline_dmgnn",
        config=config,
        client_graphs_cpu=purified_graphs,
        context=context,
        method_metadata={
            "paper_doi": "10.1016/j.patcog.2026.113693",
            "adaptation": (
                "A backdoored GCN supplies node-level counterfactual "
                "explanations. Reverse subset sampling prunes compact "
                "trigger-like ego edges before federated retraining."
            ),
        },
    )


if __name__ == "__main__":
    main()
