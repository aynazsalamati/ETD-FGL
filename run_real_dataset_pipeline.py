from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multidata_suite.config import get_config
from multidata_suite.core import build_context, fedavg, run_method, standard_local_train
from multidata_suite.defenses import (
    ETDFGLAggregator,
    FLPurifierAggregator,
    PDFLAggregator,
    _persistence_signature,
    make_flpurifier_trainer,
    purify_explanation_guided,
    purify_gsp,
)
from multidata_suite.fedtge import (
    FedTGEAggregator,
    FedTGEGCN,
    make_fedtge_trainer,
)
from multidata_suite.model import GCN, clone_state


def _summary_exists(context, slug: str) -> bool:
    return (context.base_output / slug / "final_summary.json").exists()


def calibrate_pdfl(context) -> PDFLAggregator:
    model = GCN(
        context.num_features,
        context.config.hidden_channels,
        context.num_classes,
        context.config.dropout,
    ).to(context.device)
    graphs = [graph.to(context.device) for graph in context.clean_graphs_cpu]
    features = []
    from multidata_suite.core import standard_local_train, fedavg

    for calibration_round in range(1, 5):
        global_state = clone_state(model)
        states = []
        counts = []
        for graph in graphs:
            state, _, count, _ = standard_local_train(
                global_state,
                graph,
                context.num_features,
                context.num_classes,
                context.device,
                calibration_round,
                context.config,
            )
            states.append(state)
            counts.append(count)
            features.append(_persistence_signature(state, global_state))
        averaged, _ = fedavg(states, counts, global_state, calibration_round)
        model.load_state_dict(averaged)
    return PDFLAggregator(np.stack(features), context.config.seed)


def load_checkpoint_model(context, slug: str) -> GCN:
    path = context.base_output / slug / "best_model.pt"
    checkpoint = torch.load(path, map_location=context.device, weights_only=False)
    model = GCN(
        context.num_features,
        context.config.hidden_channels,
        context.num_classes,
        context.config.dropout,
    ).to(context.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cora", "pubmed", "reddit"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config = get_config(args.dataset)
    context = build_context(project_dir, config)

    methods = []

    def execute(slug, **kwargs):
        if _summary_exists(context, slug) and not args.force:
            print(f"SKIP {slug}: existing summary found")
            summary = json.loads((context.base_output / slug / "final_summary.json").read_text(encoding="utf-8"))
        else:
            summary = run_method(context=context, output_slug=slug, **kwargs)
        methods.append(summary)

    execute(
        "fedavg_clean",
        method_name="Clean FedAvg",
        client_graphs_cpu=context.clean_graphs_cpu,
        aggregate_fn=fedavg,
    )
    execute(
        "fedavg_structural_backdoor",
        method_name="Attacked FedAvg",
        client_graphs_cpu=context.attacked_graphs_cpu,
        aggregate_fn=fedavg,
    )

    pdfl = calibrate_pdfl(context)
    execute(
        "baseline_pdfl",
        method_name="PD-FL",
        client_graphs_cpu=context.attacked_graphs_cpu,
        aggregate_fn=pdfl,
        method_metadata={"doi": "10.1016/j.cose.2023.103557", "type": "GCN federated adaptation"},
    )

    gsp_graphs = [purify_gsp(graph) for graph in context.attacked_graphs_cpu]
    execute(
        "baseline_gsp",
        method_name="GSP-FL",
        client_graphs_cpu=gsp_graphs,
        aggregate_fn=fedavg,
        method_metadata={"doi": "10.1007/s11280-025-01364-w", "type": "federated adaptation"},
    )

    execute(
        "baseline_flpurifier",
        method_name="FLPurifier-GNN",
        client_graphs_cpu=context.attacked_graphs_cpu,
        local_train_fn=make_flpurifier_trainer(config),
        aggregate_fn=FLPurifierAggregator(),
        method_metadata={"doi": "10.1109/TIFS.2024.3384846", "type": "GCN adaptation"},
    )

    attacked_model = load_checkpoint_model(context, "fedavg_structural_backdoor")
    dmgnn_graphs = [
        purify_explanation_guided(
            attacked_model,
            graph,
            target_class=config.target_class,
            trigger_size=config.trigger_size,
        )
        for graph in context.attacked_graphs_cpu
    ]
    execute(
        "baseline_dmgnn",
        method_name="DMGNN-FL",
        client_graphs_cpu=dmgnn_graphs,
        aggregate_fn=fedavg,
        method_metadata={"doi": "10.1016/j.patcog.2026.113693", "type": "federated adaptation"},
    )

    execute(
        "baseline_fedtge",
        method_name="FedTGE",
        client_graphs_cpu=context.attacked_graphs_cpu,
        local_train_fn=make_fedtge_trainer(config),
        aggregate_fn=FedTGEAggregator(),
        model_cls=FedTGEGCN,
        method_metadata={
            "method": "fedtge_adapted",
            "paper": "Energy-based Backdoor Defense Against Federated Graph Learning (ICLR 2025)",
            "official_repo": "https://github.com/ZitongShi/fedTGE",
            "official_commit": "e80950cfd612df5e6da21a9880d93735288c36a9",
            "adaptation_version": "1.0",
            "energy_epochs": 30,
            "tau": 0.85,
            "prop_layers": 1,
            "prop_alpha": 0.1,
            "supervised_epochs": 3,
            "model_architecture": (
                f"FedTGEGCN (2 GCNConv, 2 LayerNorm, "
                f"hidden={config.hidden_channels}, "
                f"dropout={config.dropout})"
            ),
            "finch_provenance": "Sarfraz et al., CVPR 2019 (vendored parameter-free FINCH)",
        },
    )

    etdfgl = ETDFGLAggregator(
        context.clean_graphs_cpu,
        context.attacked_graphs_cpu,
        attacked_model,
        config,
    )
    execute(
        "defended_etdfgl",
        method_name="ETD-FGL",
        client_graphs_cpu=context.attacked_graphs_cpu,
        aggregate_fn=etdfgl,
        method_metadata={
            "components": [
                "topology drift (9/13)",
                "round-wise model-update anomaly (4/13)",
                "trust-weighted aggregation",
            ],
            "weights": {
                "topology": 9.0 / 13.0,
                "update": 4.0 / 13.0,
                "explanation": 0.0,
            },
            "formula": "R_i^t = (9/13) * A_topo + (4/13) * A_upd",
        },
    )

    print("=" * 100)
    print(f"REAL PIPELINE COMPLETED FOR {config.display_name}")
    print(f"Output: {context.base_output}")
    print("=" * 100)


if __name__ == "__main__":
    main()
