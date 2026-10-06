from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from multidata_suite.config import DatasetConfig, get_config
from multidata_suite.core import build_context, fedavg, run_method, standard_local_train
from multidata_suite.defenses import (
    ConfigurableETDFGLAggregator,
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

DATASETS = ("cora", "pubmed", "reddit")
ALL_METHODS = (
    "clean",
    "attacked",
    "pdfl",
    "gsp",
    "flpurifier",
    "dmgnn",
    "fedtge",
    "etdfgl",
)
METHOD_LABELS = {
    "clean": "Clean FedAvg",
    "attacked": "Attacked FedAvg",
    "pdfl": "PD-FL",
    "gsp": "GSP-FL",
    "flpurifier": "FLPurifier-GNN",
    "dmgnn": "DMGNN-FL",
    "fedtge": "FedTGE",
    "etdfgl": "ETD-FGL",
}


def _tag(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_exists(context, slug: str) -> bool:
    return (context.base_output / slug / "final_summary.json").exists()


def _execute(
    context,
    *,
    slug: str,
    method_name: str,
    force: bool,
    **kwargs,
) -> dict:
    summary_path = context.base_output / slug / "final_summary.json"
    if summary_path.exists() and not force:
        print(f"SKIP {slug}: existing summary found")
        return _load_summary(summary_path)
    return run_method(
        context=context,
        output_slug=slug,
        method_name=method_name,
        **kwargs,
    )


def _load_checkpoint_model(context, slug: str) -> GCN:
    path = context.base_output / slug / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Required checkpoint was not found: {path}")
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


def _calibrate_pdfl(context) -> PDFLAggregator:
    model = GCN(
        context.num_features,
        context.config.hidden_channels,
        context.num_classes,
        context.config.dropout,
    ).to(context.device)
    graphs = [graph.to(context.device) for graph in context.clean_graphs_cpu]
    features: list[np.ndarray] = []

    # Four clean calibration rounds, matching the main multi-dataset pipeline.
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


def _ensure_attacked(context, force: bool = False) -> dict:
    return _execute(
        context,
        slug="fedavg_structural_backdoor",
        method_name="Attacked FedAvg",
        force=force,
        client_graphs_cpu=context.attacked_graphs_cpu,
        aggregate_fn=fedavg,
    )


def _run_selected_methods(
    context,
    methods: Iterable[str],
    *,
    force: bool,
) -> list[dict]:
    requested = list(dict.fromkeys(methods))
    invalid = [method for method in requested if method not in ALL_METHODS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")

    summaries: list[dict] = []

    if "clean" in requested:
        summaries.append(
            _execute(
                context,
                slug="fedavg_clean",
                method_name="Clean FedAvg",
                force=force,
                client_graphs_cpu=context.clean_graphs_cpu,
                aggregate_fn=fedavg,
            )
        )

    # All defenses are evaluated under the same attacked setting and several
    # defenses use the attacked FedAvg checkpoint. Create it once as a shared
    # dependency even if it is not explicitly requested for reporting.
    need_attacked = any(
        method in requested
        for method in ("attacked", "pdfl", "gsp", "flpurifier", "dmgnn", "fedtge", "etdfgl")
    )
    if need_attacked:
        attacked_summary = _ensure_attacked(context, force=force)
        if "attacked" in requested:
            summaries.append(attacked_summary)

    attacked_model = None
    if any(method in requested for method in ("dmgnn", "etdfgl")):
        attacked_model = _load_checkpoint_model(context, "fedavg_structural_backdoor")

    if "pdfl" in requested:
        pdfl = _calibrate_pdfl(context)
        summaries.append(
            _execute(
                context,
                slug="baseline_pdfl",
                method_name="PD-FL",
                force=force,
                client_graphs_cpu=context.attacked_graphs_cpu,
                aggregate_fn=pdfl,
                method_metadata={
                    "doi": "10.1016/j.cose.2023.103557",
                    "type": "GCN federated adaptation",
                },
            )
        )

    if "gsp" in requested:
        gsp_graphs = [purify_gsp(graph) for graph in context.attacked_graphs_cpu]
        summaries.append(
            _execute(
                context,
                slug="baseline_gsp",
                method_name="GSP-FL",
                force=force,
                client_graphs_cpu=gsp_graphs,
                aggregate_fn=fedavg,
                method_metadata={
                    "doi": "10.1007/s11280-025-01364-w",
                    "type": "federated adaptation",
                },
            )
        )

    if "flpurifier" in requested:
        summaries.append(
            _execute(
                context,
                slug="baseline_flpurifier",
                method_name="FLPurifier-GNN",
                force=force,
                client_graphs_cpu=context.attacked_graphs_cpu,
                local_train_fn=make_flpurifier_trainer(context.config),
                aggregate_fn=FLPurifierAggregator(),
                method_metadata={
                    "doi": "10.1109/TIFS.2024.3384846",
                    "type": "GCN adaptation",
                },
            )
        )

    if "dmgnn" in requested:
        assert attacked_model is not None
        dmgnn_graphs = [
            purify_explanation_guided(
                attacked_model,
                graph,
                target_class=context.config.target_class,
                trigger_size=context.config.trigger_size,
            )
            for graph in context.attacked_graphs_cpu
        ]
        summaries.append(
            _execute(
                context,
                slug="baseline_dmgnn",
                method_name="DMGNN-FL",
                force=force,
                client_graphs_cpu=dmgnn_graphs,
                aggregate_fn=fedavg,
                method_metadata={
                    "doi": "10.1016/j.patcog.2026.113693",
                    "type": "federated adaptation",
                },
            )
        )

    if "fedtge" in requested:
        summaries.append(
            _execute(
                context,
                slug="baseline_fedtge",
                method_name="FedTGE",
                force=force,
                client_graphs_cpu=context.attacked_graphs_cpu,
                local_train_fn=make_fedtge_trainer(context.config),
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
                        f"hidden={context.config.hidden_channels}, "
                        f"dropout={context.config.dropout})"
                    ),
                    "finch_provenance": "Sarfraz et al., CVPR 2019 (vendored parameter-free FINCH)",
                },
            )
        )

    if "etdfgl" in requested:
        assert attacked_model is not None
        aggregator = ETDFGLAggregator(
            context.clean_graphs_cpu,
            context.attacked_graphs_cpu,
            attacked_model,
            context.config,
        )
        summaries.append(
            _execute(
                context,
                slug="defended_etdfgl",
                method_name="ETD-FGL",
                force=force,
                client_graphs_cpu=context.attacked_graphs_cpu,
                aggregate_fn=aggregator,
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
        )

    return summaries


def _write_condition_metadata(base_output: Path, study: str, config: DatasetConfig, **extra) -> None:
    base_output.mkdir(parents=True, exist_ok=True)
    package = {
        "study": study,
        "config": asdict(config),
        **extra,
    }
    (base_output / "condition_metadata.json").write_text(
        json.dumps(package, indent=2),
        encoding="utf-8",
    )


def run_multiseed(project_dir: Path, datasets, seeds, methods, force: bool) -> None:
    for dataset in datasets:
        for seed in seeds:
            config = replace(get_config(dataset), seed=int(seed))
            base = (
                project_dir / "outputs" / "extended_experiments" /
                "multiseed" / dataset / f"seed_{seed}"
            )
            _write_condition_metadata(base, "multiseed", config, seed=int(seed))
            context = build_context(project_dir, config, base_output=base)
            _run_selected_methods(context, methods, force=force)


def run_noniid(project_dir: Path, datasets, alphas, methods, seed: int, force: bool) -> None:
    for dataset in datasets:
        for alpha in alphas:
            config = replace(get_config(dataset), seed=int(seed))
            base = (
                project_dir / "outputs" / "extended_experiments" /
                "noniid" / dataset / f"alpha_{_tag(alpha)}" / f"seed_{seed}"
            )
            _write_condition_metadata(
                base,
                "noniid",
                config,
                dirichlet_alpha=float(alpha),
            )
            context = build_context(
                project_dir,
                config,
                partition_strategy="dirichlet",
                dirichlet_alpha=float(alpha),
                base_output=base,
            )
            _run_selected_methods(context, methods, force=force)


def run_ablation(project_dir: Path, datasets, seed: int, force: bool) -> None:
    variants = [
        ("etdfgl_full", "ETD-FGL (Full final)", 9.0 / 13.0, 0.0, 4.0 / 13.0, True),
        ("etdfgl_no_update", "ETD-FGL Topology-only", 1.0, 0.0, 0.0, True),
        ("etdfgl_no_topology", "ETD-FGL Update-only", 0.0, 0.0, 1.0, True),
        ("etdfgl_no_trust", "ETD-FGL w/o Trust Weighting", 9.0 / 13.0, 0.0, 4.0 / 13.0, False),
        ("etdfgl_historical_three_channel", "Historical Three-Channel Variant (0.45/0.35/0.20)", 0.45, 0.35, 0.20, True),
    ]

    for dataset in datasets:
        config = replace(get_config(dataset), seed=int(seed))
        base = (
            project_dir / "outputs" / "extended_experiments" /
            "ablation" / dataset / f"seed_{seed}"
        )
        _write_condition_metadata(base, "ablation", config, seed=int(seed))
        context = build_context(project_dir, config, base_output=base)
        _ensure_attacked(context, force=force)
        attacked_model = _load_checkpoint_model(context, "fedavg_structural_backdoor")

        for slug, name, topology, explanation, update, use_trust in variants:
            aggregator = ConfigurableETDFGLAggregator(
                context.clean_graphs_cpu,
                context.attacked_graphs_cpu,
                attacked_model,
                config,
                topology_weight=topology,
                explanation_weight=explanation,
                update_weight=update,
                use_trust_weighting=use_trust,
            )
            _execute(
                context,
                slug=slug,
                method_name=name,
                force=force,
                client_graphs_cpu=context.attacked_graphs_cpu,
                aggregate_fn=aggregator,
                method_metadata={
                    "ablation": True,
                    "topology_weight": topology,
                    "explanation_weight": explanation,
                    "update_weight": update,
                    "use_trust_weighting": use_trust,
                },
            )


def run_sensitivity_poison(project_dir: Path, datasets, rates, seed: int, force: bool) -> None:
    for dataset in datasets:
        for rate in rates:
            config = replace(
                get_config(dataset),
                seed=int(seed),
                poison_rate=float(rate),
            )
            base = (
                project_dir / "outputs" / "extended_experiments" /
                "sensitivity_poison" / dataset /
                f"poison_{_tag(rate)}" / f"seed_{seed}"
            )
            _write_condition_metadata(base, "sensitivity_poison", config, poison_rate=float(rate))
            context = build_context(project_dir, config, base_output=base)
            _run_selected_methods(context, ("attacked", "etdfgl"), force=force)


def run_sensitivity_trigger(project_dir: Path, datasets, sizes, seed: int, force: bool) -> None:
    for dataset in datasets:
        for size in sizes:
            config = replace(
                get_config(dataset),
                seed=int(seed),
                trigger_size=int(size),
            )
            base = (
                project_dir / "outputs" / "extended_experiments" /
                "sensitivity_trigger" / dataset /
                f"trigger_{size}" / f"seed_{seed}"
            )
            _write_condition_metadata(base, "sensitivity_trigger", config, trigger_size=int(size))
            context = build_context(project_dir, config, base_output=base)
            _run_selected_methods(context, ("attacked", "etdfgl"), force=force)


def run_clients(project_dir: Path, datasets, client_counts, methods, seed: int, force: bool) -> None:
    for dataset in datasets:
        for count in client_counts:
            config = replace(
                get_config(dataset),
                seed=int(seed),
                num_clients=int(count),
            )
            if config.malicious_client_id >= config.num_clients:
                config = replace(config, malicious_client_id=0)
            base = (
                project_dir / "outputs" / "extended_experiments" /
                "clients" / dataset /
                f"clients_{count}" / f"seed_{seed}"
            )
            _write_condition_metadata(base, "clients", config, num_clients=int(count))
            context = build_context(project_dir, config, base_output=base)
            _run_selected_methods(context, methods, force=force)


def parse_methods(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ALL_METHODS
    methods = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = [method for method in methods if method not in ALL_METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown method(s): {', '.join(invalid)}. Valid: {', '.join(ALL_METHODS)}"
        )
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real extended ETD-FGL journal experiments without synthetic results."
    )
    parser.add_argument(
        "study",
        choices=["multiseed", "noniid", "ablation", "poison", "trigger", "clients"],
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=("attacked", "pdfl", "flpurifier", "etdfgl"),
        help="Comma-separated method keys or 'all'.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--poison-rates", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--trigger-sizes", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--client-counts", nargs="+", type=int, default=[5, 10, 20, 30, 50])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent

    if args.study == "multiseed":
        # Multi-seed is the statistical version of the main comparison; by
        # default the dedicated .bat file requests all seven methods.
        run_multiseed(project_dir, args.datasets, args.seeds, args.methods, args.force)
    elif args.study == "noniid":
        run_noniid(project_dir, args.datasets, args.alphas, args.methods, args.seed, args.force)
    elif args.study == "ablation":
        run_ablation(project_dir, args.datasets, args.seed, args.force)
    elif args.study == "poison":
        run_sensitivity_poison(project_dir, args.datasets, args.poison_rates, args.seed, args.force)
    elif args.study == "trigger":
        run_sensitivity_trigger(project_dir, args.datasets, args.trigger_sizes, args.seed, args.force)
    elif args.study == "clients":
        run_clients(project_dir, args.datasets, args.client_counts, args.methods, args.seed, args.force)

    print("=" * 108)
    print(f"EXTENDED STUDY COMPLETED: {args.study}")
    print(project_dir / "outputs" / "extended_experiments" / args.study)
    print("=" * 108)


if __name__ == "__main__":
    main()
