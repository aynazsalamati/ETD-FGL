from __future__ import annotations

import csv
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from baseline_suite.config import ExperimentConfig
from test_structural_backdoor import inject_clique_trigger
from train_fedavg_backdoor_cora import (
    attach_evaluation_triggers,
    build_client_graphs,
    evaluate_model,
    train_local_client,
)
from train_fedavg_cora import GCN, set_seed


StateDict = OrderedDict[str, Tensor]
LocalTrainFn = Callable[
    [
        StateDict,
        Data,
        int,
        int,
        torch.device,
        int,
    ],
    tuple[StateDict, float, int, dict[str, float]],
]
AggregateFn = Callable[
    [
        list[StateDict],
        list[int],
        StateDict,
        int,
    ],
    tuple[StateDict, dict[str, float]],
]


@dataclass
class ExperimentContext:
    project_dir: Path
    dataset: Planetoid
    full_data_cpu: Data
    clean_client_graphs_cpu: list[Data]
    attacked_client_graphs_cpu: list[Data]
    full_data: Data
    triggered_test_data: Data
    attack_victim_nodes: Tensor
    attack_metadata: dict[str, Any]
    device: torch.device


def clone_state(model: torch.nn.Module) -> StateDict:
    return OrderedDict(
        (
            key,
            value.detach().clone(),
        )
        for key, value in model.state_dict().items()
    )


def load_cora_context(
    project_dir: Path,
    config: ExperimentConfig,
) -> ExperimentContext:
    """Load the same Cora partition and structural attack used by ETD-FGL."""

    set_seed(config.seed)

    dataset_dir = project_dir / "data" / "Planetoid"
    partition_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / f"iid_stratified_{config.num_clients}_clients"
        / "client_node_indices.pt"
    )

    if not partition_path.exists():
        raise FileNotFoundError(
            "Cora partition was not found. Run partition_cora.py first:\n"
            f"{partition_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    full_data_cpu = dataset[0]
    partition_package = torch.load(
        partition_path,
        map_location="cpu",
        weights_only=False,
    )
    client_indices = partition_package["client_indices"]

    clean_client_graphs_cpu = build_client_graphs(
        full_data=full_data_cpu,
        client_indices=client_indices,
    )

    clean_malicious_client = clean_client_graphs_cpu[
        config.malicious_client_id
    ]

    attacked_client, attack_metadata = inject_clique_trigger(
        data=clean_malicious_client,
        target_class=config.target_class,
        poison_rate=config.poison_rate,
        trigger_size=config.trigger_size,
        seed=config.seed,
    )

    attacked_client_graphs_cpu = list(clean_client_graphs_cpu)
    attacked_client_graphs_cpu[
        config.malicious_client_id
    ] = attacked_client

    attack_victim_mask = (
        full_data_cpu.test_mask
        & full_data_cpu.y.ne(config.target_class)
    )
    attack_victim_nodes_cpu = attack_victim_mask.nonzero(
        as_tuple=False
    ).view(-1)

    trigger_feature_prototype = clean_malicious_client.x.mean(dim=0)

    triggered_test_data_cpu = attach_evaluation_triggers(
        data=full_data_cpu,
        victim_nodes=attack_victim_nodes_cpu,
        trigger_size=config.trigger_size,
        feature_prototype=trigger_feature_prototype,
        target_class=config.target_class,
    )

    return ExperimentContext(
        project_dir=project_dir,
        dataset=dataset,
        full_data_cpu=full_data_cpu,
        clean_client_graphs_cpu=clean_client_graphs_cpu,
        attacked_client_graphs_cpu=attacked_client_graphs_cpu,
        full_data=full_data_cpu.to(device),
        triggered_test_data=triggered_test_data_cpu.to(device),
        attack_victim_nodes=attack_victim_nodes_cpu.to(device),
        attack_metadata=attack_metadata,
        device=device,
    )


def standard_local_train(
    global_state: StateDict,
    client_data: Data,
    input_channels: int,
    output_channels: int,
    device: torch.device,
    round_number: int,
) -> tuple[StateDict, float, int, dict[str, float]]:
    del round_number
    local_state, local_loss, train_count = train_local_client(
        global_state=global_state,
        client_data=client_data,
        input_channels=input_channels,
        output_channels=output_channels,
        device=device,
    )
    return local_state, local_loss, train_count, {}


def sample_size_fedavg(
    local_states: list[StateDict],
    client_counts: list[int],
    global_state: StateDict,
    round_number: int,
) -> tuple[StateDict, dict[str, float]]:
    del global_state, round_number

    if not local_states:
        raise ValueError("No local states were supplied.")
    if len(local_states) != len(client_counts):
        raise ValueError("State/count lengths do not match.")

    total = float(sum(client_counts))
    if total <= 0:
        raise ValueError("Total client sample count must be positive.")

    averaged: StateDict = OrderedDict()

    for key in local_states[0].keys():
        reference = local_states[0][key]

        if not torch.is_floating_point(reference):
            averaged[key] = reference.clone()
            continue

        output = torch.zeros_like(reference)
        for state, count in zip(
            local_states,
            client_counts,
            strict=True,
        ):
            output.add_(state[key], alpha=float(count) / total)
        averaged[key] = output

    return averaged, {}


def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not history:
        raise ValueError("History is empty.")

    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(history)


def run_federated_baseline(
    *,
    method_name: str,
    output_slug: str,
    config: ExperimentConfig,
    client_graphs_cpu: list[Data],
    context: ExperimentContext,
    local_train_fn: LocalTrainFn = standard_local_train,
    aggregate_fn: AggregateFn = sample_size_fedavg,
    method_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one baseline under the exact ETD-FGL attack/evaluation setup."""

    output_dir = (
        context.project_dir
        / "outputs"
        / "federated_cora"
        / output_slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "training_history.csv"
    model_path = output_dir / "best_model.pt"
    summary_path = output_dir / "final_summary.json"

    client_graphs = [
        graph.to(context.device)
        for graph in client_graphs_cpu
    ]

    global_model = GCN(
        input_channels=context.dataset.num_features,
        hidden_channels=config.hidden_channels,
        output_channels=context.dataset.num_classes,
        dropout=config.dropout,
    ).to(context.device)

    print("=" * 96)
    print(f"{method_name} on attacked federated Cora")
    print("=" * 96)
    print(f"Device:             {context.device}")
    print(f"Clients:            {config.num_clients}")
    print(f"Malicious client:   {config.malicious_client_id}")
    print(f"Poison rate:        {config.poison_rate:.2f}")
    print(f"Trigger size:       {config.trigger_size}")
    print(f"Target class:       {config.target_class}")
    print("-" * 96)

    best_validation = -1.0
    best_round = 0
    rounds_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for round_number in range(1, config.max_rounds + 1):
        global_state = clone_state(global_model)

        local_states: list[StateDict] = []
        client_counts: list[int] = []
        client_losses: list[float] = []
        local_diagnostics: list[dict[str, float]] = []

        for client_id, client_data in enumerate(client_graphs):
            (
                local_state,
                local_loss,
                train_count,
                diagnostics,
            ) = local_train_fn(
                global_state,
                client_data,
                context.dataset.num_features,
                context.dataset.num_classes,
                context.device,
                round_number,
            )
            local_states.append(local_state)
            client_counts.append(train_count)
            client_losses.append(local_loss)
            local_diagnostics.append(diagnostics)

        averaged_state, aggregation_diagnostics = aggregate_fn(
            local_states,
            client_counts,
            global_state,
            round_number,
        )
        global_model.load_state_dict(averaged_state)

        metrics = evaluate_model(
            model=global_model,
            clean_data=context.full_data,
            triggered_data=context.triggered_test_data,
            attack_victim_nodes=context.attack_victim_nodes,
            target_class=config.target_class,
        )

        total_count = float(sum(client_counts))
        mean_client_loss = sum(
            loss * count
            for loss, count in zip(
                client_losses,
                client_counts,
                strict=True,
            )
        ) / total_count

        row: dict[str, Any] = {
            "round": round_number,
            "mean_client_loss": mean_client_loss,
            **metrics,
            **aggregation_diagnostics,
        }

        for client_id, diagnostics in enumerate(local_diagnostics):
            for key, value in diagnostics.items():
                row[f"client_{client_id}_{key}"] = value

        history.append(row)

        improved = metrics["validation_accuracy"] > best_validation
        if improved:
            best_validation = metrics["validation_accuracy"]
            best_round = round_number
            rounds_without_improvement = 0
            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": global_model.state_dict(),
                    "metrics": metrics,
                    "method_name": method_name,
                    "output_slug": output_slug,
                    "config": asdict(config),
                    "attack_metadata": context.attack_metadata,
                    "method_metadata": method_metadata or {},
                },
                model_path,
            )
        else:
            rounds_without_improvement += 1

        if round_number == 1 or round_number % 5 == 0 or improved:
            status = " *BEST*" if improved else ""
            print(
                f"Round {round_number:03d} | "
                f"Loss {mean_client_loss:.4f} | "
                f"Val {metrics['validation_accuracy']:.4f} | "
                f"Test {metrics['test_accuracy']:.4f} | "
                f"ASR {metrics['attack_success_rate']:.4f} | "
                f"Effect {metrics['attack_effect']:+.4f}{status}"
            )

        if rounds_without_improvement >= config.patience:
            print(f"Early stopping at round {round_number}.")
            break

    elapsed_seconds = time.perf_counter() - start_time
    save_history(history, history_path)

    checkpoint = torch.load(
        model_path,
        map_location=context.device,
        weights_only=False,
    )
    global_model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate_model(
        model=global_model,
        clean_data=context.full_data,
        triggered_data=context.triggered_test_data,
        attack_victim_nodes=context.attack_victim_nodes,
        target_class=config.target_class,
    )

    summary = {
        "method": method_name,
        "output_slug": output_slug,
        "best_round": best_round,
        "final_metrics": final_metrics,
        "elapsed_seconds": elapsed_seconds,
        "config": asdict(config),
        "method_metadata": method_metadata or {},
        "model_path": str(model_path),
        "history_path": str(history_path),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("-" * 96)
    print(f"Best round:          {best_round}")
    print(f"Clean test accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"ASR:                 {final_metrics['attack_success_rate']:.4f}")
    print(f"Trigger effect:      {final_metrics['attack_effect']:+.4f}")
    print(f"Runtime:             {elapsed_seconds:.2f} s")
    print(f"Summary:             {summary_path}")
    print("=" * 96)

    return summary
