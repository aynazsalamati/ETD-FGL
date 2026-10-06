from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from test_structural_backdoor import inject_clique_trigger
from train_fedavg_backdoor_cora import (
    attach_evaluation_triggers,
    build_client_graphs,
    evaluate_model,
    train_local_client,
)
from train_fedavg_cora import GCN, set_seed


# ============================================================
# General configuration
# ============================================================

SEED = 42
NUM_CLIENTS = 5

MAX_ROUNDS = 100
PATIENCE = 20

HIDDEN_CHANNELS = 16
DROPOUT = 0.5


# ============================================================
# Attack configuration
# ============================================================

MALICIOUS_CLIENT_ID = 0
TARGET_CLASS = 0
POISON_RATE = 0.20
TRIGGER_SIZE = 3

EPSILON = 1e-12


def load_defense_weights(
    report_path: Path,
) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    """Load fixed pilot trust weights produced by trust fusion."""

    with report_path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        report = json.load(json_file)

    clients = report["clients"]

    client_reports: dict[int, dict[str, Any]] = {}
    weights: dict[int, float] = {}

    for client_id in range(NUM_CLIENTS):
        key = str(client_id)

        if key not in clients:
            raise KeyError(
                f"Client {client_id} is missing from the trust report."
            )

        client_report = clients[key]
        weight = float(client_report["normalized_weight"])

        if weight < 0.0:
            raise ValueError(
                f"Client {client_id} has a negative aggregation weight."
            )

        client_reports[client_id] = client_report
        weights[client_id] = weight

    total_weight = sum(weights.values())

    if total_weight <= EPSILON:
        raise ValueError("The total trust weight is zero.")

    # Re-normalize defensively in case of JSON rounding.
    weights = {
        client_id: weight / total_weight
        for client_id, weight in weights.items()
    }

    return weights, client_reports


def trust_weighted_average(
    local_states: list[OrderedDict[str, Tensor]],
    client_weights: dict[int, float],
) -> OrderedDict[str, Tensor]:
    """Aggregate client models using fixed trust-aware weights."""

    if len(local_states) != NUM_CLIENTS:
        raise ValueError(
            f"Expected {NUM_CLIENTS} local states, got {len(local_states)}."
        )

    averaged_state: OrderedDict[str, Tensor] = OrderedDict()

    for parameter_name in local_states[0].keys():
        reference_tensor = local_states[0][parameter_name]

        if not torch.is_floating_point(reference_tensor):
            averaged_state[parameter_name] = reference_tensor.clone()
            continue

        weighted_parameter = torch.zeros_like(reference_tensor)

        for client_id, local_state in enumerate(local_states):
            weighted_parameter.add_(
                local_state[parameter_name],
                alpha=float(client_weights[client_id]),
            )

        averaged_state[parameter_name] = weighted_parameter

    return averaged_state


def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save defended federated training history."""

    fieldnames = [
        "round",
        "mean_client_loss",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "clean_target_rate",
        "attack_success_rate",
        "attack_effect",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    set_seed(SEED)

    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "data" / "Planetoid"

    partition_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / f"iid_stratified_{NUM_CLIENTS}_clients"
        / "client_node_indices.pt"
    )

    trust_report_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "pilot_trust_fusion"
        / "pilot_trust_report.json"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "defended_fedavg_structural_backdoor"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_defended_fedavg_cora.pt"
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "final_summary.json"

    for required_path in [partition_path, trust_report_path]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{required_path}"
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
        MALICIOUS_CLIENT_ID
    ]

    attacked_client_data, attack_metadata = inject_clique_trigger(
        data=clean_malicious_client,
        target_class=TARGET_CLASS,
        poison_rate=POISON_RATE,
        trigger_size=TRIGGER_SIZE,
        seed=SEED,
    )

    client_graphs_cpu = list(clean_client_graphs_cpu)
    client_graphs_cpu[MALICIOUS_CLIENT_ID] = attacked_client_data

    client_graphs = [
        client_data.to(device)
        for client_data in client_graphs_cpu
    ]

    attack_victim_mask = (
        full_data_cpu.test_mask
        & full_data_cpu.y.ne(TARGET_CLASS)
    )
    attack_victim_nodes_cpu = attack_victim_mask.nonzero(
        as_tuple=False
    ).view(-1)

    trigger_feature_prototype = clean_malicious_client.x.mean(dim=0)

    triggered_test_data_cpu = attach_evaluation_triggers(
        data=full_data_cpu,
        victim_nodes=attack_victim_nodes_cpu,
        trigger_size=TRIGGER_SIZE,
        feature_prototype=trigger_feature_prototype,
        target_class=TARGET_CLASS,
    )

    full_data = full_data_cpu.to(device)
    triggered_test_data = triggered_test_data_cpu.to(device)
    attack_victim_nodes = attack_victim_nodes_cpu.to(device)

    client_weights, client_trust_reports = load_defense_weights(
        trust_report_path
    )

    print("=" * 84)
    print("Trust-weighted FedAvg under structural backdoor attack on Cora")
    print("=" * 84)
    print(f"Device:                  {device}")
    print(f"Seed:                    {SEED}")
    print(f"Clients:                 {NUM_CLIENTS}")
    print(f"Malicious client:        {MALICIOUS_CLIENT_ID}")
    print(f"Target class:            {TARGET_CLASS}")
    print(f"Poisoned train victims:  {attack_metadata['poisoned_victim_count']}")
    print(f"ASR evaluation nodes:    {attack_victim_nodes_cpu.numel()}")
    print("-" * 84)

    for client_id in range(NUM_CLIENTS):
        report = client_trust_reports[client_id]
        print(
            f"Client {client_id:02d} | "
            f"Weight: {client_weights[client_id]:.4f} | "
            f"Trust: {float(report['adjusted_trust']):.4f} | "
            f"Maliciousness: {float(report['maliciousness']):.4f} | "
            f"State: {report['state']}"
        )

    print("-" * 84)

    global_model = GCN(
        input_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=dataset.num_classes,
        dropout=DROPOUT,
    ).to(device)

    best_validation_accuracy = -1.0
    best_round = 0
    rounds_without_improvement = 0
    history: list[dict[str, Any]] = []

    for round_number in range(1, MAX_ROUNDS + 1):
        global_state = OrderedDict(
            (
                parameter_name,
                parameter_value.detach().clone(),
            )
            for parameter_name, parameter_value
            in global_model.state_dict().items()
        )

        local_states: list[OrderedDict[str, Tensor]] = []
        client_losses: list[float] = []

        for client_data in client_graphs:
            local_state, local_loss, _ = train_local_client(
                global_state=global_state,
                client_data=client_data,
                input_channels=dataset.num_features,
                output_channels=dataset.num_classes,
                device=device,
            )
            local_states.append(local_state)
            client_losses.append(local_loss)

        averaged_state = trust_weighted_average(
            local_states=local_states,
            client_weights=client_weights,
        )
        global_model.load_state_dict(averaged_state)

        metrics = evaluate_model(
            model=global_model,
            clean_data=full_data,
            triggered_data=triggered_test_data,
            attack_victim_nodes=attack_victim_nodes,
            target_class=TARGET_CLASS,
        )

        mean_client_loss = sum(
            client_losses[client_id]
            * client_weights[client_id]
            for client_id in range(NUM_CLIENTS)
        )

        history.append(
            {
                "round": round_number,
                "mean_client_loss": mean_client_loss,
                **metrics,
            }
        )

        improved = (
            metrics["validation_accuracy"]
            > best_validation_accuracy
        )

        if improved:
            best_validation_accuracy = metrics["validation_accuracy"]
            best_round = round_number
            rounds_without_improvement = 0

            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": global_model.state_dict(),
                    "metrics": metrics,
                    "attack_metadata": attack_metadata,
                    "client_weights": client_weights,
                    "trust_report_path": str(trust_report_path),
                    "seed": SEED,
                    "num_clients": NUM_CLIENTS,
                    "malicious_client_id": MALICIOUS_CLIENT_ID,
                    "target_class": TARGET_CLASS,
                    "poison_rate": POISON_RATE,
                    "trigger_size": TRIGGER_SIZE,
                },
                model_path,
            )
        else:
            rounds_without_improvement += 1

        if (
            round_number == 1
            or round_number % 5 == 0
            or improved
        ):
            status = " *BEST*" if improved else ""
            print(
                f"Round {round_number:03d} | "
                f"Loss: {mean_client_loss:.4f} | "
                f"Val: {metrics['validation_accuracy']:.4f} | "
                f"Test: {metrics['test_accuracy']:.4f} | "
                f"CleanTarget: {metrics['clean_target_rate']:.4f} | "
                f"ASR: {metrics['attack_success_rate']:.4f} | "
                f"Effect: {metrics['attack_effect']:+.4f}"
                f"{status}"
            )

        if rounds_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping activated at "
                f"round {round_number}."
            )
            break

    save_history(
        history=history,
        output_path=history_path,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    global_model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate_model(
        model=global_model,
        clean_data=full_data,
        triggered_data=triggered_test_data,
        attack_victim_nodes=attack_victim_nodes,
        target_class=TARGET_CLASS,
    )

    baseline_metrics = {
        "clean_fedavg_test_accuracy": 0.7370,
        "attacked_fedavg_clean_test_accuracy": 0.7390,
        "attacked_fedavg_asr": 0.5506,
        "attacked_fedavg_attack_effect": 0.4092,
    }

    summary = {
        "best_round": best_round,
        "final_metrics": final_metrics,
        "baseline_metrics": baseline_metrics,
        "client_weights": client_weights,
        "asr_reduction": (
            baseline_metrics["attacked_fedavg_asr"]
            - final_metrics["attack_success_rate"]
        ),
        "clean_accuracy_change": (
            final_metrics["test_accuracy"]
            - baseline_metrics["attacked_fedavg_clean_test_accuracy"]
        ),
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(summary, json_file, indent=2)

    print("\n" + "=" * 84)
    print("Best trust-weighted FedAvg result")
    print("=" * 84)
    print(f"Best round:                {best_round}")
    print(f"Train accuracy:            {final_metrics['train_accuracy']:.4f}")
    print(f"Validation accuracy:       {final_metrics['validation_accuracy']:.4f}")
    print(f"Clean test accuracy:       {final_metrics['test_accuracy']:.4f}")
    print(f"Clean target rate:         {final_metrics['clean_target_rate']:.4f}")
    print(f"Attack success rate:       {final_metrics['attack_success_rate']:.4f}")
    print(f"Trigger attack effect:     {final_metrics['attack_effect']:+.4f}")
    print(
        "ASR reduction vs attacked FedAvg: "
        f"{summary['asr_reduction']:+.4f}"
    )
    print(
        "Clean accuracy change:            "
        f"{summary['clean_accuracy_change']:+.4f}"
    )
    print(f"Model saved to:            {model_path}")
    print(f"History saved to:          {history_path}")
    print(f"Summary saved to:          {summary_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
