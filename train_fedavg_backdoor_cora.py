from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from test_structural_backdoor import (
    inject_clique_trigger,
    normalize_feature_vector,
)
from train_fedavg_cora import (
    GCN,
    federated_average,
    set_seed,
)


# ============================================================
# General configuration
# ============================================================

SEED = 42
NUM_CLIENTS = 5

MAX_ROUNDS = 100
LOCAL_EPOCHS = 5
PATIENCE = 20

HIDDEN_CHANNELS = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5


# ============================================================
# Attack configuration
# ============================================================

MALICIOUS_CLIENT_ID = 0
TARGET_CLASS = 0
POISON_RATE = 0.20
TRIGGER_SIZE = 3


def calculate_accuracy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> float:
    """Calculate classification accuracy for masked nodes."""

    total = int(mask.sum().item())

    if total == 0:
        return 0.0

    predictions = logits.argmax(dim=1)

    correct = int(
        predictions[mask]
        .eq(labels[mask])
        .sum()
        .item()
    )

    return correct / total


def extend_mask(
    mask: Tensor,
    new_node_count: int,
) -> Tensor:
    """Append False entries for newly created trigger nodes."""

    extension = torch.zeros(
        new_node_count,
        dtype=torch.bool,
        device=mask.device,
    )

    return torch.cat(
        [mask.clone(), extension],
        dim=0,
    )


def attach_evaluation_triggers(
    data: Data,
    victim_nodes: Tensor,
    trigger_size: int,
    feature_prototype: Tensor,
    target_class: int,
) -> Data:
    """
    Attach one clique trigger to every evaluation victim node.

    Victim labels are not changed. This graph is used only to
    calculate attack success rate.
    """

    if victim_nodes.numel() == 0:
        raise ValueError(
            "No evaluation victim nodes were provided."
        )

    if trigger_size < 2:
        raise ValueError(
            "Trigger size must be at least 2."
        )

    attacked_data = data.clone()

    original_num_nodes = int(data.num_nodes)
    victim_count = int(victim_nodes.numel())

    trigger_node_count = (
        victim_count * trigger_size
    )

    feature_prototype = normalize_feature_vector(
        feature_prototype
    )

    trigger_features = feature_prototype.repeat(
        trigger_node_count,
        1,
    )

    trigger_labels = torch.full(
        (trigger_node_count,),
        fill_value=target_class,
        dtype=data.y.dtype,
        device=data.y.device,
    )

    attacked_data.x = torch.cat(
        [
            data.x.clone(),
            trigger_features,
        ],
        dim=0,
    )

    attacked_data.y = torch.cat(
        [
            data.y.clone(),
            trigger_labels,
        ],
        dim=0,
    )

    attacked_data.train_mask = extend_mask(
        data.train_mask,
        trigger_node_count,
    )

    attacked_data.val_mask = extend_mask(
        data.val_mask,
        trigger_node_count,
    )

    attacked_data.test_mask = extend_mask(
        data.test_mask,
        trigger_node_count,
    )

    new_edges: list[tuple[int, int]] = []

    for victim_index, victim_node in enumerate(
        victim_nodes.tolist()
    ):
        trigger_start = (
            original_num_nodes
            + victim_index * trigger_size
        )

        trigger_nodes = list(
            range(
                trigger_start,
                trigger_start + trigger_size,
            )
        )

        # Connect victim node to all trigger nodes.
        for trigger_node in trigger_nodes:
            new_edges.append(
                (victim_node, trigger_node)
            )

            new_edges.append(
                (trigger_node, victim_node)
            )

        # Create a bidirectional clique.
        for first_index in range(trigger_size):
            for second_index in range(
                first_index + 1,
                trigger_size,
            ):
                first_node = trigger_nodes[
                    first_index
                ]

                second_node = trigger_nodes[
                    second_index
                ]

                new_edges.append(
                    (first_node, second_node)
                )

                new_edges.append(
                    (second_node, first_node)
                )

    added_edge_index = torch.tensor(
        new_edges,
        dtype=torch.long,
        device=data.edge_index.device,
    ).t().contiguous()

    attacked_data.edge_index = torch.cat(
        [
            data.edge_index.clone(),
            added_edge_index,
        ],
        dim=1,
    )

    assert (
        attacked_data.num_nodes
        == original_num_nodes + trigger_node_count
    )

    return attacked_data


def build_client_graphs(
    full_data: Data,
    client_indices: dict[int, Tensor],
) -> list[Data]:
    """Create clean induced subgraphs for all clients."""

    client_graphs: list[Data] = []

    for client_id in range(NUM_CLIENTS):
        if client_id not in client_indices:
            raise KeyError(
                f"Partition for client {client_id} "
                "was not found."
            )

        client_data = full_data.subgraph(
            client_indices[client_id]
        )

        if int(client_data.train_mask.sum()) == 0:
            raise ValueError(
                f"Client {client_id} has no "
                "training nodes."
            )

        client_graphs.append(client_data)

    return client_graphs


def train_local_client(
    global_state: OrderedDict[str, Tensor],
    client_data: Data,
    input_channels: int,
    output_channels: int,
    device: torch.device,
) -> tuple[OrderedDict[str, Tensor], float, int]:
    """Train one benign or malicious local client."""

    local_model = GCN(
        input_channels=input_channels,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=output_channels,
        dropout=DROPOUT,
    ).to(device)

    local_model.load_state_dict(
        global_state
    )

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_node_count = int(
        client_data.train_mask.sum().item()
    )

    final_loss = 0.0

    for _ in range(LOCAL_EPOCHS):
        local_model.train()
        optimizer.zero_grad()

        logits = local_model(
            client_data.x,
            client_data.edge_index,
        )

        loss = F.cross_entropy(
            logits[client_data.train_mask],
            client_data.y[client_data.train_mask],
        )

        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())

    local_state = OrderedDict(
        (
            parameter_name,
            parameter_value.detach().clone(),
        )
        for parameter_name, parameter_value
        in local_model.state_dict().items()
    )

    return (
        local_state,
        final_loss,
        train_node_count,
    )


@torch.no_grad()
def evaluate_model(
    model: GCN,
    clean_data: Data,
    triggered_data: Data,
    attack_victim_nodes: Tensor,
    target_class: int,
) -> dict[str, float]:
    """
    Evaluate clean performance and structural backdoor success.
    """

    model.eval()

    clean_logits = model(
        clean_data.x,
        clean_data.edge_index,
    )

    triggered_logits = model(
        triggered_data.x,
        triggered_data.edge_index,
    )

    clean_predictions = clean_logits.argmax(
        dim=1
    )

    triggered_predictions = (
        triggered_logits.argmax(dim=1)
    )

    train_accuracy = calculate_accuracy(
        clean_logits,
        clean_data.y,
        clean_data.train_mask,
    )

    validation_accuracy = calculate_accuracy(
        clean_logits,
        clean_data.y,
        clean_data.val_mask,
    )

    test_accuracy = calculate_accuracy(
        clean_logits,
        clean_data.y,
        clean_data.test_mask,
    )

    clean_target_rate = float(
        clean_predictions[
            attack_victim_nodes
        ]
        .eq(target_class)
        .float()
        .mean()
        .item()
    )

    attack_success_rate = float(
        triggered_predictions[
            attack_victim_nodes
        ]
        .eq(target_class)
        .float()
        .mean()
        .item()
    )

    attack_effect = (
        attack_success_rate
        - clean_target_rate
    )

    return {
        "train_accuracy": train_accuracy,
        "validation_accuracy": (
            validation_accuracy
        ),
        "test_accuracy": test_accuracy,
        "clean_target_rate": (
            clean_target_rate
        ),
        "attack_success_rate": (
            attack_success_rate
        ),
        "attack_effect": attack_effect,
    }


def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save round-level results to a CSV file."""

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

    dataset_dir = (
        project_dir
        / "data"
        / "Planetoid"
    )

    partition_path = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "partitions"
        / f"iid_stratified_{NUM_CLIENTS}_clients"
        / "client_node_indices.pt"
    )

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "fedavg_structural_backdoor"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "best_backdoored_fedavg_cora.pt"
    )

    history_path = (
        output_dir
        / "training_history.csv"
    )

    if not partition_path.exists():
        raise FileNotFoundError(
            "Partition file was not found. "
            "Run partition_cora.py first."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    full_data_cpu = dataset[0]

    partition_data = torch.load(
        partition_path,
        map_location="cpu",
        weights_only=False,
    )

    client_indices = partition_data[
        "client_indices"
    ]

    clean_client_graphs_cpu = (
        build_client_graphs(
            full_data=full_data_cpu,
            client_indices=client_indices,
        )
    )

    # --------------------------------------------------------
    # Create malicious training client
    # --------------------------------------------------------

    clean_malicious_client = (
        clean_client_graphs_cpu[
            MALICIOUS_CLIENT_ID
        ]
    )

    (
        attacked_client_data,
        attack_metadata,
    ) = inject_clique_trigger(
        data=clean_malicious_client,
        target_class=TARGET_CLASS,
        poison_rate=POISON_RATE,
        trigger_size=TRIGGER_SIZE,
        seed=SEED,
    )

    client_graphs_cpu = list(
        clean_client_graphs_cpu
    )

    client_graphs_cpu[
        MALICIOUS_CLIENT_ID
    ] = attacked_client_data

    client_graphs = [
        client_data.to(device)
        for client_data in client_graphs_cpu
    ]

    # --------------------------------------------------------
    # Build triggered test graph
    # --------------------------------------------------------

    attack_victim_mask = (
        full_data_cpu.test_mask
        & full_data_cpu.y.ne(TARGET_CLASS)
    )

    attack_victim_nodes_cpu = (
        attack_victim_mask.nonzero(
            as_tuple=False
        ).view(-1)
    )

    trigger_feature_prototype = (
        clean_malicious_client.x.mean(
            dim=0
        )
    )

    triggered_test_data_cpu = (
        attach_evaluation_triggers(
            data=full_data_cpu,
            victim_nodes=(
                attack_victim_nodes_cpu
            ),
            trigger_size=TRIGGER_SIZE,
            feature_prototype=(
                trigger_feature_prototype
            ),
            target_class=TARGET_CLASS,
        )
    )

    full_data = full_data_cpu.to(device)

    triggered_test_data = (
        triggered_test_data_cpu.to(device)
    )

    attack_victim_nodes = (
        attack_victim_nodes_cpu.to(device)
    )

    print("=" * 76)
    print(
        "FedAvg training under structural "
        "backdoor attack on Cora"
    )
    print("=" * 76)
    print(f"Device:                  {device}")
    print(f"Seed:                    {SEED}")
    print(f"Clients:                 {NUM_CLIENTS}")
    print(
        f"Malicious client:        "
        f"{MALICIOUS_CLIENT_ID}"
    )
    print(
        f"Target class:            "
        f"{TARGET_CLASS}"
    )
    print(
        f"Local poisoning rate:    "
        f"{POISON_RATE:.2f}"
    )
    print(
        f"Trigger size:            "
        f"{TRIGGER_SIZE}"
    )
    print(
        f"Poisoned train victims:  "
        f"{attack_metadata['poisoned_victim_count']}"
    )
    print(
        f"ASR evaluation nodes:    "
        f"{attack_victim_nodes_cpu.numel()}"
    )
    print(
        f"Triggered test nodes:    "
        f"{triggered_test_data_cpu.num_nodes}"
    )
    print(
        f"Triggered test edges:    "
        f"{triggered_test_data_cpu.num_edges}"
    )
    print("-" * 76)

    for client_id, client_data in enumerate(
        client_graphs
    ):
        state = (
            "MALICIOUS"
            if client_id
            == MALICIOUS_CLIENT_ID
            else "BENIGN"
        )

        print(
            f"Client {client_id:02d} | "
            f"{state:9s} | "
            f"Nodes: {client_data.num_nodes:4d} | "
            f"Edges: {client_data.num_edges:4d} | "
            f"Train: "
            f"{int(client_data.train_mask.sum()):3d}"
        )

    print("-" * 76)

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

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):
        global_state = OrderedDict(
            (
                parameter_name,
                parameter_value.detach().clone(),
            )
            for parameter_name, parameter_value
            in global_model.state_dict().items()
        )

        local_states: list[
            OrderedDict[str, Tensor]
        ] = []

        client_weights: list[int] = []
        client_losses: list[float] = []

        for client_data in client_graphs:
            (
                local_state,
                local_loss,
                train_node_count,
            ) = train_local_client(
                global_state=global_state,
                client_data=client_data,
                input_channels=(
                    dataset.num_features
                ),
                output_channels=(
                    dataset.num_classes
                ),
                device=device,
            )

            local_states.append(local_state)
            client_weights.append(
                train_node_count
            )
            client_losses.append(
                local_loss
            )

        averaged_state = federated_average(
            local_states=local_states,
            client_weights=client_weights,
        )

        global_model.load_state_dict(
            averaged_state
        )

        metrics = evaluate_model(
            model=global_model,
            clean_data=full_data,
            triggered_data=(
                triggered_test_data
            ),
            attack_victim_nodes=(
                attack_victim_nodes
            ),
            target_class=TARGET_CLASS,
        )

        mean_client_loss = sum(
            loss * weight
            for loss, weight in zip(
                client_losses,
                client_weights,
                strict=True,
            )
        ) / sum(client_weights)

        history_row = {
            "round": round_number,
            "mean_client_loss": (
                mean_client_loss
            ),
            **metrics,
        }

        history.append(history_row)

        improved = (
            metrics["validation_accuracy"]
            > best_validation_accuracy
        )

        if improved:
            best_validation_accuracy = (
                metrics[
                    "validation_accuracy"
                ]
            )

            best_round = round_number
            rounds_without_improvement = 0

            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": (
                        global_model.state_dict()
                    ),
                    "metrics": metrics,
                    "attack_metadata": (
                        attack_metadata
                    ),
                    "seed": SEED,
                    "num_clients": (
                        NUM_CLIENTS
                    ),
                    "malicious_client_id": (
                        MALICIOUS_CLIENT_ID
                    ),
                    "target_class": (
                        TARGET_CLASS
                    ),
                    "poison_rate": (
                        POISON_RATE
                    ),
                    "trigger_size": (
                        TRIGGER_SIZE
                    ),
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
            status = (
                " *BEST*"
                if improved
                else ""
            )

            print(
                f"Round {round_number:03d} | "
                f"Loss: {mean_client_loss:.4f} | "
                f"Val: "
                f"{metrics['validation_accuracy']:.4f} | "
                f"Test: "
                f"{metrics['test_accuracy']:.4f} | "
                f"CleanTarget: "
                f"{metrics['clean_target_rate']:.4f} | "
                f"ASR: "
                f"{metrics['attack_success_rate']:.4f} | "
                f"Effect: "
                f"{metrics['attack_effect']:+.4f}"
                f"{status}"
            )

        if (
            rounds_without_improvement
            >= PATIENCE
        ):
            print(
                "\nEarly stopping activated "
                f"at round {round_number}."
            )
            break

    save_history(
        history=history,
        output_path=history_path,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    global_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate_model(
        model=global_model,
        clean_data=full_data,
        triggered_data=triggered_test_data,
        attack_victim_nodes=(
            attack_victim_nodes
        ),
        target_class=TARGET_CLASS,
    )

    print("\n" + "=" * 76)
    print(
        "Best FedAvg result under "
        "structural backdoor attack"
    )
    print("=" * 76)
    print(f"Best round:             {best_round}")
    print(
        f"Train accuracy:         "
        f"{final_metrics['train_accuracy']:.4f}"
    )
    print(
        f"Validation accuracy:    "
        f"{final_metrics['validation_accuracy']:.4f}"
    )
    print(
        f"Clean test accuracy:    "
        f"{final_metrics['test_accuracy']:.4f}"
    )
    print(
        f"Clean target rate:      "
        f"{final_metrics['clean_target_rate']:.4f}"
    )
    print(
        f"Attack success rate:    "
        f"{final_metrics['attack_success_rate']:.4f}"
    )
    print(
        f"Trigger attack effect:  "
        f"{final_metrics['attack_effect']:+.4f}"
    )
    print(f"Model saved to:         {model_path}")
    print(f"History saved to:       {history_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()