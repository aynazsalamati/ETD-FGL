from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


# ============================================================
# Experimental configuration
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    """Two-layer GCN used by the server and all clients."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels,
            cached=False,
            normalize=True,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            output_channels,
            cached=False,
            normalize=True,
        )

        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


def calculate_accuracy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> float:
    """Calculate node-classification accuracy."""

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


@torch.no_grad()
def evaluate_global_model(
    model: GCN,
    data: Data,
) -> tuple[float, float, float]:
    """Evaluate the global model on the complete Cora graph."""

    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    train_accuracy = calculate_accuracy(
        logits,
        data.y,
        data.train_mask,
    )

    validation_accuracy = calculate_accuracy(
        logits,
        data.y,
        data.val_mask,
    )

    test_accuracy = calculate_accuracy(
        logits,
        data.y,
        data.test_mask,
    )

    return (
        train_accuracy,
        validation_accuracy,
        test_accuracy,
    )


def train_local_client(
    global_state: OrderedDict[str, Tensor],
    client_data: Data,
    input_channels: int,
    output_channels: int,
    device: torch.device,
) -> tuple[OrderedDict[str, Tensor], float, int]:
    """
    Train one client from the current global model.

    Returns:
        local state dictionary,
        final local loss,
        number of local training nodes.
    """

    train_node_count = int(
        client_data.train_mask.sum().item()
    )

    if train_node_count == 0:
        raise ValueError(
            "A client has no training nodes."
        )

    local_model = GCN(
        input_channels=input_channels,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=output_channels,
        dropout=DROPOUT,
    ).to(device)

    local_model.load_state_dict(global_state)

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
            key,
            value.detach().clone(),
        )
        for key, value
        in local_model.state_dict().items()
    )

    return (
        local_state,
        final_loss,
        train_node_count,
    )


def federated_average(
    local_states: list[OrderedDict[str, Tensor]],
    client_weights: list[int],
) -> OrderedDict[str, Tensor]:
    """
    Perform sample-size-weighted Federated Averaging.

    Each client is weighted by its number of local training nodes.
    """

    if not local_states:
        raise ValueError(
            "No local states were provided."
        )

    if len(local_states) != len(client_weights):
        raise ValueError(
            "The number of states and weights must match."
        )

    total_weight = float(sum(client_weights))

    if total_weight <= 0:
        raise ValueError(
            "Total aggregation weight must be positive."
        )

    averaged_state: OrderedDict[str, Tensor] = OrderedDict()

    for parameter_name in local_states[0].keys():
        reference_tensor = local_states[0][parameter_name]

        if not torch.is_floating_point(reference_tensor):
            averaged_state[parameter_name] = (
                reference_tensor.clone()
            )
            continue

        weighted_parameter = torch.zeros_like(
            reference_tensor
        )

        for local_state, client_weight in zip(
            local_states,
            client_weights,
            strict=True,
        ):
            normalized_weight = (
                client_weight / total_weight
            )

            weighted_parameter.add_(
                local_state[parameter_name],
                alpha=normalized_weight,
            )

        averaged_state[parameter_name] = (
            weighted_parameter
        )

    return averaged_state


def build_client_graphs(
    full_data: Data,
    client_indices: dict[int, Tensor],
    device: torch.device,
) -> list[Data]:
    """Construct the induced local graph of every client."""

    client_graphs: list[Data] = []

    for client_id in range(NUM_CLIENTS):
        if client_id not in client_indices:
            raise KeyError(
                f"Missing partition for client {client_id}."
            )

        node_indices = client_indices[client_id]

        client_data = full_data.subgraph(
            node_indices
        ).to(device)

        train_count = int(
            client_data.train_mask.sum().item()
        )

        if train_count == 0:
            raise ValueError(
                f"Client {client_id} has no training nodes."
            )

        client_graphs.append(client_data)

    return client_graphs


def save_history(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save federated training history as CSV."""

    fieldnames = [
        "round",
        "mean_client_loss",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
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

    output_dir = (
        project_dir
        / "outputs"
        / "federated_cora"
        / "fedavg_clean"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "best_fedavg_cora.pt"
    history_path = output_dir / "training_history.csv"

    if not partition_path.exists():
        raise FileNotFoundError(
            "Partition file was not found:\n"
            f"{partition_path}\n"
            "Run partition_cora.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("Federated GCN training with FedAvg on Cora")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Maximum rounds: {MAX_ROUNDS}")

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

    client_indices = partition_data["client_indices"]

    client_graphs = build_client_graphs(
        full_data=full_data_cpu,
        client_indices=client_indices,
        device=device,
    )

    full_data = full_data_cpu.to(device)

    print("-" * 72)

    for client_id, client_data in enumerate(
        client_graphs
    ):
        print(
            f"Client {client_id:02d} | "
            f"Nodes: {client_data.num_nodes:4d} | "
            f"Edges: {client_data.num_edges:4d} | "
            f"Train nodes: "
            f"{int(client_data.train_mask.sum()):3d}"
        )

    print("-" * 72)

    global_model = GCN(
        input_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=dataset.num_classes,
        dropout=DROPOUT,
    ).to(device)

    best_validation_accuracy = -1.0
    best_test_accuracy = 0.0
    best_round = 0
    rounds_without_improvement = 0

    history: list[dict[str, Any]] = []

    for round_number in range(1, MAX_ROUNDS + 1):
        global_state = OrderedDict(
            (
                key,
                value.detach().clone(),
            )
            for key, value
            in global_model.state_dict().items()
        )

        local_states: list[
            OrderedDict[str, Tensor]
        ] = []

        client_weights: list[int] = []
        client_losses: list[float] = []

        for client_id, client_data in enumerate(
            client_graphs
        ):
            (
                local_state,
                local_loss,
                train_node_count,
            ) = train_local_client(
                global_state=global_state,
                client_data=client_data,
                input_channels=dataset.num_features,
                output_channels=dataset.num_classes,
                device=device,
            )

            local_states.append(local_state)
            client_weights.append(train_node_count)
            client_losses.append(local_loss)

        averaged_state = federated_average(
            local_states=local_states,
            client_weights=client_weights,
        )

        global_model.load_state_dict(
            averaged_state
        )

        (
            train_accuracy,
            validation_accuracy,
            test_accuracy,
        ) = evaluate_global_model(
            model=global_model,
            data=full_data,
        )

        weighted_client_loss = sum(
            loss * weight
            for loss, weight
            in zip(
                client_losses,
                client_weights,
                strict=True,
            )
        ) / sum(client_weights)

        history.append(
            {
                "round": round_number,
                "mean_client_loss": weighted_client_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        improved = (
            validation_accuracy
            > best_validation_accuracy
        )

        if improved:
            best_validation_accuracy = (
                validation_accuracy
            )

            best_test_accuracy = test_accuracy
            best_round = round_number
            rounds_without_improvement = 0

            torch.save(
                {
                    "round": round_number,
                    "model_state_dict": (
                        global_model.state_dict()
                    ),
                    "validation_accuracy": (
                        validation_accuracy
                    ),
                    "test_accuracy": test_accuracy,
                    "seed": SEED,
                    "num_clients": NUM_CLIENTS,
                    "local_epochs": LOCAL_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "dropout": DROPOUT,
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
                " *BEST*" if improved else ""
            )

            print(
                f"Round {round_number:03d} | "
                f"Loss: {weighted_client_loss:.4f} | "
                f"Train: {train_accuracy:.4f} | "
                f"Val: {validation_accuracy:.4f} | "
                f"Test: {test_accuracy:.4f}"
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
        weights_only=True,
    )

    global_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        final_train_accuracy,
        final_validation_accuracy,
        final_test_accuracy,
    ) = evaluate_global_model(
        model=global_model,
        data=full_data,
    )

    print("\n" + "=" * 72)
    print("Best clean FedAvg result on Cora")
    print("=" * 72)
    print(f"Best round: {best_round}")
    print(
        f"Train accuracy:      "
        f"{final_train_accuracy:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{final_validation_accuracy:.4f}"
    )
    print(
        f"Test accuracy:       "
        f"{final_test_accuracy:.4f}"
    )
    print(
        f"Model saved to:      {model_path}"
    )
    print(
        f"History saved to:    {history_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()