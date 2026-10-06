from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


SEED = 42
MAX_EPOCHS = 300
PATIENCE = 50
HIDDEN_CHANNELS = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network."""

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
            cached=True,
            normalize=True,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            output_channels,
            cached=True,
            normalize=True,
        )

        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
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
    """Calculate node-classification accuracy for a given mask."""

    predictions = logits.argmax(dim=1)

    correct = predictions[mask].eq(labels[mask]).sum().item()
    total = int(mask.sum().item())

    if total == 0:
        return 0.0

    return correct / total


def train_one_epoch(
    model: GCN,
    data,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train the model for one epoch."""

    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask],
    )

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: GCN,
    data,
) -> tuple[float, float, float]:
    """Evaluate train, validation, and test accuracy."""

    model.eval()

    logits = model(data.x, data.edge_index)

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


def save_history(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save training history as a CSV file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "loss",
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
    output_dir = project_dir / "outputs" / "centralized_cora"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_dir / "best_gcn_cora.pt"
    history_path = output_dir / "training_history.csv"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Centralized GCN training on Cora")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    dataset = Planetoid(
        root=str(dataset_dir),
        name="Cora",
        split="public",
        transform=NormalizeFeatures(),
    )

    data = dataset[0].to(device)

    model = GCN(
        input_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=dataset.num_classes,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {dataset.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Validation nodes: {int(data.val_mask.sum())}")
    print(f"Test nodes: {int(data.test_mask.sum())}")
    print("=" * 60)

    best_validation_accuracy = -1.0
    best_test_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
        )

        (
            train_accuracy,
            validation_accuracy,
            test_accuracy,
        ) = evaluate(model, data)

        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        improved = (
            validation_accuracy > best_validation_accuracy
        )

        if improved:
            best_validation_accuracy = validation_accuracy
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "test_accuracy": test_accuracy,
                    "seed": SEED,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            status = " *BEST*" if improved else ""

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Train: {train_accuracy:.4f} | "
                f"Val: {validation_accuracy:.4f} | "
                f"Test: {test_accuracy:.4f}"
                f"{status}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping activated at epoch {epoch}."
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        final_train_accuracy,
        final_validation_accuracy,
        final_test_accuracy,
    ) = evaluate(model, data)

    print("\n" + "=" * 60)
    print("Best centralized Cora result")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
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
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()